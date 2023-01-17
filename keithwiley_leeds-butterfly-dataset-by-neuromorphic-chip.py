from IPython.display import Image as IPyDisImg
IPyDisImg("../input/ft232h-neuroshield-4x/FT232H_NeuroShield_NeuroBricks_3_cropped_800px.jpg")
IPyDisImg("../input/ft232h-neuroshield-4x/4X FT232H_NeuroShields_2_cropped_800px.jpg")
from IPython.display import display

import os
import glob
import math
import pickle
import json
from time import sleep
from timeit import default_timer
from random import seed, randrange
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw

# The following module manages a set of pipes and interprocess communications with a set of processes
# running in terminals on the same machine. Those processes open USB/FT232H/SPI communications to one or more
# (four for this experiment) pairs of FT232H/NeuroShield boards, which contain and run NM500 neuromorphic networks.
# The module's import is commented out here, since the Kaggle environment has no knowledge
# of the module, and even if it did, there are no FT232H or NeuroShield boards physically connected to which to connect.

# Commented out since this demonstrative code cannot actually be run on Kaggle
# from Parallel_Multi_NeuroShield_Interface import *
def display_img(img, scale=1):
    try:
        if scale == 1:
            display(img)
        else:
            display(img.resize((int(round(img.size[0] * scale)), int(round(img.size[1] * scale)))))
    except NameError as e:
        print("NameError: ", e)
        # We're probably running from a shell script instead of a notebook. Just skip the display command entirely.
        pass
    except Exception as e:
        print("Exception: ", e)

def display_img_row(imgs, mode="RGB", bg_color=0x000000, border=0, scale=1, display_immediately=True):
    dim = (sum([img.size[0] for img in imgs]) + border * (len(imgs) + 1), max([img.size[1] for img in imgs]) + border * 2)
    img_row = Image.new(mode, dim, bg_color)
    
    x = border
    for i, img in enumerate(imgs):
        img_row.paste(img, (x, border))
        x += img.size[0] + border
    if display_immediately:
        display_img(img_row, scale)
    return img_row

def summarize_times(rnd, times):
    '''
    Print running time and profiling metrics
    '''
    elapsed_times = []
    for ti in range(1, len(times)):
        elapsed_times.append((times[ti][0], times[ti][1] - times[ti - 1][1]))
    elapsed_times.append(("Total", times[-1][1] - times[0][1]))
    print("Round_or_batch {} times:".format(rnd))
    desc_w = max([len(desc) for desc, et in elapsed_times])
    for desc, et in elapsed_times:
        if et < 60:
            time_str = "    {:{desc_w}}        {:>5.2f}s".format(desc, et, desc_w=desc_w)
        else:
            et_m = int(et // 60)
            et_s = et % 60
            time_str = "    {:{desc_w}} {:5}m {:>5.2f}s".format(desc, et_m, et_s, desc_w=desc_w)
        print(time_str)
def load_data(patch_wh=16, verbose_level=0):
    np.set_printoptions(edgeitems=10, linewidth=1000)

    project_dirpath = "../input/butterfly-dataset/leedsbutterfly/"
    ori_descriptions_dirpath = project_dirpath + "descriptions/"
    ori_images_dirpath = project_dirpath + "images/"
    ori_segmentations_dirpath = project_dirpath + "segmentations/"

    ori_image_filenames = os.listdir(ori_images_dirpath)
    ori_image_filenames = [file for file in ori_image_filenames if file[-3:] == 'png']
    ori_image_filenames = sorted(ori_image_filenames)

    ori_segmentation_filenames = os.listdir(ori_segmentations_dirpath)
    ori_segmentation_filenames = [file for file in ori_segmentation_filenames if file[-3:] == 'png']
    ori_segmentation_filenames = sorted(ori_segmentation_filenames)

    n_imgs_read = 0
    categorized_imgs = {}
    categories = set()
    w_min, h_min = 99999, 99999
    prev_cat = None
    for fni, fn in enumerate(ori_image_filenames):
        category = int(fn[1:3])

        new_cat = False
        if prev_cat is None or category != prev_cat:
            new_cat = True
            print("Image {:>3} of {}, Category: {}".format(fni, len(ori_image_filenames), category))
        prev_cat = category

        categories.add(category)

        num = int(fn[4:7])
        img = Image.open(ori_images_dirpath + fn)
        n_imgs_read += 1
        seg_fn = fn.replace(".png", "_seg0.png")
        seg = Image.open(ori_segmentations_dirpath + seg_fn)
        seg_arr = np.array(seg)
        seg_arr = np.where(seg_arr==1, seg_arr, 0)  # Unnecessary since we converted the image to mode 1 above

        img_masked = Image.new("RGB", img.size)
        img_masked.paste(img, mask=seg.convert("1"))

        # Find the segmentation mask bounds
        bounds = np.nonzero(seg_arr)
        bbox = (bounds[1].min(), bounds[0].min(), bounds[1].max(), bounds[0].max())

        # Crop to the segmentation mask bounds
        img_masked_crp = img_masked.crop(bbox)
        seg_crp = seg.crop(bbox)

        # The next step is to rotate the image to normalize it to a axis-aligned cardinal alignment (i.e., make the butterfly point straight up).
        # Some of the butterlies will end up pointing downwards, left, or right, but most align the correct way.
        # Somehow fixing the few erroneous alignments might have improved classification performance of course. 
        # There will be at least one mask pixel on each edge of the cropped segmentation.
        # For each edge, find the center pixel.
        # Use the pair of center pixels on opposing edges as an indication of the current rotation of the butterfly and rotate the image to normalize that pair to a horizontal (or vertical) alignment.
        seg_crp_arr = np.array(seg_crp)
        h, w = seg_crp_arr.shape
        l_pxs, r_pxs, t_pxs, b_pxs = set(), set(), set(), set()
        for y in range(seg_crp_arr.shape[0]):
            if seg_crp_arr[y][0]:
                l_pxs.add(y)
            if seg_crp_arr[y][-1]:
                r_pxs.add(y)
        for x in range(seg_crp_arr.shape[1]):
            if seg_crp_arr[0][x]:
                t_pxs.add(x)
            if seg_crp_arr[-1][x]:
                b_pxs.add(x)

        l_pxs = sorted(l_pxs)
        r_pxs = sorted(r_pxs)
        t_pxs = sorted(t_pxs)
        b_pxs = sorted(b_pxs)

        l_mid = l_pxs[0] + (l_pxs[-1] - l_pxs[0]) // 2
        r_mid = r_pxs[0] + (r_pxs[-1] - r_pxs[0]) // 2
        t_mid = t_pxs[0] + (t_pxs[-1] - t_pxs[0]) // 2
        b_mid = b_pxs[0] + (b_pxs[-1] - b_pxs[0]) // 2

        hor_dif = l_mid - r_mid
        ver_dif = t_mid - b_mid

        hor_dist = math.sqrt((l_mid - r_mid)**2 + w**2)
        ver_dist = math.sqrt((t_mid - b_mid)**2 + h**2)

        if hor_dist > ver_dist:
            hor_ang = math.atan2(hor_dif, w)
            img_masked_crp_rot = img_masked_crp
            seg_crp_rot = seg_crp
        else:
            ver_ang = math.atan2(h, ver_dif)
            img_masked_crp_rot = img_masked_crp
            seg_crp_rot = seg_crp
            
        # Find the segmentation mask bounds
        bounds = np.nonzero(seg_crp_rot)
        bbox = (bounds[1].min(), bounds[0].min(), bounds[1].max(), bounds[0].max())

        # Crop to the segmentation mask bounds
        img_masked_crp_rot_crp = img_masked_crp_rot.crop(bbox)
        seg_crp_rot_crp = seg_crp_rot.crop(bbox)

        # Normalize the size
        scale_w, scale_h = 64 / w, 64 / h
        scale = min(scale_w, scale_h)
        w_scaled = round(w * scale)
        h_scaled = round(h * scale)
        if w_scaled < patch_wh or h_scaled < patch_wh:
            print("WARNING! Cropped, rotated, cropped, resized image is thinner or shorter than the patch dimension: {}x{} < {}".format(w_scaled, h_scaled, patch_wh))
            scale =  patch_wh / min(w, h)
            w_scaled = round(w * scale)
            h_scaled = round(h * scale)
            print("Compensated image resize: {}x{}".format(w_scaled, h_scaled))
        w, h = w_scaled, h_scaled
        img_masked_crp_rot_crp_rsz = img_masked_crp_rot_crp.resize((w, h), Image.LANCZOS)
        if img_masked_crp_rot_crp_rsz.size[0] < patch_wh or img_masked_crp_rot_crp_rsz.size[1] < patch_wh:
            raise ValueError("Image is too small: {}".format(img_masked_crp_rot_crp_rsz.size()))
        seg_crp_rot_crp_rsz = seg_crp_rot_crp.resize((w, h))

        if category not in categorized_imgs:
            categorized_imgs[category] = []
        categorized_imgs[category].append((img_masked_crp_rot_crp_rsz, seg_crp_rot_crp_rsz))

        if verbose_level >= 2 or (verbose_level >= 1 and new_cat):
            display_img_row([img_masked_crp_rot_crp_rsz, seg_crp_rot_crp_rsz], bg_color=0xffffff, border=1, scale=4)

    categories = sorted(list(categories))
    
    print("Total images read: {}".format(n_imgs_read))  # len(imgs)))
    print("All categories: {}".format(categories))
    
    return categories, categorized_imgs
def create_crossfolds(categorized_imgs, crossfold_k, verbose_level=0):
    '''
    Given a set of images, divide them into three subsets (train/validation/test) based on a crossfold parameter indicating which of five folds to enact.
    The proportional sizes of the subsets will be 60%/20%/20%.
    crossfold_k: one of [0, 1, 2, 3, 4]
    '''
    if crossfold_k not in [0, 1, 2, 3, 4]:
        raise ValueError("crossfold_k must be one of [0, 1, 2, 3, 4]")
    print("crossfold_k: {}".format(crossfold_k))
    
    crossfold_shift = crossfold_k * .2
    crossfold_train_k_cutoff_range =    [round((crossfold_shift + .0) % 1, 1), min(round(crossfold_shift + .6, 1), 1.), 0., max(round(crossfold_shift - .4, 1), 0.)]
    crossfold_validate_k_cutoff_range = [round((crossfold_shift + .6) % 1, 1), round((crossfold_shift + .8) % 1.01, 1)]
    crossfold_test_k_cutoff_range =     [round((crossfold_shift + .8) % 1, 1), round((crossfold_shift + 1.) % 1.01, 1)]
    print("Crossfold cutoffs: TRAIN:", crossfold_train_k_cutoff_range, "    VALIDATE:", crossfold_validate_k_cutoff_range, "    TEST:", crossfold_test_k_cutoff_range)
    if verbose_level >= 1:
        print()

    categorized_train_imgs = {}
    categorized_validate_imgs = {}
    categorized_test_imgs = {}
    for cat in categorized_imgs:
        cutoff_train_0 =    int(round(len(categorized_imgs[cat]) * crossfold_train_k_cutoff_range[0]))
        cutoff_train_1 =    int(round(len(categorized_imgs[cat]) * crossfold_train_k_cutoff_range[1]))
        cutoff_train_2 =    int(round(len(categorized_imgs[cat]) * crossfold_train_k_cutoff_range[2]))
        cutoff_train_3 =    int(round(len(categorized_imgs[cat]) * crossfold_train_k_cutoff_range[3]))
        cutoff_validate_0 = int(round(len(categorized_imgs[cat]) * crossfold_validate_k_cutoff_range[0]))
        cutoff_validate_1 = int(round(len(categorized_imgs[cat]) * crossfold_validate_k_cutoff_range[1]))
        cutoff_test_0 =     int(round(len(categorized_imgs[cat]) * crossfold_test_k_cutoff_range[0]))
        cutoff_test_1 =     int(round(len(categorized_imgs[cat]) * crossfold_test_k_cutoff_range[1]))
        if verbose_level >= 1:
            print("Category {:>2}, Crossfold cutoffs: TRAIN: [{:>3}, {:>3}, {:>3}, {:>3}]    VALIDATE: [{:>3}, {:>3}]    TEST: [{:>3}, {:>3}]    N: {:>3}".format(
                cat, cutoff_train_0, cutoff_train_1, cutoff_train_2, cutoff_train_3, cutoff_validate_0, cutoff_validate_1, cutoff_test_0, cutoff_test_1, len(categorized_imgs[cat])))
        categorized_train_imgs[cat] =    categorized_imgs[cat][cutoff_train_0:cutoff_train_1] + categorized_imgs[cat][cutoff_train_2:cutoff_train_3]
        categorized_validate_imgs[cat] = categorized_imgs[cat][cutoff_validate_0:cutoff_validate_1]
        categorized_test_imgs[cat] =     categorized_imgs[cat][cutoff_test_0:cutoff_test_1]

    print()
    if verbose_level >= 1:
        print("Category counts ({} train, {} test):".format(sum([len(a[1]) for a in categorized_train_imgs.items()]), sum([len(a[1]) for a in categorized_test_imgs.items()])))
        print("{:>8} {:>8} {:>8} {:>8} {:>8}  {:>8}  {:>8}  {:>8}".format('CATEGORY', 'TOTAL', 'TRAIN', 'VALIDATE', 'TEST', 'TRAIN_%', 'VALID_%', 'TEST_%'))
        for cat in categorized_imgs:
            print("{:>8} {:>8} {:>8} {:>8} {:>8} {:8.2f}% {:8.2f}% {:8.2f}%".format(cat, len(categorized_imgs[cat]),
                                                            len(categorized_train_imgs[cat]),
                                                            len(categorized_validate_imgs[cat]),
                                                            len(categorized_test_imgs[cat]),
                                                            len(categorized_train_imgs[cat]) / len(categorized_imgs[cat]) * 100,
                                                            len(categorized_validate_imgs[cat]) / len(categorized_imgs[cat]) * 100,
                                                            len(categorized_test_imgs[cat]) / len(categorized_imgs[cat]) * 100))
    
    return categorized_train_imgs, categorized_validate_imgs, categorized_test_imgs
IPyDisImg("../input/butterfly-processing-images/Patch_example.png")
PATCH_IM_RGB_IDX = 0
PATCH_AR_R_IDX = 1  # Start arrays at index 1 so they can map directly onto NM500 contexts, which start at 1
PATCH_AR_G_IDX = 2
PATCH_AR_B_IDX = 3
PATCH_AR_Y_IDX = 4
PATCH_AR_CR_IDX = 5
PATCH_AR_CB_IDX = 6
PATCH_AR_S_IDX, NUM_CONTEXT_TYPES = 7, 7  # Make sure NUM_CONTEXT_TYPES is moved down here if new types are added after saturation
PATCH_IM_R_IDX = 8
PATCH_IM_G_IDX = 9
PATCH_IM_B_IDX = 10
PATCH_IM_Y_IDX = 11
PATCH_IM_CR_IDX = 12
PATCH_IM_CB_IDX = 13
PATCH_IM_S_IDX = 14
PATCH_BBUL_IDX = 15

def generate_one_patch_bbox(patch_wh, search_bounds_w, search_bounds_h, dead_zone_centers, dead_zone_size, deadzone_max_attempts):
    '''
    Generate the bounding box coordinates of a random 16x16 square within the bounds of a given image subject to restriction that a new bbox is not permitted to overlap a previously generated bbox too closely.
    The goal is to space out the random sampling of patches to loosely cover an image without too much crowding.
    
    Parameters:
        patch_wh (int): Dimensions of a patch (this is always 16).
        search_bounds_w, search_bounds_h (int): The dimensions of the image for which to produce a new bbox.
        dead_zone_centers (list of 2-tuples): List of upper-left coordinates of previously generated patches. The bbox generated during this call may not have an UL corner closer than dead_zone_size pixels from any dead_zone coordinate.
        dead_zone_size (int): Cardinal radius of exclusion around a dead_zone_center in which the UL corner of the next bbox may not reside.
        deadzone_max_attempts (int): Maximum number of attempts to make to find a bbox that doesn't violate a dead zone before giving up, presumably on the basis that the image is too small to support the requested number of patches relative to the patch-spacing restrictions.
    '''
    num_deadzone_attempts = 0
    while True:
        bbox_l = randrange(search_bounds_w) if search_bounds_w > 0 else 0
        bbox_t = randrange(search_bounds_h) if search_bounds_h > 0 else 0
        bbox_r = bbox_l + patch_wh
        bbox_b = bbox_t + patch_wh
        bbox = (bbox_l, bbox_t, bbox_r, bbox_b)

        dead = False
        for dead_zone_center in dead_zone_centers:
            if bbox_l >= dead_zone_center[0] - dead_zone_size and bbox_t >= dead_zone_center[1] - dead_zone_size and bbox_l <= dead_zone_center[0] + dead_zone_size and bbox_t <= dead_zone_center[1] + dead_zone_size:
                dead = True
                break
        if not dead:
            return bbox
        num_deadzone_attempts += 1
        if num_deadzone_attempts > deadzone_max_attempts:
            break
    return None

def generate_one_image_patches(img, seg, draw, patch_wh, patch_npxs, patch_min_coverage, num_good_patches_goal, dead_zone_centers, dead_zone_size, coverage_max_attempts, deadzone_max_attempts, transform=False, verbose_level=0):
    '''
    Given an image and its segmentation mask, generate a set of randomly sampled patches, where a patch is a small section of the image.
    For this experiment, patches are always 16x16 pixels, but that isn't a hard requirement. The only restriction is the 256-byte capacity of an NM500 pattern.
    So, for example, patches could be 32x8 pixels or any other shape that holds no more than 256 pixels.
    
    Although patches are sampled at random from the bounds of the image, there are various restrictions that can result in their disposal.
    One restriction is that patches may not overlap one another too closely. This restriction increases the likelihood that randomly sample patches will give a decent coverage of the image with minimal crowding.
    Another restriction is, given the segmentation mask of the butterfly within the image, we can exploit that a priori information to optimize our patch generation to exclude patch bounds that fall outside the segmentation mask or which fall on the border of the mask and do not include a sufficient proportion of mask pixels.
    
    For each satisfactory patch, optionally (as per a boolean parameter) generate transforms of the patch (rotations and reflections) to allow the patch to be queried against a trained network in various orientations.
    Convert each RGB patch into a set of single-band grayscale images representing various dimensional slices through the colorspace. I have currently provided red, green, blue, Y, Cr, Cb, and saturation.
    
    I didn't bother using value since that seem conceptually fairly similar to Y (luminance) and I didn't bother with hue
    because it is virtually impossible to map a circular variable such as hue onto a scalar differentiator such as the NM500's pattern difference comparator
    (which scores a neuron's pattern against a queried pattern as the summed element-wise difference between the two patterns).
    
    Parameters:
        img (PIL.Image, mode RGB)
        seg (PIL.Image, mode L)
        draw (PIL.ImageDraw)
        patch_wh (int): Dimensions of a patch (this is always 16).
        patch_npxs (int)
        patch_min_coverage (float, 0.0-1.0): Proportion of a patch which must cover the segmentation mask
        num_good_patches_goal (int): The number of patches to attempt to generate. If the image is too small relative to the spacing requirements, it may not be possible to generate this many patches.
        dead_zone_centers: see generate_one_patch_bbox()
        dead_zone_size: see generate_one_patch_bbox()
        coverage_max_attempts (int): Maximum number of attempts to make to find a bbox with sufficient coverage of the segmentation mask before giving up, presumably on the basis that the image is too small to support the requested number of patches relative to the patch-spacing restrictions.
        deadzone_max_attempts: see generate_one_patch_bbox()
        transform (bool): Indicates that a patch should be converted into multiple rotations and reflections and returned as a set of corresponding patches.
    '''
    num_good_coverage_attempts = 0
    n_good_patches = 0
    patches = []
    search_bounds_w, search_bounds_h = img.size[0] - patch_wh, img.size[1] - patch_wh
    if search_bounds_w < 0 or search_bounds_h < 0:
        raise ValueError("Negative search bounds: {} {} {} {}".format(img.size, patch_wh, search_bounds_w, search_bounds_h))
    while n_good_patches < num_good_patches_goal:
        bbox = generate_one_patch_bbox(patch_wh, search_bounds_w, search_bounds_h, dead_zone_centers, dead_zone_size, deadzone_max_attempts)
        if bbox is None:
            if verbose_level >= 2:
                print("Could not find suitable bbox")
            break

        num_black_pxs = sum(seg.crop(bbox)
                       .point(lambda x: 1 if x else 0)
                       .convert("L")
                       .point(bool)
                       .getdata())

        good_seg_overlap = num_black_pxs >= patch_npxs * patch_min_coverage

        if draw:
            red = (255 - num_black_pxs) if not good_seg_overlap else 0
            green = (num_black_pxs - 1) if good_seg_overlap else 0
            color = (0xa0 << 24) | (0x00 << 16) | (green << 8) | red
            draw.rectangle((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), None, color, 1)

        if good_seg_overlap:
            n_good_patches += 1
            dead_zone = (bbox[0] - dead_zone_size, bbox[1] - dead_zone_size, bbox[0] + dead_zone_size, bbox[1] + dead_zone_size)
            dead_zone_centers.add((bbox[0], bbox[1]))

            patch_transforms = [
                img.crop(bbox)
            ]
            if transform:
                # TODO: It would probably be faster to perform the transforms on the numpy array (see below) than on the PIL image
                patch = patch_transforms[0]
                patch_transforms.extend([
                    patch.transpose(Image.ROTATE_90),
                    patch.transpose(Image.ROTATE_180),
                    patch.transpose(Image.ROTATE_270),
                    patch.transpose(Image.FLIP_LEFT_RIGHT),
                    patch.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),
                    patch.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180),
                    patch.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270),
                ])
            
            for patch in patch_transforms:
                aRGB = np.array(patch)
                aYCC = cv2.cvtColor(aRGB, cv2.COLOR_RGB2YCR_CB)
                aHSV = cv2.cvtColor(aRGB, cv2.COLOR_RGB2HSV)
                if False:#imi % 10 == 0:
                    print("RGB/YCC min/max:    {:>3} {:>3}    {:>3} {:>3}    {:>3} {:>3}        {:>3} {:>3}    {:>3} {:>3}    {:>3} {:>3}        {:>3} {:>3}    {:>3} {:>3}    {:>3} {:>3}".format( \
                        aRGB[:,:,0].min(), aRGB[:,:,0].max(), aRGB[:,:,1].min(), aRGB[:,:,1].max(), aRGB[:,:,2].min(), aRGB[:,:,2].max(),
                        aYCC[:,:,0].min(), aYCC[:,:,0].max(), aYCC[:,:,1].min(), aYCC[:,:,1].max(), aYCC[:,:,2].min(), aYCC[:,:,2].max(),
                        aHSV[:,:,0].min(), aHSV[:,:,0].max(), aHSV[:,:,1].min(), aHSV[:,:,1].max(), aHSV[:,:,2].min(), aHSV[:,:,2].max()))
                imR, imG, imB, imY, imCr, imCb, imS = None, None, None, None, None, None, None
                if draw:
                    imR =  Image.fromarray(np.asarray(aRGB[:,:,0], dtype=np.uint8))
                    imG  = Image.fromarray(np.asarray(aRGB[:,:,1], dtype=np.uint8))
                    imB  = Image.fromarray(np.asarray(aRGB[:,:,2], dtype=np.uint8))
                    imY =  Image.fromarray(np.asarray(aYCC[:,:,0], dtype=np.uint8))
                    imCr = Image.fromarray(np.asarray(aYCC[:,:,1], dtype=np.uint8))
                    imCb = Image.fromarray(np.asarray(aYCC[:,:,2], dtype=np.uint8))
                    imS =  Image.fromarray(np.asarray(aHSV[:,:,1], dtype=np.uint8))
                patches.append((patch,
                                aRGB[:,:,0].flatten(), aRGB[:,:,1].flatten(), aRGB[:,:,2].flatten(),
                                aYCC[:,:,0].flatten(), aYCC[:,:,1].flatten(), aYCC[:,:,2].flatten(),
                                aHSV[:,:,1].flatten(),
                                imR, imG, imB,
                                imY, imCr, imCb,
                                imS,
                                (bbox[0], bbox[1])))

        num_good_coverage_attempts += 1
        if num_good_coverage_attempts >= coverage_max_attempts:
            if verbose_level >= 2:
                print("Max bbox patch attempts reached")
            break
    
    if verbose_level >= 2:
        print("Num coverage attempts, max permitted attempts: {:>3} {} {}".format(num_good_coverage_attempts, coverage_max_attempts, "" if num_good_coverage_attempts < coverage_max_attempts else "MAX ATTEMPTS REACHED!"))
        if n_good_patches != num_good_patches_goal:
            print("Num patches, goal: {:>3} {:>3} {:>3} {:>3} {}".format(n_good_patches, num_good_patches_goal, len(patches), num_good_patches_goal * (8 if transform else 1), "" if len(patches) < num_good_patches_goal * (8 if transform else 1) else "COULD NOT ACHIEVE PATCH NUM GOAL!"))
    
    return patches, dead_zone_centers
def generate_training_patches(categorized_train_imgs, patch_wh=16, verbose_level=0):
    '''
    Given a set of training images, generate a set of patches from the images and group the patches by category (i.e., label).
    There is no need to keep the patches grouped by their image of origin for training purposes.
    We only need the patches' ground truth labels for training purposes after extracting them.
    
    Parameters:
        categorized_train_imgs: A set of images, namely the train subset for a given crossfold.
    '''
    patch_npxs = patch_wh**2
    patch_min_coverage = .75  # Amount that a patch must cover the image mask in order to be used
    num_good_patches_goal = 20
    dead_zone_size = int(patch_wh * .5)  # Don't let the patches overlap too much
    coverage_max_attempts = 100
    deadzone_max_attempts = 100

    categorized_train_patches = defaultdict(list)  # Keyed by category, lists of patches ordered within category by round to steadily increase patch area overlap

    # Used a fixed random to ease debugging and comparative analytics
    seed(0)

    dead_zone_centers = {}
    num_total_patches_found = 0
    generate_patch_transforms = False

    # Generate sets of patches organized into "rounds".
    # In earlier rounds, don't let the patches overlap much to maximize spread.
    # In later rounds, let patches overlap more and more to increase coverage at the cost of overlap and redundancy.
    # In this way, the patches can be iterated in order with a presumption that they will present a wide coverage in their earlier subset and only overlap more heavily later in the set.
    for opt_round in range(4):
        opt_round_verbose_level = verbose_level if opt_round == 0 else 0

        print("dead_zone_size: {}".format(dead_zone_size))

        num_total_patches_found_one_round = 0
        for cat in categorized_train_imgs:
            if opt_round_verbose_level >= 1:
                print("Category {} has {} images".format(cat, len(categorized_train_imgs[cat])))
            categorized_train_patches_one_round_one_cat = []
            num_patches_found_histogram = Counter()
            for imi, (img, seg) in enumerate(categorized_train_imgs[cat]):
                show = opt_round_verbose_level >= 1 and imi == 0
                img2, draw = None, None
                if show:
                    img2 = img.copy()
                    draw = ImageDraw.Draw(img2, "RGBA")

                if (cat, imi) not in dead_zone_centers:
                    dead_zone_centers[(cat, imi)] = set()

                if opt_round_verbose_level >= 2:
                    print()
                
                if img.size[0] < patch_wh or img.size[1] < patch_wh:
                    raise ValueError("Image size < patch size: [{}:{}] {} {}".format(cat, imi, img.size, patch_wh))
                
                patches, dead_zone_centers_one_round = generate_one_image_patches(
                    img, seg, draw, patch_wh, patch_npxs, patch_min_coverage, num_good_patches_goal, dead_zone_centers[(cat, imi)], dead_zone_size, coverage_max_attempts,
                    deadzone_max_attempts, generate_patch_transforms, verbose_level=opt_round_verbose_level)
                dead_zone_centers[(cat, imi)].update(dead_zone_centers_one_round)

                num_patches_found_histogram[len(patches)] += 1
                categorized_train_patches_one_round_one_cat.append(patches)

                if show:
                    display_img_row([img, img2, seg], bg_color=0xffffff, border=1, scale=4)

            cat_patch_count = sum([kv[0] * kv[1] for kv in num_patches_found_histogram.most_common()])
            num_total_patches_found_one_round += cat_patch_count
            if opt_round_verbose_level >= 2:
                print("Patch count summary for this category: {:>8} total, {}".format(cat_patch_count, num_patches_found_histogram.most_common()))

            categorized_train_patches[cat].extend(categorized_train_patches_one_round_one_cat)

        num_total_patches_found += num_total_patches_found_one_round

        print("Total num patches found this round: {}".format(num_total_patches_found_one_round))
        if opt_round_verbose_level >= 1:
            print()

        dead_zone_size //= 2

    print()
    print("Total num patches found across all rounds: {}".format(num_total_patches_found))
    
    return categorized_train_patches
def convert_categorized_patches_to_dataframe(categorized_train_patches):
    '''
    Patch generation produced standard Python lists. Convert them into Pandas DataFrames.
    '''
    # The data will be split into distinct NM500 "contexts" for the various color channels, corresponding to completely isolated RBF networks
    data_prepared_train_dicts = [{}, {}, {}, {}, {}, {}, {}]

    for cat in categorized_train_patches:
        prev_ns = [len(data_prepared_train_dicts[i]) for i in range(len(data_prepared_train_dicts))]
        for patches in categorized_train_patches[cat]:
            for patch in patches:
                for context_idx, context in enumerate([
                    PATCH_AR_R_IDX,
                    PATCH_AR_G_IDX,
                    PATCH_AR_B_IDX,
                    PATCH_AR_Y_IDX,
                    PATCH_AR_CR_IDX,
                    PATCH_AR_CB_IDX,
                    PATCH_AR_S_IDX,
                ]):
                    img_arr = patch[context]
                    data_prepared_train_dicts[context_idx][len(data_prepared_train_dicts[context_idx])] = {'label': cat, 'image': img_arr}
        print("    Category {:>2} sizes (per context):".format(cat), [len(data_prepared_train_dicts[i]) - prev_ns[i] for i in range(len(data_prepared_train_dicts))])

    data_prepared_train_dfs = [pd.DataFrame.from_dict(data_prepared_dict, "index") for data_prepared_dict in data_prepared_train_dicts]
    print()
    print("Total train patches (per context):    {}".format(len(data_prepared_train_dicts[0])))
    print("Total train patches (per {} contexts): {}".format(len(data_prepared_train_dfs), len(data_prepared_train_dfs[0])))
    
    return data_prepared_train_dfs
def shuffle(df, seed=None):
    '''
    Shuffle a DataFrame's rows
    '''
    return df.reindex(np.random.RandomState(seed=seed).permutation(df.index)).reset_index().drop(columns=['index'])

def split_labels(df):
    '''
    Split a DataFrame containing rows (each row is a 256-byte pixel patch, i.e., a NM500 pattern) of mixed labels (categories) into a list of multiple DataFrames, each df containing only the rows corresponding to a single label
    '''
    df_labels = []
    for label, count in df['label'].value_counts().iteritems():
        df_label = df.loc[df['label'] == label]
        df_labels.append(df_label)
    
    return df_labels

def split_datasets(data_prepared_train_dfs, verbose_level=0):
    '''
    Given a list of DataFrames, one per context (patch color channel), shuffle its rows.
    
    Parameters:
        data_prepared_train_dfs: A list of DataFrames, one per context (color channel)
    '''
    RANDOM_SEED = 1
    data_train_labels_dfs = [[shuffle(df, RANDOM_SEED)] for df in data_prepared_train_dfs]

    n_patches_per_context = len(data_prepared_train_dfs[0])
    print("Total patches (per context, aka n_patches_per_context): {}".format(n_patches_per_context))
    
    if verbose_level >= 1:
        for context_idx in range(len(data_prepared_train_dfs)):
            if context_idx == 0:
                print("_" * 100)
                print("Context idx: {}".format(context_idx))
            df_sz_total = 0
            for data_train_label_df in data_train_labels_dfs[context_idx]:
                df_sz = len(data_train_label_df)
                df_sz_total += df_sz
                if context_idx == 0:
                    print("TRAIN LABELS N={:>5} {:>5.0f}% of train".format(df_sz, df_sz / n_patches_per_context * 100))
                a = pd.DataFrame(data_train_label_df['label'].value_counts())
                a = a.rename(columns={'label':'count'})
                a.insert(1, "%_train", [r[1]['count'] / n_patches_per_context * 100 for r in a.iterrows()], True)
                if context_idx == 0:
                    print('DF:', a)
                    print('ITERITEMS:', list(a['count'].iteritems()))
                    print()
            if context_idx == 0:
                print("Total labels: {}".format(df_sz_total))
    
    return data_train_labels_dfs, n_patches_per_context
def prepare_data(crossfold_k=0, categories=None, categorized_imgs=None):
    # The NM500 uses patterns of 256 bytes, so we will sample 256-pixel patches of 1-byte values.
    # In theory, we could use patches of nonsquare dimensions that fit within 256 pixels, say 32x8, but I did not explore that option for this experiment.
    # Instead, I only used 16x16 pixel patches. The single byte values come from various color bands, such as red/green/blue or Y/Cb/Cr, or saturation.
    # I didn't bother with hue (which is a circular variable and nearly impossible to map onto a scalar distancing function such as the NM500's) or value, which is essentially redundent with Y.
    if not categories:
        print("##### load_data " + "#" * 120)
        categories, categorized_imgs = load_data(PATCH_WH, 1)
    print("##### create_crossfolds " + "#" * 120)
    categorized_train_imgs, categorized_validate_imgs, categorized_test_imgs = create_crossfolds(categorized_imgs, crossfold_k=crossfold_k)
    print("##### generate_training_patches " + "#" * 120)
    categorized_train_patches = generate_training_patches(categorized_train_imgs, PATCH_WH, 1)
    print("##### convert_categorized_patches_to_dataframe "  + "#" * 120)
    data_prepared_train_dfs = convert_categorized_patches_to_dataframe(categorized_train_patches)
    print("##### split_datasets " + "#" * 120)
    data_train_labels_dfs, n_patches_per_context = split_datasets(data_prepared_train_dfs)
    
    return categories, categorized_imgs, categorized_validate_imgs, categorized_test_imgs, data_train_labels_dfs, n_patches_per_context

def setup_crossfold(k, force=False):
    global crossfold_k, CATEGORIES, CATEGORIZED_IMGS, categorized_validate_imgs, categorized_test_imgs, data_train_labels_dfs, n_patches_per_context
    
    if crossfold_k != k or force:
        crossfold_k = k
        CATEGORIES, CATEGORIZED_IMGS, categorized_validate_imgs, categorized_test_imgs, data_train_labels_dfs, n_patches_per_context = \
            prepare_data(crossfold_k, CATEGORIES, CATEGORIZED_IMGS)
class Parallel_Multi_NeuroShield_Interface:
    '''
    A skeleton of the true version of this class, which is imported near the top of this notebook.
    It is included here to provide the rest of the notebook dummy endpoints for various function calls.
    '''
    def __init__(self, ports):
        self.processes = []
    
    def send(self, str_list, subset_group=None):
        pass
    
    def recv(self, subset_group=None):
        return []

def connect_to_nm500_pipes():
    global nm500s
    
    ports = [
        5555,
        5556,
        5557,
        5558,
    ]

    nm500s = Parallel_Multi_NeuroShield_Interface(ports)

    nm500s.send(["set_verbose_level", 1])
    print_pipe_results(True)

    nm500s.send(["send_recv_string", "set_utilize_spi_bulk_transfer True"])
    print_pipe_results(True)

    nm500s.send(["set_verbose_level", 0])
    print_pipe_results(True)

def print_pipe_results(silent=False):
    '''
    Silent retrieval is used to retrieve and discard unneeded results. It's another way to "clear the pipes", not to be confused with clear_pipes().
    '''
    for port, r in nm500s.recv():
        if r[:len('EXCEPTION')] == 'EXCEPTION':
            print(port, "Received exception from pipe: {}".format(r))
        elif not silent:
            if r == 'None':
                print(port, "Received 'None' from pipe")
            else:
                print(port, r)

def reset_neuromem_and_interface():
    nm500s.send(["reset_neuromem"])
    print_pipe_results(True)

    nm500s.send(["reset_interface"])
    print_pipe_results(True)
def initialize_fresh_training(contexts_to_use):
    if len(contexts_to_use) > 4:
        raise ValueError("More than 4 networks (NeuroShields or contexts) requested, but only 4 are available.")
    
    verbose_level = 0
    print("set_verbose_level {}:".format(verbose_level))
    nm500s.send(["set_verbose_level", verbose_level])
    print_pipe_results(True)

    reset_neuromem_and_interface()

    data_train_labels_dfs_to_use = []
    for context_to_use, context_to_use_lbl in contexts_to_use:
        print("Num data train labels for context [{:>2} {:>2}]: {}".format(context_to_use, context_to_use_lbl, len(data_train_labels_dfs[context_to_use - 1])))
        data_train_labels_dfs_to_use.append((context_to_use, data_train_labels_dfs[context_to_use - 1]))
    print("data_train size: {} (per channel)".format([len(data_train_labels_dfs_to_use[0][1][i]) for i in range(len(data_train_labels_dfs_to_use[0][1]))], sum([len(data_train_labels_dfs_to_use[0][1][i]) for i in range(len(data_train_labels_dfs_to_use[0][1]))])))
    
    return data_train_labels_dfs_to_use

def add_patterns_to_pattern_group(nm500_idx, context, start_id, end_id, pattern_ids_stored, data_train_label_df, pattern_group, pattern_group_position_ids, verbose_level=0):
    '''
    Add patterns for one category to a group of patterns to be sent to an NM500 all at once.
    '''
    start_time = default_timer()
    
    # Send the training (learning) data
    columns = data_train_label_df.columns
    labels = data_train_label_df[columns[0]]
    rows = data_train_label_df[columns[1:]]
    
    labels_np = np.array(labels)
    rows_np = np.array(rows)
    
    if verbose_level >= 1:
        print("[{}:{}] Num rows available: {}, using rows {} - {}".format(nm500_idx, context, len(rows_np), start_id, end_id - 1))
    if end_id > len(rows_np):
        end_id = len(rows_np)
    
    pattern_group_initial_size = len(pattern_group)
    for i in range(start_id, end_id):
        label = labels_np[i]
        if (label, i) not in pattern_ids_stored:
            pattern_id = label * 1000 + i + 1
            if verbose_level >= 2:
                print("[{}:{}, {:>2}] Add pattern {} to group".format(nm500_idx, context, label, i))
            pattern_group.append((pattern_id, context, label, list(rows_np[i][0]), False))
            pattern_group_position_ids.append((label, i))
        else:
            if verbose_level >= 1:
                print("[{}:{}, {:>2}] Skipping pattern {}".format(nm500_idx, context, label, i))
    
    if verbose_level >= 1:
        print("[{}:{}] Pattern group: {:>8} prev + {:>8} added (range [{:>8}, {:>8}]) ({:>8} available) => {:>8} new".format(
            nm500_idx, context, pattern_group_initial_size, len(pattern_group) - pattern_group_initial_size, start_id, end_id, len(data_train_label_df), len(pattern_group)))
    
    if verbose_level >= 1:
        print("[{}:{}] Elapsed time to add patterns for one category to pattern group: {:8.3f}s".format(nm500_idx, context, default_timer() - start_time))

def send_training_data(nm500_idx, context, start_id, end_id, pattern_group, verbose_level=0):
    '''
    Send all patterns to the network once.
    Some will be assigned to new patterns, shrinking existing patterns' AIFs in the process.
    Some will fall within the AIF of existing patterns of the same category.
    Such patterns will be discarded (but could be added in a subsequent iteration if the existing AIFs shrink enough to make room).
    '''
    start_time = default_timer()
    
    if len(pattern_group) == 0:
        if verbose_level >= 1:
            print("[{}:{}] All patterns of this category have been stored in the network.".format(nm500_idx, context))
        return False
    
    t1 = default_timer()
    nm500s.send(["learn_multiple_patterns", pattern_group], [nm500_idx])
    if verbose_level >= 1:
        print("[{}:{}] learn_multiple_patterns send time:    {:8.3f}s".format(nm500_idx, context, default_timer() - t1))
        print("[{}:{}] Num rows sent:      {}".format(nm500_idx, context, len(pattern_group)))
    
    if verbose_level >= 1:
        print("[{}:{}] Elapsed time to send multiple learning patterns to one client: {:8.3f}s".format(nm500_idx, context, default_timer() - start_time))
    
    return True

def receive_training_response(nm500_idx, context, pattern_group, pattern_group_position_ids, prev_network_size, pattern_ids_stored, verbose_level=0):
    '''
    After sending the patterns to the NeuroShields for learning, receive the responses back.
    This response information will inform us which patterns were newly stored in the network and which were not
    (because they fell within the AIF of existing neurons of the same category).
    We can then reuse the unstored patterns in a subsequent training iteration (at which point the AIFs might have shrunk enough to make room for the patterns).
    '''
    start_time = default_timer()
    
    # There are various stages of the communications pipeline where breakdowns can occur, the most finicky being the SPI comms, so it requires multiple attempts to insure successful comms
    for attempt in range(1, 6):
        try:
            response = nm500s.recv([nm500_idx])[0]
            break
        except Exception as e:
            print("Exception <receive_training_response()>: {}".format(e), e)
            if attempt == 5:
                print("5th attempt with exception. Raising...")
                raise e
            sleep(.1)
    if verbose_level >= 2:
        print("[{}:{}] learn_multiple_patterns receive time: {:8.3f}s".format(nm500_idx, context, default_timer() - start_time))
        print("[{}:{}] learn_multiple_patterns response: {}".format(nm500_idx, context, response))
    
    pat_idx_that_filled_network = response[2]
    print("[{}:{}] pat_idx_that_filled_network: {}".format(nm500_idx, context, pat_idx_that_filled_network))
    network_size_history = [int(ns) for ns in response[1][len('Network size history: '):].split(',')]
    if verbose_level >= 1:
        print("[{}:{}] Network size history: {}".format(nm500_idx, context, network_size_history))
    if len(network_size_history) != len(pattern_group):
        print("ERROR: receive_training_response() len(network_size_history) != len(pattern_group): {} {}".format(len(network_size_history), len(pattern_group)))
    if len(network_size_history) != len(pattern_group_position_ids):
        print("ERROR: receive_training_response() len(network_size_history) != len(pattern_group_position_ids): {} {}".format(len(network_size_history), len(pattern_group_position_ids)))
    
    # To see if a pattern was stored, investigate the network size after that pattern was used for learning.
    # The network size will either be unchanged or will be incremented by exactly 1.
    for nsi, network_size in enumerate(network_size_history):
        # For the first pattern sent, compare first network-size response to last network-size response of previous iteration to see if the first pattern was stored.
        # Compare all subsequent responses to the previous response of the same iteration to see if each pattern stored.
        if (nsi == 0 and network_size == prev_network_size + 1) or network_size == network_size_history[nsi - 1] + 1:
            label = pattern_group[nsi][2]
            if verbose_level >= 1:
                print("[{}:{}] Pattern {} of label {} was stored in network. Adding to skip set.".format(nm500_idx, context, nsi, label))
            pattern_ids_stored.add(pattern_group_position_ids[nsi])
    
    if verbose_level >= 1:
        print("[{}:{}] Elapsed time to send multiple learning patterns to one client: {:8.3f}s".format(nm500_idx, context, default_timer() - start_time))
    
    return network_size_history[-1], pat_idx_that_filled_network

def train_network_asynchronously(
        nm500s, data_train_labels_dfs_to_use, start_idx, max_pats_per_cat, verbose_level, network_capacity,
        network_sizes, network_sizes_prev, pat_idcs_that_filled_networks, final_training_completed_after_full, pattern_ids_stored, category_storage_histograms
    ):
    '''
    Asynchronous training consists of first sending each context's training patterns to its NeuroShield across all four contexts and then cycling over the NeuroShields gathering their responses.
    The alternative, synchronous training, consists of sending patterns for a single context to its NeuroShield and receiving the results back before proceeding to the next context.
    As expected, experiments confirmed that asynchronous training is faster.
    '''
    num_networks = len(data_train_labels_dfs_to_use)
    num_patterns_trained = 0
    pattern_groups, pattern_group_position_idss, nm500s_to_proc = [], [], []
    for i in range(num_networks):
        pattern_groups.append([])
        pattern_group_position_idss.append([])
        nm500s_to_proc.append(False)
    for nm500_idx in range(num_networks):
        if network_sizes[nm500_idx] != network_sizes_prev[nm500_idx]:
            print("" + "-" * 100)
            if network_sizes[nm500_idx] >= network_capacity: # Use >= instead of == in order to catch the apparent high-bit errors that occasionally set network size to 16384 or 65535
                if final_training_completed_after_full[nm500_idx]:
                    print("This network is full. It will not be trained further.")
                    network_sizes_prev[nm500_idx] = network_sizes[nm500_idx]
                    continue
                else:
                    print("This network is full. It will be trained one more time, not to add neurons (which is impossible), but to shrink AIFs.")
                    final_training_completed_after_full[nm500_idx] = True
            # If we use different NeuroShields to isolate the networks, we don't actually have to use difference contexts. We could use context 0 or 1 for each NS.
            # But, in the spirit of combining the NSs into a single network, let's use distinct contexts anyway.
            context = data_train_labels_dfs_to_use[nm500_idx][0]
            if verbose_level >= 1:
                print("[{}:{}] Training NeuroShield {} with context {}".format(nm500_idx, context, nm500_idx, context))

            network_sizes_prev[nm500_idx] = network_sizes[nm500_idx]
            pattern_group = pattern_groups[nm500_idx]
            pattern_group_position_ids = pattern_group_position_idss[nm500_idx]
            t1 = default_timer()
            data_train_label_df = data_train_labels_dfs_to_use[nm500_idx][1][0]
            end_idx = min(len(data_train_label_df), start_idx + max_pats_per_cat)
            if verbose_level >= 1:
                print("{}[{}:{}] DF len {} with index range [{},{}]".format("\n" if verbose_level >= 1 else "", nm500_idx, context, len(data_train_label_df), start_idx, end_idx))
            add_patterns_to_pattern_group(nm500_idx, context, start_idx, end_idx, pattern_ids_stored[nm500_idx], data_train_label_df, pattern_group, pattern_group_position_ids, verbose_level)

            num_patterns_trained += len(pattern_group)

            sent = send_training_data(nm500_idx, context, 0, min(len(pattern_group), max_pats_per_cat), pattern_group, verbose_level)
            if sent:
                nm500s_to_proc[nm500_idx] = True

    print("" + "=" * 100)
    print("NM500s for which any patterns were learned: {}".format(nm500s_to_proc))

    for nm500_idx in range(num_networks):
        if nm500s_to_proc[nm500_idx]:
            print("" + "-" * 100)
            context = data_train_labels_dfs_to_use[nm500_idx][0]
            if verbose_level >= 1:
                print("[{}:{}] Receiving training response from NM500 idx {}".format(nm500_idx, context, nm500_idx))
                print("[{}:{}] Retrieving training results for NeuroShield {} with context {}".format(nm500_idx, context, nm500_idx, context))

            pattern_group = pattern_groups[nm500_idx]
            pattern_group_position_ids = pattern_group_position_idss[nm500_idx]
            network_sizes[nm500_idx], pat_idx_that_filled_network = \
                receive_training_response(nm500_idx, context, pattern_group, pattern_group_position_ids, network_sizes[nm500_idx], \
                pattern_ids_stored[nm500_idx], verbose_level)

            if pat_idx_that_filled_network != None:
                pat_idcs_that_filled_networks[nm500_idx] = pat_idx_that_filled_network

            category_storage_histogram = Counter()
            for pattern_id_stored in pattern_ids_stored[nm500_idx]:
                category_storage_histogram[pattern_id_stored[0]] += 1
            for k, v in category_storage_histogram.most_common():
                category_storage_histograms[nm500_idx][k] += v
    
    return num_patterns_trained

def train_network_iteratively(nm500s, data_train_labels_dfs_to_use, start_idx=0, max_pats_per_cat=math.inf, verbose_level=0):
    '''
    Iterate until convergence (until network stops growing):
        Learn all examples in random order, adding any new examples that were previously excluded due to AIF overlap but which now fit into the network since its last shrinking.
        If no new examples were accumulated into the network, then end.
    '''
    start_time = default_timer()
    
    nm500s.processes[0][2].send(["send_recv_string", "navail"])
    port, r = nm500s.processes[0][2].recv()
    network_capacity = int(r)
    print("Network capacity: {}".format(network_capacity))
    data_train_size = sum([len(data_train_labels_dfs_to_use[0][1][i]) for i in range(len(data_train_labels_dfs_to_use[0][1]))])
    print("data_train size: {} (per channel)".format(data_train_size))
    print("start_idx, max_pats_per_cat, end_idx (limited by train set size): {}, {}, {}".format(start_idx, max_pats_per_cat, min(start_idx + max_pats_per_cat, data_train_size)))
    
    num_networks = len(data_train_labels_dfs_to_use)
    
    network_sizes, network_sizes_prev = [], []  # Keep track of the network sizes so as to stop iterative training when the network sizes converge
    pat_idcs_that_filled_networks = []
    final_training_completed_after_full = []
    pattern_ids_stored = []  # Keep track of which patterns are added to the networks so as to skip them on subsequent iterations (merely an optimization)
    category_storage_histograms = []
    for i in range(num_networks):
        network_sizes.append(0)
        pat_idcs_that_filled_networks.append(None)
        network_sizes_prev.append(None)
        final_training_completed_after_full.append(False)
        pattern_ids_stored.append(set())
        category_storage_histograms.append(Counter())
    print("Initial network sizes prev and curr: {} {}".format(network_sizes_prev, network_sizes))
    
    num_patterns_trained_per_iter = []
    num_iterations = 0
    while (True in [network_sizes[i] != network_sizes_prev[i] for i in range(num_networks)]):
        iter_start_time = default_timer()
        
        print("" + "#" * 100)
        print("Network sizes prev and curr at start of iteration: {} {}".format(network_sizes_prev, network_sizes))
        print("Starting training round")
        
        num_patterns_trained = 0
        num_patterns_trained = train_network_asynchronously(
            nm500s, data_train_labels_dfs_to_use, start_idx, max_pats_per_cat, verbose_level, network_capacity,
            network_sizes, network_sizes_prev, pat_idcs_that_filled_networks, final_training_completed_after_full, pattern_ids_stored, category_storage_histograms
        )
        
        num_patterns_trained_per_iter.append(num_patterns_trained)
        
        elapsed_time = default_timer() - iter_start_time
        pats_per_sec = num_patterns_trained / elapsed_time
        
        print("" + "+" * 100)
        print("Network sizes prev and curr at end of iteration:   {} {}".format(network_sizes_prev, network_sizes))
        print("Pattern indices that filled networks at end of iteration: {}".format(pat_idcs_that_filled_networks))
        for nm500_idx in range(num_networks):
            if network_sizes[nm500_idx] == network_capacity:
                print("!!! Network {} has used every available neuron. Its training must end.".format(nm500_idx))
        
        print("Storage histograms (totals: {})".format([sum([v for k, v in category_storage_histograms[nm500_idx].most_common()]) for nm500_idx in range(num_networks)]))
        print("{:>10}\t{}".format("CATEGORY", '\t'.join(['{:>4}'.format('NW_{}'.format(v)) for v in range(num_networks)])))
        print("-" * 50)
        for cat in CATEGORIES:
            print("{:>10}\t{}".format(cat, '\t'.join(['{:>4}'.format(category_storage_histograms[nm500_idx][cat] if cat in category_storage_histograms[nm500_idx] else 0) for nm500_idx in range(num_networks)])))
        print("-" * 50)
        print("{:>10}\t{}".format("TOTAL", '\t'.join(['{:>4}'.format(sm) for sm in [sum([v for k, v in category_storage_histograms[nm500_idx].most_common()]) for nm500_idx in range(num_networks)]])))
        
        num_iterations += 1
        print("Elapsed time to process one iteration: {:8.3f}s    {} pats: {:5.3f} pats/s".format(elapsed_time, num_patterns_trained, pats_per_sec))
    
    elapsed_time = default_timer() - start_time
    
    print("" + "_" * 100)
    print("Network sizes have converged. Iterative training complete.")
    print("Elapsed time to converge {} iterations: {:8.3f}s".format(num_iterations, elapsed_time))
    
    print("Results:\nmax_pats_per_cat: {}\telapsed_time: {}\tnum_iterations: {}\tnum_patterns_trained: {}".format(
        max_pats_per_cat, elapsed_time, num_iterations, sum(num_patterns_trained_per_iter)))
    
    print("Final network sizes:")
    nm500s.send(["retrieve_network_size"])
    print_pipe_results()
def classify_image(demo, cat_test_i, cat_test, imi, num_attempts, img, seg, patch_wh, contexts_to_use, classification_method,
                   classifications_histogram, classifications_weighted_histogram, cat_diff_sums, dead_zone_centers,
                   correct_incorrect_uncertain_counts, correct_incorrect_uncertain_counts_per_cat, incorrect_test_imgs, good_neurons, bad_neurons,
                   t1, verbose_level):
    '''
    Parameters:
        demo (int, always 0 or 1): Query attempt counter for this image. If a classification is incorrect, the classification query is repeated with a higher verbosity level to produce output for analysis
        cat_test_i (int): index in category enumeration
        cat_test (int): category
        imi (int): index in image enumeration of one category
        num_attempts (int): If a classification is uncertain, this parameter enables adding additional patches for further querying to clarify the classification
        img (PIL Image, mode RGB)
        seg (PIL Image, mode L)
        patch_wh (int): Pixel width of a square patch (always 16 in this experiment)
        contexts_to_use (list): List of no more than four context ids to use (since there are only four NeuroShields)
        classification_method: best or top_k
        classifications_histogram (Counter): neuron firing counts per category
        classifications_weighted_histogram: neuron firing counts per category weighted by inverse pattern match distance
        cat_diff_sums (Counter): summed firing neuron pattern match distance, used to calculate mean match distance per category
        dead_zone_centers: Prior patch upper-left corner coordinates, to minimize patch overlap
        correct_incorrect_uncertain_counts: Performance tallies
        correct_incorrect_uncertain_counts_per_cat: Performance tallies per category
        incorrect_test_imgs: Used for debugging
        good_neurons (dict): firing counts of correctly firing neurons
        bad_neurons (dict): firing counts of incorrectly firing neurons
    '''
    if classification_method != "best" and classification_method != "top_k":
        raise ValueError("classification_method must be 'best' or 'top_k'")
    
    patch_npxs = patch_wh**2
    patch_min_coverage = .75  # Amount that a patch must cover the image mask in order to be used
    num_good_patches_goal = 1  # TEMP 10
    dead_zone_size = 0. # THIS GETS SET INSIDE THE ATTEMPT LOOP  # max(int(patch_wh * .25), 1)  # Don't let the patches overlap too much (a smaller value here allows greater overlap)
    coverage_max_attempts = 100
    deadzone_max_attempts = 100

    patch_im_idcs = [PATCH_IM_RGB_IDX] + [context_to_use[0] + NUM_CONTEXT_TYPES for context_to_use in contexts_to_use]
    
    classifications_histogram_one_pass = Counter()
    classifications_weighted_histogram_one_pass = Counter()
    cat_diff_sums_one_pass = Counter()
    
    img2, draw = None, None
    if verbose_level >= 2:
        img2 = img.copy()
        draw = ImageDraw.Draw(img2, "RGBA")

    # On subsequent attempts, allow greater overlap between patches
    if num_attempts == 0:
        dead_zone_size = max(int(patch_wh * .25), 1)  # Don't let the patches overlap too much
    elif num_attempts == 1:
        dead_zone_size = max(int(patch_wh * .125), 1)  # Don't let the patches overlap too much
    else:
        dead_zone_size = max(int(patch_wh * .0625), 1)  # Don't let the patches overlap too much

    # Generate patches for this image
    t2 = default_timer()
    patches, dead_zone_centers_one_pass = generate_one_image_patches(img, seg, draw, PATCH_WH, patch_npxs, patch_min_coverage, num_good_patches_goal, dead_zone_centers,
                                                                    dead_zone_size, coverage_max_attempts, deadzone_max_attempts, True, verbose_level=verbose_level)
    if demo == 0:
        dead_zone_centers = dead_zone_centers_one_pass
    t3 = default_timer()
    if len(patches) < num_good_patches_goal * 8:
        print("[{}:{}] Num patches: {} (Goal: {})".format(cat_test_i + 1, imi + 1, len(patches), num_good_patches_goal * 8))
    if len(patches) == 0:
        raise ValueError("Could not find any patches for {} {}".format(cat_test_i + 1, imi + 1))

    if verbose_level >= 2:
        display_img_row([img, img2, seg], bg_color=0xffffff, border=1, scale=4)
        patch_im_display_idcs = patch_im_idcs if verbose_level >= 4 else [PATCH_IM_RGB_IDX]
        step = 1 if verbose_level >= 4 else 8
        for pimi in patch_im_display_idcs:
            display_img_row([patches[pi][pimi] for pi in range(0, len(patches), step)], bg_color=0xffffffff, border=1, scale=4)

    # Classify each patch.
    # Note that rotations and reflections of a patch bbox are separate patch patterns at this point, and therefore represent separate patches within the following patch loop.
    ept12, ept23, ept13 = 0, 0, 0
    if verbose_level >= 1:
        print("[{}:{}] Classifying {} patches".format(cat_test_i + 1, imi + 1, len(patches)))
    num_patch_arrays_sent_all_nm500s = 0
    for patch_i, patch in enumerate(patches):
        if verbose_level >= 4:
            print("[{}:{}] Patch idx, loc-src, transform: {:>4} {:>4} {}".format(cat_test_i + 1, imi + 1, patch_i, patch_i // 8, patch_i % 8))
            display_img_row([patch[patch_im_idcs[0]], patch[patch_im_idcs[1]], patch[patch_im_idcs[2]], patch[patch_im_idcs[3]], patch[patch_im_idcs[4]]], bg_color=0xffffffff, border=1, scale=4)

        # Send each patch to the network for classification
        tp1 = default_timer()
        patch_ar_idcs = [context_to_use[0] for context_to_use in contexts_to_use]
        for nm500_idx, context in enumerate(patch_ar_idcs):
            img_arr = patch[context]
            nm500s.send(["classify_one_pattern_{}".format(classification_method), cat_test * 100000 + imi * 1000 + patch_i, context, img_arr], [nm500_idx])
            num_patch_arrays_sent_all_nm500s += 1

        patch_ul = patch[PATCH_BBUL_IDX]

        # Read the classification results from the network and aggregate the results
        tp2 = default_timer()
        statuses, classifications = [], []
        for nm500_idx, context in enumerate(patch_ar_idcs):
            port, r = nm500s.recv([nm500_idx])[0]
            words = r.split()
            # The NM500 supports both 'best' and 'top_k' match retrieval.
            # I wasn't sure which behavior would provide the bst classification behavior, so I coded up both.
            # It turns out that 'best' is generally superior.
            if classification_method == 'best':  # Only retrieve the best firing neuron from the entire network
                # status: 0 - no match, 4 - uncertain (firing neurons having different categories), 8 - all firing neurons have the same category
                # neuron_id: The best firing neuron's 1-indexed position in the NN500 network, which in our case has a capacity of 4032
                # classification: The label stored with the best firing neuron during training
                # distance: The summed per-element distance between the query pattern and the firing neuron's pattern, i.e., the LSUP or Manhattan distance
                #    (the NM500 also supports the L1 distance metric (max per-element distance), but I never use it).
                #    Note that a giving firing neuron's match distance must, by definition, be <= that neuron's AIF (active influence field).
                status, neuron_id, classification, distance = int(words[4]), int(words[6]), int(words[8]), int(words[10])
                if verbose_level >= 4 and status != 0 and status != 32768:
                    print("[{}:{}] [{},{}] {:>3} ({:>2},{}) ({:>3},{:>3}) {}".format(cat_test_i + 1, imi + 1, port, nm500_idx, patch_i, patch_i // 8, patch_i % 8, patch_ul[0], patch_ul[1], r))
                if status != 0 and status != 32768 and distance > 65280:
                    raise ValueError("NM500 pattern distance > 65280 (255 * 256) should be impossible:\n{}".format(r))
                statuses.append(status)
                classifications.append(classification)
                if classification != 0x7FFF and classification != 0xFFFF:
                    cat_diff_sums_one_pass[classification] += distance
                    classifications_histogram_one_pass[classification] += 1
                    classifications_weighted_histogram_one_pass[classification] += 65280 - distance  # Classifications weighted by inverse distance relative to maximum difference (255 * 256)
                    if demo == 0:
                        if classification == cat_test:
                            good_neurons[(nm500_idx, neuron_id)] += 1
                        else:
                            bad_neurons[(nm500_idx, neuron_id)] += 1
            else:  # 'top_k' -- Retrieve the top K firing neurons in decreasing order by pattern match distance
                num_recog = int(words[4])
                if verbose_level >= 4 and num_recog != 0:
                    print("[{}:{}]".format(cat_test_i + 1, imi + 1), port, r)
                nids = eval(r[r.index('Nids= ')+len('Nids= '):r.index('Dists=') - 1])  # Neuron ids, see description under 'best' above
                dists = eval(r[r.index('Dists= ')+len('Dists= '):r.index('Cats=') - 1])  # See description under 'best' above
                cats = eval(r[r.index('Cats= ')+len('Cats= '):])  # See description under 'best' above
                if len(nids) != 5 or len(dists) != 5 or len(cats) != 5:  # The middle layer is currently hard-coded to request the top 5 matches, but it could be generalized if needed
                    raise ValueError("Expected 5 nids, dists, and cats")
                for i in range(num_recog):
                    nid, dist, cat = nids[i], dists[i], cats[i]
                    classifications.append(cat)
                    if cat != 0x7FFF and cat != 0xFFFF:
                        cat_diff_sums_one_pass[cat] += dist
                        classifications_histogram_one_pass[cat] += 1
                        # I experimented with two metric for weighting the sorted top K matches since it wasn't clear which metric would provide the best results
                        # classifications_weighted_histogram_one_pass[cat] += 65280 - dist  # Classifications weighted by inverse distance relative to maximum difference (255 * 256)
                        classifications_weighted_histogram_one_pass[cat] += 5 - i  # Classifications weighted by inverse rank
                        if demo == 0:
                            if cat == cat_test:
                                good_neurons[(nm500_idx, nid)] += 1
                            else:
                                bad_neurons[(nm500_idx, nid)] += 1

        tp3 = default_timer()

        ept12 += tp2 - tp1
        ept23 += tp3 - tp2
        ept13 += tp3 - tp1
    t4 = default_timer()

    if verbose_level >= 1:
        print("[{}:{}] Classified {} patches, {} total arrays sent to all NM500s".format(cat_test_i + 1, imi + 1, len(patches), num_patch_arrays_sent_all_nm500s))

    pats_per_sec_23 = len(patches) / ept23
    pats_per_sec_13 = len(patches) / ept13
    if imi == 0:
        print("[{}:{}] Classification times for {} patches: {:8.3f}s {:8.3f}s ({:5.3f} pats/s) {:8.3f}s ({:5.3f} pats/s)".format(cat_test_i + 1, imi + 1, len(patches), ept12, ept23, pats_per_sec_23, ept13, pats_per_sec_13))

    # Sort the classification results by counts per category and generate a final classification prediction
    classifications_sorted = classifications_histogram_one_pass.most_common()
    classifications_weighted_sorted = classifications_weighted_histogram_one_pass.most_common()
    cat_diff_means_sorted = sorted([(cat, (diff_sum / classifications_histogram_one_pass[cat])) for cat, diff_sum in list(cat_diff_sums_one_pass.most_common())], key=lambda kv: kv[1])
    status = 'uncertain'
    if len(classifications_sorted) > 0:
        if classifications_sorted[0][0] != classifications_weighted_sorted[0][0]:
            print("[{}:{}] Unweighted and weighted classification orders differ".format(cat_test_i + 1, imi + 1))
        
        best_class, best_count = classifications_sorted[0]
        second_best_class, second_best_count = classifications_sorted[1] if len(classifications_sorted) >= 2 else None, 0
        best_w_class, best_w_count = classifications_weighted_sorted[0]
        second_best_w_class, second_best_w_count = classifications_weighted_sorted[1] if len(classifications_weighted_sorted) >= 2 else None, 0

        if verbose_level >= 1 or best_count == second_best_count:
            print("[{}:{}] Classifications:\n\t{}".format(               cat_test_i + 1, imi + 1, '\n\t'.join(['{:>2}: {:>3}'.format(   k, v) for k, v in classifications_sorted])))
            print("[{}:{}] Classifications weighted by inverse distance:\n\t{}".format(      cat_test_i + 1, imi + 1, '\n\t'.join(['{:>2}: {:>10}'.format(  k, v) for k, v in classifications_weighted_sorted])))
            print("[{}:{}] Classifications mean distances:\n\t{}".format(cat_test_i + 1, imi + 1, '\n\t'.join(['{:>2}: {:10.0f}'.format(k, v) for k, v in cat_diff_means_sorted])))

        # If the unweighted votes tie for best place (meaning they are arbitrarily sorted), sort them by weight.
        # TODO: Extend this to include third and higher tie-breakers.
        if best_count == second_best_count:
            if classifications_weighted_histogram_one_pass[second_best_class] > classifications_weighted_histogram_one_pass[best_class]:
                best_class, best_count, second_best_class, second_best_count = second_best_class, second_best_count, best_class, best_count
        
        if best_count > second_best_count:
            if best_class == cat_test:
                status = 'correct'
            else:
                status = 'incorrect'
                if demo == 0:
                    incorrect_test_imgs.append((cat_test, imi + 1))
                print("[{}:{}] Incorrect: {} != {}".format(cat_test_i + 1, imi + 1, best_class, cat_test))
                if verbose_level >= 4:
                    display_img_row([img, img2, seg], bg_color=0xffffff, border=1, scale=4)
                    for i in patch_im_idcs:
                        display_img_row([p[i] for p in patches], bg_color=0xffffffff, border=1, scale=4)

    et2 = t2 - t1
    et3 = t3 - t2
    et4 = t4 - t3
    if verbose_level >= 1:
        if imi == 0:
            print("[{}:{}] Elapsed times: {:8.3f}s {:8.3f}s {:8.3f}s".format(cat_test_i + 1, imi + 1, et2, et3, et4))

    if status == 'uncertain':
        print ("[{}:{}] No clear winner yet. Generating more patches and trying again.".format(cat_test_i + 1, imi + 1))
    elif demo == 0:
        correct_incorrect_uncertain_counts[status] += 1
        correct_incorrect_uncertain_counts_per_cat[cat_test][status] += 1
    
    if demo == 0:
        for cat in classifications_histogram_one_pass:
            classifications_histogram[cat] += classifications_histogram_one_pass[cat]
        for cat in classifications_weighted_histogram_one_pass:
            classifications_weighted_histogram[cat] += classifications_weighted_histogram_one_pass[cat]
        for cat in cat_diff_sums_one_pass:
            cat_diff_sums[cat] += cat_diff_sums_one_pass[cat]

    return status, t2, t3, t4

def classify_images(categorized_imgs_to_classify, contexts_to_use, classifier, classification_method, rnd, debug_filter=None):
    '''
    Parameters:
        categorized_imgs_to_classify (dict): either categorized_validate_imgs or categorized_test_imgs, keyed by category
        contexts_to_use (list): List of no more than four context ids to use (since there are only four NeuroShields)
        classifier: RBF or KNN
        classification_method: best or top_k
        rnd (int): Rounds or batches indicate which subset of the training data is being analyzed (this parameter isn't important turing final evaluation, only during training)
    '''
    if classification_method != "best" and classification_method != "top_k":
        raise ValueError("classification_method must be 'best' or 'top_k'")
    
    print("set{}".format(classifier))
    nm500s.send(["send_recv_string", "set{}".format(classifier)])
    print_pipe_results()
    print()
    
    correct_incorrect_uncertain_counts = Counter()
    correct_incorrect_uncertain_counts_per_cat = {}
    incorrect_test_imgs = []
    good_neurons, bad_neurons = Counter(), Counter()
    num_total_patches_found = 0
    
    for cat_test_i, cat_test in enumerate(categorized_imgs_to_classify):
        print("\nCategory # {} of {}, label {} has {} images".format(cat_test_i + 1, len(categorized_imgs_to_classify), cat_test, len(categorized_imgs_to_classify[cat_test])))
        correct_incorrect_uncertain_counts_per_cat[cat_test] = Counter()
        for imi, (img, seg) in enumerate(categorized_imgs_to_classify[cat_test]):
            if imi > 0:  # TEMP
                break  # TEMP
            if debug_filter and (cat_test, imi + 1) not in debug_filter:
                continue
            for demo in range(2): # Only process once, but if the classification is incorrect, run it again with verbose output
                verbose_level = 0
                if demo == 1:
                    verbose_level = 3
                if debug_filter and (cat_test, imi + 1) in debug_filter:
                    verbose_level = 3
                if verbose_level >= 1:
                    print("\n[{}:{}] Category # {} of {}, label {}, image {} of {}".format(cat_test_i + 1, imi + 1, cat_test_i + 1, len(categorized_imgs_to_classify), cat_test, imi + 1, len(categorized_imgs_to_classify[cat_test])))
                seed(cat_test * 1000 + imi)  # Use a fixed seed for each image. This will facilitate pulling a given image out and debugging it individually.
                classifications_histogram = Counter()
                classifications_weighted_histogram = Counter()
                cat_diff_sums = Counter()
                dead_zone_centers = set()
                num_attempts, MAX_ATTEMPTS = 0, 1
                while num_attempts < MAX_ATTEMPTS:
                    t1 = default_timer()
                    num_attempts += 1
                    if num_attempts > 1:
                        print("[{}:{}] Attempt # {}".format(cat_test_i + 1, imi + 1, num_attempts + 1))

                    result, t2, t3, t4 = classify_image(demo, cat_test_i, cat_test, imi, num_attempts, img, seg, PATCH_WH, contexts_to_use, classification_method,
                                                        classifications_histogram, classifications_weighted_histogram, cat_diff_sums, dead_zone_centers,
                                                        correct_incorrect_uncertain_counts, correct_incorrect_uncertain_counts_per_cat, incorrect_test_imgs, good_neurons, bad_neurons,
                                                        t1, verbose_level)
                    if result != 'uncertain':
                        break

                t5 = default_timer()
                et2 = t2 - t1
                et3 = t3 - t2
                et4 = t4 - t3
                et5 = t5 - t4
                et15 = t5 - t1
                if verbose_level >= 1:
                    if imi == 0:
                        print("[{}:{}] Elapsed times: {:8.3f}s {:8.3f}s {:8.3f}s {:8.3f}s - {:8.3f}s".format(cat_test_i + 1, imi + 1, et2, et3, et4, et5, et15))

                if demo == 0:
                    if result == 'uncertain':
                        correct_incorrect_uncertain_counts['uncertain'] += 1
                        correct_incorrect_uncertain_counts_per_cat[cat_test]['uncertain'] += 1
                        print("Uncertain")

                # if the result is correct, break out. Else, classify the same image again with verbose output
                if result == 'correct':
                    break
        print("Final counts for this category: {}".format(correct_incorrect_uncertain_counts_per_cat[cat_test]))

    result_str = "Final counts for round_or_batch {}: {}".format(rnd, correct_incorrect_uncertain_counts.most_common())
    print(result_str)
    print("{:>8} {:>12} {:>12} {:>12}".format("CAT", "CORRECT", "INCORRECT", "UNCERTAIN"))
    for cat in correct_incorrect_uncertain_counts_per_cat:
        print("{:>8} {:>12} {:>12} {:>12}".format(cat, correct_incorrect_uncertain_counts_per_cat[cat]['correct'], correct_incorrect_uncertain_counts_per_cat[cat]['incorrect'], correct_incorrect_uncertain_counts_per_cat[cat]['uncertain']))
    print("Incorrect images: {}".format(incorrect_test_imgs))
    print()
    
    return good_neurons, bad_neurons, result_str
def determine_underperforming_neurons(good_neurons, bad_neurons, criteria):
    '''
    Based on the results of classifying a dataset (presumably the validation set), use the firing performance of the neurons to filter out neurons (patterns) that underperform.
    
    Parameters:
        good_neurons (dict): correct firing counts for each neuron against the validation set
        bad_neurons (dict): incorrect firing counts for each neuron against the validation set
        criteria (list of 4 bools): Criteria by which neurons can optionally be filtered out of the network and discarded
            0: bad_count > good_count
            1: bad_count > 0 and good_count == 0
            2: (bad_count >= 3 or good_count >= 3) and bad_good_ratio >= 1.5
            3: nonfiring
    '''
    neuron_ids_to_delete = set()
    
    if criteria[0] and criteria[1]:
        raise ValueError("Incompatible filtering criteria")
    
    # Flag neurons that fire incorrectly more often that correctly for deletion
    if criteria[0] or criteria[1] or criteria[2]:
        num_bad_and_good_neurons = 0
        for i, (bad_neuron, bad_count) in enumerate(bad_neurons.most_common()):
            if bad_neuron in good_neurons:
                num_bad_and_good_neurons += 1
                good_count = good_neurons[bad_neuron]
                bad_good_balance = bad_count - good_count
                bad_good_ratio = bad_count / good_count
            else:
                good_count = 0
                bad_good_balance = bad_count
                bad_good_ratio = math.inf
            if     (criteria[0] and bad_good_balance > 0) \
                or (criteria[1] and good_count == 0) \
                or (criteria[2] and ((bad_count >= 3 or good_count >= 3) and bad_good_ratio >= 1.5)):
                neuron_ids_to_delete.add(bad_neuron)
        print("Num good neurons: {}".format(len(good_neurons)))
        print("Num bad neurons:  {}".format(len(bad_neurons)))
        print("Num bad>good neurons to delete: {}".format(len(neuron_ids_to_delete)))

    # Flag neurons that never fire for deletion
    if criteria[3]:
        all_neurons = set([(nm500_idx, neuron_id) for neuron_id in range(1, 4033) for nm500_idx in range(4)])
        all_firing_neurons = set(good_neurons).union(set(bad_neurons))
        all_nonfiring_neurons = all_neurons.difference(all_firing_neurons)
        neuron_ids_to_delete.update(all_nonfiring_neurons)
        print("Num nonfiring neurons to delete: {}".format(len(all_nonfiring_neurons)))
    
    print("Num neuron_ids_to_delete: {}".format(len(neuron_ids_to_delete)))
    
    return neuron_ids_to_delete

def filter_networks(networks, neuron_ids_to_delete):
    '''
    Given the filtering decision made by determine_underperforming_neurons(), discard designated neurons and their patterns from the networks
    '''
    pd.set_option('display.max_rows', None)

    print("Filtering networks to remove bad neurons")
    networks_filtered = []
    neuron_id_remapping = [{}, {}, {}, {}]
    neuron_id_rev_remapping = [{}, {}, {}, {}]
    for nm500_idx, (ncount, full_network_l, full_network_s) in enumerate(networks):
        context_to_use, context_to_use_lbl = contexts_to_use[nm500_idx]
        
        print("Input:    NeuroShield# : {}, ncount: {:>10}, num components: {:>10} /260: {:>10}, num to delete: {:>10}".format(nm500_idx, ncount, len(full_network_l), len(full_network_l) / 260, len(neuron_ids_to_delete)))

        neurons_filtered_flat_l = []
        filtered_network_current_length = 0
        num_deleted = 0
        for neuron_idx in range(ncount):
            neuron_id = neuron_idx + 1
            if (nm500_idx, neuron_id) not in neuron_ids_to_delete:  # neurons_to_delete_this_network:
                neuron_l = full_network_l[neuron_idx * 260 : neuron_idx * 260 + 260]
                cxt = neuron_l[0]
                if cxt != context_to_use:
                    print("Context mismatch [{} {}]: {} != {} (it has been corrected)".format(nm500_idx, neuron_idx, cxt, context_to_use))
                    print(neuron_l)
                    neuron_l[0] = context_to_use
                neurons_filtered_flat_l.extend(neuron_l)
                filtered_network_current_length += 1
                neuron_id_remapping[nm500_idx][neuron_id] = filtered_network_current_length
                neuron_id_rev_remapping[nm500_idx][filtered_network_current_length] = neuron_id
            else:
                num_deleted += 1
        print("Filtered: num deleted: {:>10}, num components: {:>10}, /260: {:>10}".format(num_deleted, len(neurons_filtered_flat_l), len(neurons_filtered_flat_l) / 260))

        neurons_filtered_flat_s = str(neurons_filtered_flat_l).replace(' ', '')
        networks_filtered.append((len(neurons_filtered_flat_l) // 260, neurons_filtered_flat_l, neurons_filtered_flat_s))
        print("Result:   network str len: {:>10}, num_comps / 260: {:>10}, ncount - num_comps / 260: {:>10}".format(len(neurons_filtered_flat_s), len(neurons_filtered_flat_l) / 260, ncount - len(neurons_filtered_flat_l) / 260))
        print()
    
    return networks_filtered, neuron_id_remapping, neuron_id_rev_remapping
def save_network_filters(good_neurons, bad_neurons, lbl, dir_path):
    network_filters_json = {
        'good_neurons': {'{}'.format(k): v for k, v in good_neurons.items()},
        'bad_neurons': {'{}'.format(k): v for k, v in bad_neurons.items()},
    }
    filename = "NM500_{}_network_filters.json".format(lbl)
    with open(dir_path + filename, 'w') as f:
        json.dump(network_filters_json, f, indent='\t')

def load_network_filters(lbl, dir_path):
    filename = "NM500_{}_network_filters.json".format(lbl)
    with open(dir_path + filename, 'r') as f:
        network_filters_json = json.load(f)
    good_neurons = Counter({eval(k): v for k, v in network_filters_json['good_neurons'].items()})
    bad_neurons = Counter({eval(k): v for k, v in network_filters_json['bad_neurons'].items()})
    return good_neurons, bad_neurons

def save_neuron_id_remapping(neuron_id_remapping, neuron_id_rev_remapping, lbl, filter_suffix, dir_path):
    neuron_id_remapping_json = {
        'forward_remapping': [{
            '{}'.format(k): v for k, v in neuron_id_remapping[nm500_idx].items()
        } for nm500_idx in range(len(neuron_id_remapping))],
        'reverse_remapping': [{
            '{}'.format(k): v for k, v in neuron_id_rev_remapping[nm500_idx].items()
        } for nm500_idx in range(len(neuron_id_remapping))]
    }
    filename = "NM500_{}_neuron_id_remapping-{}.json".format(lbl, filter_suffix)
    with open(dir_path + filename, 'w') as f:
        json.dump(neuron_id_remapping_json, f, indent='\t')

def load_neuron_id_remapping(lbl, filter_suffix, dir_path):
    filename = "NM500_{}_neuron_id_remapping-{}.json".format(lbl, filter_suffix)
    with open(dir_path + filename, 'r') as f:
        neuron_id_remapping_json = json.load(f)
    neuron_id_remapping =     [{eval(k): v for k, v in neuron_id_remapping_json['forward_remapping'][nm500_idx].items()} for nm500_idx in range(len(neuron_id_remapping_json['forward_remapping']))]
    neuron_id_rev_remapping = [{eval(k): v for k, v in neuron_id_remapping_json['reverse_remapping'][nm500_idx].items()} for nm500_idx in range(len(neuron_id_remapping_json['reverse_remapping']))]
    return neuron_id_remapping, neuron_id_rev_remapping

def read_networks_from_neuroshields(contexts_to_use, verbose_level=0):
    '''
    A network can be represented in various ways, such as:
        network_l: a list of bytes corresponding to the byte stream sent to or received from the NM500, which each neuron consists of 260 bytes:
            byte 0: the neuron's context (neurons can be assigned to nonoverlapping classification tasks, although for this experiment, I use a single context per network (per NeuroShield))
            bytes 1-256: the neuron's 255 unsigned byte pattern
            byte 257: the neuron's active influence field (AIF) (the distance (i.e., summed component-wise difference) within which a queried pattern causes this neuron to fire)
            byte 258: the neuron's minimum active influence field (if the AIF shrinks to this level, the neuron is considered degenerate)
            byte 259: the neuron's category
            The network_l representation consists of a sequential run of 260-byte neuron descriptions, i.e., for 4032 neurons, a list of 4032*260 bytes = 1048320 bytes.
            Note that network_l is not a list of 4032 lists of 260 bytes, it is a single list. This layout facilities sending the entire network over a pipe to the client for transmission over SPI to the NeuroShield.
        network_s: a comma-delimited string representation of network_l
    '''
    if verbose_level >= 1:
        print("Reading networks from NeuroShields")
    networks = []
    nm500s.send(["send_recv_string", "readNeurons"])
    for nm500_idx, (port, r) in enumerate(nm500s.recv()):
        ncount, network_l = eval(r)
        if ncount is None:
            print("[{}] ERROR: readNeurons() returned None".format(port))
            networks.append((None, None, None))
        else:
            print("[{}] ncount, len(network), len(network)/260: {} {} {}".format(port, ncount, len(network_l), len(network_l) / 260))
            if verbose_level >= 1:
                print("[{}] first neuron's context: {}".format(port, network_l[0]))
            if network_l[0] != contexts_to_use[nm500_idx][0]:
                incorrect_contexts = Counter()
                for i in range(0, len(network_l), 260):
                    if network_l[i] != contexts_to_use[nm500_idx][0]:
                        incorrect_contexts[network_l[i]] += 1
                    network_l[i] = contexts_to_use[nm500_idx][0]
                print("[{}] ERROR: After reading network from NeuroShield, {} neurons had incorrect contexts (they have been corrected).".format(port, sum(incorrect_contexts.values())))
                print("[{}] Incorrect context counts: {}".format(port, incorrect_contexts.most_common()))
                if verbose_level >= 1:
                    print("[{}] first neuron's context: {}".format(port, network_l[0]))
            network_s = "{}".format(network_l).replace(' ', '')
            networks.append((ncount, network_l, network_s))
    
    return networks

def write_flattened_networks_to_neuroshields(networks):
    '''
    An "unflattened" network is a list of lists. The outer list is a list of neurons, and the inner lists are 260-byte component (byte) lists of neuron descriptions (see read_networks_from_neuroshields() for a description of the 260 bytes).
    A "flattened" network is a single list of components (bytes) consisting of the concatenation of multiple neurons. For a 4032 neuron network, a flattened network will contain 4032*260 bytes = 1048320 bytes.
    '''
    print("Writing networks to NeuroShields")
    for nm500_idx in range(len(networks)):
        nm500s.processes[nm500_idx][2].send(["send_recv_string", "writeNeurons " + networks[nm500_idx][2]])

    responses = []
    for port, r in nm500s.recv():
        print(port, r)
        ncount = None
        if r is "None":
            pass
        elif r[:len("ERR_")] == "ERR_":
            print("[{}] Erroneous NCOUNT returned from writeNeurons() : {}".format(port, r))
            ncount = int(r[len("ERR_"):])
        elif r[:len('EXCEPTION')] == 'EXCEPTION':
            print(port, r)
            raise RuntimeError(r)
        else:
            ncount = eval(r)
        responses.append((port, ncount))
    responses = sorted(responses, key=lambda pr: pr[0])
    return responses

def save_flattened_networks(networks, lbl1, lbl2, dir_path):
    print("Saving flattened networks to disk with labels '{}' & '{}'".format(lbl1, lbl2))
    for nm500_idx, (ncount, network_flat_l, network_flat_s) in enumerate(networks):
        filename = "NM500_{}_network_desc_{}_{}.txt".format(lbl1, nm500_idx, lbl2)
        ncount2 = (network_flat_s.count(',') + 1) // 260
        if ncount2 != ncount:
            print("[{}] ncount mismatch: {} != {}".format(nm500_idx, ncount, ncount2))
        with open(dir_path + filename, 'w') as f:
            f.write("{}\n".format(ncount))
            f.write(network_flat_s)

def load_flattened_networks(nm500_indices, lbl1, lbl2, dir_path):
    print("Loading flattened networks from disk for round_or_batch {} & '{}'".format(lbl1, lbl2))
    networks = []
    for nm500_idx in nm500_indices:
        filename = "NM500_{}_network_desc_{}_{}.txt".format(lbl1, nm500_idx, lbl2)
        with open(dir_path + filename) as f:
            lines = f.readlines()
        ncount = None
        if len(lines) == 2:
            ncount = int(lines[0].strip())
        network_flat_s = lines[-1]
        
        network_l = eval(network_flat_s)
        
        if ncount is None:
            ncount = len(network_l) // 260
        networks.append((ncount, network_l, network_flat_s))
    return networks
def train_independent_filtered_networks(data_train_labels_dfs_to_use, contexts_to_use, start_batch, num_batches, max_pats_per_cat, verbose_level=0):
    '''
    Iterate over the training batches (subsets of the training data small enough to avoid saturing a network).
    For each batch, train the networks.
    Retrieve the networks from the NeuroShields and save them.
    Classify the validation set with the network for the current batch, produce correct/incorrect firing counts for each neuron, and save the results for subsequent filtering.
    Repeat with the next batch until all training data have been used.
    
    Parameters:
        data_train_labels_dfs_to_use (list): list of training patches corresponding to contexts_to_use.
        contexts_to_use (list): list of contexts (image color bands) to use. This list should not specify more contexts than we have NeuroShields available (it could, in theory, but the experiment isn't designed to support that for now). 
        start_batch (int): Batch to start from. This parameter enables restarting part way through if the process crashes before completing all batches (the SPI comms are pretty unreliable).
        num_batches (int): Number of batches to produce. The following three values should also be calculated together: num training samples, batch size, num batches
        max_pats_per_cat (int): aka batch size, number of training samples per batch. The resulting networks will very likely contain fewer neurons than this amount, since many training samples will fall within the AIF of earlier presented samples.
    '''
    for batch in range(start_batch, num_batches + 1):
        print("\n\n\n" + "#" * 100)
        print("BATCH {} ".format(batch) * 20)

        times = []
        times.append(("Start", default_timer()))

        start_idx = (batch - 1) * max_pats_per_cat

        print("data_train size: {} channels available (contexts), {} labels (categories), {} rows (patterns per channel)".format(
            len(data_train_labels_dfs), len(data_train_labels_dfs[0]), sum([len(data_train_labels_dfs[0][i]) for i in range(len(data_train_labels_dfs[0]))])))

        reset_neuromem_and_interface()
        train_network_iteratively(nm500s, data_train_labels_dfs_to_use, start_idx, max_pats_per_cat, verbose_level)
        times.append(("Train", default_timer()))

        networks = read_networks_from_neuroshields(contexts_to_use)
        if any([network[0] is None for network in networks]):
            print("An error occurred while reading networks from NeuroShields after reading networks")
            nm500s.send(["send_recv_string", "quit"])
            print_pipe_results()
            break
        times.append(("Read", default_timer()))

        contexts_to_use_lbl = ''.join(context_to_use[1] for context_to_use in contexts_to_use)
        save_flattened_networks(networks, "k-{}_{}_batch_size-{}_batch-{}".format(crossfold_k, contexts_to_use_lbl, max_pats_per_cat, batch), "full",
                                dir_path='./k{}/{}/batch_size-{}/independent networks/'.format(crossfold_k, contexts_to_use_lbl, batch_size))
        times.append(("Save full network", default_timer()))

        good_neurons, bad_neurons, result_str = classify_images(categorized_validate_imgs, contexts_to_use, "RBF", "best", batch)
        times.append(("Classify", default_timer()))

        save_network_filters(good_neurons, bad_neurons, "k-{}_{}_batch_size-{}_batch-{}".format(crossfold_k, contexts_to_use_lbl, max_pats_per_cat, batch),
                             dir_path='./k{}/{}/batch_size-{}/filters/'.format(crossfold_k, contexts_to_use_lbl, batch_size))
        times.append(("Save filters", default_timer()))

        summarize_times(batch, times)

def filter_batches(crossfold_k, num_batches, neuron_filter, contexts_to_use_lbl, max_pats_per_cat, dir_path):
    '''
    For each batch, read its networks from disk, filter (discard) neurons on the basis of the specified filter criteria, and save the filtered networks back to disk.
    
    Parameters:
        crossfold_k (int): Which crossfold to process. This experiment is hard-coded to assume five folds, so this parameter must be from [0, 1, 2, 3, 4].
        num_batches (num): How many batches to filter
        neuron_filter (list of bools): filtering criteria. See determine_underperforming_neurons() for a description.
        dir_path: Path to the top-level directory of the network storage on disk.
    '''
    filter_suffix = ''.join(['t' if tf else 'f' for tf in neuron_filter])
        
    for batch in range(1, num_batches + 1):
        print("\n\n\n" + "#" * 100)
        print("K {}".format(crossfold_k))
        print("BATCH {} ".format(batch) * 20)

        times = []
        times.append(("Start", default_timer()))
        
        lbl = "k-{}_{}_batch_size-{}_batch-{}".format(crossfold_k, contexts_to_use_lbl, max_pats_per_cat, batch)
        
        good_neurons, bad_neurons = load_network_filters(lbl, dir_path + 'filters/')
        times.append(("Load filters", default_timer()))

        neuron_ids_to_delete = determine_underperforming_neurons(good_neurons, bad_neurons, neuron_filter)
        times.append(("Determine neurons to delete", default_timer()))
        
        one_batch_networks = load_flattened_networks([0, 1, 2, 3], lbl, "full", dir_path + 'independent networks/')
        times.append(("Load full network", default_timer()))

        one_batch_networks_filtered, neuron_id_remapping, neuron_id_rev_remapping = filter_networks(one_batch_networks, neuron_ids_to_delete)
        times.append(("Filter", default_timer()))

        save_neuron_id_remapping(neuron_id_remapping, neuron_id_rev_remapping, lbl, filter_suffix,
                                 dir_path='./k{}/{}/batch_size-{}/{}/filtered remappings/'.format(k, contexts_to_use_lbl, batch_size, filter_suffix))
        times.append(("Save neuron id remapping", default_timer()))

        save_flattened_networks(one_batch_networks_filtered, lbl, "filtered-{}".format(filter_suffix),
                               dir_path='./k{}/{}/batch_size-{}/{}/independent networks/'.format(k, contexts_to_use_lbl, batch_size, filter_suffix))
        times.append(("Save filtered network", default_timer()))

        summarize_times(batch, times)

def combine_batches(num_batches, contexts_to_use_lbl, batch_size, dir_path, filter_suffix):
    '''
    Combine neurons from multiple all batches (ostensibly after filtering) into a single list of neurons.
    This list of neurons does not represent a network since the potential conflict of the AIFs must be resolve via a subsequent training round.
    The order of the conglomerated list of neurons will affect the network produced by the subsequent training, with favor given to neurons (patterns) that are presented earlier in the training process.
    Consequently, the output of this function should be sorted (see sort_combined_networks()) according to some sensible criteria before being presented for training instead of being used as is, which simply favors the earlier batches in a meaningless way.
    '''
    times = []
    times.append(("Start", default_timer()))

    # Gather all the filtered networks from all rounds
    networks = [
        [  # NM500 idx 0
            0,  # ncount
            [], # network_l (ncount*260 long)
            [], # network_s (ncount*260 comma-delimited elements long)
            [], # good firing counts per neuron (ncount long)
            [], # bad firing counts (ncount long)
        ],
        [0, [], [], [], []], [0, [], [], [], []], [0, [], [], [], []]
    ]
    for batch in range(1, num_batches + 1):
        lbl = "k-{}_{}_batch_size-{}_batch-{}".format(crossfold_k, contexts_to_use_lbl, batch_size, batch)
        one_batch_networks = load_flattened_networks([0, 1, 2, 3], lbl, "filtered-{}".format(filter_suffix), dir_path + '{}/independent networks/'.format(filter_suffix))
        
        good_neurons, bad_neurons = load_network_filters(lbl, dir_path + 'filters/')
        neuron_id_remapping, neuron_id_rev_remapping = load_neuron_id_remapping(lbl, filter_suffix, dir_path + '{}/filtered remappings/'.format(filter_suffix))
        
        for nm500_idx in [0, 1, 2, 3]:
            good_neurons_this_nm500 = {k[1]: v for k, v in good_neurons.items() if k[0] == nm500_idx}
            bad_neurons_this_nm500 =  {k[1]: v for k, v in bad_neurons.items()  if k[0] == nm500_idx}
            networks[nm500_idx][0] += one_batch_networks[nm500_idx][0]  # ncount
            networks[nm500_idx][1].extend(one_batch_networks[nm500_idx][1])  # network list
            networks[nm500_idx][2].extend(',' + one_batch_networks[nm500_idx][2])  # network string
            
            context_to_use = contexts_to_use[nm500_idx][0]
            for ni in range(one_batch_networks[nm500_idx][0]):
                cxt = one_batch_networks[nm500_idx][1][ni * 260]
                if cxt != context_to_use:
                    print("Context mismatch [{} {} {}]: {} != {}".format(batch, nm500_idx, ni, cxt, context_to_use))
                    print(one_batch_networks[nm500_idx][1][ni * 260 : ni * 260 + 260])
            
            for neuron_id in range(1, one_batch_networks[nm500_idx][0] + 1):
                neuron_rev_id = neuron_id_rev_remapping[nm500_idx][neuron_id]
                if neuron_rev_id not in good_neurons_this_nm500 and neuron_rev_id not in bad_neurons_this_nm500:
                    raise ValueError("Expected neuron_id to be in at least one of good_neurons and bad_neurons: {} {} {} {} {}".format(crossfold_k, batch, nm500_idx, neuron_id, neuron_rev_id))
                networks[nm500_idx][3].append(good_neurons_this_nm500[neuron_rev_id] if neuron_rev_id in good_neurons_this_nm500 else 0)
                networks[nm500_idx][4].append(bad_neurons_this_nm500[ neuron_rev_id] if neuron_rev_id in bad_neurons_this_nm500  else 0)
            print("Network {} size: {}".format(nm500_idx, networks[nm500_idx][0]))
    times.append(("Gather completed networks", default_timer()))
    
    summarize_times(batch, times)
    
    return networks

def sort_combined_networks(networks, sort_method, secondary_sort_weight=1.):
    '''
    This list of neurons combined by combine_batches() does not represent a network since the potential conflict of the AIFs must be resolve via a subsequent training round.
    The order of the conglomerated list of neurons will affect the network produced by the subsequent training, with favor given to neurons (patterns) that are presented earlier in the training process.
    Consequently, this function supports a variety of sorting criteria, such as:
        - Increasing AIF (this doesn't make much sense, and experiments confirmed lower performance)
        - Decreasing AIF (make judicious use of the 4032 limit on network capacity by favoring neurons with wide coverage of pattern space)
        - Increasing bad_count, secondarily sorted (to break ties) by decreasing good_count, subject to a <= 1.0 secondary weight
        - Decreasing good_count, secondarily sorted (to break ties) by increasing bad_count, subject to a <= 1.0 secondary weight
    
    Parameters:
        networks: One network per context (i.e., per NeuroShield)
        sort_method (int): A sorting method specifier
        secondary_sort_weight (float): The secondary sort weight <= 1 used by some sorting methods
    '''
    times = []
    times.append(("Start", default_timer()))

    # Sort each network
    print("Sorting gathered networks")
    sorted_networks = []
    for nfi, network in enumerate(networks):
        ncount = network[0]
        print("Network {} size: {}".format(nfi, ncount))
        # Convert the straight component list (of 260-comp neurons) into a neuron list
        neurons = []
        for neuron_idx in range(ncount):
            neuron = network[1][neuron_idx * 260 : neuron_idx * 260 + 260]
            neurons.append([neuron, network[3][neuron_idx], network[4][neuron_idx]])  # Group the neuron with its good and bad counts
        
        # Sort the neurons
        if sort_method == 0:  # Increasing AIF
            neurons = sorted(neurons, key=lambda n: n[0][-3])
        elif sort_method == 1:  # Decreasing AIF
            neurons = sorted(neurons, key=lambda n: n[0][-3], reverse=True)
        elif sort_method == 2:  # Increasing bad firing count, secondarily sorted by decreasing good firing count
            neurons = sorted(neurons, key=lambda n: n[2] - n[1] * secondary_sort_weight)
        elif sort_method == 3:  # Decreasing good firing count, secondarily sorted by increasing bad firing count
            neurons = sorted(neurons, key=lambda n: n[1] - n[2] * secondary_sort_weight, reverse=True)
            
        # Convert the neuron list back into a 260X component list
        network_l = []
        for neuron, good_count, bad_count in neurons:
            network_l.extend(neuron)
        network_s = "{}".format(network_l).replace(' ', '')
        # Store the result
        sorted_networks.append((ncount, network_l, network_s))
    times.append(("Sort gathered networks", default_timer()))

    summarize_times("combined", times)
    
    return sorted_networks

def train_combined_batch_network(networks, contexts_to_use_lbl, filter_suffix, sort_method_lbl, secondary_sort_weight, verbose_level=0):
    '''
    Train a final network from the filtered, conglomerated, sorted neurons trained on multiple batches.
    
    Parameters:
        networks: One network per context (i.e., per NeuroShield)
        filter_suffix (str): The neuron filter descriptor
        sort_method_lbl (str): The neuron sort method descriptor
        secondary_sort_weight (float): The secondary sort weight <= 1 used by some sorting methods
    '''
    times = []
    times.append(("Start", default_timer()))
    
    # Train a network from the gathered networks of all rounds completed so far
    reset_neuromem_and_interface()
    train_network_iteratively(nm500s, networks, verbose_level=verbose_level)
    times.append(("Train combined network", default_timer()))

    networks = read_networks_from_neuroshields(contexts_to_use)
    read_err = False
    if any([network[0] is None for network in networks]):
        read_err = True
        print("An error occurred while reading networks from NeuroShields after reading networks")
    times.append(("Read", default_timer()))

    save_flattened_networks(networks, "k-{}_{}_batch-{}-{}-{}".format(crossfold_k, contexts_to_use_lbl, filter_suffix, "combined", sort_method_lbl), "full",
                            dir_path='./k{}/{}/batch_size-{}/{}/{}/'.format(k, contexts_to_use_lbl, batch_size, filter_suffix, sort_method_lbl))
    times.append(("Save full combined network", default_timer()))
    
    summarize_times("combined", times)
    
    return networks
nm500s = None
connect_to_nm500_pipes()
# Only initialize CATEGORIES and CATEGORIZED_IMGS once so that subsequent calls to prepare_data() don't waste time reloading the data over again
try:
    _ = CATEGORIES
    print("CATEGORIES has already been initialized")
except NameError as e:
    print("CATEGORIES has not been initialized yet")
    CATEGORIES, CATEGORIZED_IMGS = None, None

crossfold_k = None
categorized_validate_imgs = None
categorized_test_imgs = None
data_train_labels_dfs = None
n_patches_per_context = None
PATCH_WH = 16  # In theory, patches could be any rectangle that fits within 256 bytes, e.g., 32x8, but for this experiment I only used square patches
%%time
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
# NOTE: This cell cannot be run completely without the NeuroShield hardware,
# so some of it has been commented out. Sample output is provided in the next cell.
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

setup_crossfold(k=0)

# data_train_labels_dfs is a list of DataFrames, one per context.
# We can only use 4 of them, since we only have 4 NeuroShields.
# Select 4 of the DataFrames within data_train_labels_dfs.
contexts_to_use = [ # No more than four (ideally exactly four) contexts should be enabled since I only have four NeuroShields
#     (PATCH_AR_R_IDX, 'R'),
#     (PATCH_AR_G_IDX, 'G'),
#     (PATCH_AR_B_IDX, 'B'),
    (PATCH_AR_Y_IDX, 'Y'),
    (PATCH_AR_CR_IDX, 'Cr'),
    (PATCH_AR_CB_IDX, 'Cb'),
    (PATCH_AR_S_IDX, 'S'),
]
contexts_to_use_lbl = ''.join(context_to_use[1] for context_to_use in contexts_to_use)
data_train_labels_dfs_to_use = initialize_fresh_training(contexts_to_use)

# Set the batch size to exactly the network size. This approach insures that every training sample goes into a network for evaluation against the validation set.
# However, it will very likely produce unsaturated training networks, which will waste time during evaluation (it is optimal to evaluate as many patterns at a time as possible).
# It also produces suboptimal category-pair-boundary refinement since that is best achieved by training/evaluating as many patterns at a time as possible.
batch_size = 100  # 4032  #TEMP
num_batches = 2  # n_patches_per_context // batch_size

print("n_patches_per_context, num_batches, batch_size: {} {} {}".format(n_patches_per_context, num_batches, batch_size))

# train_independent_filtered_networks(data_train_labels_dfs_to_use, contexts_to_use, 1, num_batches, batch_size)
print()
%%time
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
# NOTE: This cell cannot be run completely without results files generated by previous cells,
# so some of it has been commented out. Sample output is provided in the next cell.
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# See determine_underperforming_neurons() for explanation of the neuron_filter options
# neuron_filter = [False, False, False, True]

# for k in range(1):  # TEMP 5):
#     filter_batches(crossfold_k=k, num_batches=2,  # TEMP
#                    neuron_filter=neuron_filter, contexts_to_use_lbl=contexts_to_use_lbl, max_pats_per_cat=batch_size, dir_path='./k{}/{}/batch_size-{}/'.format(k, contexts_to_use_lbl, batch_size))
%%time
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
# NOTE: This cell cannot be run completely without results files generated by previous cells,
# so some of it has been commented out. Sample output is provided in the next cell.
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# # Steadily increment a status variable as we proceed and verify it before running each cell.
# # This design enables us to deploy the next several cells in rapid succession and "walk away", since they take a while to complete,
# # confident that if the process crashes at any intermediate point, the remaining cells will bottom out.
# status = 0

# setup_crossfold(k=0)

# neuron_filter = [False, False, False, True]
# filter_suffix = ''.join(['t' if tf else 'f' for tf in neuron_filter])
# combined_networks = combine_batches(num_batches, contexts_to_use_lbl, batch_size, "./k{}/{}/batch_size-{}/".format(crossfold_k, contexts_to_use_lbl, batch_size), filter_suffix)
# print()
# status += 1
%%time
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
# NOTE: This cell cannot be run completely without results files generated by previous cells,
# so some of it has been commented out. Sample output is provided in the next cell.
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# if status == 1:
#     secondary_sort_weight = 0

#     sort_method = 3
#     if sort_method == 0:
#         sort_method_lbl = 'aif-inc'
#     elif sort_method == 1:
#         sort_method_lbl = 'aif-dec'
#     elif sort_method == 2:
#         sort_method_lbl = 'badgood-sw{}'.format(secondary_sort_weight)
#     elif sort_method == 3:
#         sort_method_lbl = 'goodbad-sw{}'.format(secondary_sort_weight)
#     print("sort_method, sort_method_lbl: {} {}".format(sort_method, sort_method_lbl))

#     sorted_networks = sort_combined_networks(combined_networks, sort_method=sort_method, secondary_sort_weight=secondary_sort_weight)
#     print()
#     status += 1
%%time
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
# NOTE: This cell cannot be run completely without the NeuroShield hardware,
# so some of it has been commented out. Sample output is provided in the next cell.
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# status = 2
# if status == 2:
#     # Convert the networks from their (ncount, network_l, network_s) format to DataFrames for training
#     dfs_to_use = []
#     for nwi, network in enumerate(sorted_networks):
#         context_to_use = contexts_to_use[nwi][0]
#         ncount = network[0]
#         network_l = network[1]
#         rows = []
#         for ni in range(ncount):
#             neuron = network_l[ni * 260 : ni * 260 + 260]
#             cxt, aif, minif, cat = neuron[0], neuron[257], neuron[258], neuron[259]
#             comps = neuron[1:257]
#             if cxt != context_to_use:
#                 print("Context mismatch [{}:{}]: {} != {}".format(nwi, ni, cxt, context_to_use))
#                 print(neuron)
#             row = [cat, comps]
#             rows.append(row)
#         df = [pd.DataFrame(rows, columns=['label', 'image'])]
#         df_to_use = [context_to_use, df]
#         dfs_to_use.append(df_to_use)

#     # Train the networks
#     networks = train_combined_batch_network(dfs_to_use, contexts_to_use_lbl, filter_suffix, sort_method_lbl=sort_method_lbl, secondary_sort_weight=secondary_sort_weight, verbose_level=0)
#     print()
#     status += 1
%%time
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
# NOTE: This cell cannot be run completely without the NeuroShield hardware,
# so some of it has been commented out. Sample output is provided in the next cell.
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# if status == 3:
#     good_neurons, bad_neurons, result_str = classify_images(categorized_test_imgs, contexts_to_use, "RBF", "best", "combined")
    
#     dir_path = "./k{}/{}/batch_size-{}/{}/runs/{}/".format(crossfold_k, contexts_to_use_lbl, batch_size, filter_suffix, sort_method_lbl)
#     os.makedirs(dir_path, exist_ok=True)
#     filename = "k-{}_{}_batch_size-{}_{}_{}_classify_test_images.txt".format(k, contexts_to_use_lbl, batch_size, filter_suffix, sort_method_lbl)
#     with open(dir_path + filename, 'w') as f:
#         f.write(result_str + '\n')
    
#     print()
#     status += 1
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
# NOTE: This cell cannot be run completely without results files generated by previous cells,
# so some of it has been commented out. Sample output is provided in the next cell.
# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

# for contexts_to_use_lbl in ['RGBS', 'YCrCbS']:
#     for batch_size in [4032]:
#         for filter_suffix in ['ffft', 'ftft']:
#             for sort_method_lbl in ['aif-dec', 'badgood-sw1', 'badgood-sw0', 'goodbad-sw1', 'goodbad-sw0']:
#                 scores = []
#                 for k in range(5):
#                     dir_path = "./k{}/{}/batch_size-{}/{}/runs/{}/".format(k, contexts_to_use_lbl, batch_size, filter_suffix, sort_method_lbl)
#                     filename = "k-{}_{}_batch_size-{}_{}_{}_classify_test_images.txt".format(k, contexts_to_use_lbl, batch_size, filter_suffix, sort_method_lbl)
#                     if os.path.exists(dir_path + filename):
#                         with open(dir_path + filename, 'r') as f:
#                             while True:
#                                 line = f.readline()
#                                 if not line:
#                                     break
#                                 if "Final counts for round_or_batch combined:" in line:
#                                     words = line.split()
#                                     counts = eval(line[len("Final counts for round_or_batch combined: "):])
#                                     counts = {k: v for k, v in counts}
#                                     total_count = sum([v for k, v in counts.items()])
#                                     score = counts['correct'] / total_count
#                                     print("K-{}    Score: {:5.1f}%    Total: {}    Counts: {}".format(k, score * 100, total_count, counts))
#                                     scores.append(score)
#                 if len(scores) > 0:
#                     if len(scores) != 5:
#                         print("ERROR: Did not retrieve 5 scores: {}".format(scores))
#                     scores = sorted(scores)
#                     score_mean = sum(scores) / len(scores)
#                     score_median = scores[len(scores) // 2] if len(scores) % 2 == 1 else (scores[len(scores) // 2] + scores[len(scores) // 2 + 1]) / 2
#                     print("{:8} {:>5} {:15} {}    ==>    sorted/mean/median scores:    {}    {:5.1f}%    {:5.1f}%".format( \
#                         contexts_to_use_lbl, batch_size, sort_method_lbl, filter_suffix, ' '.join(["{:4.1f}%".format(score * 100) for score in scores]), score_mean * 100, score_median * 100))
# print()
