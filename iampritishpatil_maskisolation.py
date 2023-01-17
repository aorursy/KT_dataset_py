import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob, os, os.path
from scipy import ndimage
def mask_in_red(img, mask):
    # returns a copy of the image with edges of the mask added in red
    img_color = grays_to_RGB(img)
    mask_edges = cv2.Canny(mask, 100, 200) > 0  
    img_color[mask_edges, 0] = 255  # setting channel 0 = red 
    return img_color
def sobel_filter(img):
    img = ndimage.gaussian_filter(img, 8);
    sx = ndimage.sobel(img, axis=0, mode='reflect');
    sy = ndimage.sobel(img, axis=1, mode='reflect');
    processed = np.hypot(sx, sy);
    processed = np.invert(processed > 150);
    return processed
def segmentation(img):
    img = np.invert(img)
    mask = (img > img.mean()).astype(np.float)
    mask += 0.1 * img
    return mask>10;
def image_with_mask(img, mask):
    #crops the image to return the masked area.
    img = img[np.sum(mask, axis=1) > 0,...];
    img = img[...,np.sum(mask, axis=0) > 0];
    return grays_to_RGB(img);

def fimg_to_fmask(img_path):
    # convert an image file path into a corresponding mask file path 
    dirname, basename = os.path.split(img_path)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)

def mask_not_blank(mask):
    return sum(mask.flatten()) > 0

def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img)) 

def plot_image(img_array, title=None):
    img_view = plt.figure(figsize=(15,5))
    plt.title(title);
    for n, img in enumerate(img_array):
        img_view.add_subplot(1, len(img_array), n+1)
        plt.imshow(img)
    plt.show()
IMAGES_TO_SHOW = 20  # configure to taste :)
f_ultrasounds = [img for img in glob.glob("../input/train/*.tif") if 'mask' not in img]
# f_ultrasounds.sort()  
f_masks = [fimg_to_fmask(fimg) for fimg in f_ultrasounds]

images_shown = 0 

for f_ultrasound, f_mask in zip(f_ultrasounds, f_masks):

    img  = plt.imread(f_ultrasound)
    mask = plt.imread(f_mask)

    if mask_not_blank(mask):

        # plot_image(grays_to_RGB(img),  title=f_ultrasound)
        # plot_image(grays_to_RGB(mask), title=f_mask)

        f_combined = f_ultrasound + " & " + f_mask
        cropped_img = image_with_mask(img, mask);
        processed_img = segmentation(cropped_img)
        #plot_image([grays_to_RGB(img), cropped_img, processed_img], title=f_combined)
        plot_image([mask_in_red(img, mask), cropped_img, processed_img], title=f_combined)
        print('plotted:', f_combined)
        images_shown += 1

    if images_shown >= IMAGES_TO_SHOW:
        break