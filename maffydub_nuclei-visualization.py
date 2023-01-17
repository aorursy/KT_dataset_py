TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'
FIG_SIZE = (20, 30)
import math
import itertools
import os
import numpy as np
import skimage.io as io
import pandas as pd
from scipy import ndimage
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
%matplotlib inline
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
def hsv_to_rgb(h, s, v):
    h -= math.floor(h)
    h = h * 6
    if h < 1:
        [r, g, b] = [1, h, 0]
    elif h < 2:
        [r, g, b] = [2 - h, 1, 0]
    elif h < 3:
        [r, g, b] = [0, 1, h - 2]
    elif h < 4:
        [r, g, b] = [0, 4 - h, 1]
    elif h < 5:
        [r, g, b] = [h - 4, 0, 1]
    else:
        [r, g, b] = [1, 0, 6 - h]
    return [((r - 0.5) * s + 0.5) * v, ((g - 0.5) * s + 0.5) * v, ((b - 0.5) * s + 0.5) * v]

hash_colorize = np.array([[0, 0, 0] if i == 0 else
                              hsv_to_rgb(((i + 1) % 17)/17,
                                         ((i + 1) % 3)/3 / 2 + 0.5,
                                         ((i + 1) % 5)/5 * 2/3 + 1/3)
                          for i in range(256)])

def mask_to_edges(mask):
    mask_edges = np.zeros(mask.shape).astype(np.uint8)
    for ii in range(1, mask.max() + 1):
        submask = mask.copy()
        submask[submask != ii] = 0
        submask[submask == ii] = 1
        submask_edges = submask - ndimage.binary_erosion(submask, structure=np.ones((5,5)))
        mask_edges += submask_edges * ii
    return mask_edges

def visualize_mask(mask, img=None, edges=True, fill=True):
    if fill:
        if img is None:
            img = mask.copy()
            img[img > 0] = 1
        img = np.multiply(np.repeat(np.reshape(img, img.shape + (1,)), 3, 2), hash_colorize[mask])
    else:
        if img is None:
            img = np.zeros(mask.shape)
        img = np.repeat(np.reshape(img, img.shape + (1,)), 3, 2)
    if edges:
        img = np.clip(img + hash_colorize[mask_to_edges(mask)], 0, 1)
    return img
def remove_alpha(img):
    return img[:, :, 0:3]

def grayscale(img):
    if img.shape[2] == 4:
        img = remove_alpha(img)
    return img.mean(axis=2)

def normalize(img):
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = remove_alpha(img)
    img_max = img.max()
    img_min = img.min()
    img = (img - img_min) / (img_max - img_min)
    return img

def flatten_masks(masks):
    mask = np.zeros(masks[0].shape).astype(np.uint8)
    for ii, submask in enumerate(masks, 1):
        submask[submask > 0] = 1
        mask += submask * ii
    return mask

class NucleiDataset():
    def __init__(self, path, transform = None):
        self.path = path
        self.transform = transform
        self.ids = next(os.walk(path))[1]

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        return self.get(id)
    
    def get(self, id):
        path = self.path + id
        image = io.imread(path + "/images/" + id + ".png")
        masks = []
        mask_path = path + "/masks/"
        for mask in next(os.walk(mask_path))[2]:
            masks.append(io.imread(mask_path + mask))
        sample = {"id": id, "image": image, "masks": masks}
        if self.transform:
            self.transform(sample)
        return sample        

def normalize_and_flatten(sample):
    sample["image"] = normalize(grayscale(sample["image"]))
    sample["mask"] = flatten_masks(sample["masks"])
    del sample["masks"]

train_set = NucleiDataset(TRAIN_PATH)
test_set = NucleiDataset(TEST_PATH)
def plot_image(fig, sample, plot_spec):
    image = sample["image"]

    inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                    subplot_spec=plot_spec, wspace=0.1, hspace=0.1)
    for j in range(2):
        ax = plt.Subplot(fig, inner[j])
        if j == 0:
            ax.imshow(normalize(image))
        else:
            ax.imshow(visualize_mask(flatten_masks(sample["masks"]), img=normalize(grayscale(image))))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
fig = plt.figure(figsize=(FIG_SIZE[1], FIG_SIZE[1]))
outer = gridspec.GridSpec(8, 4, wspace=0.2, hspace=0.2)

for (i, sample) in enumerate(itertools.islice(train_set, 32)):
    plot_image(fig, sample, outer[i])
image_ids = []
image_sizes = []
for sample in train_set:
    image = sample["image"]
    image_ids.append(sample["id"])
    image_sizes.append([image.shape[1], image.shape[0]])
image_ids = np.array(image_ids)
image_sizes = np.array(image_sizes)
images = np.rec.fromarrays((image_ids, image_sizes[:, 0], image_sizes[:, 1]),
                           dtype=[('id', image_ids.dtype), ('w', int), ('h', int)])
pd.DataFrame.from_records(images, index=('id'))
print("Range: (%d - %d) x (%d - %d)" % (images.w.min(), images.w.max(), images.h.min(), images.h.max()))
print("Mean: %d x %d" % (images.w.mean(), images.h.mean()))
aspect_ratio = images.w / images.h
def aspect_ratio_to_str(ratio):
    if ratio >= 1:
        return "%f : 1" % (ratio,)
    else:
        return "1 : %f" % (1 / ratio,)
print("Aspect ratio range: (%s) - (%s)" % (aspect_ratio_to_str(aspect_ratio.min()), aspect_ratio_to_str(aspect_ratio.max())))
fig = plt.figure(figsize=(FIG_SIZE[1], FIG_SIZE[1]))
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

smallest_image = images[(images.w * images.h).argmin()]
biggest_image = images[(images.w * images.h).argmax()]

plot_image(fig, train_set.get(smallest_image.id), outer[0])
plot_image(fig, train_set.get(biggest_image.id), outer[1])
fig = plt.figure(figsize=(FIG_SIZE[1], FIG_SIZE[1]))
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

lowest_aspect_image = images[(images.w / images.h).argmin()]
highest_aspect_image = images[(images.w / images.h).argmax()]

plot_image(fig, train_set.get(lowest_aspect_image.id), outer[0])
plot_image(fig, train_set.get(highest_aspect_image.id), outer[1])
mask_ids = []
mask_idcs = []
mask_bounds = []
mask_pixel_counts = []
mask_overlap_counts = []
for sample in train_set:
    image = sample["image"]
    accumulated_mask = np.zeros(image.shape[0:len(image.shape)-1])
    for (mask_idx, mask) in enumerate(sample["masks"]):
        mask_ids.append(sample["id"])
        mask_idcs.append(mask_idx)
        x_positions = np.any(mask, axis=0).nonzero()[0]
        y_positions = np.any(mask, axis=1).nonzero()[0]
        mask_bounds.append([int(x_positions.min()), int(y_positions.min()), int(x_positions.max()), int(y_positions.max())])
        mask_pixel_counts.append(int(np.count_nonzero(mask)))
        mask_overlap_counts.append(int(np.count_nonzero(accumulated_mask * mask)))
        accumulated_mask += mask
mask_ids = np.array(mask_ids)
mask_idcs = np.array(mask_idcs)
mask_bounds = np.array(mask_bounds)
mask_pixel_counts = np.array(mask_pixel_counts)
mask_overlap_counts = np.array(mask_overlap_counts)
width = mask_bounds[:, 2] - mask_bounds[:, 0] + 1
height = mask_bounds[:, 3] - mask_bounds[:, 1] + 1
area = width * height
density = mask_pixel_counts / area
masks = np.rec.fromarrays((mask_ids, mask_idcs, mask_bounds[:, 0], mask_bounds[:, 1], mask_bounds[:, 2], mask_bounds[:, 3], mask_pixel_counts, width, height, area, density, mask_overlap_counts),
                          dtype=[('id', mask_ids.dtype), ('mask', int), ('x1', int), ('y1', int), ('x2', int), ('y2', int), ('count', int), ('w', int), ('h', int), ('area', int), ('density', float), ('overlap', int)])
pd.DataFrame.from_records(masks, index=('id', 'mask'))
print("Range: (%d - %d) x (%d - %d)" % (masks.w.min(), masks.w.max(), masks.h.min(), masks.h.max()))
print("Mean: %d x %d" % (masks.w.mean(), masks.h.mean()))
aspect_ratio = masks.w / masks.h
def aspect_ratio_to_str(ratio):
    if ratio >= 1:
        return "%f : 1" % (ratio,)
    else:
        return "1 : %f" % (1 / ratio,)
print("Aspect ratio range: (%s) - (%s)" % (aspect_ratio_to_str(aspect_ratio.min()), aspect_ratio_to_str(aspect_ratio.max())))
print("Mean density within bounds: %f%%" % (100. * masks.count.sum() / (masks.w * masks.h).sum(),))
print("Mean density across all pixels: %f%%" % (100. * masks.count.sum() / (images.w * images.h).sum(),))
print("Total mask overlap: %d" % (masks.overlap.sum()))
# Clip a mask out
def clip(img, mask_rec, border, pad=False):
    x1 = max(mask_rec.x1-border, 0)
    x2 = min(mask_rec.x2+border, img.shape[1])
    y1 = max(mask_rec.y1-border, 0)
    y2 = min(mask_rec.y2+border, img.shape[0])
    img = img[y1:y2, x1:x2]
    if pad:
        img = np.pad(img, ((y1-mask_rec.y1+border, mask_rec.y2+border-y2), (x1-mask_rec.x1+border, mask_rec.x2+border-x2), (0, 0)), mode='edge')
    return img

# Remove any padding added by previous clip
def reclip(img, clipped, mask_rec, border):
    x1 = max(mask_rec.x1-border, 0)
    x2 = min(mask_rec.x2+border, img.shape[1])
    y1 = max(mask_rec.y1-border, 0)
    y2 = min(mask_rec.y2+border, img.shape[0])
    clipped = clipped[y1-mask_rec.y1+border:clipped.shape[0] - mask_rec.y2-border+y2, x1-mask_rec.x1+border:clipped.shape[1] - mask_rec.x2-border+x2]
    return clipped

def plot_mask(fig, dataset, mask_rec, plot_spec):
    sample = dataset.get(mask_rec.id)
    image = sample["image"]
    mask = sample["masks"][mask_rec.mask]

    inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                    subplot_spec=plot_spec, wspace=0.1, hspace=0.1)
    for j in range(2):
        ax = plt.Subplot(fig, inner[j])
        if j == 0:
            ax.imshow(clip(normalize(image), mask_rec, 5))
        else:
            ax.imshow(reclip(image,
                    visualize_mask(clip(np.expand_dims(flatten_masks(sample["masks"]), 2), mask_rec, 5, pad=True)[..., 0],
                                   img=normalize(grayscale(clip(image, mask_rec, 5, pad=True)))), mask_rec, 5))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
masks_shuffled = np.copy(masks)
np.random.shuffle(masks_shuffled)

fig = plt.figure(figsize=(FIG_SIZE[1], FIG_SIZE[1]))
outer = gridspec.GridSpec(8, 4, wspace=0.2, hspace=0.2)

for (i, mask_rec) in enumerate(itertools.islice(masks_shuffled, 32)):
    plot_mask(fig, train_set, mask_rec, outer[i])

plt.show()
fig = plt.figure(figsize=(FIG_SIZE[1], FIG_SIZE[1]))
outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)

lowest_density_mask = masks[masks.density.argmin()]
highest_density_mask = masks[masks.density.argmax()]

plot_mask(fig, train_set, lowest_density_mask, outer[0])
plot_mask(fig, train_set, highest_density_mask, outer[1])
rgba_max = np.empty((0, 4))
rgba_min = np.empty((0, 4))
rgba_mean = np.empty((0, 4))
for sample in train_set:
    image = sample["image"]
    image = image.reshape(-1, image.shape[2])/255.0
    rgba_max = np.concatenate((rgba_max, np.expand_dims(image.max(axis=0), 0)))
    rgba_min = np.concatenate((rgba_min, np.expand_dims(image.min(axis=0), 0)))
    rgba_mean = np.concatenate((rgba_mean, np.expand_dims(image.mean(axis=0), 0)))
    #np.linalg.svd(np.cov(image.T))
    
rgba_max = rgba_max.max(axis=0)
rgba_min = rgba_min.min(axis=0)
print("Maximum: ", rgba_max)
print("Minimum: ", rgba_min)
print("Mean: ", rgba_mean.mean(axis=0))
print("Alpha channel used? ", rgba_max[3] != 255 or rgba_min[3] != 255)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
sample = train_set.get("7f34dfccd1bc2e2466ee3d6f74ff05821a0e5404e9cf2c9568da26b59f7afda5")
image = sample["image"]
colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
color_codes = np.apply_along_axis(lambda c: matplotlib.colors.rgb2hex(c / 255.0), 1, colors)
fig = plt.figure(figsize=(20, 40))
fig.suptitle('Color distribution')
ax = fig.add_subplot(2, 1, 1, projection='3d')
ax.view_init(30, 0)
ax.scatter(colors[..., 0] / 255.0 - 0.5, colors[..., 1] / 255.0 - 0.5, colors[..., 2] / 255.0 - 0.5, c=color_codes)
ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.view_init(30, 90)
ax.scatter(colors[..., 0] / 255.0 - 0.5, colors[..., 1] / 255.0 - 0.5, colors[..., 2] / 255.0 - 0.5, c=color_codes)
plt.show()
#sample = train_set.get("7f34dfccd1bc2e2466ee3d6f74ff05821a0e5404e9cf2c9568da26b59f7afda5")
sample = train_set.__getitem__(0)
image = sample["image"]
pixels = image.reshape(-1, image.shape[2])/255.0
mean = pixels.mean(axis=0)
np.linalg.svd(np.cov(pixels.T))
mask_size = np.empty((0, 2))
for sample in train_set:
    for mask in sample["masks"]:
        sum = mask.sum()
        data = [sum, sum / mask.size]
        mask_size = np.concatenate((mask_size, np.reshape(data, (1, 2))))
print("Mask size: ")
print("Maximum: ", mask_size.max(axis=0))
print("Minimum: ", mask_size.min(axis=0))
print("Mean: ", mask_size.mean(axis=0))
print("Variance: ", mask_size.var(axis=0))