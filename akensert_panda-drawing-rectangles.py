import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import cv2

import skimage.io
IMG_PATH = '../input/prostate-cancer-grade-assessment/train_images/'

data = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

data.head()
def read_img(ID, path, level=2):

    image_path = path + ID + '.tiff'

    img = skimage.io.MultiImage(image_path)

    img = img[level]

    return img



img = read_img(data.iloc[9].image_id, IMG_PATH)





# plot

plt.figure(figsize=(10, 10))

plt.imshow(img)

plt.title('original (level 2)', fontsize=20);
def img_to_gray(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray



img_gray = img_to_gray(img)





# plot

plt.figure(figsize=(10, 10))

plt.imshow(img_gray)

plt.title('original -> gray-scaled', fontsize=20);
def mask_img(img_gray, tol=240):

    mask = (img_gray < tol).astype(np.uint8)

    return mask



mask = mask_img(img_gray)





# plot

plt.figure(figsize=(10, 10))

plt.imshow(mask)

plt.title('gray-scaled -> binary mask', fontsize=20);
def compute_rect(mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = max(contours, key = cv2.contourArea) # select only the biggest rect

    rect = cv2.minAreaRect(contours)

    box = cv2.boxPoints(rect)

    box = np.int0(box)

    return rect, box



rect, box = compute_rect(mask)





# plot

img_with_rect = img.copy() # draw only on original img

cv2.drawContours(img_with_rect, [box], 0, (0, 0, 0), 2)

plt.figure(figsize=(10, 10))

plt.imshow(img_with_rect)

plt.title('binary mask -> get rect -> draw rect on original', fontsize=20);
def warp_perspective(img, rect, box):

    width = int(rect[1][0])

    height = int(rect[1][1])

    src_pts = box.astype("float32")

    dst_pts = np.array([[0, height-1],

                        [0, 0],

                        [width-1, 0],

                        [width-1, height-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped = cv2.warpPerspective(img, M, (width, height))

    return warped



img = warp_perspective(img, rect, box)





# plot

plt.figure(figsize=(10, 10))

plt.imshow(img)

plt.title('original -> rect -> warped perspective', fontsize=20);
%timeit img = read_img(data.iloc[1].image_id, IMG_PATH)

%timeit img_gray = img_to_gray(img)

%timeit mask = mask_img(img_gray)

%timeit rect, box = compute_rect(mask)

%timeit warped_img = warp_perspective(img, rect, box)
fig, axes = plt.subplots(10, 2, figsize=(10, 40))



for i, ax in enumerate(axes):

    img = read_img(data.iloc[i].image_id, IMG_PATH)

    img_gray = img_to_gray(img)

    mask = mask_img(img_gray)

    rect, box = compute_rect(mask)

    img_with_rect = img.copy() # draw only on original

    cv2.drawContours(img_with_rect, [box], 0, (0, 0, 0), 2)

    warped_img = warp_perspective(img, rect, box)

    ax[0].imshow(img_with_rect)

    ax[1].imshow(warped_img)

IMG_PATH = '../input/prostate-cancer-grade-assessment/train_images/'

data = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')



def read_img(ID, path, level=2):

    image_path = path + ID + '.tiff'

    img = skimage.io.MultiImage(image_path)

    img = img[level]

    return img



def img_to_gray(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img_gray



def mask_img(img_gray, tol=210):

    mask = (img_gray < tol).astype(np.uint8)

    return mask

# plots

fig, axes = plt.subplots(3, 3, figsize=(17, 12))



# first image

img1 = read_img(data.iloc[9].image_id, IMG_PATH)

img_gray1 = img_to_gray(img1)

mask1 = mask_img(img_gray1)



axes[0, 0].imshow(img1)

axes[0, 0].axis('off');

axes[1, 0].imshow(img_gray1)

axes[1, 0].axis('off');

axes[2, 0].imshow(mask1)

axes[2, 0].axis('off');



# second image

img2 = read_img(data.iloc[99].image_id, IMG_PATH)

img_gray2 = img_to_gray(img2)

mask2 = mask_img(img_gray2)



axes[0, 1].imshow(img2)

axes[0, 1].axis('off');

axes[1, 1].imshow(img_gray2)

axes[1, 1].axis('off');

axes[2, 1].imshow(mask2)

axes[2, 1].axis('off');





# third image

img3 = read_img(data.iloc[999].image_id, IMG_PATH)

img_gray3 = img_to_gray(img3)

mask3 = mask_img(img_gray3)



axes[0, 2].imshow(img3)

axes[0, 2].axis('off');

axes[1, 2].imshow(img_gray3)

axes[1, 2].axis('off');

axes[2, 2].imshow(mask3)

axes[2, 2].axis('off');



plt.subplots_adjust(hspace=0.0, wspace=0.0)
def compute_rects(mask, min_rect=32):

    

    contours, _ = cv2.findContours(

        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    height, width = mask.shape



    boxes = []

    dims = []

    for contour in contours:

        rect = cv2.minAreaRect(contour)

        height = rect[1][1]

        width = rect[1][0]

        if width > min_rect and height > min_rect:

            dims.append((int(width), int(height)))

            box = cv2.boxPoints(rect)

            box = np.int0(box)

            boxes.append(box)

    

    return dims, boxes



dims, boxes = compute_rects(mask1)





# plot

for box in boxes:

    cv2.drawContours(img1, [box], 0, (0, 0, 0), 2)

plt.figure(figsize=(10, 10))

plt.imshow(img1);

plt.axis('off');
def warp_perspectives(img, dims, boxes):



    imgs = []

    for (width, height), box in zip(dims, boxes):

        src_pts = box.astype("float32")

        dst_pts = np.array([[0, height-1],

                            [0, 0],

                            [width-1, 0],

                            [width-1, height-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(img, M, (width, height))

        imgs.append(warped)

    return imgs

    

imgs = warp_perspectives(img1, dims, boxes)





# plot

fig, axes = plt.subplots(1, len(imgs), figsize=(10, 10*len(imgs)))

for im, ax in zip(imgs, axes.reshape(-1)):

    ax.imshow(im)

    ax.axis('off')

plt.subplots_adjust(hspace=0.1)
IMG_PATH = '../input/prostate-cancer-grade-assessment/train_images/'

MASK_PATH = '../input/prostate-cancer-grade-assessment/train_label_masks/'



data = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')





def read_mask(ID, path, level=2):

    mask_path = path + ID + '_mask.tiff'

    mask = skimage.io.MultiImage(mask_path)

    mask = mask[level]

    return mask[:, :, 0]



def read_img(ID, path, level=2):

    image_path = path + ID + '.tiff'

    img = skimage.io.MultiImage(image_path)

    img = img[level]

    return img



def compute_rects(mask, min_rect=32):

    

    contours, _ = cv2.findContours(

        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



    height, width = mask.shape



    boxes = []

    dims = []

    for contour in contours:

        rect = cv2.minAreaRect(contour)

        height = rect[1][1]

        width = rect[1][0]

        if width > min_rect and height > min_rect:

            dims.append((int(width), int(height)))

            box = cv2.boxPoints(rect)

            box = np.int0(box)

            boxes.append(box)

    

    return dims, boxes



def warp_perspectives(img, dims, boxes):



    imgs = []

    for (width, height), box in zip(dims, boxes):

        src_pts = box.astype("float32")

        dst_pts = np.array([[0, height-1],

                            [0, 0],

                            [width-1, 0],

                            [width-1, height-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(img, M, (width, height))

        imgs.append(warped)

    return imgs



# leave these functions out for now

# def img_to_gray(img):

#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     return img_gray



# def mask_img(img_gray, tol=210):

#     mask = (img_gray < tol).astype(np.uint8)

#     return mask
ID = data.iloc[10606].image_id

mask = read_mask(ID, MASK_PATH)

img = read_img(ID, IMG_PATH)

mask_background = np.where(mask == 0, 1, 0).astype(np.uint8)

mask_benign = np.where(mask == 1, 1, 0).astype(np.uint8)

mask_cancerous = np.where(mask == 2, 1, 0).astype(np.uint8)



fig, axes = plt.subplots(1, 3, figsize=(12, 6))



axes[0].imshow(mask_background.astype(float), cmap=plt.cm.gray)

axes[0].axis('off')

axes[0].set_title('background');

axes[1].imshow(mask_benign.astype(float), cmap=plt.cm.gray)

axes[1].axis('off')

axes[1].set_title('benign');

axes[2].imshow(mask_cancerous.astype(float), cmap=plt.cm.gray)

axes[2].axis('off')

axes[2].set_title('cancerous');

#plt.imshow(mask.astype(np.uint8));

# hardcoding this

dims, boxes = compute_rects(mask_benign)

img_benign = warp_perspectives(img, dims, boxes)



dims, boxes = compute_rects(mask_cancerous.astype(np.uint8))

img_cancerous = warp_perspectives(img, dims, boxes)



fig, axes = plt.subplots(1, 3, figsize=(10, 8))

axes[0].imshow(img_benign[1])

axes[0].set_title("benign")

axes[1].imshow(img_benign[0])

axes[1].set_title("benign")

axes[2].imshow(img_cancerous[0])

axes[2].set_title("cancerous");
ID = data.iloc[10612].image_id

mask = read_mask(ID, MASK_PATH, level=1)

img = read_img(ID, IMG_PATH, level=1)

mask_background = np.where(mask == 0, 1, 0).astype(np.uint8)

mask_benign = np.where(mask == 1, 1, 0).astype(np.uint8)

mask_cancerous = np.where(mask == 5, 1, 0).astype(np.uint8)



fig, axes = plt.subplots(3,1, figsize=(8, 8))



axes[0].imshow(mask_background.astype(float), cmap=plt.cm.gray)

axes[0].axis('off')

axes[0].set_title('background');

axes[1].imshow(mask_benign.astype(float), cmap=plt.cm.gray)

axes[1].axis('off')

axes[1].set_title('benign');

axes[2].imshow(mask_cancerous.astype(float), cmap=plt.cm.gray)

axes[2].axis('off')

axes[2].set_title('cancerous');

# rewriting compute_rects to take into account hierarchies 

def compute_rects(mask, min_rect=32):

    

    contours, hierarchy = cv2.findContours(

        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    height, width = mask.shape



    boxes = []

    dims = []

    for (contour, hier) in zip(contours, hierarchy[0]):

        rect = cv2.minAreaRect(contour)

        height = rect[1][1]

        width = rect[1][0]

        if width > min_rect and height > min_rect and hier[3] == -1:

            dims.append((int(width), int(height)))

            box = cv2.boxPoints(rect)

            box = np.int0(box)

            boxes.append(box)

    

    return dims, boxes



dims, boxes = compute_rects(mask_cancerous, min_rect=32)





# plot

img_with_rects = img.copy()

mask_cancerous_with_rects = mask_cancerous.copy()

for box in boxes:

    cv2.drawContours(mask_cancerous_with_rects, [box], 0, 1, 5)

    cv2.drawContours(img_with_rects, [box], 0, 1, 5)



fig, axes = plt.subplots(1,2, figsize=(20, 20))

axes[0].imshow(mask_cancerous_with_rects)

axes[0].axis('off')

axes[0].set_title('Too many rectangles are drawn and they are also too small')

axes[1].imshow(img_with_rects)

axes[1].axis('off');

axes[1].set_title('Original');
def dilate_mask(mask, kernel_size=0.015, iterations=5):

    # 'kernel_size' and 'iterations' needs finetunning

    h, w = mask.shape

    kernel_size = int(min(h, w) * kernel_size)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)

    return dilated_mask



mask_cancerous_dilated = dilate_mask(mask_cancerous)



# plot

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].imshow(img)

axes[0].axis('off')

axes[0].set_title("original")

axes[1].imshow(mask_cancerous)

axes[1].axis('off')

axes[1].set_title("mask")

axes[2].imshow(mask_cancerous_dilated)

axes[2].axis('off')

axes[2].set_title("mask dilated");
# rewriting compute_rects to take into account hierarchies 

def compute_rects(mask, min_rect=32):

    

    contours, hierarchy = cv2.findContours(

        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    height, width = mask.shape



    boxes = []

    dims = []

    for (contour, hier) in zip(contours, hierarchy[0]):

        rect = cv2.minAreaRect(contour)

        height = rect[1][1]

        width = rect[1][0]

        if width > min_rect and height > min_rect and hier[3] == -1:

            dims.append((int(width), int(height)))

            box = cv2.boxPoints(rect)

            box = np.int0(box)

            boxes.append(box)

    

    return dims, boxes



dims, boxes = compute_rects(mask_cancerous_dilated, min_rect=32)





# plot

img_with_rects = img.copy()

for box in boxes:

    cv2.drawContours(mask_cancerous_dilated, [box], 0, 1, 5)

    cv2.drawContours(img_with_rects, [box], 0, 1, 5)



fig, axes = plt.subplots(1,2, figsize=(20, 20))

axes[0].imshow(mask_cancerous_dilated)

axes[0].axis('off')

axes[0].set_title('Still too many rectangles?')

axes[1].imshow(img_with_rects)

axes[1].axis('off');

axes[1].set_title('Original');
# let's rewrite dilate_mask to remove the "spurious" islands

def erode_dilate_mask(mask, 

                      kernel_size_closing=0.015, 

                      kernel_size_opening=0.025,

                      kernel_size_final_dilate=0.025,

                      final_dilate_iterations=5):

    

    """

    This implementation is likely far from perfect and probably also overly complicated.

    

    skimage.morphology.remove_small_objects could, in one line of code,

    also be used to filter out some small islands. 

    

    Note:

      'kernel_size_closing', 'kernel_size_opening', 'kernel_size_final_dilate' and 

      'final_dilate_iterations' all needs to be carefully experimented with and finetuned

    """

    



    h, w = mask.shape

    

    # the bigger the closing, the more we keep islands that are sufficiently big

    kernel_size_closing = int(min(h, w) * kernel_size_closing)

    # the bigger the opening, the more we remove islands of bigger sizes

    kernel_size_opening = int(min(h, w) * kernel_size_opening) 

    # the bigger the final dilate, the bigger rectangles we get

    kernel_size_final_dilate = int(min(h, w) * kernel_size_final_dilate) 

    

    # obtain elliptic kernel

    # e.g. np.ones((kernel_size, kernel_size) could also be used as a kernel

    kernel_closing = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE, (kernel_size_closing, kernel_size_closing))

    kernel_opening = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE, (kernel_size_opening, kernel_size_opening))

    kernel_final_dilate = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE, (kernel_size_final_dilate, kernel_size_final_dilate))

    

    # cv2.MORPH_CLOSE = erode -> dilation

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_closing)

    # cv2.MORPH_OPEN = dilation -> erode

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)

    # final dilate

    mask = cv2.dilate(mask, kernel_final_dilate, iterations=final_dilate_iterations)

    

    return mask



mask_cancerous_dilated = erode_dilate_mask(mask_cancerous)



# plot

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].imshow(img)

axes[0].axis('off')

axes[0].set_title("original")

axes[1].imshow(mask_cancerous)

axes[1].axis('off')

axes[1].set_title("mask")

axes[2].imshow(mask_cancerous_dilated)

axes[2].axis('off')

axes[2].set_title("mask eroded + dilated");
dims, boxes = compute_rects(mask_cancerous_dilated, min_rect=32)



# plot

img_with_rects = img.copy()

for box in boxes:

    cv2.drawContours(mask_cancerous_dilated, [box], 0, 1, 10)

    cv2.drawContours(img_with_rects, [box], 0, 1, 5)



fig, axes = plt.subplots(1,2, figsize=(14, 8))

axes[0].imshow(mask_cancerous_dilated)

axes[0].axis('off')

axes[0].set_title('Now only two rectangles are drawn')

axes[1].imshow(img_with_rects)

axes[1].axis('off');

axes[1].set_title('Original');
dims, boxes = compute_rects(mask_cancerous_dilated, min_rect=32)



imgs = warp_perspectives(img, dims, boxes)



fig, axes = plt.subplots(1, len(imgs), figsize=(8 * len(imgs), 8))



if isinstance(axes, np.ndarray):

    axes = axes.reshape(-1)

else:

    axes = [axes]

    

for ax, img in zip(axes, imgs):

    ax.imshow(img)

    ax.axis('off')