import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# all classes 
classes = ["brightpixel",
            "narrowband",
            "narrowbanddrd",
            "noise",
            "squarepulsednarrowband",
            "squiggle",
            "squigglesquarepulsednarrowband"]
num_images = 2
for _class in classes:
    # start off by observing images
    path = os.path.join("../input/primary_small/train", _class)
    image_files = os.listdir(path)
    random_images = random.sample(range(0, len(image_files)-1), num_images)
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(12, 14), squeeze=False)
    fig.tight_layout()
    for l in range(1):
        for m in range(num_images):
            axes[l][m].imshow(cv2.imread(os.path.join(path, image_files[random_images[m]]), 0), cmap="gray")
            axes[l][m].axis("off")
            axes[l][m].set_title(_class)
# done displaying
def display(image):
    fig = plt.figure(figsize=(9, 11))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()
_random = random.choice(os.listdir(os.path.join("../input/primary_small/train/narrowband")))
# lets read the image and display it
narrowband = cv2.imread(os.path.join("../input/primary_small/train/narrowband", _random))
display(narrowband)
# convert from BGR to Grayscale
narrowband = cv2.cvtColor(narrowband, cv2.COLOR_BGR2GRAY)
display(narrowband)
# now let's extract some features from the image
low = np.min(narrowband)
high = np.max(narrowband)
mean = np.mean(narrowband)
std = np.std(narrowband)
variance = np.var(narrowband)
# print
print("Min: {}".format(low))
print("Max: {}".format(high))
print("Mean: {}".format(mean))
print("Standard Deviation: {}".format(std))
print("Variance: {}".format(variance))
clipped = np.clip(narrowband, mean-3.5*std, mean+3.5*std)
# print
print("Min: {}".format(np.min(clipped)))
print("Max: {}".format(np.max(clipped)))
display(clipped)
# Gaussian blurr
gaussian = cv2.GaussianBlur(narrowband, (3, 3), 1)
print("Min: {}".format(np.min(gaussian)))
print("Max: {}".format(np.max(gaussian)))
print("Mean: {}".format(np.mean(gaussian)))
display(gaussian)
# lets do a morphological closing on the clipped image which is dilation + erosion
morphed = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.float32))
display(morphed)
# we'll start by applying sobel edge detection along x-axis
sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 2)
display(sobelx)
# let's apply sobel ege detection along y-axis
sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 2)
display(sobely)
blended = cv2.addWeighted(src1=sobelx, alpha=0.7, src2=sobely, beta=0.3, gamma=0)
display(blended)
def process_image(image):
    # grayscale conversion
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # clip intensities
    mean = np.mean(image)
    std = np.std(image)
    image = np.clip(image, mean-3.5*std, mean+3.5*std)
    # morph close 
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.float32))
    # gradient in both directions
    sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 2)
    sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 2)
    # blend 
    blended = cv2.addWeighted(src1=sobelx, alpha=0.7, src2=sobely, beta=0.3, gamma=0)
    return blended

for _class in classes:
    # start off by observing images
    path = os.path.join("../input/primary_small/train", _class)
    image_files = os.listdir(path)
    random_images = random.sample(range(0, len(image_files)-1), num_images)
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(11, 12), squeeze=False)
    fig.tight_layout()
    for l in range(1):
        for m in range(num_images):
            axes[l][m].imshow(process_image(cv2.imread(os.path.join(path, image_files[random_images[m]]))), cmap="gray")
            axes[l][m].axis("off")
            axes[l][m].set_title(_class)
# done displaying