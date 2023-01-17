import os

import cv2

import numpy as np

import matplotlib.pyplot as plt





# For reproducibility purposes

np.random.seed(42)



def get_random_images(images_path, n=5):

    image_collection = []

    

    for i in np.random.permutation(len(images_path))[:n]:

        image = cv2.imread(images_path[i])

        image_collection.append(image)

        

    return image_collection





# We obtain the path of all the original images

base_path = "../input/Captcha Dataset/original"

images_path = list(map(lambda x: os.path.join(base_path, x), os.listdir(base_path)))



random_images = get_random_images(images_path, n=9)



plt.figure(figsize=(9, 3))



for i, image in enumerate(random_images):

    plt.subplot(3, 3, i + 1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')



plt.suptitle('Random Captchas')

plt.tight_layout()

plt.show()
def plot_histogram(image, colorspace):

    chans = cv2.split(image)

    

    plt.title(f"{colorspace} Histogram")

    plt.xlabel("Bins")

    plt.ylabel("# of Pixels")

    

    for (chan_data, channel_name) in zip(chans, list(colorspace)):

        if colorspace == "BGR":

            hist = cv2.calcHist([chan_data], [0], None, [256], [0, 256])

            plt.plot(hist, label=channel_name, color=channel_name.lower())

        elif colorspace == "HSV":

            if channel_name == "H":

                hist = cv2.calcHist([chan_data], [0], None, [180], [0, 256])

            else:

                hist = cv2.calcHist([chan_data], [0], None, [256], [0, 256])

                

            plt.plot(hist, label=channel_name)

        

    

    plt.xlim([0, 256])

    plt.legend(loc="upper right")
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)

plt.imshow(cv2.cvtColor(random_images[0], cv2.COLOR_BGR2RGB))

plt.axis('off')

plt.subplot(3, 1, 2)

plot_histogram(image, "BGR")

plt.subplot(3, 1, 3)

plot_histogram(cv2.cvtColor(random_images[0], cv2.COLOR_BGR2HSV), "HSV")

plt.tight_layout()

plt.show()
random_images = get_random_images(images_path)



for image in random_images:

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    channels = cv2.split(image)

    

    plt.figure(figsize=(20, 10))

    

    for index, channel in enumerate(channels):

        plt.subplot(1, 3, 1 + index)

        plt.imshow(channel, cmap="gray")

        

    plt.show()
def threshold_image(image, level):

    # image should be in BGR space

    

    # Conversion and extraction of `value` channel

    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)[:, :, 2]

    

    # Thresholding at specified level

    ret, thresh_img = cv2.threshold(image, level, 255, cv2.THRESH_BINARY_INV)

    

    return thresh_img





# Plotting at different threshold levels

levels = np.linspace(0, 255, 10).astype('uint8')



plt.figure(figsize=(10, 5))



for i, level in enumerate(levels):

    plt.subplot(5, 2, i + 1)

    

    plt.title(f'Threshold = {level}')    

    plt.imshow(threshold_image(random_images[0], level), cmap='gray')

    plt.axis('off')



plt.tight_layout()

plt.show()
plt.figure(figsize=(5, 5))



for i, image in enumerate(random_images):

    thresholded = threshold_image(image, 99)

    

    plt.subplot(5, 1, i + 1)

    plt.imshow(thresholded, cmap='gray')

    plt.axis('off')



plt.tight_layout()

plt.show()
threshold_level = 99



# 1: Load the images

for image in random_images:

    # 2, 3: Convert to HSV, obtain the value channel and threshold it

    binarized = threshold_image(image, threshold_level)

    

    # 4: Obtain the contours over the binarized image

    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    # 4.1: Create a copy of the binary image to draw over it

    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)

    

    # 4.2: Contour drawing

    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)

        

        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)

        

    # 5: Visualization

    

    plt.figure(figsize=(15, 5))

    

    plt.subplot(1, 2, 1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    

    plt.subplot(1, 2, 2)

    plt.imshow(gray)

    plt.axis('off')

    

    plt.show()
threshold_level = 99

area_threshold = 100



# NEW: Kernel for the closing operation

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))



# 1: Load the images

for image in random_images:

    # 2, 3: Convert to HSV, obtain the value channel and threshold it

    binarized = threshold_image(image, threshold_level)

    

    # NEW: Apply a closing operation

    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

    

    # 4: Obtain the contours over the binarized image

    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    # 4.1: Create a copy of the binary image to draw over it

    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)

    

    # 4.2: Contour drawing

    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)

        

        # NEW: Apply an area threshold

        if w * h > area_threshold:

            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)

        

    # 5: Visualization

    

    plt.figure(figsize=(15, 5))

    

    plt.subplot(1, 2, 1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    

    plt.subplot(1, 2, 2)

    plt.imshow(gray)

    plt.axis('off')

    

    plt.show()
def get_letters(image, threshold_level=99, area_threshold=100):

    letters_bin = []

    rects = []

    

    # 2, 3: Convert to HSV, obtain the value channel and threshold it

    binarized = threshold_image(image, threshold_level)

    

    # 3.1: Apply a closing operation

    binarized = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)

    

    # 4: Obtain the contours over the binarized image

    cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    # 4.2: Create a copy of the binary image to draw over it

    gray = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)

    

    # Creation of bounding rects

    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)

        

        # 4.4: Apply an area threshold

        if w * h > area_threshold:

            rects.append((x, y, w, h))

    

    # Sort by position along the x axis. This is necessary because the order does matter when

    # typing the answer to a captcha

    rects = sorted(rects, key=lambda x: x[0])

    

    for (x, y, w, h) in rects:

        # 4.5: Extract the letter and append it into an array

        letters_bin.append(binarized[y:(y + h), x:(x + w)])

            

            

    return letters_bin
image = get_random_images(images_path, n=1)[0]



plt.figure()

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.axis('off')

plt.show()



letters = get_letters(image)



plt.figure()

for i, letter in enumerate(letters):

    plt.subplot(len(letters), 1, i + 1)

    plt.imshow(letter, cmap='gray')

    plt.axis('off')

    

plt.tight_layout()

plt.show()