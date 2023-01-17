import cv2

import matplotlib.pyplot as plt

import os

import numpy as np

import random
def load_data(data_directory):

    directories = [d for d in os.listdir(data_directory)

                  if(os.path.isdir(os.path.join(data_directory, d)))]

    

    labels = []

    images = []

    for d in directories:

        label_directory = os.path.join(data_directory, d)

        filenames = [os.path.join(label_directory, f) for f in os.listdir(label_directory)]

        for f in filenames:

            images.append(cv2.imread(f, 1))

            labels.append(ord(d))

    return labels, images
os.listdir('../input/hand-gestures/project')
ROOT_PATH = '../input/hand-gestures/project'

train_directory = os.path.join(ROOT_PATH, 'train')

labels, images = load_data(train_directory)
labels_array = np.array(labels)

images_array = np.array(images)



## see the arrays

# print length of images array

print('Total number of images: ', images_array.size)

# print length of labels array

print('Total number of lables: ', labels_array.size)



## count number of classes and print them

print('Total number of distinct classes are {} and they are {}'.format(

    len(set(labels_array)), [chr(lbl) for lbl in set(labels_array)]))
### randomly select 5 images to show

hand_signs = random.sample(images, 5)



for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(hand_signs[i])

plt.show()
# randomly select 5 images to get shape and min, max values

hand_signs = random.sample(images, 5)



for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(hand_signs[i])

    plt.show()

    print('shape: {}, min: {}, max: {}'.format(

    hand_signs[i].shape, hand_signs[i].min(), hand_signs[i].max()))

## get unique image labels

unique_labels = set(labels)



## initialize the figure

plt.figure(figsize = (15, 15))



## set a counter 

i = 1

# loop through eachh label

for label in unique_labels:

    # pick first image for each label

    image = images[labels.index(label)]

    # create 64 subplots

    plt.subplot(8, 8, i)

    # off axis

    plt.axis('off')

    # add title to each subplot

    plt.title('Label {} [{}]'.format(chr(label), labels.count(label)))

    # show the first image of the label

    plt.imshow(image)

    # add 1 to the counter

    i += 1

# show plot

plt.show()
edged_img = []

for img in images_array:

    edged_img.append(cv2.Canny(img, 0.66 * np.mean(img), 1.33 * np.mean(img)))
hand_signs = random.sample(edged_img, 5)



for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(hand_signs[i], cmap = 'gray')

    plt.show()

    print('shape: {}, min: {}, max: {}'.format(

    hand_signs[i].shape, hand_signs[i].min(), hand_signs[i].max()))
## resize image

images_resized = [cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA) for image in edged_img]
kernel = np.ones((3,3),np.uint8)

images_cleaned = [cv2.dilate(img, kernel, iterations = 1) for img in images_resized]
## show resized images

hand_signs = random.sample(list(range(4852)), 5)

for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(images_cleaned[hand_signs[i]], cmap = 'gray')

plt.show()
images_cleaned = np.array(images_cleaned)
print(labels_array.shape)

print(images_cleaned.shape)
images_cleaned = images_cleaned.reshape(4852, 28*28)

images_cleaned.shape
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(images_cleaned, labels_array, test_size = 0.3, random_state = 1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
svc_model = svm.SVC()

trainGrid = GridSearchCV(estimator = svc_model, cv = 5, param_grid = [

  {'C': [1,2,3,4,5,6,7,8,9], 'kernel': ['linear']},

  {'C': [1,2,3,4,5], 'gamma': [0.1, 0.001, 0.00001, 0.0000001,0.000000001], 'kernel': ['rbf']},

 ], n_jobs = -1)

trainGrid.fit(X_train, y_train)

print(trainGrid.best_score_)

print("train score - " + str(trainGrid.score(X_train, y_train)))

print("test score - " + str(trainGrid.score(X_test, y_test)))

print(trainGrid.best_params_)
os.listdir('../input/asltestimages/testcaptures')
ROOT_PATH = '../input/asltestimages/testcaptures'

test_directory = os.path.join(ROOT_PATH, 'testCaptures')

labels_test, images_test = load_data(test_directory)
## show images

hand_signs = random.sample(list(range(len(images_test))), 5)

for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(images_test[hand_signs[i]])

plt.show()
bg = None

# Function - To find the running average over the background

def run_avg(image, aWeight):

    global bg

    # initialize the background

    if bg is None:

        bg = image.copy().astype("float")

        return



    # compute weighted average, accumulate it and update the background

    cv2.accumulateWeighted(image, bg, aWeight)

# Function - To segment the region of hand in the image

def segment(image, threshold=25):

    global bg

    # find the absolute difference between background and current frame

    diff = cv2.absdiff(bg.astype("uint8"), image)



    # threshold the diff image so that we get the foreground

    thresholded = cv2.threshold(diff,

                                threshold,

                                255,

                                cv2.THRESH_BINARY)[1]



    # get the contours in the thresholded image

    (_, cnts, _) = cv2.findContours(thresholded.copy(),

                                    cv2.RETR_EXTERNAL,

                                    cv2.CHAIN_APPROX_SIMPLE)



    # return None, if no contours detected

    if len(cnts) == 0:

        return

    else:

        # based on contour area, get the maximum contour which is the hand

        segmented = max(cnts, key=cv2.contourArea)

        return (thresholded, segmented)
images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_test]

images_gray = [cv2.GaussianBlur(img, (7, 7), 0) for img in images_gray]
hand_signs = random.sample(list(range(len(images_gray))), 5)

for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(images_gray[hand_signs[i]], cmap = 'gray')

plt.show()
images_segmented = []

num_frames = 0

for img in images_gray:

    if num_frames < 30:

        run_avg(img, 0.5)

    else:

        hand = segment(img)

        if hand is not None:

                (thresholded, segmented) = hand

        images_segmented.append(segmented)
images_segmented
hand_signs = random.sample(list(range(len(images_segmented))), 5)

for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(images_segmented[hand_signs[i]], cmap = 'gray')

plt.show()
hand_signs = random.sample(list(range(len(segmented_test))), 5)

for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(segmented_test[hand_signs[i]], cmap = 'gray')

plt.show()
test_edged = []

for img in images_test:

    test_edged.append(cv2.Canny(img, 0.66 * np.mean(img), 1.33 * np.mean(img)))
## show cleaned images

hand_signs = random.sample(list(range(len(test_edged))), 5)

for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(test_edged[hand_signs[i]], cmap = 'gray')

plt.show()
test_images_resized = [cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA) for image in test_edged]
## show cleaned images

hand_signs = random.sample(list(range(len(test_images_resized))), 5)

for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(test_images_resized[hand_signs[i]], cmap = 'gray')

plt.show()
kernel = np.ones((3,3),np.uint8)

test_images_cleaned = [cv2.fastNlMeansDenoising(img,None,15,7,21) for img in test_images_resized]

#test_images_cleaned = [cv2.erode(img, kernel, iterations = 1) for img in test_images_cleaned]

#test_images_cleaned = [cv2.dilate(img, kernel, iterations = 1) for img in test_images_cleaned]
## show cleaned images

hand_signs = random.sample(list(range(len(test_images_cleaned))), 5)

for i in range(len(hand_signs)):

    plt.subplot(1, 5, i+1)

    plt.axis('off')

    plt.imshow(test_images_cleaned[hand_signs[i]], cmap = 'gray')

plt.show()
test_images_resized = np.array(test_images_resized)

test_images_resized = test_images_cleaned.reshape(test_images_resized.shape[0], test_images_resized.shape[1]*test_images_resized.shape[2])

test_images_resized.shape
test_pred = trainGrid.predict(test_images_resized)
test_pred.shape
labels_test = np.array(labels_test)

labels_test.shape
test_pred
labels_test
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy: "+str(accuracy_score(labels_test, test_pred)*100)+"%")

print('\n')

print(classification_report(labels_test, test_pred))
from sklearn.externals import joblib
filename = 'svcModel.sav'

joblib.dump(trainGrid, filename)