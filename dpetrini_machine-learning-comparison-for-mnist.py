# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.externals import joblib

from sklearn.svm import LinearSVC

# Decision Tree

from sklearn.model_selection import cross_val_predict

from sklearn.tree import DecisionTreeClassifier

# Ramdom Forest Classifier

from sklearn.ensemble import RandomForestClassifier



import time

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# split the data

def splitdata_train_test(data, fraction_training):

    #ramdomize dataset order

    np.random.seed(0)

    np.random.shuffle(data)

    split = int(data.shape[0]*fraction_training)

    training_set = data[:split]

    testing_set = data[split:]

    

    return training_set, testing_set



#data = pd.read_csv('../input/train.csv')

#testingData = pd.read_csv('../input/test.csv')



data = np.genfromtxt('../input/train.csv', delimiter = ",", dtype = "uint8")

target = data[1:, 0]

digits = data[1:, 1:].reshape(data.shape[0] - 1, 28, 28) # remove head line



# Split for 70% train / 30% test

(digits_train, digits_test) = splitdata_train_test(digits, 0.7)

(target_train, target_test) = splitdata_train_test(target, 0.7)



# Some sanity check

print("Train data shape", digits_train.shape, "Target shape", target_train.shape)

print("Unique elements in targets: ", (np.unique(target_train)))
from skimage import feature

from scipy import ndimage



# Histogram of Oriented Gradients Feature

class HOG:

    def __init__(self, orientations = 9, pixelsPerCell = (9, 9),

        cellsPerBlock = (3, 3), block_norm = 'L2-Hys'):  

        self.orientations = orientations

        self.pixelsPerCell = pixelsPerCell

        self.cellsPerBlock = cellsPerBlock

        self.block_norm = block_norm    # changing from default to L2-Hys, improved a lot



    def describe(self, image):

        # compute HOG for the image

        hist = feature.hog(image, orientations = self.orientations,

            pixels_per_cell = self.pixelsPerCell,

            cells_per_block = self.cellsPerBlock,

            block_norm = self.block_norm) 



        # return the HOG features

        return hist

    

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    # initialize the dimensions of the image to be resized and

    # grab the image size

    dim = None

    (h, w) = image.shape[:2]



    # if both the width and height are None, then return the

    # original image

    if width is None and height is None:

        return image



    # check to see if the width is None

    if width is None:

    # calculate the ratio of the height and construct the

    # dimensions

        r = height / float(h)

        dim = (int(w * r), height)



    # otherwise, the height is None

    else:

        # calculate the ratio of the width and construct the

        # dimensions

        r = width / float(w)

        dim = (width, int(h * r))



    # resize the image

    resized = cv2.resize(image, dim, interpolation = inter)



    # return the resized image

    return resized



def deskew(image, width):

    # grab the width and height of the image and compute

    # moments for the image

    (h, w) = image.shape[:2]

    moments = cv2.moments(image)

    

    # deskew the image by applying an affine transformation

    skew = moments["mu11"] / moments["mu02"]

    M = np.float32([

        [1, skew, -0.5 * w * skew],

        [0, 1, 0]])

    image = cv2.warpAffine(image, M, (w, h),

        flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)



    # resize the image to have a constant width

    image = resize(image, width = width)

    

    # return the deskewed image

    return image



def center_extent(image, size):

    # grab the extent width and height

    (eW, eH) = size



    # handle when the width is greater than the height

    if image.shape[1] > image.shape[0]:

        image = resize(image, width = eW)



    # otherwise, the height is greater than the width

    else:

        image = resize(image, height = eH)



    # allocate memory for the extent of the image and

    # grab it

    extent = np.zeros((eH, eW), dtype = "uint8")

    offsetX = (eW - image.shape[1]) // 2

    offsetY = (eH - image.shape[0]) // 2

    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image



    # compute the center of mass of the image and then

    # move the center of mass to the center of the image

    #(cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")

    (cY, cX) = np.round(ndimage.measurements.center_of_mass(extent)).astype("int32")

    (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)

    M = np.float32([[1, 0, dX], [0, 1, dY]])

    extent = cv2.warpAffine(extent, M, size)



    # return the extent of the image

    return extent
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.externals import joblib

from sklearn.svm import LinearSVC

# Decision Tree

from sklearn.model_selection import cross_val_predict

from sklearn.tree import DecisionTreeClassifier

# Ramdom Forest Classifier

from sklearn.ensemble import RandomForestClassifier



import time

import cv2



from scipy import ndimage

from skimage import feature





# Accuracy

def calculate_accuracy(predicted, actual):



    count = 0

    for i in range(len(predicted)):

        if (predicted[i] == actual[i]):

            count += 1



    return (count / len(actual))



# HOG parameters to explore

#intervalPPC = [4, 5, 6]                  # Pixels Per Cell (7-> very bad 0.60 or error)

#intervalOrient = [6, 12, 15, 18, 24]     # Orientations

#intervalCPB = [1, 2, 3]                  # Cels Per Block



#intervalPPC = [5]                  # Pixels Per Cell (7-> very bad 0.60 or error)

#intervalOrient = [24]     # Orientations

#intervalCPB = [2] 



# HOG parameters to explore

intervalPPC = [4, 5, 6]                  # Pixels Per Cell (7-> very bad 0.60 or error)

intervalOrient = [6, 15, 24]     # Orientations

intervalCPB = [2, 3]                  # Cels Per Block





# ML Algorithms

mlAlgoList = ["SVM", "DTC", "Random Forest"]



# statistics 

avgScore = {}

avgTime = {}

maxScore = {}

totalPass = 0



for algo in mlAlgoList:

    avgScore[algo] = 0

    avgTime[algo] = 0

    maxScore[algo] = 0



# Output header

print ("Feature: HOG")

print (" _______________________________________________________________________________________________")

print ("|    ML Algo    | orientations  | pixelsPerCell | cellsPerBlock |     Score     |     Time      |")



# Loop through HOG parameters and calculate features for each model and predict

for i in intervalOrient:



    orientations = i



    for j in intervalPPC:



        hogPixelsPerCell = (j, j)



        for k in intervalCPB:



            cellsPerBlock = (k, k)



            for algo in mlAlgoList:



                start = time.perf_counter()  # Measure time



                # initialize the HOG descriptor for the variations of parameters

                hog = HOG(orientations = orientations, pixelsPerCell = hogPixelsPerCell,

                      cellsPerBlock = cellsPerBlock, block_norm = 'L2-Hys')



                data = [] # Clear data

                

                # Create model

                for image in digits_train:



                    # deskew the image, center it

                    image = deskew(image, 20)

                    image = center_extent(image, (20, 20))



                    # describe the image and update the data matrix

                    hist = hog.describe(image)

                    data.append(hist)



                # Calculate each model for this HOG descriptor configuration

                if (algo == "SVM"):

                    model = LinearSVC(random_state = 42) # train the SVM model

                elif (algo == "DTC"):

                    model = DecisionTreeClassifier()     # train DTC model

                elif(algo == "Random Forest"):

                    model = RandomForestClassifier(n_estimators = 50) # train RTC model



                # Fit to the selected model

                model.fit(data, target_train)



                data = [] # Clear data



                # Prepare images and calculates features (HOG)

                for image in digits_test:



                    # deskew the image, center it

                    image = deskew(image, 20)

                    image = center_extent(image, (20, 20))

    

                    # describe the image and update the data matrix

                    hist = hog.describe(image)			

                    data.append(hist)



                # Create predictions

                predicted = model.predict(data)



                # Calculate score for selected predition

                modelScore = calculate_accuracy(predicted, target_test)



                endTime = time.perf_counter() - start



                # Gather statistics

                avgScore[algo] += modelScore

                avgTime[algo] += endTime

                if (maxScore[algo] < modelScore):

                    maxScore[algo] = modelScore



                totalPass += 1



                print ("|{:^15s}|{:^15d}|{:^15s}|{:^15s}|{:^15.4f}|{:^15.2f}|".format(algo, orientations, str(hogPixelsPerCell), str(cellsPerBlock), modelScore, endTime))



print (" _______________________________________________________________________________________________")



# Print summary

totalPass /= len(mlAlgoList)

print("Summary totalPass each: 	{:2.0f}".format(totalPass))



# Average score for each one

print("_______AVG Score________")

for i in avgScore.keys():

    print("[{:^15s}]: {:^6.2f}".format(i, avgScore[i]/totalPass))



# Average time for each one

print("_______AVG Time_________")

for i in avgTime.keys():

    print("[{:^15s}]: {:^6.2f}".format(i, avgTime[i]/totalPass))



# Max score for each one

print("_______Max Score________")

for i in maxScore.keys():

    print("[{:^15s}]: {:^6.4f}".format(i, maxScore[i]))