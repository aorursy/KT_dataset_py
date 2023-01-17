# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# create model

model = LinearRegression()
#  train sample

x = np.array([5,15,25,35,45,55]).reshape((-1,1))

y = np.array([5,20,14,32,22,38])
# start training

model.fit(x,y)
# print output

print('score',model.score(x,y))

print('intercept',model.intercept_)

print('corefficient',model.coef_)
# test

model.predict(x)
# identical way to predict

y_pred = model.intercept_ + model.coef_ * x

print('predicted response:', y_pred, sep='\n')
# imoport data set from sklearn

from sklearn import datasets

from sklearn.model_selection import train_test_split 



cancer = datasets.load_breast_cancer()
print("Features:", cancer.feature_names)

print("Labels: ", cancer.target_names) 

print('Shape:', cancer.data.shape)

print('Top5 row data:', len(cancer.data[0:5][0]))

print('Cancer labels (0:malignant, 1:benign)', cancer.target)
# spliting train set and test set

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=0)
# creating model

#Import svm model

from sklearn import svm

#Create a svm Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets

clf.fit(X_train, y_train)

# Predict the response for test dataset

y_pred = clf.predict(X_test) 
print(y_pred)
# Evaluate model

from sklearn import metrics 

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 

import argparse

import errno

import scipy.misc

import dlib

import cv2



from skimage.feature import hog



# initialization

image_height = 48

image_width = 48

window_size = 24

window_step = 6

ONE_HOT_ENCODING = False

SAVE_IMAGES = False

GET_LANDMARKS = False

GET_HOG_FEATURES = False

GET_HOG_WINDOWS_FEATURES = False

SELECTED_LABELS = []

IMAGES_PER_LABEL = 500

OUTPUT_FOLDER_NAME = "fer2013_features"



# parse arguments and initialize variables:

parser = argparse.ArgumentParser()

parser.add_argument("-j", "--jpg", default="no", help="save images as .jpg files")

parser.add_argument("-l", "--landmarks", default="yes", help="extract Dlib Face landmarks")

parser.add_argument("-ho", "--hog", default="yes", help="extract HOG features")

parser.add_argument("-hw", "--hog_windows", default="yes", help="extract HOG features from a sliding window")

parser.add_argument("-o", "--onehot", default="no", help="one hot encoding")

parser.add_argument("-e", "--expressions", default="0,1,2,3,4,5,6", help="choose the faciale expression you want to use: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral")

args = parser.parse_args()

if args.jpg == "yes":

    SAVE_IMAGES = True

if args.landmarks == "yes":

    GET_LANDMARKS = True

if args.hog == "yes":

    GET_HOG_FEATURES = True

if args.hog_windows == "yes":

    GET_HOG_WINDOWS_FEATURES = True

if args.onehot == "yes":

    ONE_HOT_ENCODING = True

if args.expressions != "":

    expressions  = args.expressions.split(",")

    for i in range(0,len(expressions)):

        label = int(expressions[i])

        if (label >=0 and label<=6 ):

            SELECTED_LABELS.append(label)

if SELECTED_LABELS == []:

    SELECTED_LABELS = [0,1,2,3,4,5,6]

print( str(len(SELECTED_LABELS)) + " expressions")



# loading Dlib predictor and preparing arrays:

print( "preparing")

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

original_labels = [0, 1, 2, 3, 4, 5, 6]

new_labels = list(set(original_labels) & set(SELECTED_LABELS))

nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))

try:

    os.makedirs(OUTPUT_FOLDER_NAME)

except OSError as e:

    if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):

        pass

    else:

        raise



def get_landmarks(image, rects):

    # this function have been copied from http://bit.ly/2cj7Fpq

    if len(rects) > 1:

        raise BaseException("TooManyFaces")

    if len(rects) == 0:

        raise BaseException("NoFaces")

    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])



def get_new_label(label, one_hot_encoding=False):

    if one_hot_encoding:

        new_label = new_labels.index(label)

        label = list(np.zeros(len(new_labels), 'uint8'))

        label[new_label] = 1

        return label

    else:

        return new_labels.index(label)



def sliding_hog_windows(image):

    hog_windows = []

    for y in range(0, image_height, window_step):

        for x in range(0, image_width, window_step):

            window = image[y:y+window_size, x:x+window_size]

            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),

                                            cells_per_block=(1, 1), visualise=False))

    return hog_windows



print( "importing csv file")

data = pd.read_csv('fer2013.csv')



for category in data['Usage'].unique():

    print( "converting set: " + category + "...")

    # create folder

    if not os.path.exists(category):

        try:

            os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)

        except OSError as e:

            if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):

               pass

            else:

                raise

    

    # get samples and labels of the actual category

    category_data = data[data['Usage'] == category]

    samples = category_data['pixels'].values

    labels = category_data['emotion'].values

    

    # get images and extract features

    images = []

    labels_list = []

    landmarks = []

    hog_features = []

    hog_images = []

    for i in range(len(samples)):

        try:

            if labels[i] in SELECTED_LABELS and nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:

                image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))

                images.append(image)

                if SAVE_IMAGES:

                    scipy.misc.imsave(category + '/' + str(i) + '.jpg', image)

                if GET_HOG_WINDOWS_FEATURES:

                    features = sliding_hog_windows(image)

                    f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),

                                            cells_per_block=(1, 1), visualise=True)

                    hog_features.append(features)

                    hog_images.append(hog_image)

                elif GET_HOG_FEATURES:

                    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),

                                            cells_per_block=(1, 1), visualise=True)

                    hog_features.append(features)

                    hog_images.append(hog_image)

                if GET_LANDMARKS:

                    scipy.misc.imsave('temp.jpg', image)

                    image2 = cv2.imread('temp.jpg')

                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]

                    face_landmarks = get_landmarks(image2, face_rects)

                    landmarks.append(face_landmarks)            

                labels_list.append(get_new_label(labels[i], one_hot_encoding=ONE_HOT_ENCODING))

                nb_images_per_label[get_new_label(labels[i])] += 1

        except Exception as e:

            print( "error in image: " + str(i) + " - " + str(e))



    np.save(OUTPUT_FOLDER_NAME + '/' + category + '/images.npy', images)

    if ONE_HOT_ENCODING:

        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)

    else:

        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/labels.npy', labels_list)

    if GET_LANDMARKS:

        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/landmarks.npy', landmarks)

    if GET_HOG_FEATURES or GET_HOG_WINDOWS_FEATURES:

        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_features.npy', hog_features)

        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/hog_images.npy', hog_images)