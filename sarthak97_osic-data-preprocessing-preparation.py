# Importing relevant packages

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train_df.head()
train_df.shape
train_df.describe()
# Create checkpoint before deleting 'patient' feature

original_df = train_df.copy()



# Delete the feature now

train_df = train_df.drop(['Patient'], axis=1)

train_df.head()
# Create a checkpoint

df_without_patient = train_df.copy()



# Convert the categorical variables now...

train_df = pd.get_dummies(train_df, drop_first=True)

train_df.head()
# Create a checkpoint

df_with_dummies = train_df.copy()



# Now scale each feature in dataframe

scaler = StandardScaler()

scaler.fit(train_df)

train_df = scaler.transform(train_df)

train_df
# Convert numpy array into dataframe

final_train_df = pd.DataFrame(train_df)



# get column headers from checkpoint df

col_names = df_with_dummies.columns



# Set column headers in final dataframe

final_train_df.columns = col_names



final_train_df.head()
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test_df.head()
test_df.shape
test_df.describe()
# Create checkpoint before deleting 'patient' feature

original_test_df = test_df.copy()



# Delete the feature now

test_df = test_df.drop(['Patient'], axis=1)

test_df.head()
# Create a checkpoint

test_df_without_patient = test_df.copy()



# Convert the categorical variables now...

test_df = pd.get_dummies(test_df, drop_first=True)

test_df.head()
# Create a checkpoint

test_df_with_dummies = test_df.copy()



# Now scale each feature in dataframe

scaler = StandardScaler()

scaler.fit(test_df)

test_df = scaler.transform(test_df)

test_df
# Convert numpy array into dataframe

final_test_df = pd.DataFrame(test_df)



# get column headers from checkpoint df

col_names = test_df_with_dummies.columns



# Set column headers in final dataframe

final_test_df.columns = col_names



final_test_df.head()
final_train_df.to_csv('final_train.csv', index=False)

final_test_df.to_csv('final_test.csv', index=False)

'''

1.Hit commit and run at the right hand corner of the kernel.

2.Wait till the kernel runs from top to bottom.

3.Checkout the 'Output' Tab from the Version tab. Or go to the snapshot of your kernel and checkout the 'Output' tab.

  Your csv file will be there!!

4. Download it.

'''
!pip install imutils
# Importing relevant packages



import os

import cv2

import glob

import imutils

import pydicom

import matplotlib.pyplot as plt

%matplotlib inline
# Convert .dcm image to .png image



path = '../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm'

file = path.split('/')[-1]

outdir = './'



# read dcm file

dcm_img = pydicom.dcmread(path)



# get pixel arrays, replace .dcm extension with .png and place image in output directory

img = dcm_img.pixel_array

cv2.imwrite(outdir + file.replace('.dcm','.png'),img)
image = cv2.imread("./1.png")

plt.imshow(image)
# Gaussian Blur

blur = cv2.GaussianBlur(image, (7,7), 0)

plt.imshow(blur)
# Median Blur

median_blur = cv2.medianBlur(image, 5)

plt.imshow(median_blur)
# Flip vertically 

flip_vertical = cv2.flip(image, flipCode=0)

plt.imshow(flip_vertical)
# Flip horizontally 

flip_horizontal = cv2.flip(image, flipCode=1)

plt.imshow(flip_horizontal)
# Flip both horizontally and vertically

flip_both = cv2.flip(image, flipCode=-1)

plt.imshow(flip_both)
edged = cv2.Canny(image, 100, 200)

plt.imshow(edged)
# clockwise rotation

rotate_clock = imutils.rotate(image, -45)

plt.imshow(rotate_clock)
# counter clockwise rotation

rotate_counter = imutils.rotate(image, 90)

plt.imshow(rotate_counter)
_, thresh1 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

plt.imshow(thresh1)
_, thresh2 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

plt.imshow(thresh2)
# Erosion

erode = cv2.erode(thresh1, (5,5), iterations=1)

plt.imshow(erode)
dilate = cv2.dilate(thresh1, (5,5), iterations=1)

plt.imshow(dilate)