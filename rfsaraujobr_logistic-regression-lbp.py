# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour
from skimage.data import imread
from sklearn.ensemble import RandomForestClassifier
import time
import cv2
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize

# To calculate a normalized histogram 


from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality


# matplotlib setup
%matplotlib inline
from pylab import rcParams
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from skimage.feature import local_binary_pattern
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from pathlib import Path
input_path = Path('../input')
train_path = input_path / 'train'
test_path = input_path / 'test'

cameras = os.listdir(train_path)

train_images = []
for camera in cameras:
    for fname in sorted(os.listdir(train_path / camera)):
        train_images.append((camera, fname))

train = pd.DataFrame(train_images, columns=['camera', 'fname'])
#print(train.shape)

test_images = []
for fname in sorted(os.listdir(test_path)):
    test_images.append(fname)

test = pd.DataFrame(test_images, columns=['fname'])
#print(test.shape)
nt = 10
train_images2 = train_images[:nt]
train_images2 = train_images2 + train_images[1000:1000+nt] 
train_images = train_images2
test_images = test_images[:nt]
X_train = []
y_train = []

for i in train_images:
    i_path = "../input/train/" + i[0] + '/' + i[1];
    img_aux = cv2.imread(i_path)
    im = img_aux
    train_image = i_path
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    # Append image path in X_name
    # Append histogram to X_name
    X_train.append(hist)
    # Append class label in y_test
    y_train.append(i[0])
    
#Para submissao
X_test = []
X_test_name = []

for i in test_images:
    i_path = "../input/test/" + '/' + i;
    X_test_name.append(i)
    img_aux = cv2.imread(i_path)
    im = img_aux
    train_image = i_path
    # Convert to grayscale as LBP works on grayscale image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    # Append image path in X_name
    # Append histogram to X_name
    X_test.append(hist)

    
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

output = pd.DataFrame(y_pred)
output.to_csv("submission.csv")
