# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

filename = "PixelsAndInchesY/PixelsAndInchesKehlin.csv"

dataFile = pd.read_csv("../input/pixelandinchesy/PixelsAndInchesKehlin.csv")

wristPose = pd.read_csv("../input/posenetwristtracking/poseNetTracking.csv")
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



X = dataFile[["Pixels"]]

y = dataFile["Inches"]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

#Need to add train test split



pixelPred = LinearRegression(normalize=True)

pixelPred.fit(X,y)

y_pred = pixelPred.predict(X_test)



print(y_pred)

print("Test", y_test)

#accuracy prediction 

r2_score(y_test, y_pred)
import matplotlib 

wristPoseNew = wristPose.iloc[1:, 0:2]

wristPoseNew
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



X = wristPoseNew[["Pixels"]]

y = wristPoseNew["Inches"]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

#Need to add train test split



pixelPred = LinearRegression(normalize=True)

pixelPred.fit(X,y)

y_pred = pixelPred.predict(X_test)



# print(y_pred,y_test)

#accuracy prediction 

r2_score(y_test, y_pred)
%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



# rWristIndex = []



# for i in range(len(y_test)):

#     rWristIndex.append(i)



plt.scatter(y_pred,y_test)



# plt.plot(y_pred)


