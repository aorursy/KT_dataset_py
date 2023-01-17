# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Import the libraries

# Author: kbv71

# Many useful insights : https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/51411

import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,roc_curve,recall_score,classification_report,mean_squared_error,confusion_matrix
import os
print(os.listdir("../input"))

# It is a bulky dataset. There are multiple options to deal with it. 

##A. We retrieve only min(nPos,nNeg) rows from both classes and get a smaller yet more balenced data

##B. We use efficient data tools like Dask 

##Skip the attribute_time because if it is filled, it will be anyways denoted by 1 in label data.

# Set a random state

random_state = np.random.RandomState(2)

# preset the data types

dtyp = {'ip': np.int64, 'app': np.int16,'device': np.int16,'os': np.int16,'channel': np.int16,'is_attributed' : np.int16}


print("LOADING DATA..........................")

# TRAINING DATA

print("TRAINING DATA")

dfTrain = dd.read_csv("../input/train_sample.csv",blocksize=1e05)

print("original dataframe")

print(dfTrain.head())

del dfTrain['attributed_time']

#nRows = dfTrain[dfTrain.ip > 0].ip.value_counts().compute()

nRows = len(dfTrain)

print("nRows = ", nRows)

print(dfTrain.astype(dtyp))

# Create new features out of time. Year and month are skipped.

dfTrain['click_time'] =  dd.to_datetime(dfTrain['click_time'])

# the given data is of 4 days. So useful data is day and hours

dfTrain['click_time_day'] = dfTrain['click_time'].dt.day

dfTrain['click_time_hour'] = dfTrain['click_time'].dt.hour

del dfTrain['click_time']

dfTrain.columns = ['ip', 'app', 'device', 'os','channel','is_attributed','click_time_day','click_time_hour']

print("dfTrain.columns",dfTrain.columns)

dfTrain.astype(dtyp)
print("---------")
print(dfTrain.info())
# Find the ratio of positive / negetive to check for imbalance.

nPos = dfTrain.is_attributed.sum().compute()

nNeg = nRows - nPos

r = np.longdouble(nPos/nRows)

print("positive cases in training set: ", 100.0*r, "%")

print(nPos)
# Create a balenced dataset

posEx = dfTrain [ (dfTrain['is_attributed'] == 1) ]

sampledNegEx =  dfTrain [ (dfTrain['is_attributed'] == 0) ].sample(frac=r,random_state=random_state)

newTrainsubs = [posEx, sampledNegEx]

dfTrainBal = dd.concat(newTrainsubs)
# Split the balanced dataset to create cross validation set


#print(XTrainBal.head())

# create a randomly selected cross validation set
#
#train_test_split(XTrainBal[features], XTrainBal['is_attributed'], test_size=0.33, random_state=random_state)

dTrain = pd.DataFrame()

dCV = pd.DataFrame()

dTrain, dCV = dfTrainBal.random_split([0.70,0.30], random_state=random_state)

#print(dTrain.head())
#Get X and y

yTrain = dTrain['is_attributed']

XTrain = dTrain.drop('is_attributed',axis=1).compute()

yCV = dCV['is_attributed']
XCV = dCV.drop('is_attributed',axis=1).compute()

print(yTrain.head())
# Create classifiers

clfSVM = svm.SVC(kernel='linear', probability=True,random_state=random_state)

clfLR = LogisticRegression()

clfRF = RandomForestClassifier(n_estimators=100,random_state=random_state)

# Train the classifier on training set

clfSVM.fit(XTrain, yTrain)

clfLR.fit(XTrain, yTrain)

clfRF.fit(XTrain, yTrain)
# Apply on Cross validation set

yCVPredSVM = clfSVM.predict(XCV)

yCVPredLR = clfLR.predict(XCV)

yCVPredRF = clfRF.predict(XCV)
# Check error matrix


CMSVM = confusion_matrix(yCV,yCVPredSVM) 

CMLR = confusion_matrix(yCV,yCVPredLR)

CMRF = confusion_matrix(yCV,yCVPredRF)

print("0,0 : true nagetive \n 0,1 : False positive \n 1,0 : False negetive \n 1,1 : True positive ")

print(CMSVM)

print(CMLR)

print(CMRF)

# random samlping helps evaluate the algorithms 