# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv") #read csv file
test=pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
sample=pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
data.describe() #describe the dataset
corr=data.corr()
data.isnull().sum() # used to find any nullvalue
x=data[['timeindex','currentBack','motorTempBack','positionBack','refPositionBack','refVelocityBack','trackingDeviationBack','velocityBack','currentFront','motorTempFront','positionFront','refPositionFront','refVelocityFront','trackingDeviationFront','velocityFront']]
y=data['flag']
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)
# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42)
ranfor.fit(X_train, y_train)
Y_pred_ranfor = ranfor.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_ranfor = accuracy_score(y_test, Y_pred_ranfor)
print("Random Forest: " + str(accuracy_ranfor * 100))
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred_ranfor)
cm
# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, Y_pred_ranfor))
test=test[['timeindex','currentBack','motorTempBack','positionBack','refPositionBack','refVelocityBack','trackingDeviationBack','velocityBack','currentFront','motorTempFront','positionFront','refPositionFront','refVelocityFront','trackingDeviationFront','velocityFront']]
pred=ranfor.predict(test)
pred
sample["flag"]=pred
sample.tail()
sample.to_csv("submit_9.csv",index=False)