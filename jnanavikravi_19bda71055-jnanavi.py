#import the libraries necessary

import pandas as pd

import numpy as np

import seaborn as sn

import matplotlib.pyplot as plt





#load train and test data

train_data_path='../input/bda-2019-ml-test/Train_Mask.csv'

test_data_path='../input/bda-2019-ml-test/Test_Mask_Dataset.csv'
#read the data into a dataframe

train=pd.read_csv(train_data_path)

test=pd.read_csv(test_data_path)
#check the shape of the data(dimension)

train.shape
#displayfirst few values of the data

train.head()
#check for null(missing) values

train.isnull().sum()
#description (summary) of the data

train.describe()
#chcek the correlation between the features

train.corr()
#plot the heatmap

corrMatrix = train.corr()

sn.heatmap(corrMatrix, annot=True)



plt.show()
#get the labels of the columns

train.columns
#assign dependent(y) and independent(x) variables

X = train[['timeindex', 'currentBack', 'motorTempBack', 'positionBack','refPositionBack', 'refVelocityBack', 'trackingDeviationBack','velocityBack', 'currentFront', 'motorTempFront', 'positionFront','refPositionFront', 'refVelocityFront', 'trackingDeviationFront','velocityFront']]

y = train['flag']
#splitting the data into train data and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2231,random_state=42)
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets 

clf.fit(X_train,y_train)



#test prediction for train data

y_pred=clf.predict(X_test)
#check the classification report(score)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
#predict value for test data

ytest=clf.predict(test)
#check the classification report(score)

from sklearn.metrics import classification_report

print(classification_report(y_test,ytest))
#dataframe the columns necessary for the submission file

timeindex=test['timeindex']

result = pd.DataFrame({'Sl.No': timeindex,'flag':ytest})



#submissin file

result.to_csv('submission.csv',index=False)