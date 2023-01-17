# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
#Lets have a look into the matadata 

data.info()
data.describe()
#Lets have a look into some sample data

data.head()
#Provide features for X and label for y

X = data.drop('diabetes',axis=1)

y = data['diabetes']

# Split the data into traing and test sets

X_train, X_test, y_train, y_test = train_test_split(

                                    X, y, random_state=42, test_size=.33)

#Initialize Random Forest Classifier

rfc = RandomForestClassifier()

#Fit model on the training Data

rfc.fit(X_train,y_train)

#Make prediction

predictions = rfc.predict(X_test)
#Generate Confusion Matrix

conf_matrix = confusion_matrix(predictions,y_test)

print(conf_matrix)
#Lets calculate Precision, Recall and F1 score for label 0 and 1

#For Label 0

tp = conf_matrix[0,0]

fp = conf_matrix[1,0]

fn = conf_matrix[0,1]



precision  = tp / (tp + fp)

recall     = tp / (tp + fn)

f1_score   = 2*( precision * recall)/(precision + recall)



print('precision, recall and f1-score for label 0')

print('The precision for label 0 is: {0:.2f}'.format(precision))

print('The recall for label 0 is: {0:.2f}'.format(recall))

print('The f1-score for label 0 is: {0:.2f}'.format(f1_score))

print('\n')



#For Label 1 



tp = conf_matrix[1,1]

fp = conf_matrix[0,1]

fn = conf_matrix[1,0]



precision  = tp / (tp + fp)

recall     = tp / (tp + fn)

f1_score   = 2*( precision * recall)/(precision + recall)



print('precision, recall and f1-score for label 1')

print('The precision for label 1 is: {0:.2f}'.format(precision))

print('The recall for label 1 is: {0:.2f}'.format(recall))

print('The f1-score for label 1 is: {0:.2f}'.format(f1_score))

print(classification_report(predictions,y_test))