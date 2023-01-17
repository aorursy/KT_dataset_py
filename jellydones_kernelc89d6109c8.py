# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

data = pd.read_csv("../input/mushrooms.csv")



#Peep the data

data.head(6)





# Any results you write to the current directory are saved as output.
## Check the type of classifications ('p' = posionous, 'e' = edible)

data['class'].unique()

##Check the shape ( 8124, instances,  and 23 attributes including label)

data.shape
##Convert the data into interger values

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

    

data.head()
#Seperate the features and labels



#all features no labels

X = data.iloc[:,1:23]

#Only Labels

y = data.iloc[:,0]



X.head()

y.head()
#Standardize the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

X
#Split the data into training and test data ( 80-20 )

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = .2, random_state = 3 ) 
#Check data

xTest

xTrain

yTrain

yTest





#Try creating a Random Forest

from sklearn.ensemble import RandomForestClassifier



model_RR = RandomForestClassifier()
model_RR.fit(xTrain, yTrain)
y_prob = model_RR.predict_proba(xTest)[:,1] #Postive class predicition probabilities

y_pred = np.where(y_prob > 0.5, 1, 0) #Threshold the probabilities to give class predtictions



model_RR.score(xTest, y_pred)