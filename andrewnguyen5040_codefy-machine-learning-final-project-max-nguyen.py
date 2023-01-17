# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

#I used random forest, because it's a really good model for classifing numerical data into categorical.

from sklearn.ensemble import RandomForestClassifier



#reading in the csv data

data = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")



#assigning all numerical data to variable X

x = data[["sepal_length","sepal_width","petal_length","petal_width"]]



#assigning the species data to variable Y

y = data["species"]



#Using method train_test_split from sklearn to divide the data into a 70% training data, and a 30% testing data

Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size = .3)



#Creating the model

model = RandomForestClassifier()



#giving the model the training data

model.fit(Xtrain,Ytrain)



#testing the model on new data, to make sure it doesn't over/underfit the data

accuracy = model.score(Xtest,Ytest)



#Rounding the accuracy 

accuracy = round(accuracy,4)



#turning it into a %

accuracy *= 100



#formatting it with an fstring. Similar to .format

print(f"Accuracy of model: {accuracy}%")
