#Feature Importance

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

print(os.listdir("../input"))

Data_Set1= pd.read_csv("../input/diabetes.csv")

Data_Set1.tail()
#check for null values-- nothing found

Data_Set1.isnull().sum()
#prepare x axis 

x=Data_Set1[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]

x.head()
#prepare y axis

y=Data_Set1["Outcome"]

y.head()
#Feature importance of each feature of your dataset by using the feature importance property of the model.

#Feature importance gives you a score for each feature of your data, the higher the score more important or 

#relevant is the feature towards your output variable.

#Feature importance is an inbuilt class that comes with Tree Based Classifiers, we will be using Extra Tree 

#Classifier for extracting the top 5 features for the dataset.

model = ExtraTreesClassifier()

model.fit(x,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(8).plot(kind='barh')

plt.show()


