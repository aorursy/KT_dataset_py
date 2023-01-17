import numpy as np 
import pandas as pd #For loading dataset
import matplotlib.pyplot as plt # Visualization Library
import seaborn as sns # Visualization Library
import os
print(os.listdir("../input"))
dataset = pd.read_csv('../input/Iris.csv')
dataset.info()
dataset.head()
%matplotlib inline
sns.countplot(x=dataset['Species'],data=dataset)
sns.distplot(dataset['SepalLengthCm'], kde = False)
sns.distplot(dataset['SepalWidthCm'], kde = False)
sns.distplot(dataset['PetalLengthCm'], kde = False)
sns.distplot(dataset['PetalWidthCm'], kde = False)
sns.set_style("ticks")
sns.pairplot(dataset,hue = 'Species',diag_kind = "kde",kind = "scatter",palette = "husl")
sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', data=dataset, fit_reg=False,hue='Species') 
sns.lmplot(x='PetalLengthCm', y='PetalWidthCm', data=dataset, fit_reg=False,hue='Species') 
#Seprating dependent and independent variable
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values

#Split dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)