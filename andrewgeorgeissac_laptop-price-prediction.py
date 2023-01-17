#Importing libraries
import pandas as pd
import numpy as np
#getting the data
dataset=pd.read_csv('../input/laptop-price-predictor/laptop_pricing.csv')
x=dataset.iloc[:, 1:-1].values #features data
y=dataset.iloc[:, -1].values #price data

#Splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#Training the logistic regression model from the training set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
#predicting the result
print(classifier.predict(sc.transform([[2,5,2.3,4,1000,0,0,15.6]]))) #example