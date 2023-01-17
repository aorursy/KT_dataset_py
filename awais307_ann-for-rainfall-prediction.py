import numpy as np
import csv
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
dataset=[] 
dataset = pd.read_csv('../input/Input_Data.csv')
x = dataset.iloc[1:,1:7].values 
y = dataset.iloc[1:,-1].values 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
classifier = LogisticRegression(random_state = 0) 
classifier.fit(x_train,y_train) 
y_pred = classifier.predict(x_test)
print(y_pred[5:500])
