#Import relevant modules
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import tree
import graphviz
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime
data = pd.read_csv('../input/weatherAUS.csv')
print (data.head(10))
print (data.shape)
#Drop column RISK_MM
data = data.drop(["RISK_MM"],axis =1)

#transform categorical data to integers.
le = preprocessing.LabelEncoder()
#Convert the columns returning errors to strings
data['WindGustDir']=data['WindGustDir'].astype(str)
data['WindDir9am']=data['WindDir9am'].astype(str)
data['WindDir3pm']=data['WindDir3pm'].astype(str)
data['RainToday']=data['RainToday'].astype(str)
data['RainTomorrow'] = data['RainTomorrow'].replace({"No":0, "Yes":1})

#Transform the other columns while leaving the date column the same
data.loc[:,'Location':'RainToday'] = data.loc[:,'Location':'RainToday'].apply(le.fit_transform)
transf_data = data
#test = test.apply(le.fit_transform)
print (transf_data.head())
#Group data by Date and Location and aggregate the other columns accordingly
transf_data = (transf_data.groupby(['Date']).agg({'MinTemp':'mean', 'MaxTemp': 'mean','Rainfall': 'mean','Evaporation': 'mean','Sunshine': 'mean','WindGustDir': 'mean','WindGustSpeed': 'mean','WindDir9am': 'mean','WindDir3pm': 'mean',
                                              'WindSpeed9am': 'mean','WindSpeed3pm': 'mean','Humidity9am': 'mean','Humidity3pm': 'mean','Pressure9am': 'mean','Pressure3pm': 'mean','Cloud9am': 'mean','Cloud3pm': 'mean',
                                                'Temp9am': 'mean','Temp3pm': 'mean','RainToday': lambda x: x.mode()[0],'RainTomorrow': lambda x: x.mode()[0]}))
print (transf_data.head())
#Separate the label/class from the features
features = transf_data.drop(['RainTomorrow'], axis =1)
label = transf_data['RainTomorrow']
#split the data into training and testing components
features_train, features_test, label_train, label_test = train_test_split(
features, label, test_size=0.33, random_state=42)
print (features_test.head())
 #Set up the Adaboost Classifier model
abc = AdaBoostClassifier(base_estimator=None, n_estimators=300, learning_rate=1.0,  random_state=None)
abc = abc.fit(features_train,label_train)
accuracy = abc.score(features_test, label_test)
mse = mean_squared_error(label_test, abc.predict(features_test)) 
print ('The accuracy level is', accuracy.round(2))
print ('The mean square error is', mse.round(2))
#Extract current day features
current_feat = features[-1:]
predict = abc.predict(current_feat)
print ("Will it rain in any part of Australia tomorrow?")
if predict == 0:
    print ('No')
else:
    print ('Yes')