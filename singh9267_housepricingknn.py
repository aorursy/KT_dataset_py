import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv("../input/kc_house_data.csv")
data.head()
data.shape
data.info
data.columns
x=data[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']]
y=data['price']
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.4,random_state=5)
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)
knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
acc=metrics.accuracy_score(y_test,y_pred)
print (acc)