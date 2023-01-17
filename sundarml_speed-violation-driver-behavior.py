# My first ML solution to a practical problem of over speeding of vehicles and unsafe Driver behavior
# data set is from GPS fitted in busses
# 60 kmph is the speed limit, target is classified based on over speed distance travelled
# test your sample by entering your value at x_sample at 18 th line and see the result

import os
print(os.listdir("../input"))


import pandas as pd     
import numpy as np
import math 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('../input/prm_1_2.csv')
data.drop(['Date&Time','OverSpeed','Duration_overspeed'],axis=1,inplace=True)

knn = KNeighborsClassifier(n_neighbors=1)
X = np.array(data['over_speed_distance_travelled']).reshape(-1,1)
Y = np.array(data['Target'])
X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0)

knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(data.head(5))
#print(y_pred)
print("The test score of this model is",np.mean(y_pred==y_test))
x_sample = np.array([4000,3000,768]).reshape(-1,1) # my sample speed violation data
print("My Prediction is",knn.predict(x_sample))
