#KNN algorithm with standard scaling.
import os
print(os.listdir("../input"))

import pandas as pd
df = pd.read_csv('../input/pima-indians-diabetes.csv')
print(df.head())


nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
#renaming column names for better understanding 
df.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness","insulin","bmi","Diabetes_Pedigree_Function","age","outcome"]
df.head()
df.info()
# It seems that data has no any null entry. However missing value can be encoded in number of different ways.
#skin_thickness equal to zero, glucose equal to zero.Here zero,for all intent and purposes, is a missing value.

#handling missing data 
import numpy as np
df.glucose.replace(0,np.nan,inplace = True)
df.insulin.replace(0,np.nan,inplace = True)
df.blood_pressure.replace(0,np.nan,inplace = True)
df.bmi.replace(0,np.nan,inplace = True)
df.skin_thickness.replace(0,np.nan,inplace = True)
df.age.replace(0,np.nan,inplace = True)
df.Diabetes_Pedigree_Function.replace(0,np.nan,inplace = True)
df.info()
df.head()
#filling the NaN values with mean value, mainly targetting skin_thickness.
df = df.fillna(df.mean())
df.head()
df.describe()
#Range of insulin is quite large,Need of scaling it.
from sklearn.preprocessing import scale
df['insulin'] = scale(df['insulin'])

#Lets use KNN algorithm first.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
y = df['outcome'].values
X = df.drop('outcome',axis =1).values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42,stratify = y)


#instead of randomly selecting the values of n_neighbors, its better to plot the accuracy curve and then select the value of n_neighbors.
import matplotlib.pyplot as plt
import pylab
import numpy as np
neighbors  = np.arange(1,10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)
plt.title('KNN : Accuracy Curve')
plt.plot(neighbors,test_accuracy,label = 'Testing Accuracy')
plt.plot(neighbors,train_accuracy,label = 'Training Accuracy')
plt.xlabel('No of Neighbors')
plt.ylabel('Accuracy')
pylab.legend(loc = 'upper right')
plt.show()
    
    
#Accuracy curve suggest that there is a sweet spot in the middle where testing accuracy is maximum.
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
knn.score(X_test,y_test)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
knn.score(X_test,y_test)
#After scaling the insulin accuracy got increased by 3%. Without scaling the efficiecy was around 72%.
knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
knn.score(X_test,y_test)


