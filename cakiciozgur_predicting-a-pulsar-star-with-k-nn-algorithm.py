# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/pulsar_stars.csv")
data.head()
data.info()
data.isnull().sum()
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace("-","_")
data.info()
y=data.target_class.values
x_data=data.drop(["target_class"],axis=1)
x=(x_data-np.min(x_data))/((np.max(x_data))-(np.min(x_data)))
x.head()
Class_0=data[data.target_class==0]
Class_1=data[data.target_class==1]
plt.scatter(Class_0.standard_deviation_of_the_integrated_profile,Class_0.excess_kurtosis_of_the_integrated_profile,color="red",label="Class 0",alpha=0.5)
plt.scatter(Class_1.standard_deviation_of_the_integrated_profile,Class_1.excess_kurtosis_of_the_integrated_profile,color="blue",label="Class 1",alpha=0.5)
plt.xlabel("standard_deviation_of_the_integrated_profile")
plt.ylabel("excess_kurtosis_of_the_integrated_profile")
plt.legend()
plt.show()
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
# K-NN MODEL
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3) # n_neigbors = k
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
print("{}-NN Score : {}".format(5,knn.score(x_test,y_test)))
# why find k value ?
score_list=[]
for each in range(5,20):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(5,20),score_list)



