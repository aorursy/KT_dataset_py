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
data=pd.read_csv("../input/column_2C_weka.csv")
data.head()
A=data[data.iloc[:,-1]=="Abnormal"]

N=data[data.iloc[:,-1]=="Normal"]
A.info()
plt.scatter(A.lumbar_lordosis_angle,A.sacral_slope,color="red",label="Anormal")

plt.scatter(N.lumbar_lordosis_angle,N.sacral_slope,color="green",label="Normal")

plt.xlabel("lumbar_lordosis_angle")

plt.ylabel("sacral_slope")

plt.show()
data["class"]=[1 if each =="Abnormal" else 0 for each in data.iloc[:,-1]]



y=data["class"].values

x_data=data.drop(["class"],axis=1)
y
#Normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#Split 



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=29)
#KNN Model

from sklearn.neighbors import KNeighborsClassifier

KNN=KNeighborsClassifier(n_neighbors=14)

KNN.fit(x_train,y_train)

prediction=KNN.predict(x_test)
prediction
#KNN SCORE PRİNT

print("{} KNN Score: {}".format(14,KNN.score(x_test,y_test))) 
#MAX Knn Result Search

score_list=[]



for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



plt.plot(range(1,15),score_list)

plt.show()
## RESULT

# 14 K good result