# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For visualization

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/column_2C_weka.csv')
data.head(10)
f, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
data.info()
data["class"] = [1 if(each == "Abnormal") else 0 for each in data["class"]]



y= data["class"].values

x_data= data.drop(["class"],axis=1)
x= (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x,y)

prediction = knn.predict(x)

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))
score_list = []

for each in range(1,20):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,20),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()