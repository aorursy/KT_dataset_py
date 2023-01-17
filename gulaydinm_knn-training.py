# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/column-2C-weka.csv')
data.head()
data.tail()
data.info()
Normal   =   data[data["class"] == 'Normal']

Abnormal =   data[data["class"] == 'Abnormal']
f,ax = plt.subplots(figsize = (12,12))

sns.heatmap (data.corr(),annot = True , fmt = '.1f' ,ax=ax)

plt.show()
plt.scatter(Normal.pelvic_incidence ,Normal.degree_spondylolisthesis , label = "Normal" ,color = "red" ,alpha = 0.3)

plt.scatter(Abnormal.pelvic_incidence  ,Abnormal.degree_spondylolisthesis , label = "Abnormal",color = "green" ,alpha =0.3)

plt.xlabel("pelvic_incidence")

plt.ylabel("sacral_slope")

plt.legend()

plt.show()
data["class"] = [1 if each == "Normal" else 0  for each in data["class"]]

y = data["class"].values

x_data = data.drop("class" , axis =1 )
x = (x_data - np.min(x_data) / np.max(x_data) - np.min(x_data)) 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 1,test_size =0.3)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 17) #n_neighbors is our k value.

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print( "if k == {} , knn score is = {}".format(17,knn.score(x_test,y_test)) )
score_list = []

for each in range (1,20):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,20),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()