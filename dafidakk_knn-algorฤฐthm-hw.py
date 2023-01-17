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
dataframe=pd.read_csv('../input/column_2C_weka.csv')

dataframe.head()

dataframe.info() # seeking NaN values
dataframe=dataframe.rename(columns={'class': 'posture'})

dataframe.head()
A = dataframe[dataframe.posture=='Abnormal']

N = dataframe[dataframe.posture=='Normal']
plt.scatter(A.pelvic_incidence,A.lumbar_lordosis_angle,color='red',label='well_posture',alpha=0.3)

plt.scatter(N.pelvic_incidence,N.lumbar_lordosis_angle,color='green',label='bad_posture',alpha=0.3)

plt.xlabel('pelvic_incidence') 

plt.ylabel('lumbar_lordosis_angle') 

plt.legend()

plt.show()

dataframe.posture=[1 if each=="Abnormal" else 0 for each in dataframe.posture]

y=dataframe.posture.values

x_data=dataframe.drop(["posture"],axis=1)
# normalization

x= (x_data-np.max(x_data))/(np.max(x_data)-np.min(x_data))
x.head()
# train test split

from sklearn.model_selection import train_test_split



x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=1)
y_train
# knn model

from sklearn.neighbors import KNeighborsClassifier



knn= KNeighborsClassifier(n_neighbors=13) # n_neighbors=k

knn.fit(x_train,y_train)

prediction= knn.predict(x_test)
# accuracy(score)

print("{} nn score: {} ".format(13,knn.score(x_test,y_test)))
# find K values

score_list=[]

for each in range(1,15):

    knn2=KNeighborsClassifier(n_neighbors= each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,15),score_list)

plt.xlabel('k values')

plt.ylabel('accuracy')

plt.show()