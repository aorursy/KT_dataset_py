# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

#print(data)

#data.info()
Abnormal=data[data['class']=='Abnormal']

Normal=data[data['class']=='Normal']

#Abnormal.info()

#Normal.info()
plt.scatter(Abnormal.pelvic_incidence,Abnormal.lumbar_lordosis_angle,color="red",label= "Abnormal", alpha=0.3)

plt.scatter(Normal.pelvic_incidence,Normal.lumbar_lordosis_angle,color="green",label= "Normal", alpha=0.3)

plt.xlabel="pelvic_incidence"

plt.ylabel="lumbar_lordosis_angle"

plt.legend()

plt.show()

data['class']=[1 if each=="Normal" else 0 for each in data['class']]

y=data['class'].values

x_data=data.drop(["class"],axis=1)
#normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#KNN model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)
print(" {} score: {}".format(3,knn.score(x_test,y_test)))
score_list=[]

for each in range(1,25):

    knn=KNeighborsClassifier(n_neighbors=each)

    knn.fit(x_train,y_train)

    score_list.append(knn.score(x_test,y_test))

    print(" {} score: {}".format(each,knn.score(x_test,y_test)))

plt.plot(range(1,25),score_list)

plt.xlabel="k"

plt.ylabel="score"

plt.show()