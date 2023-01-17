# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/wall-following-robot/sensor_readings_24.csv")





print("Data size - ", data.shape)






dataa = np.loadtxt("/kaggle/input/wall-following-robot/sensor_readings_24.csv", delimiter=',', dtype=np.str)



data = pd.DataFrame(dataa[:,:24], dtype=np.float)

data = pd.concat([data, pd.DataFrame(dataa[:, 24], columns=['Label'])], axis=1)

                      

print("Data size - ", data.shape)



data.head()
data.groupby(['Label']).count()[0]
def parse_values(each):

    if  (each =="Move-Forward"):

       return 1

    elif (each =="Slight-Right-Turn"):

       return 2

    elif (each =="Sharp-Right-Turn"):

       return 3

    else:

       return 4



data['Label'].apply(parse_values)


y= data.Label.values

x_data=data.drop(["Label"],axis=1)
x =(x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("accuracy:",nb.score(x_test,y_test))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print("accuracy of svm algo:",svm.score(x_test,y_test))
y_pred =knn.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt 



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5 ,linecolor ="red",fmt =".0f",ax=ax)

plt.show()
y_pred =svm.predict(x_test)

y_true=y_test

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt 



f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5 ,linecolor ="red",fmt =".0f",ax=ax)

plt.show()