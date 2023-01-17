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
data=pd.read_csv("/kaggle/input/data.csv")


data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.head()
M=data[data.diagnosis=="M"]

B=data[data.diagnosis=="B"]



#Radius mean Area mean virtualization

plt.scatter(M.radius_mean,M.area_mean,color="red",label="Kötü")

plt.scatter(B.radius_mean,B.area_mean,color="green",label="İyi",alpha=0.5)

plt.xlabel("radius mean")

plt.ylabel("area mean")

plt.legend()

plt.show()
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis ]

y=data.diagnosis.values

x_data=data.drop(["diagnosis"],axis=1)
#normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=29)
#naive bayes using

from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()

nb.fit(x_train,y_train)

print("Accuracy of Bayes Algorithm",nb.score(x_test,y_test))
#Decision Tree using 

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()



dt.fit(x_train,y_train)



print ("score: ",dt.score(x_test,y_test))

#random forest using

from sklearn.ensemble import RandomForestClassifier 

rfc=RandomForestClassifier(n_estimators=100,random_state=29)

rfc.fit(x_train,y_train)

print("Random Forest Accuracy: ",rfc.score(x_test,y_test))



#

y_pred=rfc.predict(x_test)

y_true=y_test
#confusion matrix using and virtualization

from sklearn.metrics import confusion_matrix



cm=confusion_matrix(y_true,y_pred)
#virtualization matrix using seaborn

import seaborn as sns



f,ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
