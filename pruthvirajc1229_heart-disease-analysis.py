# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.head()
data.target.value_counts()
data.sex.value_counts()
sns.countplot(x='target',data=data)

plt.show()
sns.countplot(x='sex', data=data)

plt.show()
negativeDisease =len(data[data.target==0])

positiveDisease = len(data[data.target==1])

X=Percentage_of_patients_dont_have_Disease = (negativeDisease/len(data.target)*100)

Y=Percentage_of_patients_have_Disease = (positiveDisease/len(data.target)*100)

print(X)

print(Y)
data.groupby('sex').mean()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))

plt.title("Hear Disease Freaquency for Ages")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.savefig("hdaa.png")

plt.show()
pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(20,6))

plt.title("Heart Disease Frequency for Sex")

plt.xlabel("Sex(0=Female,1=Male)")

plt.xticks(rotation=0)

plt.legend(["Have Disease", "Dont have Disease"])

plt.ylabel("Frequency")

plt.show()
plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)])

plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)])

plt.legend(["Not Disease", "Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
pd.crosstab(data.slope,data.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Slope')

plt.xlabel('The Slope of The Peak Exercise ST Segment ')

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()

pd.crosstab(data.fbs,data.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
a = pd.get_dummies(data['cp'], prefix = "chest_pain")

b = pd.get_dummies(data['thal'], prefix = "Max heart rate achieved")

c = pd.get_dummies(data['slope'], prefix = "slope of the peak")

frames = [data, a, b, c]

data = pd.concat(frames, axis = 1)

data.head()
data = data.drop(columns = ['cp', 'thal', 'slope'])

data.head()
data = pd.get_dummies(data, drop_first=True)
x= data

print("shape of x:",x.shape)


y = data['target']



data = data.drop('target', axis = 1)



print("Shape of y:", y.shape)
y.value_counts()
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.naive_bayes import GaussianNB

model=GaussianNB()

model.fit(x_train,y_train)

y_model=model.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_model)*100
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

mat= confusion_matrix(y_test,y_model)

plt.rcParams['figure.figsize']=(5,5)

sns.heatmap(mat,annot=True,annot_kws={'size':15}, cmap='PuBu')

cr = classification_report(y_test, y_model)

print(cr)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

lr_pred=lr.predict(x_test)

lr_pred
from sklearn.tree import DecisionTreeClassifier

dtc= DecisionTreeClassifier()

dtc.fit(x_train,y_train)

dtc_pre= dtc.predict(x_test)

accuracy_score(y_test,dtc_pre)*100
from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier(n_estimators=1000, random_state=1)

rfc.fit(x_train,y_train)

rfc_pre= rfc.predict(x_test)



accuracy_score(y_test,rfc_pre)*100