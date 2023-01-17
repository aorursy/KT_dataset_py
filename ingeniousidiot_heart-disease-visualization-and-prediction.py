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
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
data=pd.read_csv('../input/heart-disease-uci/heart.csv')

data.head()
data.describe()
data.target.value_counts()
sns.countplot(x="target", data=data, palette="Blues")

plt.show()
sns.countplot(x='sex', data=data, palette="Blues")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(8,6))

pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
pd.crosstab(data.fbs,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
pd.crosstab(data.cp,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.show()
plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c='green')

plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)],c='red')

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
x=data.iloc[:,:-1].values

y=data.iloc[:,12].values
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)

imputer = imputer.fit(x[:,0:13])   

x[:, 0:13] = imputer.transform(x[:, 0:13])
#from sklearn.preprocessing import MinMaxScaler

#sc_X = MinMaxScaler()

#x = sc_X.fit_transform(x)



#from sklearn.preprocessing import Normalizer

#sc_X = Normalizer()

#x = sc_X.fit_transform(x)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

x = sc_X.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)

model.fit(x_train,y_train)

y_predict = model.predict(x_test)



#from sklearn.tree import DecisionTreeClassifier

#model = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state =7)

#model.fit(x_train,y_train)

#y_predict=model.predict(x_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_predict)

print(score)