# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplot inline





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
read='/kaggle/input/pima-indians-diabetes-database/diabetes.csv'

t=pd.read_csv(read)

t.head()
fig, ax = plt.subplots(4,2, figsize=(16,16))

sns.distplot(t.Age, bins = 20, ax=ax[0,0]) 

sns.distplot(t.Pregnancies, bins = 20, ax=ax[0,1]) 

sns.distplot(t.Glucose, bins = 20, ax=ax[1,0]) 

sns.distplot(t.BloodPressure, bins = 20, ax=ax[1,1]) 

sns.distplot(t.SkinThickness, bins = 20, ax=ax[2,0])

sns.distplot(t.Insulin, bins = 20, ax=ax[2,1])

sns.distplot(t.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0]) 

sns.distplot(t.BMI, bins = 20, ax=ax[3,1]) 
sns.regplot(x='SkinThickness', y= 'Insulin', data=t)
sns.pairplot(data=t,hue='Outcome')
fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(18,18))

plt.suptitle('Violin Plots',fontsize=24)

sns.violinplot(x="Pregnancies", data=t,ax=ax[0,0],palette='Set3')

sns.violinplot(x="Glucose", data=t,ax=ax[0,1],palette='Set3')

sns.violinplot (x ='BloodPressure', data=t, ax=ax[1,0], palette='Set3')

sns.violinplot(x='SkinThickness', data=t, ax=ax[1,1],palette='Set3')

sns.violinplot(x='Insulin', data=t, ax=ax[2,0], palette='Set3')

sns.violinplot(x='BMI', data=t, ax=ax[2,1],palette='Set3')

sns.violinplot(x='DiabetesPedigreeFunction', data=t, ax=ax[3,0],palette='Set3')

sns.violinplot(x='Age', data=t, ax=ax[3,1],palette='Set3')

plt.show()
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier



X = t.iloc[:, :-1]

y = t.iloc[:, -1]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


LR = LogisticRegression()



#fiting the model

LR.fit(X_train, y_train)



#prediction

y_pred = LR.predict(X_test)



#Accuracy

print("Accuracy ", LR.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()
DT = DecisionTreeClassifier()



#fiting the model

DT.fit(X_train, y_train)



#prediction

y_pred = DT.predict(X_test)



#Accuracy

print("Accuracy ", DT.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()