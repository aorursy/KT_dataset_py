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

%matplotlib inline



breast = pd.read_csv("../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")
breast
breast.head()
breast.info()
breast.shape
breast.isnull().sum()
sns.heatmap(breast.isnull(),yticklabels=False)
breast.hist(figsize=(9,10),bins=10)
breast.plot()
fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(18,18))

plt.suptitle('Violin Plots',fontsize=24)

sns.violinplot(x="mean_radius", data=breast,ax=ax[0,0],palette='Set3')

sns.violinplot(x="mean_texture", data=breast,ax=ax[0,1],palette='Set3')

sns.violinplot(x="mean_perimeter", data=breast,ax=ax[1,0],palette='Set3')

sns.violinplot(x="mean_area", data=breast,ax=ax[1,1],palette='Set3')

sns.violinplot(x="mean_smoothness", data=breast,ax=ax[2,0],palette='Set3')

sns.violinplot(x="diagnosis", data=breast,ax=ax[2,1],palette='Set3')
x_label=breast['mean_radius']

y_label=breast['mean_perimeter']

plt.hist2d(x_label,y_label)
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LinearRegression



X = breast.iloc[:, :-1]

y = breast.iloc[:, -1]





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
AR = LinearRegression()



#fiting the model

AR.fit(X_train, y_train)



#prediction

y_pred = AR.predict(X_test)



#Accuracy

print("Accuracy ", AR.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

#cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

#plt.show()
model = GradientBoostingClassifier()



#fiting the model

model.fit(X_train, y_train)



#prediction

y_pred = model.predict(X_test)



#Accuracy

print("Accuracy ", model.score(X_test, y_test)*100)



#Plot the confusion matrix

sns.set(font_scale=1.5)

cm = confusion_matrix(y_pred, y_test)

sns.heatmap(cm, annot=True, fmt='g')

plt.show()