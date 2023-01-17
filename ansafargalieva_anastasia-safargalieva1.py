# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/heart.csv")
data.head()
data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
data.head()
data['sex'][data['sex'] == 'Female'] = 0

data['sex'][data['sex'] == 'Male'] = 1
data.head()
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier



X = data.iloc[:, :-1]

y = data.iloc[:, -1]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Model

LR = LogisticRegression()



#fiting the model

LR.fit(X_train, y_train)



#prediction

y_pred = LR.predict(X_test)



#Accuracy

print("Accuracy ", LR.score(X_test, y_test)*100)
