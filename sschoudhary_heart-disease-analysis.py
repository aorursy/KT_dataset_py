# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

heart = pd.read_csv("../input/heart-disease-uci/heart.csv")

heart.head()
heart.columns = ['Age', 'Sex', 'Chest_Pain_type', 'Resting_BP', 'Cholesterol', 'Fasting_Blood_Sugar', 'Rest_ECG', 'Max_Heart_Rate',

       'Exercise_Induced_Angina', 'ST_Depression', 'ST_Slope', 'No.of_Major_Vessels', 'Thalassemia', 'Target']

heart.head()

# looks better now
heart.info()
import warnings

warnings.filterwarnings("ignore")

heart['Sex'][heart['Sex'] == 0] = 'Female'

heart['Sex'][heart['Sex'] == 1] = 'Male'

heart['Chest_Pain_type'][heart['Chest_Pain_type'] == 1] = 'Typical Angina'

heart['Chest_Pain_type'][heart['Chest_Pain_type'] == 2] = 'aTypical Angina'

heart['Chest_Pain_type'][heart['Chest_Pain_type'] == 3] = 'Non-Anginal Pain'

heart['Chest_Pain_type'][heart['Chest_Pain_type'] == 4] = 'Asymptomatic'

heart['Fasting_Blood_Sugar'][heart['Fasting_Blood_Sugar'] == 0] = 'Lower than 120mg/ml'

heart['Fasting_Blood_Sugar'][heart['Fasting_Blood_Sugar'] == 1] = 'Greater than 120mg/ml'

heart['Rest_ECG'][heart['Rest_ECG'] == 0] = 'Normal'

heart['Rest_ECG'][heart['Rest_ECG'] == 1] = 'ST-T Wave Abnormality'

heart['Rest_ECG'][heart['Rest_ECG'] == 2] = 'Left Ventricular Hypertrophy'

heart['Exercise_Induced_Angina'][heart['Exercise_Induced_Angina'] == 0] = 'no'

heart['Exercise_Induced_Angina'][heart['Exercise_Induced_Angina'] == 1] = 'yes'

heart['ST_Slope'][heart['ST_Slope'] == 1] = 'Upsloping'

heart['ST_Slope'][heart['ST_Slope'] == 2] = 'Flat'

heart['ST_Slope'][heart['ST_Slope'] == 3] = 'Downsloping'

heart['Thalassemia'][heart['Thalassemia'] == 1] = 'Normal'

heart['Thalassemia'][heart['Thalassemia'] == 2] = 'Fixed defect'

heart['Thalassemia'][heart['Thalassemia'] == 3] = 'Reversable defect'
# import seaborn as sns

# sns.countplot(x='Sex', data=heart)

# sns.countplot(x='Chest_Pain_type', data=heart)

# sns.countplot(x='Fasting_Blood_Sugar', data=heart)

# sns.countplot(x='Rest_ECG', data=heart)

# sns.countplot(x='Exercise_Induced_Angina', data=heart)

# sns.countplot(x='ST_Slope', data=heart)

# sns.countplot(x='Thalassemia', data=heart)
heart.info()
heart.head()
heart = pd.get_dummies(heart, drop_first=True)

heart.head()
X= heart.drop('Target',axis=1)

y=heart['Target']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_predict = lr.predict(X_test)

lr_confusion_matrix = confusion_matrix(y_test, lr_predict)

lr_accuracy_score = accuracy_score(y_test, lr_predict)

print(lr_confusion_matrix)

print(lr_accuracy_score)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics

rf=RandomForestClassifier(n_estimators=250)

rf.fit(X_train,y_train)

Rfpred=rf.predict(X_test)

confusion_matrix(y_test, Rfpred)

metrics.accuracy_score(y_test,Rfpred)
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

model_SVC=SVC()

model_SVC.fit(X_train, y_train)

predictions=model_SVC.predict(X_test)

confusion_matrix(y_test, predictions)

classification_report(y_test,predictions,digits=4)

metrics.accuracy_score(y_test,predictions)