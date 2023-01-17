# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

diabetes = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

diabetes.columns 
diabetes.head()
print("Diabetes data set dimensions : {}".format(diabetes.shape))
diabetes.groupby('Outcome').size()
diabetes.groupby('Outcome').hist(figsize=(9, 9))
diabetes.isnull().sum()

diabetes.isna().sum()
#no living human should have 0 blood pressure

print("Total : ", diabetes[diabetes.BloodPressure == 0].shape[0])
# 0 is invalid reading for glucose

print("Total : ", diabetes[diabetes.Glucose == 0].shape[0])
# 0 is invalid for skin thickness

print("Total : ", diabetes[diabetes.SkinThickness == 0].shape[0])
print("Total : ", diabetes[diabetes.BMI == 0].shape[0])
print("Total : ", diabetes[diabetes.Insulin == 0].shape[0])
#remove row with 0

diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]

print(diabetes_mod.shape)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = diabetes_mod[feature_names]

y = diabetes_mod.Outcome
#import libraries

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
#initialize

models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVC', SVC()))

models.append(('LR', LogisticRegression()))

models.append(('DT', DecisionTreeClassifier()))

models.append(('GNB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

models.append(('GB', GradientBoostingClassifier()))
#evaluation method - train test split

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)
names = []

scores = []

for name, model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    scores.append(accuracy_score(y_test, y_pred))

    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})

print(tr_split)
# evaluation method k fold cross validation

from sklearn.model_selection import KFold

names = []

scores = []

for name, model in models:

    

    kfold = KFold(n_splits=10, random_state=10) 

    score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()

    

    names.append(name)

    scores.append(score)

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})

print(kf_cross_val)
#plot the accuracy

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)

axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:

    height = p.get_height()

    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 

    

plt.show()