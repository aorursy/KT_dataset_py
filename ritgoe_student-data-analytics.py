# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import seaborn as sns

%matplotlib inline



dataset = pd.read_csv('../input/xAPI-Edu-Data.csv')
dataset.head()
labelEncoder = LabelEncoder()

dataset['Class'] = labelEncoder.fit_transform(dataset['Class'])

# 0 - High,1 - Low, 2 - Med
sns.barplot(x='Class',y='raisedhands',data=dataset)

# More raised hands seems directly related to Class
sns.countplot(x='Class',hue='Relation',data=dataset)

# 'Class' seems to be more related to moms so we will keep Moms here
relation_data = pd.get_dummies(dataset['Relation'])

dataset = pd.concat([dataset,relation_data],axis=1)

# Now drop Relation and Father column

dataset.drop(['Relation','Father'],axis=1,inplace=True)
# Semester vs Class

sns.countplot(x='Class',hue='Semester',data=dataset)
# Second semester seems more related to Class

semester_data = pd.get_dummies(dataset['Semester'])

dataset = pd.concat([dataset,semester_data],axis=1)

dataset.drop(['Semester','F'],axis=1,inplace=True)
# Student absence days

sns.countplot(x='Class',hue='StudentAbsenceDays',data=dataset)
# Above-7 seems more promising then Under-7

absence_data = pd.get_dummies(dataset['StudentAbsenceDays'])

dataset = pd.concat([dataset,absence_data],axis=1)

dataset.drop(['StudentAbsenceDays','Under-7'],axis=1,inplace=True)
# Gender

sns.countplot(x='Class',hue='gender',data=dataset)
gender_data = pd.get_dummies(dataset['gender'])

dataset = pd.concat([dataset,gender_data],axis=1)

dataset.drop(['gender','M'],axis=1,inplace=True)
# Parent school satisfaction

sns.countplot(x='Class',hue='ParentschoolSatisfaction',data=dataset)
# Keeping 'Bad'

satisfac_data = pd.get_dummies(dataset['ParentschoolSatisfaction'])

dataset = pd.concat([dataset,satisfac_data],axis=1)

dataset.drop(['ParentschoolSatisfaction','Good'],axis=1,inplace=True)
# Parents answering survey

sns.countplot(x='Class',hue='ParentAnsweringSurvey',data=dataset)
# Seems like Class = Low student parents are not answering questions

# Keeping 'No'

survey_data = pd.get_dummies(dataset['ParentAnsweringSurvey'])

dataset = pd.concat([dataset,survey_data],axis=1)

dataset.drop(['ParentAnsweringSurvey','Yes'],axis=1,inplace=True)
# Subject

sns.countplot(x='Topic',hue='Class',data=dataset)
topic_data = pd.get_dummies(dataset['Topic'],drop_first=True)

dataset = pd.concat([dataset,topic_data],axis=1)

dataset.drop(['Topic'],axis=1,inplace=True)
# Section

sns.countplot(x='Class',hue='SectionID',data=dataset)
section_data = pd.get_dummies(dataset['SectionID'])

dataset = pd.concat([dataset,section_data],axis=1)

dataset.drop(['SectionID','C'],axis=1,inplace=True)
sns.countplot(x='Class',hue='GradeID',data=dataset)
sns.countplot(x='Class',hue='StageID',data=dataset)
stage_data = pd.get_dummies(dataset['StageID'])

dataset = pd.concat([dataset,stage_data],axis=1)

dataset.drop(['StageID','HighSchool'],axis=1,inplace=True)
dataset.drop(['GradeID','PlaceofBirth'],axis=1,inplace=True)
sns.countplot(x='NationalITy',hue='Class',data=dataset)
dataset.drop(['NationalITy'],axis=1,inplace=True)
dataset.rename(columns={'S': 'Second_sem', 'F': 'Female','No':'NoSurvey','Bad':'BadSatisfaction'}, inplace=True)
# Now lets seperate input and output

X = dataset.drop('Class',axis=1)

y = dataset['Class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_test[0]
from sklearn.svm import SVC

svc = SVC(C=100,kernel='rbf',gamma=0.1)

svc.fit(X_train,y_train)
svc.score(X_train,y_train)
svc.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train,y_train)
dtc.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 150)

rfc.fit(X_train,y_train)
rfc.score(X_test,y_test)