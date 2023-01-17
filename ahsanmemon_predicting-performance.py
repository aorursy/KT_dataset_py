# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler,LabelEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/xAPI-Edu-Data.csv")
# Testing Null Values
df.isnull().sum()
df['Class'].value_counts().plot(kind = 'bar')
df['Topic'].value_counts().plot(kind = 'bar')
df.gender.value_counts().plot(kind = 'bar')
df.NationalITy.value_counts().plot(kind = 'bar')
df.dtypes
df.select_dtypes(exclude=['object'])
# Selecting all columns that need transformation
df.select_dtypes(include=['object']).columns
# Encoding Labels
label=LabelEncoder()
def encode_labels(df,labels_to_encode):
    for column in labels_to_encode:
        df[column] = label.fit_transform(df[column])
    return df

df_labelled = encode_labels(df,['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
       'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
       'ParentschoolSatisfaction', 'StudentAbsenceDays'])

# Applying Machine Learning
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

x=df_labelled.iloc[:,:-1]
y=df_labelled.iloc[:,-1]

# Score Before scaling
svc=SVC(gamma='auto') #Default hyperparameters
acc_before_scaling=cross_val_score(svc,x,y,cv=10).mean()
print("Accuracy Score Before scaling: ",acc_before_scaling)

# Scaling
sc = StandardScaler()
sc.fit(x)
X_std = sc.transform(x)

# Score After scaling
svc_std=SVC(gamma='auto') #Default hyperparameters
acc_after_scaling=cross_val_score(svc,X_std,y,cv=10).mean()
print("Accuracy Score After scaling: ",acc_after_scaling)
