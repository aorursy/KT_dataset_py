import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
df=pd.read_csv("../input/xAPI-Edu-Data.csv")

df.head()
df.isnull().sum()
df['Class'].value_counts()
sns.countplot(df.Class)
df['Topic'].value_counts()
sns.countplot(df.Topic.sort_index(ascending=False))

plt.xticks(rotation=45)
pd.crosstab(df['Class'],df['Topic'])
df.columns
sns.countplot(df.gender)
df.NationalITy.value_counts()
sns.countplot(df.NationalITy)

plt.xticks(rotation=45)
print(df.dtypes)
label=LabelEncoder()
gender=label.fit_transform(df['gender'])

Nationality=label.fit_transform(df['NationalITy'])

PlaceOfBirth=label.fit_transform(df['PlaceofBirth'])

StageID=label.fit_transform(df['StageID'])

GradeID=label.fit_transform(df['GradeID'])

SectionID=label.fit_transform(df['SectionID'])

Topic=label.fit_transform(df['Topic'])

Semester=label.fit_transform(df['Semester'])

Relation=label.fit_transform(df['Relation'])

ParentAnsweringSurvey=label.fit_transform(df['ParentAnsweringSurvey'])

ParentschoolSatisfaction=label.fit_transform(df['ParentschoolSatisfaction'])

StudentAbsenceDays=label.fit_transform(df['StudentAbsenceDays'])

Class=label.fit_transform(df['Class'])
df.drop(['gender','NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic',

         'Semester','Relation','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays','Class'],

       axis=1,inplace=True)
df.head()
df['gender']=gender

df['NationalITy']=Nationality

df['PlaceofBirth']=PlaceOfBirth

df['StageID']=StageID

df['GradeID']=GradeID

df['SectionID']=SectionID

df['Topic']=Topic

df['Semester']=Semester

df['Relation']=Relation

df['ParentAnsweringSurvey']=ParentAnsweringSurvey

df['ParentschoolSatisfaction']=ParentschoolSatisfaction

df['StudentAbsenceDays']=StudentAbsenceDays

df['Class']=Class
df.head()
plt.figure(figsize=(12,10))

sns.heatmap(df.corr())
x=df.iloc[:,:-1]

y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.svm import SVC

from sklearn import metrics

svc=SVC() #Default hyperparameters

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

print('Accuracy Score:',end='')

print(metrics.accuracy_score(y_test,y_pred))

acc_before_scaling=metrics.accuracy_score(y_test,y_pred)
sc = StandardScaler()

sc.fit(x_train)

X_train_std = sc.transform(x_train)

X_test_std = sc.transform(x_test)
from sklearn.svm import SVC

from sklearn import metrics

svc=SVC() #Default hyperparameters

svc.fit(X_train_std,y_train)

y_pred=svc.predict(X_test_std)

print('Accuracy Score:',end='')

print(metrics.accuracy_score(y_test,y_pred))

acc_after_scaling=metrics.accuracy_score(y_test,y_pred)
print(metrics.classification_report(y_test, y_pred))
svm = SVC(kernel='linear', C=2.0, random_state=0)

svm.fit(X_train_std, y_train)



y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' %metrics.accuracy_score(y_test, y_pred))

acc_linear_svc=metrics.accuracy_score(y_test, y_pred)
print(metrics.classification_report(y_test, y_pred))
svm = SVC(kernel='rbf', random_state=0, gamma=2, C=1.0)

svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' %metrics.accuracy_score(y_test, y_pred))

acc_svc_circular_rbf=metrics.accuracy_score(y_test, y_pred)
print(metrics.classification_report(y_test, y_pred))
models = pd.DataFrame({

    'Model': ['SVM Before Scaling', 'SVM After Scaling', 'SVC Linear', 

              'SVC Circular'],

    'Score': [acc_before_scaling, acc_after_scaling, acc_linear_svc, 

              acc_svc_circular_rbf]})

models.sort_values(by='Score', ascending=False)