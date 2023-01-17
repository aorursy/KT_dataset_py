import numpy as np

import pandas as pd

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline

from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier, plot_importance
data = pd.read_csv('../input/xAPI-Edu-Data.csv')

# Any results you write to the current directory are saved as output.

data.head()
data.describe()
print(data.shape)
data.columns
data.isnull().sum()
data['gender'].value_counts()
data['NationalITy'].value_counts()
data['PlaceofBirth'].value_counts()
data['StageID'].value_counts()
data['GradeID'].value_counts()
data['Topic'].value_counts()
data['Semester'].value_counts()
data['Relation'].value_counts()
data['raisedhands'].value_counts()
data['ParentschoolSatisfaction'].value_counts()
data['ParentAnsweringSurvey'].value_counts()
data['StudentAbsenceDays'].value_counts()
data['Class'].value_counts()
fig, axarr  = plt.subplots(2,2,figsize=(10,10))

sns.countplot(x='Class', data=data, ax=axarr[0,0], order=['L','M','H'])

sns.countplot(x='gender', data=data, ax=axarr[0,1], order=['M','F'])

sns.countplot(x='StageID', data=data, ax=axarr[1,0])

sns.countplot(x='Semester', data=data, ax=axarr[1,1])
fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))

sns.countplot(x='Topic', data=data, ax=axis1)

sns.countplot(x='NationalITy', data=data, ax=axis2)
fig, axarr  = plt.subplots(2,2,figsize=(10,10))

sns.countplot(x='gender', hue='Class', data=data, ax=axarr[0,0], order=['M','F'], hue_order=['L','M','H'])

sns.countplot(x='gender', hue='Relation', data=data, ax=axarr[0,1], order=['M','F'])

sns.countplot(x='gender', hue='StudentAbsenceDays', data=data, ax=axarr[1,0], order=['M','F'])

sns.countplot(x='gender', hue='ParentAnsweringSurvey', data=data, ax=axarr[1,1], order=['M','F'])
fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))

sns.countplot(x='Topic', hue='gender', data=data, ax=axis1)

sns.countplot(x='NationalITy', hue='gender', data=data, ax=axis2)
fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))

sns.countplot(x='NationalITy', hue='Relation', data=data, ax=axis1)

sns.countplot(x='NationalITy', hue='StudentAbsenceDays', data=data, ax=axis2)
fig, axarr  = plt.subplots(2,2,figsize=(10,10))

sns.barplot(x='Class', y='VisITedResources', data=data, order=['L','M','H'], ax=axarr[0,0])

sns.barplot(x='Class', y='AnnouncementsView', data=data, order=['L','M','H'], ax=axarr[0,1])

sns.barplot(x='Class', y='raisedhands', data=data, order=['L','M','H'], ax=axarr[1,0])

sns.barplot(x='Class', y='Discussion', data=data, order=['L','M','H'], ax=axarr[1,1])
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sns.barplot(x='gender', y='raisedhands', data=data, ax=axis1)

sns.barplot(x='gender', y='Discussion', data=data, ax=axis2)
fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))

sns.boxplot(x='Class', y='Discussion', data=data, order=['L','M','H'], ax=axis1)

sns.boxplot(x='Class', y='VisITedResources', data=data, order=['L','M','H'], ax=axis2)
fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))

sns.pointplot(x='Semester', y='VisITedResources', hue='gender', data=data, ax=axis1)

sns.pointplot(x='Semester', y='AnnouncementsView', hue='gender', data=data, ax=axis2)
fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))

sns.regplot(x='raisedhands', y='VisITedResources', data=data, ax=axis1)

sns.regplot(x='AnnouncementsView', y='Discussion', data=data, ax=axis2)
plot = sns.countplot(x='Class', hue='Relation', data=data, order=['L', 'M', 'H'], palette='Set1')

plot.set(xlabel='Class', ylabel='Count', title='Gender comparison')

plt.show()
Features = data.drop('gender',axis=1)

Target = data['gender']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])

    
Features = data.drop('Semester',axis=1)

Target = data['Semester']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])
Features = data.drop('ParentAnsweringSurvey',axis=1)

Target = data['ParentAnsweringSurvey']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])
Features = data.drop('Relation',axis=1)

Target = data['Relation']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])
Features = data.drop('ParentschoolSatisfaction',axis=1)

Target = data['ParentschoolSatisfaction']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])
Features = data.drop('StudentAbsenceDays',axis=1)

Target = data['StudentAbsenceDays']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])
Features = data.drop('Class',axis=1)

Target = data['Class']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])
X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2, random_state=52)
Logit_Model = LogisticRegression()

Logit_Model.fit(X_train,y_train)








Prediction = Logit_Model.predict(X_test)

Score = accuracy_score(y_test,Prediction)

Report = classification_report(y_test,Prediction)


print(Prediction)
print(Score)
print(Report)
xgb = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,seed=10)

xgb_pred = xgb.fit(X_train, y_train).predict(X_test)

print (classification_report(y_test,xgb_pred))
print(accuracy_score(y_test,xgb_pred))
plot_importance(xgb)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=20, criterion='entropy')

model.fit(X_train, y_train)

model.score(X_test,y_test)
