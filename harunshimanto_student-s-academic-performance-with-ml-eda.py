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
print('Percentage',data.gender.value_counts(normalize=True))

data.gender.value_counts(normalize=True).plot(kind='bar')
data['NationalITy'].value_counts()
print('Percentage',data.NationalITy.value_counts(normalize=True))

data.NationalITy.value_counts(normalize=True).plot(kind='bar')
data['PlaceofBirth'].value_counts()
print('Percentage',data.PlaceofBirth.value_counts(normalize=True))

data.PlaceofBirth.value_counts(normalize=True).plot(kind='bar')
data['StageID'].value_counts()
print('Percentage',data.StageID.value_counts(normalize=True))

data.StageID.value_counts(normalize=True).plot(kind='bar')
data['GradeID'].value_counts()
print('Percentage',data.GradeID.value_counts(normalize=True))

data.GradeID.value_counts(normalize=True).plot(kind='bar')
data['Topic'].value_counts()
print('Percentage',data.Topic.value_counts(normalize=True))

data.Topic.value_counts(normalize=True).plot(kind='bar')
data['Semester'].value_counts()
print('Parcentage',data.Semester.value_counts(normalize=True))

data.Semester.value_counts(normalize=True).plot(kind='bar')
data['Relation'].value_counts()
print('Parcentage',data.Relation.value_counts(normalize=True))

data.Relation.value_counts(normalize=True).plot(kind='bar')
data['raisedhands'].value_counts()
#print('Parcentage',df.raisedhands.value_counts(normalize=True))

#df.raisedhands.value_counts(normalize=True).plot(kind='bar')

color_brewer = ['#41B5A3','#FFAF87','#FF8E72','#ED6A5E','#377771','#E89005','#C6000D','#000000','#05668D','#028090','#9FD35C',

                '#02C39A','#F0F3BD','#41B5A3','#FF6F59','#254441','#B2B09B','#EF3054','#9D9CE8','#0F4777','#5F67DD','#235077','#CCE4F9','#1748D1',

                '#8BB3D6','#467196','#F2C4A2','#F2B1A4','#C42746','#330C25']

fig = {

  "data": [

    {

      "values": data["raisedhands"].value_counts().values,

      "labels": data["raisedhands"].value_counts().index,

      "domain": {"x": [0, .95]},

      "name": "Raisedhands Parcentage",

      "hoverinfo":"label+percent+name",

      "hole": .7,

      "type": "pie",

      "marker": {"colors": [i for i in reversed(color_brewer)]},

      "textfont": {"color": "#FFFFFF"}

    }],

  "layout": {

        "title":"Raisedhands Parcentage",

        "annotations": [

            {

                "font": {

                    "size": 15

                },

                "showarrow": False,

                "text": "Raisedhands Parcentage",

                "x": 0.47,

                "y": 0.5

            }

        ]

    }

}

iplot(fig, filename='donut')
data['ParentschoolSatisfaction'].value_counts()
print('Parcentage',data.ParentschoolSatisfaction.value_counts(normalize=True))

data.ParentschoolSatisfaction.value_counts(normalize=True).plot(kind='bar')
data['ParentAnsweringSurvey'].value_counts()
print('Parcentage',data.ParentAnsweringSurvey.value_counts(normalize=True))

data.ParentAnsweringSurvey.value_counts(normalize=True).plot(kind='bar')
data['StudentAbsenceDays'].value_counts()
print('Parcentage',data.StudentAbsenceDays.value_counts(normalize=True))

data.StudentAbsenceDays.value_counts(normalize=True).plot(kind='bar')
data['Class'].value_counts()
print('Parcentage',data.Class.value_counts(normalize=True))

data.Class.value_counts(normalize=True).plot(kind='bar')
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

sns.swarmplot(x='gender', y='AnnouncementsView', data=data, ax=axis1)

sns.swarmplot(x='gender', y='raisedhands', data=data, ax=axis2)
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
sns.pairplot(data,hue='Class')
import networkx as nx



g= nx.Graph()

g = nx.from_pandas_dataframe(data,source='gender',target='PlaceofBirth')

print (nx.info(g))





plt.figure(figsize=(10,10)) 

nx.draw_networkx(g,with_labels=True,node_size=50, alpha=0.5, node_color="blue")

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