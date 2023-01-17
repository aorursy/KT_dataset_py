import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from plotly import __version__

from plotly.offline import download_plotlyjs,init_notebook_mode,iplot,plot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

import warnings

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format
df_train = pd.read_csv("../input/healthcare-dataset-stroke-data/train_2v.csv")

df_test = pd.read_csv("../input/healthcare-dataset-stroke-data/test_2v.csv")
df_train.head(2)
df_train.info()
def findMissingValue(df):

    for fn in df.columns:

        targetNum = len(df)

        x= df[fn].describe()[0]

        if x !=targetNum:

            missingValue = targetNum-x

            percentOfMV = round(float((missingValue/targetNum)*100),2)

            print(fn + ' has missing value = '+str(missingValue)+' ('+str(percentOfMV)+'%)')

        else:

            print(fn+ ' = No Missing Value')
findMissingValue(df_train)
findMissingValue(df_test)
df_train[df_train['smoking_status'].isna()].count()[0]
df_test[df_test['smoking_status'].isna()].count()[0]
df_train[df_train['smoking_status'].isna()]['work_type'].unique()
df_train[df_train['smoking_status'].isna()].groupby('work_type')['stroke'].count().iplot(kind='bar')
df_train[df_train['smoking_status'].notna()].groupby('work_type')['stroke'].count().iplot(kind='bar')
df_train['smoking_status'].fillna(value='unknown',inplace=True)

df_test['smoking_status'].fillna(value='unknown',inplace=True)
df_train.dropna(inplace=True)

df_test.dropna(inplace=True)
findMissingValue(df_train)
findMissingValue(df_test)
sns.pairplot(df_train)
df_train[df_train['stroke']==1]['gender'].iplot(kind='hist')
df_train['stroke'].value_counts().iplot(kind='bar')
minor = df_train[df_train['age']<=18]

young = df_train[(df_train['age']>=19) & (df_train['age']<=40)]

middle = df_train[(df_train['age']>=41) & (df_train['age']<=60)]

elderly = df_train[df_train['age']>=61]
import plotly.graph_objects as go



labels = ['minor','young','middle','elderly']

values = [(len(minor)/41938)*100,(len(young)/41938)*100,(len(middle)/41938)*100,(len(elderly)/41938)*100]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
minor_stroke = df_train[(df_train['age']<=18) & (df_train['stroke']==1)]

young_stroke = df_train[(df_train['age']>=19) & (df_train['age']<=40) & (df_train['stroke']==1)]

middle_stroke = df_train[(df_train['age']>=41) & (df_train['age']<=60) & (df_train['stroke']==1)]

elderly_stroke = df_train[(df_train['age']>=61) & (df_train['stroke']==1)]
import plotly.graph_objects as go



labels = ['minor','young','middle','elderly']

values = [(len(minor_stroke)/643)*100,(len(young_stroke)/643)*100,(len(middle_stroke)/643)*100,(len(elderly_stroke)/643)*100]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
df = pd.get_dummies(df_train)
df=df.drop('id',axis=1)
df2 = pd.get_dummies(df_test)
df2=df2.drop('id',axis=1)
X= df.drop(['stroke'],axis=1)

y= df['stroke']
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.linear_model import LogisticRegression





bbc_lr = BalancedBaggingClassifier(base_estimator=LogisticRegression(),

                                sampling_strategy='auto',

                                replacement=False,

                                random_state=1)



y_train = df['stroke']

X_train = df.drop(['stroke'], axis=1, inplace=False)





bbc_lr.fit(X_train, y_train)





prediction = bbc_lr.predict(X_test)



bbc_lr.score(X_test,y_test)

print('The Logistic Regression Accuracy is {:.2f} %'.format(bbc_lr.score(X_test,y_test)*100))



print('\n')



print(classification_report(y_test,prediction))

print('\n')

print(confusion_matrix(y_test,prediction))
from sklearn.tree import DecisionTreeClassifier



#Create an object of the classifier.

bbc_dt = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),

                                sampling_strategy='auto',

                                replacement=False,

                                random_state=0)



y_train = df['stroke']

X_train = df.drop(['stroke'], axis=1, inplace=False)



#Train the classifier.

bbc_dt.fit(X_train, y_train)



prediction = bbc_dt.predict(X_test)



bbc_dt.score(X_test,y_test)

print('The Decision Tree Accuracy is {:.2f} %'.format(bbc_dt.score(X_test,y_test)*100))



print('\n')



print(classification_report(y_test,prediction))

print('\n')

print(confusion_matrix(y_test,prediction))
from sklearn.ensemble import RandomForestClassifier

#Create an object of the classifier.

bbc_rf = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(),

                                sampling_strategy='auto',

                                replacement=False,

                                random_state=0)



y_train = df['stroke']

X_train = df.drop(['stroke'], axis=1, inplace=False)



#Train the classifier.

bbc_rf.fit(X_train, y_train)



prediction = bbc_rf.predict(X_test)



bbc_rf.score(X_test,y_test)

print('The Random Forest Accuracy is {:.2f} %'.format(bbc_rf.score(X_test,y_test)*100))



print('\n')



print(classification_report(y_test,prediction))

print('\n')

print(confusion_matrix(y_test,prediction))
from sklearn.svm import SVC

#Create an object of the classifier.

bbc_sv = BalancedBaggingClassifier(base_estimator=SVC(random_state=1),

                                sampling_strategy='auto',

                                replacement=False,

                                random_state=1)



y_train = df['stroke']

X_train = df.drop(['stroke'], axis=1, inplace=False)



#Train the classifier.

bbc_sv.fit(X_train, y_train)



prediction = bbc_sv.predict(X_test)



bbc_sv.score(X_test,y_test)

print('The Support Vector Accuracy is {:.2f} %'.format(bbc_sv.score(X_test,y_test)*100))



print('\n')



print(classification_report(y_test,prediction))

print('\n')

print(confusion_matrix(y_test,prediction))
# Saving Model

import pickle

saved_model = pickle.dumps(bbc_sv)
# Load the Pickled model

bbc_sv_from_pickle = pickle.loads(saved_model)
# Using the loaded pickle model to make predictions

df2['stroke']= bbc_sv_from_pickle.predict(df2)
df_test['stroke']= df2['stroke']
df_test.head()
sns.pairplot(df_test)
df_test[df_test['stroke']==1]['gender'].iplot(kind='hist')
df_test['stroke'].value_counts().iplot(kind='bar')
minor = df_test[df_test['age']<=18]

young = df_test[(df_test['age']>=19) & (df_test['age']<=40)]

middle = df_test[(df_test['age']>=41) & (df_test['age']<=60)]

elderly = df_test[df_test['age']>=61]
import plotly.graph_objects as go



labels = ['minor','young','middle','elderly']

values = [(len(minor)/len(df_test['age']))*100,(len(young)/len(df_test['age']))*100,(len(middle)/len(df_test['age']))*100,(len(elderly)/len(df_test['age']))*100]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
minor_stroke = df_test[(df_test['age']<=18) & (df_test['stroke']==1)]

young_stroke = df_test[(df_test['age']>=19) & (df_test['age']<=40) & (df_test['stroke']==1)]

middle_stroke = df_test[(df_test['age']>=41) & (df_test['age']<=60) & (df_test['stroke']==1)]

elderly_stroke = df_test[(df_test['age']>=61) & (df_test['stroke']==1)]
import plotly.graph_objects as go



labels = ['minor','young','middle','elderly']

values = [(len(minor_stroke)/len(df_test[df_test['stroke']==1]))*100,(len(young_stroke)/len(df_test[df_test['stroke']==1]))*100,(len(middle_stroke)/len(df_test[df_test['stroke']==1]))*100,(len(elderly_stroke)/len(df_test[df_test['stroke']==1]))*100]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()