%matplotlib inline
import numpy as np 

import pandas as pd 

import plotly_express as px

import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf

cf.go_offline(False)

cf.set_config_file(sharing = 'public',theme = 'space')

from cufflinks import iplot
train = pd.read_csv('../input/titanic/train.csv')

test =  pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
train.shape,test.shape
#Columns with missing values and percetage of missing values in them

null_columns = [col for col in train.columns if train[col].isnull().sum()>1]

for col in null_columns:

    print(col,': {},count {}'.format(train[col].isnull().mean(),train[col].isnull().sum()))
for col in null_columns:

    df = train.copy()

    df[col] = np.where(df[col].isna(),1,0)

    df.groupby(col)['Survived'].value_counts().iplot(kind = 'bar',xTitle = '(Missing,Survived)',yTitle = 'count')
num_cols = [col for col in train.columns if train[col].dtypes!='O']#Features with numerical values

conti_cols = ['Age','Fare']#Features with continuous numerical values
#Let's find out the disribution of continuous varibles by plotting histograms.

for col in conti_cols:

    df = train.copy()

    df[col].iplot(kind = 'hist',linecolor = 'white',xTitle = col,yTitle = 'count')
for col in conti_cols:

    df = train.copy()

    df.groupby('Survived')[col].mean().iplot(kind = 'bar',xTitle = 'Survived',yTitle = 'mean'+col)
for col in conti_cols:

    df = train.copy()

    fig = px.histogram(data_frame=df,color='Survived',x = col,barmode='group',template='plotly_dark')

    fig.show()
#Let's find out the outliers by ploting box plots for continuous variables

for col in conti_cols:

    df = train.copy()

    fig = px.box(y= col,data_frame=df,width=600,height=400,template = 'plotly_dark')

    fig.show()
disc_cols = [col for col in num_cols if col not in conti_cols+['PassengerId']]#Features with discrete numerical values

disc_cols
for col in disc_cols:

    df = train.copy()

    df.groupby('Survived')[col].value_counts().iplot(kind = 'bar',xTitle = '(Survived,'+col+')',yTitle = 'count')
cat_col = [col for col in train.columns if train[col].dtypes == 'O']

cat_col
train[cat_col].head()
for col in cat_col:

    print(col,"cardinality is {}".format(train[col].nunique()))
for col in ['Sex','Embarked']:

    df = train.copy()

    fig = px.histogram(x= col,data_frame=df,color = 'Survived',height = 400,width = 600,barmode='group',template = 'plotly_dark')

    fig.show()
fig = px.bar(x ='Cabin',data_frame = train,barmode='group',template='plotly_dark')

fig.show()
df = train.copy()

df['deck'] = train.Cabin.str[0]
df.deck.unique()
train['missingcabin'] = np.where(train.Cabin.isna(),1,0)
train.missingcabin.value_counts()
train['deck'] = df.deck.fillna('Z')
train.deck.value_counts()
df = test.copy()

df['deck'] = df.Cabin.str[0]
df.deck.unique()
test['missingcabin'] = np.where(test.Cabin.isna(),1,0)

test['deck'] = df.deck.fillna('Z')

test.deck.value_counts()
train['title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['title'].value_counts()
train['title'].value_counts()
rep = {'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master','Dr':'Miss','Rev':'Mr','Col':'Mr','Mile':'Miss','Major':'Mr','Ms':'Rare','Countess':'Mrs','Lady':'Mrs','Jonkheer':'Mrs','Mme':'Rare','Don':'Rare','Capt':'Rare','Sir':'Rare'}



train['title'] = train.title.map(rep)
train.title.value_counts()
rep = {'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master','Col':'Mr','Rev':'Mr','Dr':'Miss','Dona':'Rare','Ms':'Rare'}

test['title'] = test.title.map(rep)

test.title.value_counts()
train.head()
test.head()
train.drop(['PassengerId','Name'],axis = 1,inplace = True)

test.drop(['PassengerId','Name'],axis = 1,inplace = True)
train.shape,test.shape
train['famsize'] = train['SibSp']+train['Parch']+1

test['famsize'] = test['SibSp']+test['Parch']+1
train.head()
test.head()
train.drop(['Cabin','Ticket'],axis = 1,inplace = True)

test.drop(['Cabin','Ticket'],axis = 1,inplace = True)
x = train.groupby(['Pclass','title']).mean()['Age']# class and title wise age for impotation of missing values

print(x)
x.iplot(kind = 'bar',yTitle='Mean-Age',xTitle='(class,Title)',linecolor = 'white')
def apply_age(title,Pclass):

    if(title=='Master' and Pclass==1):

        age=5

    elif (title=='Miss' and Pclass==1):

        age=31

    elif (title=='Mr' and Pclass==1):

        age=42

    elif (title=='Mrs' and Pclass==1):

        age=40

    elif (title=='Rare' and Pclass==1):

        age=46

    elif (title=='Master' and Pclass==2):

        age=2

    elif (title=='Mr' and Pclass==2):

        age=33

    elif (title=='Mrs' and Pclass==2):

        age=33

    elif (title=='Miss' and Pclass==2):

        age=23

    elif (title=='Rare' and Pclass==2):

        age=28

    elif (title=='Master' and Pclass==3):

        age=5

    elif (title=='Mr' and Pclass==3):

        age=28

    elif (title=='Miss' and Pclass==3):

        age=16

    elif (title=='Mrs' and Pclass==3):

        age=33

    else:

        age=30 # mean age considered from describe()

    return age
y = test.groupby(['Pclass','title']).mean()['Age']

print(y)
y.iplot(kind = 'bar',xTitle = '(Class,Title)',yTitle = "mean-age")
train['Agemissing'] = np.where(train.Age.isna(),1,0)
age_null = train[train.Age.isna()]

age_null['Age'] = age_null.apply(lambda row : apply_age(row['title'],row['Pclass']), axis = 1) 

train['Age'].fillna(value=age_null['Age'],inplace=True)
def apply_age_test(title,Pclass):

    if(title=='Master' and Pclass==1):

        age=8

    elif (title=='Miss' and Pclass==1):

        age=31

    elif (title=='Mr' and Pclass==1):

        age=42

    elif (title=='Mrs' and Pclass==1):

        age=43

    elif (title=='Rare' and Pclass==1):

        age=40

    elif (title=='Master' and Pclass==2):

        age=4

    elif (title=='Mr' and Pclass==2):

        age=33

    elif (title=='Mrs' and Pclass==2):

        age=33

    elif (title=='Miss' and Pclass==2):

        age=17

    elif (title=='Rare' and Pclass==2):

        age=28

    elif (title=='Master' and Pclass==3):

        age=5

    elif (title=='Mr' and Pclass==3):

        age=28

    elif (title=='Miss' and Pclass==3):

        age=16

    elif (title=='Mrs' and Pclass==3):

        age=33

    else:

        age=30 # mean age considered from describe()

    return age
test['Agemissing'] = np.where(test.Age.isna(),1,0)
age_null_test = test[test.Age.isna()]

age_null_test['Age'] = age_null_test.apply(lambda row : apply_age(row['title'],row['Pclass']), axis = 1) 

test['Age'].fillna(value=age_null_test['Age'],inplace=True)
test.Age.isna().sum()
cat_train = train.copy()

cat_test = test.copy()
train = pd.get_dummies(train,drop_first=True)

test = pd.get_dummies(test,drop_first=True)
train.head()
test.head()
col = [col for col in train.columns if col not in test.columns]

print(col)
train.drop('deck_T',axis = 1,inplace = True)
train.shape
y = train['Survived']

train.drop('Survived',axis = 1,inplace = True)
pclass = {1:3,2:2,3:1}

train['Pclass'] = train.Pclass.map(pclass)
train.Pclass.value_counts()
test['Pclass'] = test.Pclass.map(pclass)
train.info()
test.info()
test.Fare.fillna(test.Fare.median(),inplace=True)
test.info()
test.Agemissing.value_counts()
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier,XGBRFClassifier

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report,accuracy_score,f1_score,roc_auc_score
rf = RandomForestClassifier(n_estimators=150,max_depth=4,random_state=42)
rf.fit(train,y)
print(classification_report(y,rf.predict(train)))
roc_auc_score(y,rf.predict(train))
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = rf.predict(test)
submission.to_csv('Submission1',index=False)
lgb = LGBMClassifier(n_estimators=200,max_depth=5,learning_rate=0.01,random_state=42)

lgb.fit(train,y)

print(classification_report(y,lgb.predict(train)),roc_auc_score(y,lgb.predict(train)))
submission['Survived'] = lgb.predict(test)

submission.to_csv('submission2',index = False)