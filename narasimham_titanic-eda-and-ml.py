# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH = "../input/titanic/"

train=pd.read_csv(PATH + 'train.csv')

test=pd.read_csv(PATH+ 'test.csv')
train.head()
train.info()
cols = ['Survived','Pclass','Sex', 'SibSp',

       'Parch', 'Embarked']



fig,ax = plt.subplots(math.ceil(len(cols)/3),3,figsize=(20, 12))

ax = ax.flatten()

for a,s in zip(ax,cols):

    sns.countplot(x =s,data = train,ax =a)   
from sklearn.preprocessing import LabelEncoder

df=train

label_encoder=LabelEncoder()

df['Sex']=label_encoder.fit_transform(df['Sex'])

test['Sex']=label_encoder.fit_transform(test['Sex'])
df['Cabin']=df['Cabin'].fillna(0)

for row in range(891):

    if(df.loc[row,'Cabin']==0):

        df.loc[row,'Ca']=0

    else:

        df.loc[row,'Ca']=1

        

print(df[df['Ca']==1].mean(),'\n\n',df[df['Ca']==0].mean())
test['Cabin']=test['Cabin'].fillna(0)

for row in range(418):

    if(test.loc[row,'Cabin']==0):

        test.loc[row,'Ca']=0

    else:

        test.loc[row,'Ca']=1



print(test[test['Ca']==1].mean(),'\n\n',test[test['Ca']==0].mean())
print(df[df['Sex']==0].mean(),'\n\n',df[df['Sex']==1].mean())
print(test[test['Sex']==0].mean(),'\n\n',test[test['Sex']==1].mean())
df['Age']=df['Age'].fillna(0)

print(df[(df['Sex']==0) & (df['Age']==0)].count(),'\n\n',df[(df['Sex']==1) & (df['Age']==0)].count())
for row in range(891):

    if ('Mr.' in df.loc[row,'Name'] or 'Don.' in df.loc[row,'Name'] or 'Jonkheer.' in df.loc[row,'Name']):

        df.loc[row,'Sal']=1;

    elif 'Mrs.' in df.loc[row,'Name']:

        df.loc[row,'Sal']=2;

    elif 'Master.' in df.loc[row,'Name']:

        df.loc[row,'Sal']=3;

    elif ('Miss.' in df.loc[row,'Name'] or 'Ms.' in df.loc[row,'Name'] or 'Mlle.' in df.loc[row,'Name'] or 'Mme.' in df.loc[row,'Name']):

        df.loc[row,'Sal']=4;

    elif 'Rev.' in df.loc[row,'Name']: 

        df.loc[row,'Sal']=5;

    elif 'Dr.' in df.loc[row,'Name']: 

        df.loc[row,'Sal']=6;

    elif ('Major.' in df.loc[row,'Name'] or 'Col.' in df.loc[row,'Name'] or 'Capt.' in df.loc[row,'Name']): 

        df.loc[row,'Sal']=7;

    else: 

        df.loc[row,'Sal']=8;

        

print("Mean age Mr.:",df[df['Sal']==1].Age.mean(),"Mean survival rate of Mr.:",df[df['Sal']==1].Survived.mean())

print("Mean age Mrs.:",df[df['Sal']==2].Age.mean(),"Mean survival rate of Mrs.:",df[df['Sal']==2].Survived.mean())

print("Mean age Master.:",df[df['Sal']==3].Age.mean(),"Mean survival rate of Master.:",df[df['Sal']==3].Survived.mean())

print("Mean age Miss.:",df[df['Sal']==4].Age.mean(),"Mean survival rate of Miss.:",df[df['Sal']==4].Survived.mean())

print("Mean age Rev.:",df[df['Sal']==5].Age.mean(),"Mean survival rate of Rev.:",df[df['Sal']==5].Survived.mean())

print("Mean age Dr.:",df[df['Sal']==6].Age.mean(),"Mean survival rate of Dr.:",df[df['Sal']==6].Survived.mean())

print("Mean age Military titles:",df[df['Sal']==7].Age.mean(),"Mean survival rate of Military titles:",df[df['Sal']==7].Survived.mean())

print("Mean age Royal.:",df[df['Sal']==8].Age.mean(),"Mean survival rate of Royal.:",df[df['Sal']==8].Survived.mean())
Mr_rows=df[(df['Age']==0) & (df['Sal']==1)].index

Mrs_rows=df[(df['Age']==0) & (df['Sal']==2)].index

Master_rows=df[(df['Age']==0) & (df['Sal']==3)].index

Ms_rows=df[(df['Age']==0) & (df['Sal']==4)].index

Rev_rows=df[(df['Age']==0) & (df['Sal']==5)].index

Dr_rows=df[(df['Age']==0) & (df['Sal']==6)].index

Military_rows=df[(df['Age']==0) & (df['Sal']==7)].index

Royal_rows=df[(df['Age']==0) & (df['Sal']==8)].index



print('Nans in Mr:',Mr_rows)

print('Nans in Mrs:',Mrs_rows)

print('Nans in Master:',Master_rows)

print('Nans in Miss:',Ms_rows)

print('Nans in Rev:',Rev_rows)

print('Nans in Dr:',Dr_rows)

print('Nans in Military:',Military_rows)

print('Nans in Royal:',Royal_rows)
print("Mean age of Men:",df[(df['Age']!=0) & (df['Sal']==1)].Age.mean())

print("Mean age of Women:",df[(df['Age']!=0) & (df['Sal']==2)].Age.mean())

print("Mean age of Master:",df[(df['Age']!=0) & (df['Sal']==3)].Age.mean())

print("Mean age of Miss:",df[(df['Age']!=0) & (df['Sal']==4)].Age.mean())

print("Mean age of Rev:",df[(df['Age']!=0) & (df['Sal']==5)].Age.mean())

print("Mean age of Dr:",df[(df['Age']!=0) & (df['Sal']==6)].Age.mean())

print("Mean age of Military:",df[(df['Age']!=0) & (df['Sal']==7)].Age.mean())

print("Mean age of Royal:",df[(df['Age']!=0) & (df['Sal']==8)].Age.mean())
df.loc[Mr_rows,'Age']=df[(df['Age']!=0) & (df['Sal']==1)].Age.mean()

df.loc[Mrs_rows,'Age']=df[(df['Age']!=0) & (df['Sal']==2)].Age.mean()

df.loc[Master_rows,'Age']=df[(df['Age']!=0) & (df['Sal']==3)].Age.mean()

df.loc[Ms_rows,'Age']=df[(df['Age']!=0) & (df['Sal']==4)].Age.mean()

df.loc[Rev_rows,'Age']=df[(df['Age']!=0) & (df['Sal']==5)].Age.mean()

df.loc[Dr_rows,'Age']=df[(df['Age']!=0) & (df['Sal']==6)].Age.mean()

df.loc[Ms_rows,'Age']=df[(df['Age']!=0) & (df['Sal']==7)].Age.mean()

df.loc[Ms_rows,'Age']=df[(df['Age']!=0) & (df['Sal']==8)].Age.mean()
sns.barplot(df['Sal'],df['Survived'])
df.info()
df['Embarked'].unique()
df[(df['Embarked']!='S') & (df['Embarked']!='C') & (df['Embarked']!='Q')]
df.loc[61,'Embarked']='S'

df.loc[829,'Embarked']='S'
rows1=df[(df['Age']!=0) & (df['Sal']==1)].index

rows2=df[(df['Age']!=0) & (df['Sal']==2)].index

rows3=df[(df['Age']!=0) & (df['Sal']==3)].index

rows4=df[(df['Age']!=0) & (df['Sal']==4)].index

rows5=df[(df['Age']!=0) & (df['Sal']==5)].index

rows6=df[(df['Age']!=0) & (df['Sal']==6)].index

rows7=df[(df['Age']!=0) & (df['Sal']==7)].index

rows8=df[(df['Age']!=0) & (df['Sal']==8)].index



df.loc[:,'Mr']=0

df.loc[:,'Mrs']=0

df.loc[:,'Master']=0

df.loc[:,'Miss']=0

df.loc[:,'Rev']=0

df.loc[:,'Dr']=0

df.loc[:,'Military']=0

df.loc[:,'Royal']=0
df.loc[rows1,'Mr']=1

df.loc[rows2,'Mrs']=1

df.loc[rows3,'Master']=1

df.loc[rows4,'Miss']=1

df.loc[rows5,'Rev']=1

df.loc[rows6,'Dr']=1

df.loc[rows7,'Military']=1

df.loc[rows5,'Royal']=1
S_rows=df[df['Embarked']=='S'].index

Q_rows=df[df['Embarked']=='Q'].index

C_rows=df[df['Embarked']=='C'].index

df['S']=0

df['Q']=0

df['C']=0



df.loc[S_rows,'S']=1

df.loc[Q_rows,'Q']=1

df.loc[C_rows,'C']=1
Sibrows=df[(df['SibSp']!=0)].index

Parrows=df[(df['Parch']!=0)].index

df.loc[Sibrows,'SibSp']=1

df.loc[Parrows,'Parch']=1
df['Age'].hist()

plt.title('Train set:Age distribution of people')

plt.show()



test['Age'].hist()

plt.title('Test set:Age distribution of people')

plt.show()

df['Fare'].hist()

plt.title('Train set:Fare distribution')

plt.show()



test['Fare'].hist()

plt.title('Test set:Fare distribution')

plt.show()
df.drop(['PassengerId','Name','Ticket','Fare','Cabin','Sal','Embarked'],axis=1)
#features=['Mr','Mrs','Master','Miss','Rev','Dr','Military','Royal','Pclass','S','Q','C']

features=['Sex','Pclass','Parch','Mr','Mrs','Master','Miss','Rev','Dr','Military','Royal']

X=df.loc[:,features]

Y=df.loc[:,'Survived']
from sklearn.model_selection import train_test_split



X_test,X_train,Y_test,Y_train=train_test_split(X,Y, test_size=0.2, random_state=1)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



DT=DecisionTreeClassifier(criterion="entropy",random_state=0)

DT.fit(X_train,Y_train)

y_pred=DT.predict(X_test)

DT_score=accuracy_score(Y_test, y_pred)

print("DecisionTree Score:",DT_score)
DTimportance = DT.feature_importances_

# summarize feature importance

for i,v in enumerate(DTimportance):

    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

plt.bar([x for x in range(len(DTimportance))], DTimportance)

plt.grid()

plt.show()
from sklearn.ensemble import RandomForestClassifier



RF=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)

RF.fit(X_train,Y_train)

RF_score=accuracy_score(Y_test, RF.predict(X_test))



print("RandomForest Score:",RF_score)
RFimportance=RF.feature_importances_

# summarize feature importance

for i,v in enumerate(RFimportance):

    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

plt.bar([x for x in range(len(RFimportance))], RFimportance)

plt.grid()

plt.show()
from sklearn.ensemble import BaggingClassifier



BC=BaggingClassifier(random_state=0)

BC.fit(X_train,Y_train)

BC_score=accuracy_score(Y_test, BC.predict(X_test))



print(BC_score)
from sklearn.ensemble import GradientBoostingClassifier



GBC=GradientBoostingClassifier(random_state=0)

GBC.fit(X_train,Y_train)

GBC_score=accuracy_score(Y_test, GBC.predict(X_test))



print(GBC_score)
from xgboost import XGBRegressor



XGB= XGBRegressor()

XGB.fit(X_train,Y_train)

XGB_score=accuracy_score(Y_test, GBC.predict(X_test))



print(XGB_score)
XGBimportance = XGB.feature_importances_

# summarize feature importance

for i,v in enumerate(XGBimportance):

    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance

plt.bar([x for x in range(len(XGBimportance))], XGBimportance)

plt.grid()

plt.show()
for row in range(418):

    if ('Mr.' in test.loc[row,'Name'] or 'Don.' in test.loc[row,'Name'] or 'Jonkheer.' in test.loc[row,'Name']):

        test.loc[row,'Sal']=1;

    elif 'Mrs.' in test.loc[row,'Name']:

        test.loc[row,'Sal']=2;

    elif 'Master.' in test.loc[row,'Name']:

        test.loc[row,'Sal']=3;

    elif ('Miss.' in test.loc[row,'Name'] or 'Ms.' in test.loc[row,'Name'] or 'Mlle.' in test.loc[row,'Name'] or 'Mme.' in test.loc[row,'Name']):

        test.loc[row,'Sal']=4;

    elif 'Rev.' in test.loc[row,'Name']: 

        test.loc[row,'Sal']=5;

    elif 'Dr.' in test.loc[row,'Name']: 

        test.loc[row,'Sal']=6;

    elif ('Major.' in test.loc[row,'Name'] or 'Col.' in test.loc[row,'Name'] or 'Capt.' in test.loc[row,'Name']): 

        test.loc[row,'Sal']=7;

    else: 

        test.loc[row,'Sal']=8;

trows1=test[(test['Age']!=0) & (test['Sal']==1)].index

trows2=test[(test['Age']!=0) & (test['Sal']==2)].index

trows3=test[(test['Age']!=0) & (test['Sal']==3)].index

trows4=test[(test['Age']!=0) & (test['Sal']==4)].index

trows5=test[(test['Age']!=0) & (test['Sal']==5)].index

trows6=test[(test['Age']!=0) & (test['Sal']==6)].index

trows7=test[(test['Age']!=0) & (test['Sal']==7)].index

trows8=test[(test['Age']!=0) & (test['Sal']==8)].index



test.loc[:,'Mr']=0

test.loc[:,'Mrs']=0

test.loc[:,'Master']=0

test.loc[:,'Miss']=0

test.loc[:,'Rev']=0

test.loc[:,'Dr']=0

test.loc[:,'Military']=0

test.loc[:,'Royal']=0



test.loc[trows1,'Mr']=1

test.loc[trows2,'Mrs']=1

test.loc[trows3,'Master']=1

test.loc[trows4,'Miss']=1

test.loc[trows5,'Rev']=1

test.loc[trows6,'Dr']=1

test.loc[trows7,'Military']=1

test.loc[trows5,'Royal']=1
testSibrows=test[(test['SibSp']!=0)].index

testParrows=test[(test['Parch']!=0)].index

test.loc[testSibrows,'SibSp']=1

test.loc[testParrows,'Parch']=1





testS_rows=test[test['Embarked']=='S'].index

testQ_rows=test[test['Embarked']=='Q'].index

testC_rows=test[test['Embarked']=='C'].index

test['S']=0

test['Q']=0

test['C']=0

test.loc[testS_rows,'S']=1

test.loc[testQ_rows,'Q']=1

test.loc[testC_rows,'C']=1
test.head()
test1=test.loc[:,features]
DT_test=DT.predict(test1)

np.savetxt('DT.csv', DT_test, delimiter ='\n')
RF_test=RF.predict(test1)

np.savetxt('RF.csv', RF_test, delimiter ='\n')
BC_test=BC.predict(test1)

np.savetxt('BC.csv', BC_test, delimiter ='\n')
GBC_test=GBC.predict(test1)

np.savetxt('GBC.csv', GBC_test, delimiter ='\n')
XGB_test=XGB.predict(test1)

np.savetxt('XGB.csv', GBC_test, delimiter ='\n')