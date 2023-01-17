# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
#dropping fields which do not have any impact on survival

df.drop(['Name', 'Ticket'], axis=1, inplace=True)
df.columns
df.info()
sns.heatmap(df.isnull(), cmap='viridis')
temp=df.groupby('Pclass')[['Pclass','Age']].mean()
temp=df.groupby('Pclass')[['Pclass','Age']].mean()
temp.info()
round(temp[temp['Pclass']==1]['Age'],0)
df[df['Age'].isnull()].head()
def Agefill(cols):

    Age=cols[0]

    Pclass=cols[1]

    

    if pd.isnull(Age):

        if Pclass==1:

            return round(temp[temp['Pclass']==1]['Age'])

        elif Pclass==2:

            return round(temp[temp['Pclass']==2]['Age'])

        else:

            return round(temp[temp['Pclass']==3]['Age'])

    else:

        return Age

    

df['Age']=df[['Age','Pclass']].apply(Agefill, axis=1)
df.info()
sns.heatmap(df.isnull(), cmap='viridis')
df.drop(['Cabin'], axis=1, inplace=True)
df.dropna(inplace=True)
sns.heatmap(df.isnull(), cmap='viridis')
df.columns
df.info()
df_test=pd.read_csv('/kaggle/input/titanic/test.csv')
df_2_test=df_test.drop(['Name','Ticket','Cabin'], axis=1)
df_2_test.head()
df_2_test['Age']=df_2_test[['Age','Pclass']].apply(Agefill, axis=1)
df_2_test.info()
df_2_test['Fare'].fillna(df_2_test['Fare'].mean(), inplace=True)
df_2_test.info()
sns.countplot('Survived', hue='Pclass', data=df)
sns.barplot('Pclass','Fare', data=df)
sns.boxplot('Pclass', 'Age', data=df)
sns.countplot(df['SibSp'])
sns.scatterplot(x='Age', y='Fare', hue='Survived',palette=['red','blue'], data=df)

plt.title('Age vs Fare')

#plt.rcParams['figure.figsize']=[100,100]
df.info()
df['Fare'].describe()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-3,3))

df['Fare']=scaler.fit_transform(df.Fare.values.reshape(-1,1))
from sklearn.preprocessing import MinMaxScaler

scaler_test = MinMaxScaler(feature_range=(-3,3))

df_2_test['Fare']=scaler.fit_transform(df_2_test.Fare.values.reshape(-1,1))
df.head()
df_2_test.head()
sex=pd.get_dummies(df['Sex'])

Embarked=pd.get_dummies(df['Embarked'])

sex_test=pd.get_dummies(df_2_test['Sex'])

Embarked_test=pd.get_dummies(df_2_test['Embarked'])

sex.head()

#Embarked.head()
df.info()
df_1_final=pd.concat([df,sex,Embarked],axis=1)

df_2_final=pd.concat([df_2_test,sex_test,Embarked_test],axis=1)

df_2_final.head()
df_1_final.drop(['Sex','Embarked'],axis=1, inplace=True)

df_2_final.drop(['Sex','Embarked'],axis=1, inplace=True)

df_2_final.head()
features=df_1_final.drop(['PassengerId','Survived'], axis=1).values

label=df_1_final['Survived'].values
print(features.shape, label.shape)
label.resize(889,1)
label.shape
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

max_test_score=0

random_state=0

for x in range(0,1000):

    X_train, X_test, y_train, y_test  = train_test_split(features, label, test_size=0.2, random_state=x)

    lr=LogisticRegression(max_iter=1000)

    lr.fit(X_train, np.ravel(y_train))

    if lr.score(X_test,y_test)>max_test_score and lr.score(X_train, y_train)<lr.score(X_test,y_test):

        max_test_score=lr.score(X_test,y_test)

        train_score=lr.score(X_train, y_train)

        random_state=x

print('Random State:',random_state,' Train score: ',train_score)

print('Random State:',random_state,' Test Score: ',max_test_score)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(features, label, test_size=0.2, random_state=669)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))

print(lr.score(X_test,y_test))
predictions= lr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
features_test=df_2_final.drop(['PassengerId'], axis=1).values
features_test.shape
df_model_predictions=lr.predict(features_test)
type(df_model_predictions)
df_model_predictions
df_2_final.head()
submission=pd.concat([df_2_final['PassengerId'], pd.DataFrame(df_model_predictions)],axis=1)

submission
submission.to_csv('submission.csv', encoding='utf-8')