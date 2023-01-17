# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading the training data

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train.head()
#Loading the testing data

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test.head()
# What type of values are stored in the columns?

df_train.info()
#Looking at the columns

df_train.columns
# Let's look at some statistical information about our dataframe.

df_train.describe()
print(df_train.shape,df_test.shape)
df_train[['Pclass','Survived']].groupby('Pclass').mean().sort_values(by = 'Survived',ascending = False)
df_train[['Sex','Survived']].groupby('Sex').mean().sort_values(by = 'Survived',ascending = False)
df_train[['SibSp','Survived']].groupby('SibSp').mean().sort_values(by = 'Survived',ascending = False)
df_train[['Parch','Survived']].groupby('Parch').mean().sort_values(by = 'Survived',ascending = False)
#Sex Analysis using bar chart and Pie chart

df_train['Sex'].value_counts().plot.bar()
f,ax=plt.subplots(1,2,figsize=(12,7))

colors = ['#ff9999','#66b3ff']

df_train['Sex'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],colors=colors,shadow=True)

ax[0].set_title('Sex')

ax[0].set_ylabel('')

sns.countplot('Sex',data=df_train,ax=ax[1])

ax[1].set_title('Sex')

plt.show()
#Embarked Analysis Using Bar and Pie Chart

f,ax=plt.subplots(1,2,figsize=(12,7))

colors = ['#ff9999','#66b3ff','#66ff7a']

df_train['Embarked'].value_counts().plot.pie(explode=[0,0.2,0.2],autopct='%1.1f%%',ax=ax[0],colors=colors,shadow=True)

ax[0].set_title('Embarked')

ax[0].set_ylabel('')

sns.countplot('Embarked',data=df_train,ax=ax[1])

ax[1].set_title('Embarked')

plt.show()
#Pclass (Ticket class) Analysis using bar chart and Pie chart

df_train['Pclass'].value_counts().plot.bar()
f,ax=plt.subplots(1,2,figsize=(12,7))

colors = ['#ff9999','#66b3ff','#66ff7a']

df_train['Pclass'].value_counts().plot.pie(explode=[0,0.2,0.2],autopct='%1.1f%%',ax=ax[0],colors=colors,shadow=True)

ax[0].set_title('Pclass')

ax[0].set_ylabel('')

sns.countplot('Pclass',data=df_train,ax=ax[1])

ax[1].set_title('Pclass')

plt.show()
plt.subplots(figsize = (12,7))

sns.barplot(x = 'Pclass',y = 'Survived',data = df_train)

plt.title('Passeneger Class Distribution - Survived vs Non Survived',fontsize = 10)

plt.xlabel('Passenger Class')

labels = ['1st','2nd','3rd']

val = [0,1,2]

plt.xticks(val, labels)





#print percentages of 1st vs. 2nd and 3rd class

print('Percentage of First class passeneger who survived:',df_train['Survived'][df_train['Pclass']==1].value_counts(normalize = True)[1]*100)

print('Percentage of Second class passeneger who survived:',df_train['Survived'][df_train['Pclass']==2].value_counts(normalize = True)[1]*100)

print('Percentage of Third class passeneger who survived:',df_train['Survived'][df_train['Pclass']==3].value_counts(normalize = True)[1]*100)
#create a subplot

f,ax=plt.subplots(1,2,figsize=(12,7))



# create bar plot using groupby

df_train[['Sex','Survived']].groupby('Sex').mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')



# create count plot

sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
# create subplot plot

f,ax=plt.subplots(1,2,figsize=(12,7))



df_train['Pclass'].value_counts().plot.bar(color = ['r','g','b'],ax = ax[0])

ax[0].set_title('Number Of Passengers By Pclass')

ax[0].set_ylabel('Count')



# create count plot

sns.countplot('Pclass',hue = 'Survived',data = df_train)

plt.title('Pclass:Survived vs Dead')

plt.show()
# create subplot plot



f,ax=plt.subplots(1,2,figsize=(18,8))



#Creating violin plot

sns.violinplot(x = 'Pclass',y = 'Age',hue = 'Survived',data = df_train,ax = ax[0])

plt.title('Passeneger Class & Age vs Survived')



sns.violinplot(x = 'Sex',y = 'Age',hue = 'Survived',data = df_train,ax = ax[1])

plt.title('Sex & Age vs Survived')

plt.show()
plt.figure(figsize = (12,7))

sns.boxplot(x = 'Pclass',y = 'Age',data = df_train)

plt.show()
plt.figure(figsize = (12,7))

sns.scatterplot(x = 'Age',y = 'Fare',data = df_train,hue = 'Survived')

plt.show()
plt.figure(figsize = (12,7))

sns.heatmap(df_train.corr(),cmap = 'ocean',annot = True)

plt.show()
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())
df_train = df_train.drop('Cabin',axis = 1)

df_test = df_test.drop('Cabin',axis = 1)
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train['Sex'] = df_train['Sex'].map({'female':0,'male':1})



df_test['Sex'] = df_test['Sex'].map({'female':0,'male':1})
# df_train = df_train['Embarked'].dropna()

# df_test = df_train['Embarked'].dropna()





df_train['Embarked'] = df_train['Embarked'].map({'S':1,'Q':2,'C':3})

df_test['Embarked'] = df_test['Embarked'].map({'S':1,'Q':2,'C':3})
df_train['Family']=df_train['SibSp']+df_train['Parch']+1

df_test['family']=df_test['SibSp']+df_train['Parch']+1
df_train.isna().sum()
df_train['Embarked'].isna().sum()
df_train['Embarked'].mode()
df_train['Embarked'] = df_train['Embarked'].fillna(1.0)
df_train = df_train.drop(['Name','Ticket','SibSp','Parch'],axis = 1)

df_test = df_test.drop(['Name','Ticket','SibSp','Parch'],axis = 1)

df_test.head()

df_train.head()
#df_test.head()
df_test['Fare'] = df_test['Fare'].replace(np.nan,df_test['Fare'].mean())
y = df_train['Survived']

X = df_train.drop('Survived',axis = 1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

pred = logmodel.fit(X,y)
from sklearn.metrics import accuracy_score
predictions = pred.predict(df_test)
print(predictions)
df_test['Survived'] = predictions
data = df_test[['PassengerId','Survived']]
print(data)
data.to_csv('submission.csv',index = False)