# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks



%matplotlib inline

sns.set_style('whitegrid')

init_notebook_mode(connected=True)

cufflinks.go_offline()
titanic = pd.read_csv('../input/titanic/train.csv')
titanic.head()
titanic.info()
plt.figure(figsize=(12,6))

sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
plt.figure(figsize=(12,6))

sns.countplot(titanic['Survived'])
plt.figure(figsize=(12,6))

sns.countplot(titanic['Survived'],hue=titanic['Sex'])
titanic['Pclass'].value_counts()
plt.figure(figsize=(12,6))

sns.countplot(titanic['Pclass'],hue=titanic['Sex'])
fg=sns.FacetGrid(data=titanic,col='Sex',height=6)

fg.map(sns.countplot,'Survived',hue=titanic['Pclass'],palette='rainbow').add_legend()
titanic['Age'].iplot(kind='hist',bins=30)
corr = titanic.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,6))

sns.heatmap(corr,mask=mask,cmap='coolwarm',annot=True)
sns.boxplot(x='Pclass', y='Age', data=titanic)
def inputAge(data):

    age=data[0]

    _class= data[1]

    if pd.isnull(age):

        if _class==1:

            return 37

        elif _class==2:

            return 29

        else:

            return 25

    else:

        return age
titanic['Age'] = titanic[['Age','Pclass']].apply(inputAge,axis=1)
plt.figure(figsize=(12,6))

sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
titanic.drop('Cabin',axis=1,inplace=True)
titanic.dropna(inplace=True)
plt.figure(figsize=(12,6))

sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sex=pd.get_dummies(titanic['Sex'],drop_first=True)

embark=pd.get_dummies(titanic['Embarked'],drop_first=True)

titanic.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([titanic,sex,embark],axis=1)
train.head()
from sklearn.linear_model import LogisticRegression
X_train= train.drop('Survived',axis=1)

y_train = train['Survived']



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
test= pd.read_csv('../input/titanic/test.csv')
test['Age'] = test[['Age','Pclass']].apply(inputAge,axis=1)
test.drop('Cabin',axis=1,inplace=True)
import plotly.express as px

fig = px.box(test,x='Pclass', y="Fare")

fig.show()
boolean = pd.isnull(test['Fare'])

test[boolean]
test.loc[152,'Fare'] = 7.89
sex=pd.get_dummies(test['Sex'],drop_first=True)

embark=pd.get_dummies(test['Embarked'],drop_first=True)

test.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test = pd.concat([test,sex,embark],axis=1)
predict = logmodel.predict(test)
df= pd.read_csv('../input/titanic/gender_submission.csv')
df['Survived'] = predict
df.to_csv('submission.csv', index=False)