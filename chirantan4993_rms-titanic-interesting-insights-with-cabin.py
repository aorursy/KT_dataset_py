import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
titanic=pd.read_csv('../input/titanic/train.csv')

titanic.head()
titanic.dtypes
cat_feat=['Name','Sex','Ticket','Cabin','Embarked']
titanic.drop(cat_feat, axis=1,inplace=True)

titanic.head()
titanic.describe()
titanic.groupby('Survived').mean()
titanic.groupby(titanic['Age'].isnull()).mean() #it will return True or False - if the rows have mising values or not

for i in ['Age', 'Fare']:

    died = list(titanic[titanic['Survived'] == 0][i].dropna())

    survived = list(titanic[titanic['Survived'] == 1][i].dropna())

    xmin = min(min(died), min(survived))

    xmax = max(max(died), max(survived))

    width = (xmax - xmin) / 40

    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))

    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))

    plt.legend(['Did not survive', 'Survived'])

    plt.title('Overlaid histogram for {}'.format(i))

    plt.show()

    
for i, col in enumerate(['Pclass','SibSp','Parch']): #enumerate returns the index of the list and actual items in the list 

    plt.figure(i) 

    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2) #you can also explore other kinds of plot in seaborn documentation
titanic['family_count']=titanic['SibSp']+titanic['Parch']

sns.catplot(x='family_count',y='Survived', data=titanic, kind='point', aspect=2)
titanic.drop('PassengerId', axis=1,inplace=True)

titanic.head()
titanic.groupby(titanic['Age'].isnull()).mean() #it will return True or False - if the rows have mising values or not

titanic['Age'].fillna(titanic['Age'].mean(),inplace=True)

titanic.isna().sum()
titanic.head(10)
titanic.drop(['SibSp','Parch'], axis=1,inplace=True)

titanic.head(5)
titanic=pd.read_csv('../input/titanic/train.csv')

titanic.head()
# Drop all continuous features

cont_feat = ['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Fare']

titanic.drop(cont_feat, axis=1, inplace=True)

titanic.head()
titanic.info()
titanic.groupby(titanic['Cabin'].isnull()).mean()
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1) #this is like if-else condition 

# using np.where - if it is null show '0' else '1'

titanic.head(10)
for i, col in enumerate(['Cabin_ind', 'Sex', 'Embarked']):

    plt.figure(i)

    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2)
titanic.pivot_table('Survived',index='Sex',columns='Embarked',aggfunc='count')
titanic.pivot_table('Survived',index='Cabin_ind',columns='Embarked',aggfunc='count')
import numpy as np

import pandas as pd



titanic = pd.read_csv('../input/titanic/train.csv')

titanic.drop(['Name', 'Ticket'], axis=1, inplace=True)

titanic.head()
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)

titanic.head()
gender_num = {'male':0 , 'female':1 }

titanic['Sex']=titanic['Sex'].map(gender_num)

titanic.head(5)
titanic.drop(['Cabin', 'Embarked'], axis=1, inplace=True)

titanic.head(5)
from sklearn.model_selection import train_test_split
titanic_1=pd.read_csv('../input/titanic/train.csv')

titanic_1.head()
features=titanic_1.drop('Survived',axis=1)

labels=titanic_1['Survived']
X_train,X_test,y_train,y_test=train_test_split(features,labels,test_size=0.4,random_state=42)

X_test,X_val,y_train,y_val=train_test_split(X_test,y_test,test_size=0.5,random_state=42)

print(len(labels),len(y_train),len(y_val),len(y_test))