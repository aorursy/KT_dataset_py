# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/titanic/train.csv',index_col=False)
df.sample(5)
df.shape
df.info()
df.describe()


#filling the age column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
df['Age'].median()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
df['Age'].mean()
df['Age']=df['Age'].fillna(df['Age'].mean())
#change the type of age
df['Age']=df['Age'].astype('int32')
#dealing with the Embarked
df['Embarked'].value_counts()

df['Embarked']=df['Embarked'].fillna('S')
df[df['Cabin'].isna()]['Pclass'].value_counts()

df.drop(columns=['Cabin'],inplace=True)
df.columns
df.isnull().sum()
df.drop(columns=['PassengerId'],inplace=True)
import seaborn as sns
sns.countplot(df['Survived'],hue=df['Pclass'])
sns.countplot(df['Survived'],hue=df['Sex'])
sns.distplot(df[df['Survived']==1]['Age'],bins=10)#blue              #updation 1
sns.distplot(df[df['Survived']==0]['Age'],bins=10) #orange

#histogram is like barchart for numerical data
#probabilty density function
#conclusion: there is a dependency between Age and Survived
sns.distplot(df[df['Survived']==1]['Fare'])
sns.distplot(df[df['Survived']==0]['Fare'])
#conclusion: there is a dependency between Fare and Survived
sns.countplot(df['Survived'],hue=df['Embarked'])
sns.catplot(x="SibSp", col = 'Survived', data=df, kind = 'count')
sns.catplot(x="Parch", col = 'Survived', data=df, kind = 'count')
df['family']=df['SibSp']+df['Parch']+1
df.sample(5)
df['family_size']=0
def family_size(no):
    if no==1:
        return "alone"
    elif no>1 and no<=4:
        return "small"
    else:
        return "large"
#very important
df['family_size']=df['family'].apply(family_size)
df.sample(5)
#percentage
df.groupby('family_size').mean()['Survived']
df.reset_index(drop=True, inplace=True)
df.sample(2)
df.drop(columns=['Name','Ticket'],inplace=True,axis=1)
df.sample(2)
from sklearn.preprocessing import LabelEncoder
LR=LabelEncoder()
df['Sex']=LR.fit_transform(df['Sex'])
df.sample(2)
#df['Embarked']=LR.fit_transform(df['Embarked'])
df=pd.get_dummies(df,columns=['Embarked','family_size'],drop_first=True)

df.sample(5)
df.Fare=df.Fare.astype('int64')
df.sample(5)
#Extract the input features
X=df.iloc[:,1:].values
#target
y=df.iloc[:,0].values

X.shape
y.shape
#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
X_train.shape
X_test.shape
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf.fit(X_train,y_train)
pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred))
from sklearn.metrics import f1_score
print(f1_score(y_test,pred))
