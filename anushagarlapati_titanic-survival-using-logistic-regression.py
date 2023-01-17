# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filename='/kaggle/input/titanic-machine-learning-from-disaster/train.csv'
df=pd.read_csv(filename,encoding='ISO-8859-1')
df.head() 
df.isnull()
sns.heatmap(df.isnull(),yticklabels=False)
df.isnull().sum()
sns.countplot("Survived",data=df)
#style=white, dark, whitegrid, darkgrid, ticks
sns.set_style('whitegrid')
sns.countplot("Survived",data=df)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df,palette='rainbow')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')
sns.distplot(df['Age'].dropna(),kde=True,color='green',bins=30)
df['Age'].hist(bins=40,color='darkred',alpha=0.5)
sns.countplot('SibSp',data=df)
df['Fare'].hist(color='green',bins=40,figsize=(8,4))
df['Parch'].hist(color='red',bins=40,figsize=(8,4))
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
#We'll use these average age values to impute based on Pclass for Age.
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(df.isnull(),yticklabels='False')
df.isnull().sum()
df.drop('Cabin',axis=1,inplace=True)
df.head()
df.dropna(inplace=True)
df.head()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False)
df.info()
pd.get_dummies(df['Embarked'],drop_first=True).head()
sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df.head()
train = pd.concat([df,sex,embark],axis=1)
train.head()
X=train.drop('Survived',axis=1)
X.head()
y=train['Survived']
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,  random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
predictions
from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y_test,predictions)
accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)
accuracy
predictions
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
