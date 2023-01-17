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
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
# Collect the data into pandas dataframe 

df = pd.read_csv('../input/titanic/train.csv')

df.head()
# Check the info of Dataset

df.info()
# Check How many Data is missing.

df.isnull()
#Let's Visualise which values are missing the most:)

sns.heatmap(df.isnull(),yticklabels=False ,cbar=False ,cmap='viridis')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=df)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=df)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')
sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=40)
sns.countplot(x='SibSp',data=df)
df['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')
def impute_age(cols):

    Age= cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        if Pclass==2:

            return 29

        else:

            return 24

    return Age
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(df.isnull(),yticklabels=False ,cbar=False ,cmap='viridis')
df.drop('Cabin',axis=1,inplace=True)
df.head()
pd.get_dummies(df['Embarked'],drop_first=True).head()
sex = pd.get_dummies(df['Sex'],drop_first=True)

embarked = pd.get_dummies(df['Embarked'],drop_first=True)

#Delete Features like name,ticket etc which are of no use.

df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
df.head()
df = pd.concat([df,sex,embarked],axis=1)
df.head()
df.drop('Survived',axis=1).head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(df.drop('Survived',axis=1),df['Survived'],test_size =0.3, random_state = 102)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train,y_train)
predictions = lr.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = confusion_matrix(y_test,predictions)
accuracy
accuracy = accuracy_score(y_test,predictions)

accuracy
predictions