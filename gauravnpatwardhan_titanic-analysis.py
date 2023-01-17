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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv("../input/titanic/train.csv") 

test = pd.read_csv("../input/titanic/test.csv")
train.head()
test.head()
print(train.shape,test.shape)
train.isnull().sum()
test.isnull().sum()
sns.countplot("Sex",data=train,order=train['Sex'].value_counts().index)

plt.title("Count of Gender")

plt.show()
f,ax=plt.subplots(1,1, figsize=(25,10))

sns.countplot(train['Age'],ax=ax,color='green')

ax.tick_params('x',rotation=70)

ax.set_title('Y')

plt.show()
fig=sns.heatmap(train[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']].corr(),

                annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':15})

fig=plt.gcf()

fig.set_size_inches(18,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,5))

train['Survived'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Percentage of Survived and Death  guests')

ax[0].set_ylabel('Count')

sns.countplot('Survived',data=train,ax=ax[1],order=train['Survived'].value_counts().index)

ax[1].set_title('Count of Survived and Death guests')

plt.show()
sns.countplot(train['Pclass'],hue=train['Survived'])
sns.countplot(train['Sex'],hue=train['Survived'])
sns.countplot(train['Fare'],hue=train['Survived'])
sns.countplot(train['Embarked'],hue=train['Survived'])
gender=pd.read_csv("../input/titanic/gender_submission.csv")
gender.head()
gender['Survived'].hist()