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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split , cross_validate

from sklearn.metrics import accuracy_score , precision_score

from sklearn.preprocessing import StandardScaler , OneHotEncoder



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
train = '../input/titanic/train.csv'

train = pd.read_csv(train)



test = '../input/titanic/test.csv'

test = pd.read_csv(test)
train.head()
test.head()
train[train.Survived<1].Sex.value_counts()        #that could not survived
train[train.Survived>0].Sex.value_counts()         #the one survived
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))



women = train[train['Sex']=='female']



men = train[train['Sex']=='male']



ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
sns.barplot(x='Pclass',y='Survived',data=train)

plt.show()
train['Pclass'].isnull().sum()
train['Sex'].isnull().sum()
train['Age'].isnull().sum()
train["Age"].describe()
train['Age'].fillna(value=29,inplace = True)
train.head(10)  
features = ['Pclass']





ss = StandardScaler()



train_ss = pd.DataFrame(data = train)

train_ss[features] = ss.fit_transform(train_ss[features])
train_ss.head(11)
train_ss.replace(to_replace ="male",value =1,inplace = True) 

train_ss.replace(to_replace ="female",value =0,inplace = True)
train_ss
test.head()
test['Age'].isnull().sum()
test["Age"].describe()
test['Age'].fillna(value=30,inplace = True)
features = ['Pclass']





ss = StandardScaler()



test_ss = pd.DataFrame(data = test)

test_ss[features] = ss.fit_transform(test_ss[features])
test_ss.replace(to_replace ="male",value =1,inplace = True) 

test_ss.replace(to_replace ="female",value =0,inplace = True)
y_train = train_ss[['Survived']]



print(y_train)
ft = ['Pclass','Sex','Age','SibSp','Parch']



x =(train_ss[ft])

x_test =(test_ss[ft])


decision_tree = DecisionTreeClassifier()

decision_tree.fit(x, y_train)

pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x, y_train) * 100, 2)

acc_decision_tree
output = pd.DataFrame({'PassengerId': test_ss.PassengerId, 'Survived': pred})

output.to_csv('my_new_submission.csv', index=False)

print("Your submission was successfully saved!")
print(output)