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
train_set=pd.read_csv('/kaggle/input/titanic/train.csv')
test_set=pd.read_csv('/kaggle/input/titanic/test.csv')
train_set.info()
train_set.describe()
train_set.head()
tot_data_missing= train_set.isnull().sum()
print(tot_data_missing)
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline
women=train_set[train_set['Sex']=='female']
men= train_set[train_set['Sex']=='male']
plt.figure(1)
plt.title('Women')
plt.hist(women[women['Survived']==1].Age,label="Survived", bins=18, color="silver")
plt.hist(women[women['Survived']==0].Age,label="Not Survived", bins=40, color="orange")
plt.xlabel("Age")
plt.legend()
plt.figure(2)
plt.title('Men')
plt.hist(men[men['Survived']==1].Age,label="Survived", bins=18, color="silver")
plt.hist(men[men['Survived']==0].Age,label="Not Survived", bins=40, color="orange")
plt.xlabel("Age")
plt.legend()
FacetGrid = sns.FacetGrid(train_set, row='Embarked', size=4.0, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_set)
corr=train_set.corr()

colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,cmap=colormap,xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
dataset=[train_set,test_set]
for data in dataset:
    data['relation']=data['SibSp']+data['Parch']
    data.loc[data['relation'] > 0, 'not_alone'] = 0
    data.loc[data['relation'] == 0, 'not_alone'] = 1
    data['not_alone'] = data['not_alone'].astype(int)
train_set['not_alone'].value_counts()
train_set=train_set.drop(['PassengerId','Cabin'],axis=1)
train_set.columns.values
dataset=[train_set,test_set]
for data in dataset:
    mean=train_set["Age"].mean()
    std=train_set["Age"].std()
    is_null=data["Age"].isnull().sum()
    random_age=np.random.randint(mean - std, mean + std, size = is_null)
    age=data["Age"].copy()
    age[np.isnan(age)]=random_age
    data["Age"]=age
    data["Age"]=train_set["Age"].astype(int)    
train_set["Age"].isnull().sum()
train_set["Embarked"].describe()
common_value = 'S'
data = [train_set, test_set]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
    
train_set.info()
dataset = [train_set, test_set]

for data in dataset:
    data['Fare'] = data['Fare'].fillna(0)
    data['Fare'] = data['Fare'].astype(int)
train_set=train_set.drop(['Name','Ticket'],axis=1)
genders={"male": 0,"female":1}
ports = {"S": 0, "C": 1, "Q": 2}
dataset=[train_set,test_set]
for data in dataset:
    data['Sex'] = data['Sex'].map(genders)
    data['Embarked'] = data['Embarked'].map(ports)
    
train_set['Embarked'].head()
corr=train_set.corr()

colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,cmap=colormap,xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
test_set=test_set.drop(['Name'],axis=1)
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
X_train = train_set.drop(["Survived"], axis=1)
Y_train = train_set["Survived"]
X_test  = test_set.drop(["PassengerId","Cabin","Ticket"], axis=1).copy()
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, Y_train)


Y_pred = lr.predict(X_test)

acc_log = round(lr.score(X_train, Y_train) * 100, 2)

print(acc_log)

my_submission = pd.DataFrame({'PassengerId': list(test_set['PassengerId']), 'Survived': Y_pred})
my_submission.to_csv('submission.csv', index=False)