

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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print('Train data columns:\n',train_data.columns,'\n')

print('Test dataset columns:\n',test_data.columns)
# shape of our train data(no of rows,no of columns)

print('Shape of our train dataset:\n',train_data.shape,'\n')

print('Shape of our test datset:\n',test_data.shape)
# descriptive anlalysis of numerical variables

print('Train data numerical feature analysis:')

train_data.describe()
print('Test data numerical feature analysis:')

test_data.describe()
# check data for null

print('Null values in Train dataset:\n',train_data.isnull().sum(),'\n')

print('Null values in Test dataset:\n',test_data.isnull().sum())
# unique values in each feature of dataset

print('Unique values in each feature of Train dataset:\n',train_data.nunique(),'\n')

print('Unique values in each feature of Test dataset:\n',test_data.nunique(),'\n')
import matplotlib.pyplot as plt

train_data['Sex'].value_counts().plot(kind='barh')

plt.show()
train_data['Survived'].value_counts().plot(kind='barh')

plt.show()
train_data.Age.plot(kind='hist')

plt.show()
import seaborn as sns

sns.countplot('Pclass',data=train_data)

plt.show()
sns.distplot(train_data['Fare'], kde=True)

plt.show()
sns.catplot(x ="Sex", hue ="Survived",kind ="count", data = train_data) 

plt.show()
sns.catplot(x ="Embarked", hue ="Survived",kind ="count", data = train_data) 

plt.show()
# grouping pclass and survived features of the dataset and plotting them using seaborn heatmap which gives us the survival rate based on passenger class

group = train_data.groupby(['Pclass', 'Survived']) 

ps = group.size().unstack()

sns.heatmap(ps, annot = True,cmap='Wistia_r')

plt.show()
train_data['Cabin'].fillna('0',inplace=True)



age=train_data['Age'].mean()

print(age)

train_data['Age'].fillna('29',inplace=True)



# emb=train_data['Embarked'].mode()

emb='S'

train_data['Embarked'].fillna(emb,inplace=True)
train_data.isnull().sum()
age_test=test_data['Age'].mean()

print('Avg age :',age)

test_data['Age'].fillna('29',inplace=True) #filling Nan values with Avg age value



test_data['Cabin'].fillna('0',inplace=True) 



fare=test_data['Fare'].mean()

test_data['Fare'].fillna(fare,inplace=True)

print('-'*20)



test_data.isna().sum()
# librabries needed for our model building and predicting

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
# since we need numerical data to process

# Converting 'object'-dtype to 'int' dtype
train_data['Age']=pd.get_dummies(train_data['Age'])

train_data['Sex']=pd.get_dummies(train_data['Sex'])

train_data['Embarked']=pd.get_dummies(train_data['Embarked'])
test_data['Age']=pd.get_dummies(test_data['Age'])

test_data['Sex']=pd.get_dummies(test_data['Sex'])

test_data['Embarked']=pd.get_dummies(test_data['Embarked'])
train_data.head()
test_data.head()
features = ["Pclass", "Sex","Age", "SibSp", "Parch","Fare","Embarked"] #features we are considering for model building and predicting



y = train_data["Survived"] #target variable to predict



X =train_data[features] #training the model

To_test=test_data[features] #data that needed to test the model

print(X.isna().sum().sum())

print(y.isna().sum().sum())

print(To_test.isna().sum())
# Using Decision Tree Classifier to predict



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X,y)



Y_pred_DT = decision_tree.predict(To_test)



acc_decision_tree = round(decision_tree.score(X, y) * 100, 2)

print('Accuracy score-training data:',acc_decision_tree)
RF = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

RF.fit(X, y)

predictions = RF.predict(To_test)



acc_RF = round(RF.score(X, y) * 100, 2)

print('Accuracy score-training data:',acc_RF)
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X, y)

Y_pred = knn.predict(To_test)

acc_knn = round(knn.score(X, y) * 100, 2)

print('Accuracy score-training data:',acc_knn)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
