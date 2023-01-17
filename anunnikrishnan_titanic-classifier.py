# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

from sklearn.svm import SVC

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
df  = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()
#### find the missing values

df.isnull().sum()
df.shape
df  = df.drop('Cabin',axis = 1)
df.columns
df.isnull().sum()
df = df.dropna()
df.shape
df.isnull().sum()
df.head()
df.drop(['Name','Ticket'] , axis =1, inplace  = True)
df.dtypes
from sklearn import preprocessing



le_gender = preprocessing.LabelEncoder()

df['Sex'] = le_gender.fit_transform(df['Sex'])

le_embarked = preprocessing.LabelEncoder()

df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

df.isnull().sum()
df.corr()

X = df.drop(['Survived','PassengerId'],axis = 1 )

y = df['Survived']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



#from sklearn.neighbors import KNeighborsClassifier



#model = KNeighborsClassifier(n_neighbors = 5)

#model.fit(X_train, y_train)



model = SVC()

model.fit(X_train, y_train)
pred = model.predict(X_test)
pred
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
confusion_matrix(y_test,pred)
f1_score(y_test,pred)
accuracy_score(y_test,pred)
test_data  = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.columns
test_data.head()

pass_id = test_data['PassengerId']

print(type(pass_id))

test_data = test_data.drop(['PassengerId','Ticket'],axis =1)
### transform

test_data['Sex'] = le_gender.transform(test_data['Sex'])

test_data['Embarked'] = le_embarked.transform(test_data['Embarked'])
test_data.drop(['Cabin','Name'],axis =1,inplace = True)
test_data.isnull().sum()
mean_age =  test_data['Age'].mean()

print(mean_age)



test_data['Age'] = test_data['Age'].fillna(mean_age)
mean_fare =  test_data['Fare'].mean()

print(mean_fare)



test_data['Fare'] = test_data['Fare'].fillna(mean_fare)
test_data.isnull().sum()
submission_df =  pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
X_train.columns
test_data

pred =  model.predict(test_data)
pred
len(pass_id)

print(type(pass_id))
len(pred)

print(type(pred))
pred = pd.DataFrame(pred)

pred.columns = ['Survived']

pred
my_submission_df =  pd.concat([pass_id,pred],axis =1)
my_submission_df
my_submission_df.to_csv("submission.csv")

my_submission_df.columns
# this is a sample provided by kaggle present in submission.csv

submission_df



#### 1. remove pass id and pass to model , 2.  output concat with test data and then keep only needed columns