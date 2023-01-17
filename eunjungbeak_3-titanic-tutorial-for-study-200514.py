import pandas as pd
import numpy as np

import os
for dirname,_, filenames in os.walk('/kaggle/input'):
    for filename in filenames : 
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
gender_submission.head()
train.shape, test.shape, gender_submission.shape
train.describe()
train.isnull().sum() / 891
test.isnull().sum()
train['Age'].mean()
# 그냥 train['Age'].fillna(train['Age'].mean())만 하면 실행만 하는 꼴임, 원래의 trainage에걸 덮어씌워줘야한다!

train['Age'] = train['Age'].fillna(train['Age'].mean())
train.isnull().sum()
train['Age'].mean()
test['Age'] = test['Age'].fillna(test['Age'].mean())
test.isnull().sum()

# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
train['Fare'].isnull().sum()
test['Fare'].isnull().sum()
test['Fare'].mean()
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test['Fare'].isnull().sum()
train['Sex'].describe()

# 여기서 unique는 값의 개수이고 이것만 따로 확인하는게 unique()
train['Sex'].unique()
train['Sex'] = train['Sex'].replace("male",0)
train['Sex'] = train['Sex'].replace("female",1)
train['Sex']
train['Sex'] = train['Sex'].replace("male",0)
train['Sex'] = train['Sex'].replace("female",1)
train['Sex']
test['Sex'] = test['Sex'].replace("male",0)
test['Sex'] = test['Sex'].replace("female",1)
test['Sex'].unique()
train['Embarked'].unique()
train['Embarked'] = train['Embarked'].replace("S",0)
train['Embarked'] = train['Embarked'].replace("C",1)
train['Embarked'] = train['Embarked'].replace("Q",2)
train['Embarked'].unique()
test['Embarked'] = test['Embarked'].replace("S",0)
test['Embarked'] = test['Embarked'].replace("C",1)
test['Embarked'] = test['Embarked'].replace("Q",2)
test['Embarked'].unique()
train = train.drop(['Name', 'Cabin', 'Ticket'],1)
train
test = test.drop(['Name', 'Cabin', 'Ticket'],1)
test
# 직접 입력해 보세요!
y = train['Survived']
y
X = train.drop(['Survived'],1)
X
X.shape, y.shape
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 30, test_size = 0.3)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
(pred_train == y_train).mean()
pred_train = model.predict(X_valid)
(pred_train == y_valid).mean()
# 직접 입력해 보세요!
# 직접 입력해 보세요!
pred_test = model.predict(test)
pred_test
gender_submission['Survived']=pred_test
gender_submission.to_csv("submission_final.csv", index=False)
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
# 직접 입력해 보세요!
