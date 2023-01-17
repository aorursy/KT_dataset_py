import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# kaggle dataset 확인
train = pd.read_csv("/kaggle/input/titanic/train.csv",)

print(train.shape)

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

print(test.shape)

test.head()
train.describe()

# 분석의 목적은 Survived를 예측하는 것이며, 여기서는 살았냐 죽었냐에 따른 0,1로 나뉨
train.isnull().sum()
test.isnull().sum()
train['Age'].describe()
train['Age'] = train['Age'].fillna(train['Age'].mean())

# Age Column에서 빈값을 그냥 평균값으로 대충 채워주자
test['Age'] = test['Age'].fillna(test['Age'].mean())

# Test도 동일
train.isnull().sum()
round(train['Age'].mode(),0)
test.isnull().sum()
train.isnull().sum() 

# Cabin Null의 비중이 너무 높음, 전체의 77%정도인데... 이건 뭘로 채워도 문제가 될 것 같음
train['Cabin'].unique()

# 게다가 값들이 뭔가 Null 값을 채우기 만만치 않아보임
train = train.drop(['Cabin'],1)

# 이거는 그냥 지워버리자... 도저히...
test = test.drop(['Cabin'],1)

# train도 지웠으니까 test도 같이 지워줘야 함 -> 안지우면 가르쳐주지도 않았는데 시험보게 하는 꼴
train.isnull().sum()

# 잘 지워졌나 확인
test.isnull().sum()

# 테스트도 확인
train.groupby(['Embarked','Pclass'])['Fare'].median()
train.isnull().sum()
test.isnull().sum()
train['Embarked'].value_counts()

# 다음은 Embarked를 해볼 예정, 이거는 Train에서 2개만 비어있으니 그냥 가장 많은 값으로 넣으면 될듯
train['Embarked'] = train['Embarked'].fillna('S')

# fillna -> NA를 채워라, 'S' 값으로
train.isnull().sum()

# 잘 채워진 걸 알 수 있음
test.isnull().sum()

# Test는 뭐 없으니까 Pass!
train.isnull().sum()
test.isnull().sum()

# Test의 Fare에 Null값이 1개가 있음.....
train['Fare'].describe()

# Fare는 요금에 대한 값인데, 이것도 하나만 비어있으니 대충 평균으로 떄려넣자
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

test.isnull().sum() 

# 잘 채워졌나 확인하는 습관을 가지자
train['Sex']

# Male, Female로 되어있는데 이건 기계는 못읽음... 바꿔줘야함
train['Sex']=train['Sex'].replace("male",0)

train['Sex']=train['Sex'].replace("female",1)

train['Sex'].head()

# 남자는 0 여자는 1로 바꿔줌, 잘 바뀌었는지 확인
test['Sex']=test['Sex'].replace("male",0)

test['Sex']=test['Sex'].replace("female",1)

test['Sex'].head()

# Test도 동일하게
train.head()
train['Name'].unique()[:30]

# Unique는 해당 Column의 변수들

# 이건 지금은 죽어도 못하겠음...
train = train.drop(['Name'],1)

train.head()

# Train 에서 Name을 시원하게 지워버리고 확인
test = test.drop(['Name'],1)

test.head()

# test 에서 Name을 시원하게 지워버리고 확인
train.head()
train['Ticket'].unique()[:30]

# Train Ticket의 변수들을 출력... 근데 이것도 숫자가 아니고 힘드네...
train = train.drop(['Ticket'],1)

train.head()
test = test.drop(['Ticket'],1)

test.head()

# 잘...날렸다
train['Embarked'].unique()

# 이건 3개밖에 안되니까 바꿀 수 있겠다... 직접 실습 진행
train['Embarked']=train['Embarked'].replace("S",0)

train['Embarked']=train['Embarked'].replace("C",1)

train['Embarked']=train['Embarked'].replace("Q",1)

train['Embarked'].head()

# 남자는 0 여자는 1로 바꿔줌, 잘 바뀌었는지 확인
test['Embarked']=test['Embarked'].replace("S",0)

test['Embarked']=test['Embarked'].replace("C",1)

test['Embarked']=test['Embarked'].replace("Q",1)

test['Embarked'].head()

# 남자는 0 여자는 1로 바꿔줌, 잘 바뀌었는지 확인
train.head()

# 잘 바뀐거같은데..
test.head()

#이것도...
y = train['Survived']

X = train.drop(['Survived'],1)
print(X.shape, y.shape)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=30, test_size =0.3)

# 이런 양식에 넣으면 알아서 나눠짐, test_size에 비율을 넣으면 그만큼 Validation으로 나눠짐
print(X_train.shape,y_train.shape, X_valid.shape, y_valid.shape)
268/(623 + 268)

# 대략 30%로 나눠짐을 확인
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()



# from lightgbm import LGBMClassifier

# model = LGBMClassifier()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)

(pred_train == y_train).mean()
pred_train = model.predict(X_valid)

(pred_train == y_valid).mean()
test.isnull().sum()
pred_test = model.predict(test)
pred_test

# 우리가 예측한 값이 잘 나왔나 확인
gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

print(gender_submission.shape)

gender_submission.head()

# OMR 카드를 출력
gender_submission['Survived'] = pred_test

# gender_submission 정답부분 Survived에 우리가 예측한 pred_test를 넣어준다
gender_submission

# 잘 들어갔는지 확인
gender_submission.to_csv("gender_submission_final.csv", index=False)

# gender_submission 파일을 저장한다.(Output)
#=== Decision Tree에서만 Visualization이 가능함 ===#

!pip install pydotplus

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(model, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())