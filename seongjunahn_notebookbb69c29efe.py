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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
drop_train=train_data.drop(['Cabin','Ticket', 'PassengerId'],axis=1)

drop_test=test_data.drop(['Cabin','Ticket', 'PassengerId'],axis=1)
drop_train.isnull().sum()
drop_test.isnull().sum()
S=drop_train[drop_train["Embarked"]=="S"].shape[0]

Q=drop_train[drop_train["Embarked"]=="Q"].shape[0]

C=drop_train[drop_train["Embarked"]=="C"].shape[0]

S,Q,C
drop_train=drop_train.fillna({"Embarked":"S"})

#train 데이터에 Embarked 데이터 중 2개가 NA이므로 가장 많은 S값을 넣어준다

#데이터 총량에 비해 NA값은 매우 작으므로 임의로 조절해도 큰 오차는 나지 않을 것이다
embarked_mapping={"S":1,"C":2,"Q":3}

drop_train['Embarked']=drop_train['Embarked'].map(embarked_mapping)

drop_test['Embarked']=drop_test['Embarked'].map(embarked_mapping)

#Embarked의 문자열을 숫자로 전처리
sex_mapping={"male":0,"female":1}

drop_train['Sex']=drop_train['Sex'].map(sex_mapping)

drop_test['Sex']=drop_test['Sex'].map(sex_mapping)

#Sex의 문자열을 숫자로 전처리
drop_test["Fare"].fillna(drop_test["Fare"].mean() , inplace=True)

#test 데이터에 Fare 데이터 중 1개ork NA이므로 평균값을 넣어준다

#데이터 총량에 비해 NA값은 매우 작으므로 임의로 조절해도 큰 오차는 나지 않을 것이다
drop_train['Name']=drop_train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

drop_train['Name']

#NA값이 많은 Age데이터를 메꾸기 위해 호칭 별 나이의 평균값을 NA에 넣는다

#쉼표를 기준으로 호칭과 이름으로 나뉘기 때문에 쉼표를 기준점으로 함
drop_train['Name'].value_counts()

#Mr, Master, Dr, Rev(목사), Col(대령 혹은 남자에 대한 경칭), Major, Jonkheer(귀족), Sir, Don(귀족), Capt

#위는 전부 남자를 부르는 호칭인데, Master는 어린 남자의 호칭이니까 이거만 빼고 다 Mr로 묶는다

#Miss, Mrs, Mlle(마드모아젤), Countess(백작부인), Mme(마담), Lady, Ms

#위는 전부 여자 호칭으로, Miss(미혼),Mrs(기혼) 둘로 묶는다

#Dr 밑으로 정렬된 호칭들은 총 데이터량과 비교하면 적은 양이기 때문에 분류가 틀려도 큰 오차가 생기지 않을 것이다
drop_train['Name']=drop_train['Name'].replace(['Dr','Rev','Col','Major','Jonkheer','Sir','Don','Capt'],'Mr')

drop_train['Name']=drop_train['Name'].replace(['Mlle','the Countess','Ms'],'Miss')

drop_train['Name']=drop_train['Name'].replace(['Mme','Lady'],'Mrs')
drop_train['Name'].value_counts()
drop_train.groupby('Name')['Age'].mean()
drop_train.loc[(drop_train['Age'].isnull())&(drop_train['Name']=='Master'),'Age']=4.5

drop_train.loc[(drop_train['Age'].isnull())&(drop_train['Name']=='Miss'),'Age']=21.9

drop_train.loc[(drop_train['Age'].isnull())&(drop_train['Name']=='Mr'),'Age']=33

drop_train.loc[(drop_train['Age'].isnull())&(drop_train['Name']=='Mrs'),'Age']=35.9
drop_test['Name']=drop_test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

drop_test['Name']

#위처럼 test데이터도 전처리
drop_test['Name'].value_counts()
drop_test['Name']=drop_test['Name'].replace(['Dr','Rev','Col','Major','Jonkheer','Sir','Capt'],'Mr')

drop_test['Name']=drop_test['Name'].replace(['Mlle','the Countess','Ms'],'Miss')

drop_test['Name']=drop_test['Name'].replace(['Mme','Lady','Dona'],'Mrs')
drop_test['Name'].value_counts()
drop_test.groupby('Name')['Age'].mean()
drop_test.loc[(drop_test['Age'].isnull())&(drop_test['Name']=='Master'),'Age']=7.4

drop_test.loc[(drop_test['Age'].isnull())&(drop_test['Name']=='Miss'),'Age']=21.7

drop_test.loc[(drop_test['Age'].isnull())&(drop_test['Name']=='Mr'),'Age']=32.3

drop_test.loc[(drop_test['Age'].isnull())&(drop_test['Name']=='Mrs'),'Age']=38.9
drop_train.isnull().sum()
drop_test.isnull().sum()
drop_train['Name']=drop_train['Name'].replace('Mr','1')

drop_train['Name']=drop_train['Name'].replace('Miss','2')

drop_train['Name']=drop_train['Name'].replace('Mrs','3')

drop_train['Name']=drop_train['Name'].replace('Master','4')
drop_test['Name']=drop_test['Name'].replace('Mr','1')

drop_test['Name']=drop_test['Name'].replace('Miss','2')

drop_test['Name']=drop_test['Name'].replace('Mrs','3')

drop_test['Name']=drop_test['Name'].replace('Master','4')

#문자열을 전부 숫자로 전처리
#train과 test 모두 전처리 완료 후 prediction
from sklearn.ensemble import RandomForestClassifier



y = drop_train["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch","Embarked","Age","Fare"]

X = pd.get_dummies(drop_train[features])

X_test = pd.get_dummies(drop_test[features])



model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")