import pandas as pd 

import numpy as np
#Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # setting seaborn default for plots
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train_and_test = [train,test] #데이터 병합
#데이터에 Title이라는 새로운 열을 만들어 Title에서 추출한 Title을 넣어주기

for dataset in train_and_test:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.') 

    

    #’([A-Za-z]+).‘는 정규표현식인데, 공백으로 시작하고, .으로 끝나는 문자열을 추출할 때 저렇게 표현한다.





test.head(5)
#추출한 Title을 가진 사람이 몇 명이 존재하는지 성별과 함께 표현

pd.crosstab(train['Title'], train['Sex'])
for dataset in train_and_test:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Don','Dr', 'Jonkheer',

                                             'Major', 'Master','Rev', 'Sir'], 'Mr')

    dataset['Title'] = dataset['Title'].replace(['Ms','Mlle','Mme','Mrs','Countess','Lady',], 'Miss')

pd.crosstab(train['Title'], train['Sex'])
pd.isnull(train['Sex'])
pd.isnull(test['Sex'])
for dataset in train_and_test:

    dataset['Title'] = dataset['Title'].astype(str)
for dataset in train_and_test:

    dataset['Sex'] = dataset['Sex'].astype(str)
for dataset in train_and_test:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].astype(str)
#여기서부터 병합한 파일에서 에러가 자꾸 나서 따로 함



train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

train.head()
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1 # 자신을 포함해야하니 1을 더하기

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train.head()
# pclass와 연관시키지 않고 binning한 평균으로 적용해보려했으나

# distribution이 매우 비대칭적

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')

# testset 에 있는 nan value 를 평균값으로 치환합니다.

test.loc[test.Fare.isnull(), 'Fare'] = test['Fare'].mean()



# outlier의 영향을 줄이기 위해 Fare 에 log 를 취하

# dataFrame 의 특정 columns 에 공통된 작업(함수)를 적용하고 싶으면 아래의 map, 또는 apply 를 사용하면 매우 손쉽게 적용

# 우리가 지금 원하는 것은 Fare columns 의 데이터 모두를 log 값 취하는 것인데, 파이썬의 간단한 lambda 함수를 이용해 간단한 로그를 적용하는 함수를 map 에 인수로 넣어주면, 

# Fare columns 데이터에 그대로 적용이 됩



train['Fare'] = train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

test['Fare'] = test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
# Cabin, Ticket 제거

train = train.drop(['Cabin','Ticket'], axis=1)

test = test.drop(['Cabin','Ticket'], axis=1)
# 이제 필요 없어진 것들 : Name, SibSp, Parch

train = train.drop(['Name', 'SibSp', 'Parch'], axis=1)

test = test.drop(['Name', 'SibSp', 'Parch'], axis=1)
print(train.head())

print(test.head())
# One-hot-encoding for categorical variables

train = pd.get_dummies(train)

test = pd.get_dummies(test)



# Categorical Feature에 대해 one-hot encoding과 train data와 label을 분리

train_label = train['Survived']

train_data = train.drop('Survived', axis=1)

test_data = test.drop("PassengerId", axis=1).copy()
# scikit-learn 라이브러리를 불러오기



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



# 학습시키기 전에는 주어진 데이터가 정렬되어있어 학습에 방해가 될 수도 있으므로 섞어주기

from sklearn.utils import shuffle

train_data, train_label = shuffle(train_data, train_label, random_state = 5)
# 모델 학습과 평가에 대한 pipeline을 만들기

# scikit-learn에서 제공하는 fit()과 predict()를 사용하면 매우 간단하게 학습과 예측을 할 수 있어서 

# 그냥 하나의 함수만 만들면 편하게 사용가능



def train_and_test(model):

    model.fit(train_data, train_label)

    prediction = model.predict(test_data)

    accuracy = round(model.score(train_data, train_label) * 100, 2)

    print("Accuracy : ", accuracy, "%")

    return prediction



# Logistic Regression

log_pred = train_and_test(LogisticRegression())

# SVM

svm_pred = train_and_test(SVC())

#kNN

knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))

# Random Forest

rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))

# Navie Bayes

nb_pred = train_and_test(GaussianNB())
submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": rf_pred

})



submission.to_csv('submission_rf.csv', index=False)