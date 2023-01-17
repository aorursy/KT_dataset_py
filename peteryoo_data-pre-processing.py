from pprint import pprint

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Load Titanic dataset

titanic = pd.read_csv('../input/train.csv', index_col='PassengerId')



# 데이터 샘플을 확인

titanic.head(10)
# 변수 갯수, 널값, 타입을 확인

titanic.info()
#숫자형 변수의 분포를 확인

titanic.describe()
# 변수간의 상관관계를 분석

correlation = titanic.corr()

sns.heatmap(correlation, annot=True, cbar=True, cmap="YlGnBu")
list = []



#  age에 대해 연령대 별로 변수를 분할

for i in range (0, len(titanic)):

    age = titanic.iloc[i].Age

    

    if age < 20 :

        age = 'child'

    elif (age >= 20) & (age < 30):

        age = 'adult20'

    elif (age >= 30) & (age < 40):

        age = 'adult30'

    elif (age >= 40) & (age < 50):

        age = 'adult40'

    elif (age >= 50) & (age < 60):

        age = 'adult50'

    elif (age >= 60) & (age < 70):

        age = 'adult60'

    elif (age >= 70) & (age < 99):

        age = 'adult70'

    else:

        age = 'unknown'

    

    list.append(age)

    

titanic['Age_modified'] = list

Age_dummies = pd.get_dummies(titanic.Age_modified, prefix = 'Age')
# embark, sex에 대해 one-hot 변수로 변환

Embarked_dummies = pd.get_dummies(titanic.Embarked, prefix = 'Embarked')

Sex_dummies = pd.get_dummies(titanic.Sex, prefix = 'Sex')

data = pd.concat([titanic, Age_dummies, Embarked_dummies, Sex_dummies], axis = 1)



data = data.drop(['Name', 'Sex', 'Age', 'Age_modified', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1)
# 전처리 끝난 변수들을 확인

data.columns.values