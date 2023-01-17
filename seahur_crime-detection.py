# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/sf-crime/train.csv', parse_dates=['Dates'])

# train['Dates'] = train['Dates'].astype('datetime64')

test = pd.read_csv('/kaggle/input/sf-crime/test.csv', parse_dates=['Dates'])
train.shape
train.head(30)
test.head()
train['year'] = train['Dates'].dt.year

train['month'] = train['Dates'].dt.month

train['day'] = train['Dates'].dt.day

train['DayOfWeek'] = train['Dates'].dt.weekday

train['hour'] = train['Dates'].dt.hour

train['minute'] = train['Dates'].dt.minute
# 지역마다 발생하는 범죄 다르기에 구분 

# case=False (T가 default: 대소문자구분한다는말)

# train['Address'].str.contains('Block', case=False)

train['Block'] =  train['Address'].str.contains('Block', case=False)

# boolean형식, T=1, F=0으로 알아서 인식함.

train.head()
train['AV'] =  train['Address'].str.contains('AV', case=False)

# boolean형식, T=1, F=0으로 알아서 인식함.

train.head()
train['ST'] =  train['Address'].str.contains('ST', case=False)

# boolean형식, T=1, F=0으로 알아서 인식함.

train.head()
test['year'] = test['Dates'].dt.year

test['month'] = test['Dates'].dt.month

test['day'] = test['Dates'].dt.day

test['DayOfWeek'] = test['Dates'].dt.weekday

test['hour'] = test['Dates'].dt.hour

test['minute'] = test['Dates'].dt.minute
test['Block'] =  test['Address'].str.contains('Block', case=False)

# boolean형식, T=1, F=0으로 알아서 인식함.

test.head()
test['AV'] =  test['Address'].str.contains('AV', case=False)

# boolean형식, T=1, F=0으로 알아서 인식함.

test.head()
test['ST'] =  test['Address'].str.contains('ST', case=False)

# boolean형식, T=1, F=0으로 알아서 인식함.

test.head()
# 도움이 되는 이유.(시각화가 안될 때는 통계량을 보자)

train.groupby(['Category'])['Block'].mean()
train.groupby(['Category'])['AV'].mean()
# 값이 편차가 크다고 무조건 도움되는 것은 아니다.

# 각 카테고리에 포함되는 갯수가 작을 때, 도움이 안된다. 

train.groupby(['Category'])['ST'].mean()
# 각 카테고리에 포함되는 갯수가 작을 때, 도움이 안된다. 

# X 변수일 때는 1000이하인 것들을 others로 묶을 수 있으나, Y변수일 때는 건들지 못한다.

train['Category'].value_counts()
y = train['Category']
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(train['PdDistrict'])

le.classes_
train['PdDistrict'] = le.transform(train['PdDistrict'])
train.head()
test['PdDistrict'] = le.transform(test['PdDistrict'])

test.head()
# PdDistrict랑 급이 다르다. 범주가 23000여개가 있다.

train.Address.nunique()
# 심지어 train과 test셋의 Address 범주 갯수가 다르다. 알파벳 순서로 넘버 배정. 배정된 숫자가 달라짐

test.Address.nunique()
# column은 series 형식이라서 각각 더해야 함.

le.fit(list(train['Address']) + list(test['Address']))
train['Address'] = le.transform(train['Address'])

test['Address'] = le.transform(test['Address'])
train.head()
test.head()
# X변수도 범주형이고 Y변수도 범주형일 때 시각화 어떻게 하는가?

import matplotlib.pyplot as plt

import seaborn as sns



# sns.catplot(x="Category", y="year", kind="bar", data=train)

a, b = plt.subplots(1,1, figsize=(20,12))

sns.boxplot(train['Category'], train['year'])

plt.xticks(rotation=75)
train['Dates'].dt.week
test['Dates'].dt.week
y = train['Category']

train = train.drop(['Dates', 'Resolution', 'Descript', 'Category'], axis=1)

test = test.drop(['Id', 'Dates'], axis=1)
# 데이터 셋이 많다. 같은 트리 계열인 LGBM을 사용해보자. 트리 계열에서는 one-hot encoding이 필요 없다.

#from lightgbm import LGBMClassifier

#lgbm = LGBMClassifier(n_estimators=100)
# lgbm.fit(train, y)
# preds = lgbm.predict_proba(test)
# preds
from catboost import CatBoostClassifier

cat = CatBoostClassifier(task_type='GPU')

cat.fit(train, y)
preds = cat.predict_proba(test)
sample = pd.read_csv('/kaggle/input/sf-crime/sampleSubmission.csv')

sample.head()
sample.iloc[:,1:] = preds

sample.head()
sample.to_csv("baseline_lgbm.csv", index=False)
# train set은 홀수 week, test set은 짝수 week --> day(일)이 필요 없는 정보가 된다.

# lgbm parameter 중에서 categorical_feature 언급해주면 가중치 분산이 줄어들어서 효율적으로 예측 가능 할 듯.

# 시각화.
