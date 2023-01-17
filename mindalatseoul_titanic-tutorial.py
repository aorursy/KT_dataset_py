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

# train dataset의 주소는 '/kaggle/input/titanic/train.csv'
# test dataset의 주소는 '/kaggle/input/titanic/test.csv'
# train set으로 머신러닝 모델을 학습 시키고 
# test set으로 예측을 합니다 
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
answer = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# 수집된 데이터 파악
# train set의 사이즈 파악 
row_number = train.shape[0]
col_number = train.shape[1]
column_list = train.columns

print('train 데이터셋은 {}행의 데이터와 {} 등 {}열의 column이 있습니다.'.format(row_number,', '.join(column_list),col_number))
# ','.join(column_list) 는 column_list라는 리스트의 요소들을 , 기준으로 연결해주는 함수입니다 
# join 함수 참고 링크 https://www.geeksforgeeks.org/join-function-python/ 
# 각 변수별로 생존율을 계산해봅니다  
# 계산하기 위해서 변수를 집어 넣으면 생존율을 계산해주는 함수를 미리 만들어 봅니다 
# python 의 메소드에 대한 참고자료 : https://www.w3schools.com/python/python_functions.asp 
def calc_ratio(x):
    # x는 리스트 형태로 다룹니다
    df = train.groupby(x).agg({'Survived':[np.sum,'count']})['Survived']
    df['생존율'] = round(df['sum']/df['count']*100,1)
    print(','.join(x)+'에 따른 생존율을 아래와 같습니다')
    return df 
print('티켓 등급이 높을 수록 생존율이 높죠, 우스갯소리로, 구명보트가 VIP에게 우선으로 제공되기 때문이라 합니다')
calc_ratio(['Pclass'])

print('여성의 생존율은 74%가 넘는데, 남성의 생존율은 18%죠, 그 당시엔 Lady First 같은 신사도가 있었다고 하는데요. 그럼 지금은..?')
calc_ratio(['Sex'])
print('승선 항구에 따라서도 생존율이 나뉩니다')
calc_ratio(['Embarked'])

# 나이와 티켓가격의 경우, 숫자형대로 되어있기 때문에 구간에 따라서 번위를 지정할 필요가 있습니다  
# 나이는 10살 단위로 나누어 age_range를 만들고 
# age_range를 groupby해 생존율을 구함  
train['Age_range']=round(train['Age']/10,0)*10
print('나이가 어리거나, 높으면 생존율이 높지만, 10대 ~ 60대의 생존율은 35~40%로 비등비등 하죠')
calc_ratio(['Age_range'])
# 티겟은 100달러 단위로 range를 나눠봅니다 
train['fare_range'] = round(train['Fare']/100,0)*100
print('티켓 가격이 500달러 이상이면, 생존율이 100%이고, 100달러 미만인 경우는 30%에 그칩니다')
calc_ratio(['fare_range'])
import matplotlib.pyplot as plt 
import seaborn as sns  
plt.figure(figsize=(10,10))
sns.heatmap(data = train.corr(),annot=True,fmt='.2f',cmap='Blues')
plt.show 
# 우선 데이터 프레임을 필요한 요소만 남겨봅니다  
train2 = train.loc[:,['Pclass','Age','Sex','Embarked']].copy() 
train2.head()
# 결측값은 상황에 따라 계산하는 방법이 무궁무진하지만 
# 현재는 단순 평균 값으로 결측값을 대체합니다  
train2['Age'] = train2['Age'].fillna(train2['Age'].mean())
# 이중 Sex와 Embarked를 숫자로 바꾸는데 
# 성별은 2가지, 승선항구는 3가지가 있습니다
print(train2['Sex'].unique())
print(train2['Embarked'].unique())
# 모델을 활용하기 위해 단순히 문자열을 숫자로 바꾸는 목적이며 
# 각각 
# male -> 0, femail -> 1  
# S -> 0, C -> 1, Q -> 2, nan -> 3
# 로 변경합니다. 
train2['Sex'] = train2['Sex'].replace("male",0)
train2['Sex'] = train2['Sex'].replace("female",1)

train2['Embarked'] = train2['Embarked'].replace("S",0)
train2['Embarked'] = train2['Embarked'].replace("C",1)
train2['Embarked'] = train2['Embarked'].replace("Q",2)
train2['Embarked'] = train2['Embarked'].replace(np.nan,3)
X = train2
y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 30, test_size = 0.3)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier()

model.fit(X_train,y_train)
predict_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)
# 4가지 변수를 활용해 학습한 결과, 예측의 정확도는 91%에 달합니다
# 이제 test 데이터를 적용해 실제 예측을 합니다  
# 우리는 아직 test 데이터를 정제 하지 않았고  
# 어떻게 정제해야할지를 알기 때문에  
# 바로 test 데이터의 정제를 진행합니다
test2 = test[['Pclass','Sex','Age','Embarked']].copy() 

test2['Age'] = test2['Age'].fillna(test2['Age'].mean())

test2['Sex'] = test2['Sex'].replace("male",0)
test2['Sex'] = test2['Sex'].replace("female",1)

test2['Embarked'] = test2['Embarked'].replace("S",0)
test2['Embarked'] = test2['Embarked'].replace("C",1)
test2['Embarked'] = test2['Embarked'].replace("Q",2)
test2['Embarked'] = test2['Embarked'].replace(np.nan,3)

# 그리고 정제된 test2를 바로 model.predict에 적용합니다  
predict_result = model.predict(test2)
# predict_result를 앞서 불러왔던 answer와 비교합니다  
my_score = accuracy_score(predict_result,answer['Survived'])
print('accuracy_score on train dataset : ', my_score)
