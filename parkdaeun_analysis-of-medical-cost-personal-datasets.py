# 파이썬3에는 유용한 분석 라이브러리가 많이 설치되어 있음

# 캐글/파이썬 이미지로 정의됨 : https://github.com/kaggle/docker-python      <-



import numpy as np # 선형대수

import pandas as pd # 데이터 처리, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # 시각화

# Input 데이터 파일은 "../input" 디렉터리에서 사용할 수 있다.

# 예를 들어 실행을 클릭하거나 Shift+Enter를 누르면 입력 디렉토리에 파일이 나열



import os # os 모듈 호출

print(os.listdir("../input")) 



# 현재 디렉토리에 쓴 결과는 Output으로 저장됨
data = pd.read_csv('../input/insurance.csv')
data.info()
data.columns
data.head(5)
data.describe()
data.corr()
f,ax = plt.subplots(figsize = (10, 10))

sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt='.1f', ax=ax) # https://kiddwannabe.blog.me/221205309816

plt.show()
data.corr()['charges'].sort_values()
data.plot.scatter(x='age', y='charges', figsize=(10, 5))
data.plot.scatter(x='charges', y='bmi', figsize=(10, 5))
data.boxplot(column="charges",by="children", figsize=(10, 5))
data.boxplot(column="charges",by="smoker", figsize=(10, 5))
pd.crosstab(data["sex"],data["region"],margins=True)
data.boxplot(column="charges", by="region", figsize=(10,5))
data['sex'] = data['sex'].map({'female' : 0, 'male' : 1})

data['smoker'] = data['smoker'].map({'yes' : 0, 'no' : 1})

data['region'] = data['region'].map({'southeast' : 0, 'northeast' : 1, 'southwest' : 3, 'northwest' : 4})
'''

def smoker(yes):

    if yes =="yes":

        return 1

    else:

        return 0

data["smoker"]=data["smoker"].apply(smoker)

def sex(s):

    if s =="male":

        return 1

    else:

        return 0

data["sex"]=data["sex"].apply(sex)

def region(region):

    if region =="southwest":

        return 1

    else:

        return 0

data["region"]=data["region"].apply(sex)

'''
sns.pairplot(data)

plt.show()
data.corr()['charges'].sort_values()
data.corr()
sns.pairplot(data)

plt.show()
x = data.drop(["charges", "region"],axis =1)
y =data["charges"]
from sklearn.model_selection import train_test_split

# sklearn.cross_validation -> sklearn.model_selection으로 이름 변경

#훈련용, 테스트용으로 데이터셋 분리

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import r2_score  # For find accuracy with R2 Score
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3) #데이터의 30퍼센트를 평가용으로 쓰기로 함

model = LinearRegression()

model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)

y_test_pred = model.predict(x_test)



print(model.score(x_test, y_test))
forest = RandomForestRegressor(n_estimators = 100, # 생성할 의사결정 트리의 갯수

                              random_state = 1, max_depth=5, 

                              n_jobs = -1) # 컴퓨터의 코어를 얼마나 사용할 것인지. -1은 컴퓨터의 모든 코어 사용

# Create a instance for Random Forest Regression model

forest.fit(x_train,y_train)



# Prediction with training dataset:

y_pred_RFR_train = forest.predict(x_train)



# Prediction with testing dataset:

y_pred_RFR_test = forest.predict(x_test)

accuracy_RFR_train = r2_score(y_train, y_pred_RFR_train)

print("Training Accuracy for Random Forest Regression Model: ", accuracy_RFR_train)



# Find testing accuracy for this model:

accuracy_RFR_test = r2_score(y_test, y_pred_RFR_test)

print("Testing Accuracy for Random Forest Regression Model: ", accuracy_RFR_test)



print(model.score(x_test, y_test))