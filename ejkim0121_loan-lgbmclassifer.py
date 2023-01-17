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
train=pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")

train.head()
test=pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-test.csv")

test.head()
y=train["SeriousDlqin2yrs"]

#학습을 시킬 때 정답값이 들어X

#정답값 저장해두는 기능도 있음

#Column 수를 맞춰 주기 위해서
train=train.drop(["SeriousDlqin2yrs","Unnamed: 0"],axis=1)

test=test.drop(["SeriousDlqin2yrs","Unnamed: 0"],1)

train.head()

 #기본적으로 행을 제거하는 것으로 되어 있기 때문에, 열 제거를 위해서는 axis=1

    
#RandomForest는 대용량 데이터 셋에 대해서는 약하고, 속도가 느리다. 

#카테고리형 데이터 유 ; 트리형모델

#데이터양 많, 비선형성 높을수록

#모델링
#분류모델 #학습

from lightgbm import LGBMClassifier

lgb=LGBMClassifier(colsample_bytree=0.5, subsample=0.8,num_leaves=20,n_estimators=1200,learning_rate=0.0075)

lgb.fit(train,y)



#"colsample_bytree=0.5"의 의미 : 트리형성시 랜덤하게 컬럼이 선택됨. 

# 학습시에 중요하지 않다고 판단되는 칼럼은 학습X. 상대적으로 중요하지 않은 칼럼에서도 유의마한 데이터 존재 가능. > 일반화 능력 떨어짐. "칼럼에대한 과적합?"

#ensemble effect  각 모델/트리 다른 학습 방식 보유. 다양한 방식의 학습 적용 가능해짐.



#subsample 데이터 추출 랜덤



#num_leaves 데이터가 양이 많아지고 복잡해짐에 따라 number or leaves가 많아지는 게 좋음. 세분화 & 다분화. 단, 데이터가 단순한데 num leaves가 너무 커지게 되면 과적합이 생김. 

#feature의 개수



#n_estimators 나무의 갯수. 기본 셋팅이 100개. 



#learning rate 기본값 0.1 낮춰주면 좋아짐. ***Learning_rate를 낮추게 되면 학습량이 줄어들기때문에 n_estimators도 바례하여 높여주기.

#예측 0과 1

#result=lgb.predict(test)

#확률예측

result=lgb.predict_proba(test)
result[:,1]

#0[0=파산X일확률,1=파산일확률]

#result[행,열]/ : 가져오지 X / 0부터 카운트
pd.DataFrame(lgb.feature_importances_)
importance_df=pd.DataFrame(lgb.feature_importances_).rename(columns={0:"importance"})

importance_df['columns']=train.columns

importance_df['importance']=(importance_df['importance']/importance_df['importance'].values.sum())*100

#importance_df['importance']=importance_df.sort_values("importance",ascending=False)

importance_df=importance_df.sort_values("importance",ascending=False)

importance_df.head(10)

#각 features의 중요성 판단
sample=pd.read_csv("/kaggle/input/GiveMeSomeCredit/sampleEntry.csv")

sample.head()
sample["Probability"]=result[:,1]

sample.head()
sample.to_csv("20191007_LGBMClassifier.csv",index=False)

#index Column에서 제거해주기
train.head()