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
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
alldata = pd.concat([train,test])

pd.options.display.max_columns=999

alldata 
#트리모델은 카테고리적 데이터 학습을 잘하는데, 숫자적인 데이터를 학습하기가 힘들다

#숫자적인 데이터 -> y 값에 선형적인 관계를 준다

train.corr()["SalePrice"].sort_values(ascending=False)
alldata2 = alldata.drop("SalePrice",1)

# Onehot encoding in one line

pd.get_dummies(alldata2)
alldata3=pd.get_dummies(alldata2)

alldata3=alldata3.fillna(-1) #handle missing values

alldata3
#컬럼의 범위가 달라서 표준화를 할꺼임 : 어떤 컬럼의 범위가 넓다면 그컬럼이 중요도가 높다고 생각하기때문에, 범위를 맞춰주기위해

from sklearn.preprocessing import StandardScaler



ss = StandardScaler()

alldata4 = ss.fit_transform(alldata3)

alldata4
alldata4=pd.DataFrame(alldata4,columns=alldata3.columns)

alldata4
train2 = alldata4[:len(train)]

test2 =alldata4[len(train):]
from sklearn.linear_model import Ridge



#alpha 값 학습할 때 가중치설정 default alpha=1 

#최적의 알파값찾기

rg = Ridge(20)



rg.fit(train2,train["SalePrice"])

result = rg.predict(test2)
#train test split 여기서는 사용하면 안좋음

#왜? 1. 데이터갯수 너무 적음 2. 평가셋의 역할을 못함

#그래서 교차검증을 해야함



from sklearn.model_selection import cross_val_score



np.sqrt(-cross_val_score(rg, train2, train["SalePrice"], 

                cv=10, n_jobs=-1, 

                scoring="neg_mean_squared_error").mean()) #10개평균

#model name, train data set ,train targetdata , cv몇개? 기본5, 많으면 좋음 , 4/-1? 다쓰겟다

#점수가 높으면 높을 수록좋음
sub=pd.read_csv("/kaggle/input/home-data-for-ml-course/sample_submission.csv")

sub
sub["SalePrice"]=result

sub.to_csv("sub_linear.csv", index=0)