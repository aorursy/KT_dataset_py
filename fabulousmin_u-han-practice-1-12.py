# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt 
import matplotlib
%matplotlib inline

import seaborn as sns 
import missingno as msno

#provides a small toolset of flexible and easy-to-use missing data visualizations and utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of your dataset. Just pip install missingno to get started.
#설치
#conda install -c conda-forge missingno 

import xgboost as xgb
#설치
#conda install -c conda-forge xgboost 
import warnings 
sns.set(style='white', context = 'notebook', palette='deep')
#seaborn style API
#https://seaborn.pydata.org/api.html#style-api
#style 파라미터 확인법 axes_style(),  plotting_context(), color_palette()
#seaborn 그래프 종류 https://seaborn.pydata.org/examples/index.html
# Any results you write to the current directory are saved as output.
np.random.seed(1989)
#시드값 설정, 아무거나해도 되지만 여기선 커널이랑 결과값 똑같이 하려고1989로 설정
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape :", train.shape)
print("Test shape :", test.shape) 
train.head()
#train 헤드확인
print(train.info())
#train.csv 각 칼럼 메타데이터 확인
print(test.info())
#test.csv 각 칼럼 메타데이터 확인
targets = train['target'].values
sns.set(style="darkgrid")
#테마설정 
#밝은거 좋아하면 "whitegrid"

ax = sns.countplot(x = targets)
#taget 칼럼 그래프 

for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))
#바그래프 위에 퍼센트 값 계산 
    
plt.title('Distribution of Target', fontsize=20)
plt.xlabel('Claim', fontsize=20)
plt.ylabel('Frequency [%]', fontsize=20)
#제목, x축, y축 레이블설정
ax.set_ylim(top=700000)
#y축 최대값설정
print(train.id.nunique()) #The number of unique elements
print(train.shape[0]) #id 칼럼 값 갯수 
print('Id is unique.') if train.id.nunique() == train.shape[0] else print('Oh no')
#if train 유니크한id값의 갯수 == train id칼럼갯수랑 같으면 모든 값이 unique 

print('Train and test sets are distinct.') if len(np.intersect1d(train.id.values, test.id.values)) == 0 else print('Oh no')
#train id 값, test id값 중 겹치는 값 있는지 확인
#cf.  print(np.intersect1d(train.id.values, test.id.values)) 
#1차원 배열 2개 받아서  겹치는 값만 1차원배열으로 리턴 
print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] else print('Oh no')
#id값 갯수랑 train테이블의 모든 칼럼갯수중 가장 작은 값 똑같으면 missing value 없다. 


train_null = train
train_null = train_null.replace(-1, np.NaN)
# -1 값을 np.NaN으로 바꿔줌 그래야 msno라이브러리에서 null로 알아먹음 
#안바꿔주면 -1 도 값 있는것으로 판단
msno.matrix(df=train_null.iloc[:, :], figsize=(20, 14), color=(0.8, 0.5, 0.2))   
#train_null 데이터프레임 null값 시각화 
#cf. :param figsize: The size of the figure to display.
#:param color: The color of the filled columns. Default is `(0.25, 0.25, 0.25)`. -디폴드 검정


test_null = test
test_null = test_null.replace(-1, np.NaN)
msno.matrix(df=test_null.iloc[:,:], figsize=(20,14), color=(0.8, 0.5, 0.2))

#train_null 과 같은 방식으로 -1을 null으로 바꿔줌
train_null = train_null.loc[:, train_null.isnull().any()]
test_null = test_null.loc[:, test_null.isnull().any()]
#Null값 있는 칼럼 추출

print(train_null.columns)
print(test_null.columns)
#null 값 포함하고 있는 칼럼 출력
print('Columns \t Number of NaN')
for column in train_null.columns:
    print('{}:\t {}'.format(column,len(train_null[column][np.isnan(train_null[column])])))
#각 칼럼별 null값 갯수 



