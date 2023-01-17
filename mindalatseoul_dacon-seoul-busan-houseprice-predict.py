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
import pandas as pd 
import numpy as np
import os
import time 

# 데이터 시각화
import matplotlib 
import matplotlib.pyplot as plt  
matplotlib.rc('font',family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
# 이 경진대회에서는 각 아파트의 가격 추이와, 주변 아이돌봄센터 및 공원의 정보를 활용해 주택 가격을 예측하는 것
# 즉, 데이터 분석의 목표는 "아이돌폼센터와 공원의 여부가 주택 가격 예측에 도움이 되느냐 안되는냐를 판단해보기"
# 이에 필요한 가설은 크게 3가지로 나눠 검정을 해볼 수 있음 
# 가설1
# 귀무가설 : 어린이집의의 유무 또는 특징은 아파트 가격 변화에 영향이 없다 
# 대립가설 : 어린이집의의 유무 또는 특징은 아파트 가격 변화에 영향이 있다 

# 가설2 
# 귀무가설 : 공원의 유무 또는 특징은 아파트 가격 변화에 영향이 없다  
# 대립가설 : 공원의 유무 또는 특징은 아파트 가격 변화에 영향이 있다  

# 가설3 
# 귀무가설 : 어린이집 & 공원의 유무 또는 특징은 아파트 가격 변화에 영향이 없다 
# 대립가설 : 어린이집 & 공원의 유무 또는 특징은 아파트 가격 변화에 영향이 있다
# 가설 검증을 위한 데이터 수집 : 
# 모델링에 활용할 train 데이터 
pth = '/kaggle/input/seoul-busan-house-price-compatition-dacon/'
train = pd.read_csv(pth+'train.csv')
# 어린이집 정보 
dcc = pd.read_csv(pth+'day_care_center.csv')
# 공원 정보 
park = pd.read_csv(pth+'park.csv')

# 에측할 2017년의 데이터
test = pd.read_csv(pth+'test.csv')
submission = pd.read_csv(pth+'submission.csv')
# train 데이터 정제 
# train 데이터에서 손봐야할 곳은
# 1. column 명을 다루기 쉽게 바꾸기  
# 2. 시와 동의 정보는 있지만, 구의 정보가 없어서 이를 채워 넣을 방법을 찾아야 함 
# 3. 아파트의 면적을 평 단위로 바꾸기  
# 4. 거래년월 정보를 날짜 타입으로 바꾸기(지금은 숫자 타입)
# 5. 거래일은 상,중,하순으로 나뉘어져 있는데, 하순을 하나로 통일하기 
# ==========
train = pd.read_csv(pth+'train.csv')
# 1. columns명을 다루기 쉽게 바꾸기 
train.rename(columns={'apartment_id':'aptid',
                      'exclusive_use_area':'area',
                     'year_of_completion':'yearComp',
                     'transaction_year_month':'months',
                     'transaction_date':'dates',
                     'transaction_real_price':'price'},inplace=True)

# 2. 아파트의 id 마다 구에 대한 정보를 추가
apt = train[['aptid','city','dong']].drop_duplicates().reset_index(drop=True).copy() 
dist = apt[['city','dong']].drop_duplicates().reset_index(drop=True).copy()
gu_dong = pd.read_excel(pth+'seoul_busan_gu_dong.xlsx')
gu_dong.columns = ['city','gu','dong2','dong']
gu_dong = gu_dong.drop(columns=['dong2']).drop_duplicates()
for i in dist.index:
    dong = dist.loc[i,'dong'].split(' ')[0]
    gu = gu_dong.loc[gu_dong['dong'] == dong,'gu'].values[0]
    dist.loc[i,'gu'] = gu
apt = pd.merge(apt,dist,on=['city','dong'])
train = pd.merge(train,apt,on=['aptid','city','dong'])

# 3. 아파트의 면적을 평 단위로 바꾸기  
train['area']=round(train['area']/3.3,1)

# 4. 거래 정보를 날짜 타입으로 바꾸기 
train['months']=pd.to_datetime(train['months'],format='%Y%m').dt.strftime('%Y-%m')

# 5. 거래일은 상, 중, 하순으로 나뉘어져 있는데, 하순을 하나로 통일하기, 각각 상순 = 1, 중순 = 2, 하순 = 3으로 변경 
train['dates'] = train['dates'].apply(lambda x:1 if x == '1~10' else(
                                               2 if x == '11~20' else 
                                               3 ))
# 국공립 어린이집을 분류하는 기준 찾기 
care = dcc.groupby(['city','gu']).agg({'day_care_baby_num':np.mean,'teacher_num':np.mean})
care.fillna(care['teacher_num'].mean(),inplace=True)
care['teacher_ratio'] = care['day_care_baby_num']/care['teacher_num']
care = care.astype(int).copy() 
care = care.reset_index().drop(columns=['day_care_baby_num','teacher_num'])
print(care['teacher_ratio'].describe())

care2 = dcc.groupby(['city','gu','day_care_type']).agg({'day_care_name':'count'}).\
pivot_table(index=['city','gu'],columns='day_care_type')['day_care_name'].fillna(0)
care2['nat_ratio']=care2['국공립']/care2.sum(axis=1)*100
care2['private_ratio']=care2['민간']/care2.sum(axis=1)*100
care = care2[['nat_ratio','private_ratio']].astype(int).reset_index().copy() 
care.head()
# EDA를 통해 가설1을 조금 더 수정 
# 가설1 
# 귀무가설 : 국공립 어린이집의 비중에 따라 아파트 가격 상승률은 차이가 없다
# 대립가설 : 국공립 어린이집의 비중에 따라 아파트 가격 상승률은 차이가 있다
train2 = pd.merge(train,care,on=['city','gu'],how='outer').fillna(0)
train2['daycareratio']=train2['nat_ratio'].apply(lambda x : 0 if x >= 0 and x < 10 else(
                                                            1 if x >= 10 and x < 20 else(
                                                            2 if x >= 20 and x < 30 else(
                                                            3 if x >= 30 and x < 40 else(
                                                            4)))))

daycare_price = train2.groupby(['daycareratio','months']).agg({'price':np.mean}).\
                pivot_table(index='months',columns='daycareratio')['price']
daycare_price
(daycare_price.pct_change(periods=12)*100).mean()
sample_result = [] 
for i in range(400):
    df_sample = train2.sample(n=100000,replace=True).copy() 
    daycare_price = df_sample.groupby(['daycareratio','months']).agg({'price':np.mean}).\
                    pivot_table(index='months',columns='daycareratio')['price']
    sample_result.append((daycare_price.pct_change(periods=12)*100).mean())
pd.DataFrame(sample_result).hist()
daycare_price = train2.groupby(['daycareratio','months','dates']).agg({'price':np.mean}).\
                pivot_table(index=['months','dates'],columns='daycareratio')['price']
(daycare_price.pct_change(periods=36)*100).mean()
sample_result = [] 
for i in range(400):
    df_sample = train2.sample(n=100000,replace=True).copy() 
    daycare_price = df_sample.groupby(['daycareratio','months','dates']).agg({'price':np.mean}).\
                    pivot_table(index=['months','dates'],columns='daycareratio')['price']
    sample_result.append((daycare_price.pct_change(periods=36)*100).mean())
pd.DataFrame(sample_result).hist()
# 가설을 검증하기 위한 간단한 t-test
import numpy as np
from scipy import stats
sample = pd.DataFrame(sample_result)
tTestResult = stats.ttest_ind(sample.loc[:,1],sample.loc[:,2],equal_var=False)
tTestResult
# LinearRegression 활용  
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train2["city"] = le.fit_transform(train2["city"])
le = LabelEncoder()
train2["gu"] = le.fit_transform(train2["gu"])
le = LabelEncoder()
train2["dong"] = le.fit_transform(train2["dong"])

train2['Year'] = pd.to_datetime(train2['months']).dt.year
train2['Months'] = pd.to_datetime(train2['months']).dt.month

train3 = train2[['aptid','city','gu','dong','area','Year','Months','dates','floor','nat_ratio','price']].copy()

from sklearn.model_selection import train_test_split
X = train3.drop(columns=['price'])
y = train3['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True, n_jobs=None)
lm.fit(X_train, y_train)
accuracy = lm.score(X_test, y_test)
print("Linear Regression test file accuracy:"+str(accuracy))
residuals = y_test-lm.predict(X_test)
residuals.describe()
# 적합도 검증 - 결정계수
SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared)
from sklearn.metrics import mean_squared_error
print('score = ', accuracy)
print('Mean_Squared_Error = ', mean_squared_error(lm.predict(X_test), y_test))
print('RMSE = ', mean_squared_error(lm.predict(X_test), y_test)**0.5)
# test 데이터 정제 
def edit_df(train,care):
    # 1. columns명을 다루기 쉽게 바꾸기 
    train.rename(columns={'apartment_id':'aptid',
                          'exclusive_use_area':'area',
                         'year_of_completion':'yearComp',
                         'transaction_year_month':'months',
                         'transaction_date':'dates',
                         'transaction_real_price':'price'},inplace=True)

    # 2. 아파트의 id 마다 구에 대한 정보를 추가
    apt = train[['aptid','city','dong']].drop_duplicates().reset_index(drop=True).copy() 
    dist = apt[['city','dong']].drop_duplicates().reset_index(drop=True).copy()
    gu_dong = pd.read_excel(pth+'seoul_busan_gu_dong.xlsx')
    gu_dong.columns = ['city','gu','dong2','dong']
    gu_dong = gu_dong.drop(columns=['dong2']).drop_duplicates()
    for i in dist.index:
        dong = dist.loc[i,'dong'].split(' ')[0]
        gu = gu_dong.loc[gu_dong['dong'] == dong,'gu'].values[0]
        dist.loc[i,'gu'] = gu
    apt = pd.merge(apt,dist,on=['city','dong'])
    train = pd.merge(train,apt,on=['aptid','city','dong'])

    # 3. 아파트의 면적을 평 단위로 바꾸기  
    train['area']=round(train['area']/3.3,1)

    # 4. 거래년열 정보를 날짜 타입으로 바꾸기 
    train['months']=pd.to_datetime(train['months'],format='%Y%m').dt.strftime('%Y-%m')

    # 5. 거래일은 상, 중, 하순으로 나뉘어져 있는데, 하순을 하나로 통일하기, 각각 상순 = 1, 중순 = 2, 하순 = 3으로 변경 
    train['dates'] = train['dates'].apply(lambda x:1 if x == '1~10' else(
                                                   2 if x == '11~20' else 
                                                   3 ))
    train2 = pd.merge(train,care,on=['city','gu'],how='outer').fillna(0)
    train2['daycareratio']=train2['nat_ratio'].apply(lambda x : 0 if x >= 0 and x < 10 else(
                                                            1 if x >= 10 and x < 20 else(
                                                            2 if x >= 20 and x < 30 else(
                                                            3 if x >= 30 and x < 40 else(
                                                            4)))))
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train2["city"] = le.fit_transform(train2["city"])
    le = LabelEncoder()
    train2["gu"] = le.fit_transform(train2["gu"])
    le = LabelEncoder()
    train2["dong"] = le.fit_transform(train2["dong"])
    train2['Year'] = pd.to_datetime(train2['months']).dt.year
    train2['Months'] = pd.to_datetime(train2['months']).dt.month
    train3 = train2[['aptid','city','gu','dong','area','Year','Months','dates','floor','nat_ratio']].copy()
    return train3
test2 = edit_df(test,care)
test2
result = test2.copy()
result['price'] = lm.predict(test2) 
result
result['daycareratio']=result['nat_ratio'].apply(lambda x : 0 if x >= 0 and x < 10 else(
                                                            1 if x >= 10 and x < 20 else(
                                                            2 if x >= 20 and x < 30 else(
                                                            3 if x >= 30 and x < 40 else(
                                                            4)))))
daycare_price = result.groupby(['daycareratio','Year','Months','dates']).agg({'price':np.mean}).\
                        pivot_table(index=['Year','Months','dates'],columns='daycareratio')['price']
daycare_price.plot()
# randomForest 활용
from sklearn.model_selection import train_test_split
X = train3.drop(columns=['price'])
y = train3['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)
print("test file accuracy:"+str(accuracy))
residuals = y_test-rf.predict(X_test)
print(residuals.describe())
# 적합도 검증 - 결정계수
SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared)
from sklearn.metrics import mean_squared_error
print('score = ', accuracy)
print('Mean_Squared_Error = ', mean_squared_error(rf.predict(X_test), y_test))
print('RMSE = ', mean_squared_error(rf.predict(X_test), y_test)**0.5)
result = test2.copy()
result['price'] = rf.predict(test2) 
result
result['daycareratio']=result['nat_ratio'].apply(lambda x : 0 if x >= 0 and x < 10 else(
                                                            1 if x >= 10 and x < 20 else(
                                                            2 if x >= 20 and x < 30 else(
                                                            3 if x >= 30 and x < 40 else(
                                                            4)))))
daycare_price = result.groupby(['daycareratio','Year','Months','dates']).agg({'price':np.mean}).\
                        pivot_table(index=['Year','Months','dates'],columns='daycareratio')['price']
daycare_price.plot()
# 그라디언트 부스팅 
residuals = y_test-rf.predict(X_test)
print(residuals.describe())
# 적합도 검증 - 결정계수
SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared)
from sklearn.metrics import mean_squared_error
print('score = ', accuracy)
print('Mean_Squared_Error = ', mean_squared_error(rf.predict(X_test), y_test))
print('RMSE = ', mean_squared_error(rf.predict(X_test), y_test)**0.5)
residuals = y_test-rf.predict(X_test)
print(residuals.describe())
# 적합도 검증 - 결정계수
SSE = (residuals**2).sum()
SST = ((y-y.mean())**2).sum()
R_squared = 1 - (SSE/SST)
print('R_squared = ', R_squared)
from sklearn.metrics import mean_squared_error
print('score = ', accuracy)
print('Mean_Squared_Error = ', mean_squared_error(rf.predict(X_test), y_test))
print('RMSE = ', mean_squared_error(rf.predict(X_test), y_test)**0.5)
result = test2.copy()
result['price'] = rf.predict(test2) 
result
result['daycareratio']=result['nat_ratio'].apply(lambda x : 0 if x >= 0 and x < 10 else(
                                                            1 if x >= 10 and x < 20 else(
                                                            2 if x >= 20 and x < 30 else(
                                                            3 if x >= 30 and x < 40 else(
                                                            4)))))
daycare_price = result.groupby(['daycareratio','Year','Months','dates']).agg({'price':np.mean}).\
                        pivot_table(index=['Year','Months','dates'],columns='daycareratio')['price']
daycare_price.plot()