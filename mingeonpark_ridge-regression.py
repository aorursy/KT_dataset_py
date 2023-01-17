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
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
cost_df = pd.read_csv('/kaggle/input/manufacturing-cost/EconomiesOfScale.csv')
print(cost_df.shape)
cost_df.head()
cost_df.info()
cost_df['Number of Units'].describe()
plt.figure(figsize=(8,4))
plt.xticks(range(0,20,1))
sns.distplot(cost_df['Number of Units'])
cost_df['Manufacturing Cost'].describe()
plt.figure(figsize=(8,4))
plt.xticks(range(0,120,10))
sns.distplot(cost_df['Manufacturing Cost'])
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
X_features = cost_df['Number of Units']
Y_target = cost_df['Manufacturing Cost']

print(type(X_features))
print(type(Y_target))
X_train,X_test,Y_train,Y_test = train_test_split(X_features,Y_target,test_size=0.2,random_state=156)
print(type(X_train))
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
lr_reg=LinearRegression()

#선형회귀의 fit 메서드는 X_train으로 2차원 array를 기대함 -> 칼럼이 하나 뿐이어서 Series로 되었기 때문에 DataFrame으로 변경함
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

lr_reg.fit(X_train,Y_train)
print('회귀 계수 값 : ',np.round(lr_reg.coef_,2))
print('절편 값 : ',np.round(lr_reg.intercept_,2))
pred = lr_reg.predict(X_test)
mse = mean_squared_error(Y_test,pred)
rmse = np.sqrt(mse)

print("rmse : ",round(rmse,3))
from sklearn.model_selection import cross_val_score
cost_df1 = cost_df.copy()

Y_target = cost_df1['Manufacturing Cost']
X_features = cost_df1['Number of Units']

lr=LinearRegression()

print(Y_target.isnull().sum()) #Y_target의 결측치(NaN) 없음
print(X_features.isnull().sum()) #X_features의 결측치(NaN) 없음

cost_df1 = cost_df1.astype('float')

cost_df1.dtypes

#cross_val_score를 사용하려면 X_features가 데이터 프레임이어야 함.
X_features = pd.DataFrame(X_features)
print(type(X_features))
neg_mse_scores = cross_val_score(lr,X_features,Y_target,scoring="neg_mean_squared_error",cv=5)
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print('5개 fold의 평균 RMSE:',round(avg_rmse,3))
print(X_features.shape)
print(type(X_features))
print(Y_target.shape)
print(type(Y_target))
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=10)

neg_mse_scores = cross_val_score(ridge,X_features,Y_target,scoring="neg_mean_squared_error",cv=5)
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print('5개 fold의 평균 RMSE:',round(avg_rmse,3))
alphas = [0,0.1,1,10,20,50,80,100,200,300,400,500,600,700,800]

for alpha in alphas:
  ridge = Ridge(alpha=alpha)

  neg_mse_scores = cross_val_score(ridge,X_features,Y_target,scoring="neg_mean_squared_error",cv=5)
  rmse_scores = np.sqrt(-1*neg_mse_scores)
  avg_rmse = np.mean(rmse_scores)

  print('alpha가 {0}일 때 5개 fold의 평균 RMSE: {1:.3f}'.format(alpha,avg_rmse))
  #alpha가 400 정도일 때, 7.448로 가장 좋은 점수를 보임, 그 이후로는 다시 성능 저하
X_features_log = np.log1p(X_features)
Y_target_log = np.log1p(Y_target)
print(type(X_features_log))
print(type(Y_target_log))
alphas = [0,0.1,1,10,20,50,80,100,200,300,400,500,600,700,800]

for alpha in alphas:
  ridge = Ridge(alpha=alpha)

  neg_mse_scores = cross_val_score(ridge,X_features_log,Y_target_log,scoring="neg_mean_squared_error",cv=5)
  rmse_scores = np.sqrt(-1*neg_mse_scores)
  avg_rmse = np.mean(rmse_scores)

  print('alpha가 {0}일 때 5개 fold의 평균 RMSE: {1:.3f}'.format(alpha,avg_rmse))
  #alpha가 0 일때 => 아예 규제가 없을때 , 0.139의 성능을 보임 -> 아주 성능이 향상됨 (8.05 -> 7.448 -> 0.139)
cost_df_outlier_delete = cost_df.copy()
cost_df_outlier_delete.info()
sns.distplot(cost_df_outlier_delete['Number of Units'])
quantile_25 = np.percentile(cost_df_outlier_delete['Number of Units'],25)
quantile_75 = np.percentile(cost_df_outlier_delete['Number of Units'],75)

print("1/4분위 수 : ",quantile_25)
print("3/4분위 수 : ",quantile_75)

iqr = quantile_75 - quantile_25
iqr_weight = iqr * 1.5 # weight는 1.5

print("iqr_weight : ",iqr_weight)

lowest_val = quantile_25 - iqr_weight
highest_val = quantile_75 + iqr_weight

outlier_index = cost_df_outlier_delete[(cost_df_outlier_delete['Number of Units'] < lowest_val) | (cost_df_outlier_delete['Number of Units'] > highest_val)].index

print(outlier_index)
print(cost_df_outlier_delete['Number of Units'][991:1000])
cost_df_outlier_delete.drop(outlier_index,axis=0,inplace=True)
cost_df_outlier_delete.shape
X_features = cost_df_outlier_delete['Number of Units']
Y_target = cost_df_outlier_delete['Manufacturing Cost']

X_features = pd.DataFrame(X_features)

print(X_features.shape)
print(Y_target.shape)

print(type(X_features))
print(type(Y_target))
alphas = [0,0.1,1,10,20,50,80,100,200,300,400,500,600,700,800]

for alpha in alphas:
  ridge = Ridge(alpha=alpha)

  neg_mse_scores = cross_val_score(ridge,X_features_log,Y_target_log,scoring="neg_mean_squared_error",cv=5)
  rmse_scores = np.sqrt(-1*neg_mse_scores)
  avg_rmse = np.mean(rmse_scores)

  print('alpha가 {0}일 때 5개 fold의 평균 RMSE: {1:.3f}'.format(alpha,avg_rmse))
  #alpha가 0 일때 => 아예 규제가 없을때 , 0.139의 성능을 보임 -> 아주 성능이 향상됨 (8.05 -> 7.448 -> 0.139)
