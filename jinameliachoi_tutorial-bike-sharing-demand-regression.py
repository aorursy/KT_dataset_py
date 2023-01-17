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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)



data = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

print(data.shape)

data.head(3)
data.info()
# datatime 칼럼의 경우 가공이 필요

# 년, 월, 일, 시간과 같이 4가지 속성으로 분리

# 일단, dtype을 datetime으로 변경



data['datetime'] = data.datetime.apply(pd.to_datetime)



# 4가지 추출

data['year'] = data.datetime.apply(lambda x: x.year)

data['month'] = data.datetime.apply(lambda x: x.month)

data['day'] = data.datetime.apply(lambda x: x.day)

data['hour'] = data.datetime.apply(lambda x: x.hour)

data.head(3)
# `casual` 칼럼은 사전에 등록하지 않은 사용자의 자전거 대여 횟수

# `registered` 칼럼은 사전에 등록한 사용자의 자전거 대여 횟수

# 두 칼럼을 더해진 것이 `count`이기 때문에 제거 



drop_columns = ['datetime', 'casual', 'registered']

data.drop(drop_columns, axis=1, inplace=True)
# 이번 대회에서 요구한 성능 평가 방법은 RMSLE

from sklearn.metrics import mean_squared_error, mean_absolute_error



# log 값 변환 시 NaN 등의 이슈로 log()가 아닌 log1p()를 이용해 RMSLE 계산

def rmsle(y, pred):

    log_y = np.log1p(y)

    log_pred = np.log1p(pred)

    squared_error = (log_y - log_pred) ** 2

    rmsle = np.sqrt(np.mean(squared_error))

    return rmsle



# sklearn의 mean_squared_error 이용해 RMSE 계산

def rmse(y, pred):

    return np.sqrt(mean_squared_error(y, pred))



# MSE, RMSE, RMSLE 모두 계산

def evaluate_regr(y, pred):

    rmsle_val = rmsle(y, pred)

    rmse_val = rmse(y, pred)

    mae_val = mean_absolute_error(y, pred)

    print('RMSLE: {0:.3f}, RMSE: {1:.3f}, MAE: {2:.3f}'.format(rmsle_val, rmse_val, mae_val))
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso



y_target = data['count']

x_features = data.drop(['count'], axis=1, inplace=False)



x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size = .3, random_state=0)



lr = LinearRegression()

lr.fit(x_train, y_train)

pred = lr.predict(x_test)



evaluate_regr(y_test, pred)
# 예측 오류로 비교적 큰 값이 나옴

# 실제 값과 예측 값이 어느 정도 차이나는 지 dataframe 칼럼으로 만들어 오류 값이 가장 큰 순으로 5개만 확인

def get_top_error_data(y_test, pred, n_tops=5):

    # dataframe의 칼럼으로 실제 대여 횟수와 예측값을 서로 비교할 수 있도록 생성

    result_df = pd.DataFrame(y_test.values,

                            columns = ['real_count'])

    result_df['predicted_count'] = np.round(pred)

    result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count'])

    

    # 예측값과 실제 값이 가장 큰 데이터 순으로 출력

    print(result_df.sort_values('diff', ascending=False)[:n_tops])

    

get_top_error_data(y_test, pred, n_tops=5)
# 예측 값과 실제 값의 격차가 클 때 살펴볼 것,

# 1) target 변수의 분포 확인

# 2) feature들의 분포 확인



y_target.hist()
# 0~200 사이에 왜곡되어 있는 값. 스케일링 필요 

y_log_transform = np.log1p(y_target)

y_log_transform.hist()
# 이를 적용하여 다시 학습 후 평가 필요 

y_target_log = np.log1p(y_target)



x_train, x_test, y_train, y_test = train_test_split(x_features, y_target_log,

                                                   test_size = .3,

                                                   random_state = 0)



lr = LinearRegression()

lr.fit(x_train, y_train)

pred = lr.predict(x_test)



# 로그 변환 된건 다시 expm1 이용

y_test_exp = np.expm1(y_test)

pred_exp = np.expm1(pred)



evaluate_regr(y_test_exp, pred_exp)
# RMSLE 오류는 줄었지만 RMSE는 늘었다. 이유는?

coef = pd.Series(lr.coef_,

                index = x_features.columns)

coef_sort = coef.sort_values(ascending=False)

sns.barplot(x = coef_sort.values,

           y = coef_sort.index)
# `year` 피처의 회귀 계수 값이 독보적으로 큰 값을 가지고 있음

# 이 변수는 카테고리 변수인데 원핫인코딩을 하지 않아서 발생

# 카테고리 변수들에 원핫인코딩 적용



x_features_ohe = pd.get_dummies(x_features,

                               columns = ['year', 'month', 'day', 'hour',

                                         'holiday', 'workingday', 'season', 'weather'])
# 원핫인코딩 적용

x_train, x_test, y_train, y_test = train_test_split(x_features_ohe, y_target_log,

                                                   test_size=.3, random_state = 0)



def get_model_predict(model, x_train, x_test, y_train, y_test, is_expm1=False):

    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    if is_expm1:

        y_test = np.expm1(y_test)

        pred = np.expm1(pred)

    print('###', model.__class__.__name__, '###')

    evaluate_regr(y_test, pred)



# 모델별로 평가 수행

lr = LinearRegression()

ridge = Ridge()

lasso = Lasso()



for model in [lr, ridge, lasso]:

    get_model_predict(model, x_train, x_test, y_train, y_test, is_expm1=True)
coef = pd.Series(lr.coef_, index=x_features_ohe.columns)

coef_sort = coef.sort_values(ascending=False)[:20]

sns.barplot(x=coef_sort.values,

           y=coef_sort.index)
# 다른 모델 이용

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



rf = RandomForestRegressor(n_estimators=500)

gbm = GradientBoostingRegressor(n_estimators=500)

xgb = XGBRegressor(n_estimators=500)

lgbm = LGBMRegressor(n_estimators=500)



for model in [rf, gbm, xgb, lgbm]:

    get_model_predict(model, x_train.values, x_test.values, y_train.values, y_test.values,

                     is_expm1=True)