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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import preprocessing
warnings.filterwarnings("ignore")
%matplotlib inline

train_df = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')
test_df = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
print(train_df.shape, test_df.shape)
train_df.head()
train_df.info()
# null값이 많은 데이터나 text위주의 데이터는 제외했다. 하지만, belongs_to_collection과 같은 피처는 revenue를 결정하는 데 유의미할 것 같아 drop하지 않았다.
train_df = train_df[['belongs_to_collection', 'budget', 'genres', 'original_language', 'popularity', 'release_date', 'runtime', 'spoken_languages', 'cast', 'crew', 'revenue']]
train_df.head()
isnull_series = train_df.isnull().sum()
print('\n ### Null 칼럼과 그 건수 ### \n', isnull_series[isnull_series>0].sort_values(ascending=False))
# fillna를 이용해서 null을 0으로 바꿔준 후,
# lambda 식을 통해 시리즈가 제작되지 않은 경우 0, 제작된 경우 1로 데이터를 변환했다.
train_df['belongs_to_collection'] = train_df['belongs_to_collection'].fillna(0)
train_df['belongs_to_collection'] = train_df['belongs_to_collection'].apply(lambda x : 0 if x == 0 else 1)
train_df['belongs_to_collection'].value_counts()
train_df['budget'].hist()
train_df['original_language'] = train_df['original_language'].apply(lambda x : 1 if x == "en" else 0)
train_df['original_language'].value_counts()
train_df['popularity'].hist()
train_df['release_date'].head()
def date(x):
    x=str(x)
    year=x.split('/')[2]
    if int(year)<20:
        return x[:-2]+'20'+year
    else:
        return x[:-2]+'19'+year
train_df['release_date']=train_df['release_date'].apply(lambda x: date(x))
train_df['release_date']=train_df['release_date'].apply(lambda x : pd.datetime.strptime(x, '%m/%d/%Y'))
train_df['release_year']=train_df['release_date'].apply(lambda x : x.year)
train_df['release_month']=train_df['release_date'].apply(lambda x : x.month)
train_df['release_day']=train_df['release_date'].apply(lambda x : x.day)
train_df.head()
train_df['runtime'].hist()
train_df['spoken_languages'] = train_df['spoken_languages'].astype(str)
train_df['spoken_languages'] = train_df['spoken_languages'].apply(lambda x:x.count("name"))
train_df['spoken_languages'].value_counts()
train_df['spoken_languages'][train_df['spoken_languages']==0]=1
train_df['spoken_languages'].value_counts()
train_df['cast'] = train_df['cast'].astype(str)
train_df['cast'] = train_df['cast'].apply(lambda x:x.count("cast_id"))
train_df['cast'].value_counts()
train_df['crew'] = train_df['crew'].astype(str)
train_df['crew'] = train_df['crew'].apply(lambda x:x.count("credit_id"))
train_df['crew'].value_counts()
train_df['genres'] = train_df['genres'].astype(str)
train_df['genres'] = train_df['genres'].apply(lambda x:x.count("name"))
train_df['genres'].value_counts()
train_df['revenue'].hist()
# 최종 train_df
train_df.head()
train_df.info()
#runtime 값이 null인 데이터를 발견해 평균값으로 대체
train_df = train_df.drop('release_date', axis = 1)
train_df['runtime'] = train_df['runtime'].fillna(train_df['runtime'].mean())
train_df.head()
print('\n ### 학습 데이터 정보 ### \n')
print(train_df.info())
print('\n ### 데이터 세트의 Shape ### \n:', train_df.shape)
print('\n ### 전체 피처의 type ### \n', train_df.dtypes.value_counts())
isnull_series = train_df.isnull().sum()
print('\n ### Null 칼럼과 그 건수 ### \n', isnull_series[isnull_series>0].sort_values(ascending=False))
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    mae_val = mean_absolute_error(y, pred)
    print('RMSLE:{0:.3f}, RMSE:{1:.3F}, MAE:{2:.3F}'.format(rmsle_val, rmse_val, mae_val))
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def get_scaled_data(method='None', input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_transform(input_data)
    elif method == 'MinMax':
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == 'Log' :
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data
                
    return scaled_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target = train_df['revenue']
X_data = train_df.drop('revenue', axis=1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.5)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

evaluate_regr(y_test, pred)
y_target.hist()
y_target = get_scaled_data(method = 'Log', input_data = y_target)
y_target.hist()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target,
                                                    test_size = 0.5, random_state = 156)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

evaluate_regr(y_test, pred)
X_data = get_scaled_data(method = 'Standard', input_data = X_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target,
                                                    test_size = 0.5, random_state = 156)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

evaluate_regr(y_test, pred)
X_data = get_scaled_data(method = 'MinMax', input_data = X_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size = 0.5,
                                                   random_state = 156)

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

evaluate_regr(y_test, pred)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y_target = train_df['revenue']
y_target = np.log1p(y_target)
X_data = train_df.drop('revenue', axis=1, inplace = False)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target,
                                                    test_size = 0.5, random_state = 156)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_preds = lr.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)

print('MSE : {0:.3f}, RMSE : {1:.3F}'.format(mse, rmse))
print('Variance score : {0:.3f}'.format(r2_score(y_test, y_preds)))
col_names = ['belongs_to_collection', 'budget', 'genres', 'original_language', 'popularity', 'runtime', 'spoken_languages', 'cast', 'crew', 'release_year', 'release_month', 'release_day']
fig, axs = plt.subplots(figsize=(16,8), ncols = 4, nrows = 3)
for i, feature in enumerate(col_names):
    row = int(i/4)
    col = i%4
    
    sns.regplot(x=feature, y='revenue', data=train_df, ax=axs[row][col])
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 1000)
neg_mse_scores = cross_val_score(rf_reg, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(' 5 교차 검증의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 교차 검증의 개별 RMSE scores: ', np.round(rmse_scores, 2))
print(' 5 교차 검증의 평균 RMSE: {0:.3f}'.format(avg_rmse))
def get_model_cv_prediction(model, X_data, y_target):
    neg_mse_scores=cross_val_score(model, X_data, y_target, scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-1*neg_mse_scores)
    avg_rmse = np.mean(rmse_scores)
    print('#### ', model.__class__.__name__, ' ####')
    print(' 5 교차 검증의 평균 RMSE: {0:.3f}'.format(avg_rmse))
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

dt_reg = DecisionTreeRegressor(random_state=0, max_depth=4)
rf_reg = RandomForestRegressor(random_state=0, n_estimators=1000)
gb_reg = GradientBoostingRegressor(random_state=0, n_estimators=1000)
xgb_reg = XGBRegressor(n_estimators=1000)
lgb_reg = LGBMRegressor(n_estimators=1000)

models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg]
for model in models:
    get_model_cv_prediction(model, X_data, y_target)
rf_reg = RandomForestRegressor(n_estimators = 1000)

rf_reg.fit(X_data, y_target)

feature_series = pd.Series(data=rf_reg.feature_importances_, index = X_data.columns)
feature_series = feature_series.sort_values(ascending=False)
sns.barplot(x=feature_series, y=feature_series.index)
