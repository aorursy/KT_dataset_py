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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



data_org = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data = data_org.copy()

data.head(3)
# target 변수는 마지막 칼럼 `SalePrice`

# 데이터 정보 요약

print('데이터 세트의 Shape: ', data.shape)

print('\n전체 피처의 타입 \n', data.dtypes.value_counts())

isnull_series = data.isnull().sum()

print('\nNull 칼럼과 그 건수: \n', isnull_series[isnull_series > 0].sort_values(ascending=False))
# target 변수 분포 확인하기 

plt.title('Original Sale Price Histogram')

sns.distplot(data['SalePrice'])
# log transformation

plt.title('Log Transformation Sale Price Histogram')

log_SalePrice = np.log1p(data['SalePrice'])

sns.distplot(log_SalePrice)
# null 값이 많은 피처 삭제 

# 단순식별자 `id` 삭제 

# 나머지 Null 피처는 숫자형의 경우 평균값으로 대체



# SalePrice log transformation

original_SalePrice = data['SalePrice']

data['SalePrice'] = np.log1p(data['SalePrice'])



# null이 너무 많은 칼럼과 불필요한 칼럼 삭제 

data.drop(["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "Id"], axis=1, inplace=True)



# drop하지 않는 숫자형 null 칼럼은 평균값으로 대체 

data.fillna(data.mean(), inplace=True)
# null 값이 있는 피처명과 타입 추출

null_column_count = data.isnull().sum()[data.isnull().sum() > 0]

print('## Null 피처의 type :\n', data.dtypes[null_column_count.index])
# 이제 null 값은 문자형 피처에만 존재 

# 문자형은 원핫인코딩을 하면서 null값 대체 별도로 필요 없음(none으로 처리하기 때문에)

print('get_dummies() 수행 전 데이터 shape: ', data.shape)

data_ohe = pd.get_dummies(data)

print('get_dummies() 수행 후 데이터 shape: ', data_ohe.shape)



null_column_count = data_ohe.isnull().sum()[data_ohe.isnull().sum() > 0]

print('## Null 피처의 type :\n', data_ohe.dtypes[null_column_count.index])
def get_rmse(model):

    pred = model.predict(x_test)

    mse = mean_squared_error(y_test, pred)

    rmse = np.sqrt(mse)

    print(model.__class__.__name__, '로그 변환된 RMSE:', np.round(rmse, 3))

    return rmse



def get_rmses(models):

    rmses = []

    for model in models:

        rmse = get_rmse(model)

        rmses.append(rmse)

    return rmses
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



y_target = data_ohe['SalePrice']

x_features = data_ohe.drop('SalePrice', axis=1, inplace=False)

x_train, x_test, y_train, y_test = train_test_split(x_features, y_target,

                                                   test_size = .3,

                                                   random_state = 0)



# linear regression, ridge, lasso 실행하기

lr = LinearRegression()

ridge = Ridge()

lasso = Lasso()



lr.fit(x_train, y_train)

ridge.fit(x_train, y_train)

lasso.fit(x_train, y_train)



models = [lr, ridge, lasso]

get_rmses(models)
# 회귀 계수 시각화. 모델별로 어떠한 피처 회귀 계수로 구성되는 지 확인

# 피처가 많으니 상위 10개, 하위 10개만 살펴보기 

def get_top_bottom_coef(model, n=10):

    coef = pd.Series(model.coef_, index=x_features.columns)

    

    coef_high = coef.sort_values(ascending=False).head(n)

    coef_low = coef.sort_values(ascending=False).tail(n)

    return coef_high, coef_low
def visualize_coefficient(models):

    fig, axs = plt.subplots(figsize=(24, 10), nrows=1, ncols=3)

    fig.tight_layout()

    

    # 입력 인자로 받은 list 객체인 models에서 차례로 model을 추출해 회귀 계수 시각화 

    for i_num, model in enumerate(models):

        # 상위 10개, 하위 10개 회귀 계수 구하고 concat으로 결합

        coef_high, coef_low = get_top_bottom_coef(model)

        coef_concat = pd.concat([coef_high, coef_low])

        # ax subplot에 barchar로 표현. 한 화면에 표현하기 위해 tick label 위치와 font 조정

        axs[i_num].set_title(model.__class__.__name__+ ' Coefficients', size=25)

        axs[i_num].tick_params(axis='y', direction='in', pad=-120)

        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):

            label.set_fontsize(22)

        sns.barplot(x=coef_concat.values,

                   y=coef_concat.index, ax=axs[i_num])

        

# 앞 예제에서 학습한 모델의 회귀 계수 시각화 

models = [lr, ridge, lasso]

visualize_coefficient(models)
from sklearn.model_selection import cross_val_score



def get_avg_rmse_cv(models):

    

    for model in models:

        # 분할하지 않고 전체 데이터로 cross_val_score() 수행.

        # 모델별 CV RMSE값과 평균 RMSE 출력

        rmse_list = np.sqrt(-cross_val_score(model, x_features, y_target,

                                            scoring = 'neg_mean_squared_error',

                                            cv = 5))

        rmse_avg = np.mean(rmse_list)

        print('\n{0} CV RMSE 값 리스트: {1}'.format(model.__class__.__name__, np.round(rmse_list, 3)))

        print('\n{0} CV 평균 RMSE 값: {1}'.format(model.__class__.__name__, np.round(rmse_avg, 3)))

        

# 앞 예제에서 학습한 lr, ridge, lasso 모델의 CV RMSE 값 출력

models = [lr, ridge, lasso]

get_avg_rmse_cv(models)
# lasso의 경우, 처음 회귀 계수 형태도 비이상적이며 ols나 릿지보다 폴드 세트 학습도 성능이 떨어진다

# ridge와 lasso 모델에 대해 alpha 하이퍼 파라미터를 변화시키며 최적의 값 도출해보자

from sklearn.model_selection import GridSearchCV



def print_best_params(model, params):

    grid_model = GridSearchCV(model, param_grid=params,

                             scoring='neg_mean_squared_error', cv=5)

    grid_model.fit(x_features, y_target)

    rmse = np.sqrt(-1 * grid_model.best_score_)

    print('{0} 5 CV 시 최적 평균 RMSE 값: {1}, 최적 alpha: {2}'.format(model.__class__.__name__,

                                                              np.round(rmse, 4), grid_model.best_params_))

    

ridge_params = {'alpha' : [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}

lasso_params = {'alpha' : [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]}

print_best_params(ridge, ridge_params)

print_best_params(lasso, lasso_params)
# 앞의 최적화 alpha 값으로 학습 데이터로 학습, 테스트 데이터로 예측 및 평가 수행

lr = LinearRegression()

ridge = Ridge(alpha=12)

lasso = Lasso(alpha=0.001)



lr.fit(x_train, y_train)

ridge.fit(x_train, y_train)

lasso.fit(x_train, y_train)



# 모든 모델의 RMSE 출력

models = [lr, ridge, lasso]

get_rmses(models)



# 모든 모델의 회귀 계수 시각화

visualize_coefficient(models)
# 데이터 세트 추가적으로 가공하여 모델 튜닝하기 

# 1) 피처 데이터 세트의 데이터 분포도 

# 2) 이상치 확인



from scipy.stats import skew



# object가 아닌 숫자형 피처의 칼럼 index 객체 추출

features_index = data.dtypes[data.dtypes != 'object'].index

# data에 칼럼 index를 []로 입력하면 해당하는 칼럼 데이터 세트 반환, skew 호출

skew_features = data[features_index].apply(lambda x: skew(x))

# skew 정도가 1 이상인 칼럼만 추출

skew_features_top = skew_features[skew_features > 1]

print(skew_features_top.sort_values(ascending=False))
# 추출된 왜곡 정도가 높은 피처들 로그 변환 실시 

data[skew_features_top.index] = np.log1p(data[skew_features_top.index])
# 변수 로그 변환하였으므로 다시 원핫인코딩 적용한 data_ohe 만들고, 모델링까지

data_ohe = pd.get_dummies(data)

y_target = data_ohe['SalePrice']

x_features = data_ohe.drop('SalePrice', axis=1, inplace=False)

x_train, x_test, y_train, y_test = train_test_split(y_target, x_features,

                                                   test_size=.2,

                                                   random_state=156)



# 피처를 로그 변환한 후 최적 하이퍼 파라미터와 RMSE 출력

ridge_params = {'alpha' : [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}

lasso_params = {'alpha' : [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]}

print_best_params(ridge, ridge_params)

print_best_params(lasso, lasso_params)
visualize_coefficient(models)
# 이상치 확인하기 

# `GrLivArea` 주거 공간 크기 변수 확인하기 

plt.scatter(x=data_org['GrLivArea'],

            y=data_org['SalePrice'])

plt.ylabel('SalePrice', fontsize=15)

plt.xlabel('GrLivArea', fontsize=15)

plt.show()
# `GrLivArea`가 4000평방비트 이상임에도 가격이 500,000달러 이하인 데이터는 모두 이상치로 간주하고 삭제

# GrLivArea와 SalePrice 모두 로그 변환되었으므로 이를 반영한 조건 생성

cond1 = data_ohe['GrLivArea'] > np.log1p(4000)

cond2 = data_ohe['SalePrice'] < np.log1p(500000)

outlier_index = data_ohe[cond1 & cond2].index



print('이상치 레코드 index :', outlier_index.values)

print('이상치 삭제 전 data_ohe shape: ', data_ohe.shape)



# dataframe의 인덱스를 이용해 이상치 레코드 삭제 

data_ohe.drop(outlier_index, axis=0, inplace=True)

print('이상치 삭제 후 data_ohe shape: ', data_ohe.shape)
y_target = data_ohe['SalePrice']

x_features = data_ohe.drop('SalePrice', axis=1, inplace=False)

x_train, x_test, y_train, y_test = train_test_split(x_features, y_target,

                                                   test_size=.2,

                                                   random_state=156)



# 피처를 로그 변환한 후 최적 하이퍼 파라미터와 RMSE 출력

ridge_params = {'alpha' : [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}

lasso_params = {'alpha' : [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]}

print_best_params(ridge, ridge_params)

print_best_params(lasso, lasso_params)
# 앞의 최적화 alpha 값으로 학습 데이터로 학습, 테스트 데이터로 예측 및 평가 수행

lr = LinearRegression()

ridge = Ridge(alpha=8)

lasso = Lasso(alpha=0.001)



lr.fit(x_train, y_train)

ridge.fit(x_train, y_train)

lasso.fit(x_train, y_train)



# 모든 모델의 RMSE 출력

models = [lr, ridge, lasso]

get_rmses(models)



# 모든 모델의 회귀 계수 시각화

visualize_coefficient(models)
# xgboost 회귀 트리

from xgboost import XGBRegressor



xgb_params = {'n_estimators':[1000]}

xgb = XGBRegressor(n_estimators=1000,

                  learning_rate = 0.05,

                  colsample_bytree=0.5,

                  subsample=0.8,

                  tree_method='gpu_hist',

                  random_state=0)

print_best_params(xgb, xgb_params)
# lightGBM 회귀 트리

from lightgbm import LGBMRegressor



lgbm_params = {'n_estimators':[1000]}

lgbm = LGBMRegressor(n_estimators=1000,

                    learning_rate=0.05,

                    num_leaves=4,

                    subsample=0.6,

                    colsample_bytree=0.4,

                    reg_lambda=10,

                    n_jobs=-1,

                    tree_method='gpu_hist',

                    random_state=0)

print_best_params(lgbm, lgbm_params)
xgb.fit(x_train, y_train)

lgbm.fit(x_train, y_train)
# 모델의 중요도 상위 20개의 피처명과 그때의 중요도값을 Series로 반환.

def get_top_features(model):

    ftr_importances_values = model.feature_importances_

    ftr_importances = pd.Series(ftr_importances_values, index=x_features.columns  )

    ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

    return ftr_top20



def visualize_ftr_importances(models):

    # 2개 회귀 모델의 시각화를 위해 2개의 컬럼을 가지는 subplot 생성

    fig, axs = plt.subplots(figsize=(24,10),nrows=1, ncols=2)

    fig.tight_layout() 

    # 입력인자로 받은 list객체인 models에서 차례로 model을 추출하여 피처 중요도 시각화. 

    for i_num, model in enumerate(models):

        # 중요도 상위 20개의 피처명과 그때의 중요도값 추출 

        ftr_top20 = get_top_features(model)

        axs[i_num].set_title(model.__class__.__name__+' Feature Importances', size=25)

        #font 크기 조정.

        for label in (axs[i_num].get_xticklabels() + axs[i_num].get_yticklabels()):

            label.set_fontsize(22)

        sns.barplot(x=ftr_top20.values, y=ftr_top20.index , ax=axs[i_num])



# 앞 예제에서 print_best_params( )가 반환한 GridSearchCV로 최적화된 모델의 피처 중요도 시각화    

models = [xgb, lgbm]

visualize_ftr_importances(models)
# 개별 회귀 모델의 예측 결괏값을 혼합해 이를 기반으로 최종 회귀값 예측

def get_rmse_pred(preds):

    for key in preds.keys():

        pred_value = preds[key]

        mse = mean_squared_error(y_test, pred_value)

        rmse = np.sqrt(mse)

        print('{0} 모델의 RMSE: {1}'.format(key, rmse))



# ridge & lasso

# 개별 모델의 학습

ridge = Ridge(alpha=8)

lasso = Lasso(alpha=0.001)



ridge.fit(x_train, y_train)

lasso.fit(x_train, y_train)



# 개별 모델 예측

ridge_pred = ridge.predict(x_test)

lasso_pred = lasso.predict(x_test)



# 개별 모델 예측값 혼합으로 최종 예측값 도출

pred = 0.4 * ridge_pred + 0.6 * lasso_pred

preds = {'최종 혼합': pred,

        'Ridge': ridge_pred,

        'Lasso': lasso_pred}



# 최종 혼합 모델, 개별 모델의 RMSE 값 출력

get_rmse_pred(preds)
# xgb & lightgbm

xgb = XGBRegressor(n_estimators=1000,

                  learning_rate=0.05,

                  colsample_bytree=0.5,

                  subsample=0.8,

                  tree_method='gpu_hist',

                  random_state=0)

lgbm = LGBMRegressor(n_estimators=1000,

                    learning_rate=0.05,

                    num_leaves=4,

                    subsample=0.6,

                    colsample_bytree=0.4,

                    reg_lambda=10,

                    n_jobs=-1,

                    tree_method='gpu_hist',

                    random_state=0)



xgb.fit(x_train, y_train)

lgbm.fit(x_train, y_train)



xgb_pred = xgb.predict(x_test)

lgbm_pred = lgbm.predict(x_test)



pred = 0.5 * xgb_pred + 0.5 * lgbm_pred

preds = {'최종 혼합': pred,

        'XGB' : xgb_pred,

        'LGBM' : lgbm_pred}



get_rmse_pred(preds)
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error



# 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수

def get_stacking_base_datasets(model, x_features_n, y_target_n, x_test_n, n_folds):

    # 지정된 n_folds 값으로 KFold 생성

    kf = KFold(n_splits=n_folds,

              shuffle=False,

              random_state=0)

    # 추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화

    train_fold_pred = np.zeros((x_train_n.shape[0], 1))

    test_pred = np.zeros((x_test_n.shape[0], n_folds))

    print(model.__class__.__name__, 'model 시작')

    

    for folder_counter, (train_index, valid_index) in enumerate(kf.split(x_train_n)):

        # 입력된 학습 데이터에서 기반 모델이 학습/예측할 폴드 데이터 추출

        print('\t 폴드 세트: ', folder_counter, '시작')

        x_tr = x_train_n[train_index]

        y_tr = y_train_n[train_index]

        x_te = x_train_n[valid_index]

        

        # 폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행

        model.fit(x_tr, y_tr)

        # 폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장

        train_fold_pred[valid_index, :] = model.predict(x_te).reshape(-1, 1)

        # 입력된 원본 텍스트 데이터를 폴드 세트 내 학습된 기반 모델에서 예측 후 데이터 저장

        test_pred[:, folder_counter] = model.predict(x_test_n)

        

    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성

    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)

    

    # train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터

    return train_fold_pred, test_pred_mean
# get_stacking_base_dataset은 ndarray 인자로 사용

x_train_n = x_train.values

y_train_n = y_train.values

x_test_n = x_test.values



# 각 개별 기반 모델이 생성한 학습용/테스트용 데이터 반환

ridge_train, ridge_test = get_stacking_base_datasets(ridge, x_train_n, y_train_n, x_test_n, 5)

lasso_train, lasso_test = get_stacking_base_datasets(lasso, x_train_n, y_train_n, x_test_n, 5)

xgb_train, xgb_test = get_stacking_base_datasets(xgb, x_train_n, y_train_n, x_test_n, 5)

lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm, x_train_n, y_train_n, x_test_n, 5)
# 개별 모델이 반환한 학습 및 테스트용 데이터 세트를 스태킹 형태로 결합

stack_final_x_train = np.concatenate((ridge_train, lasso_train, xgb_train, lgbm_train), axis=1)

stack_final_x_test = np.concatenate((ridge_test, lasso_test, xgb_test, lgbm_test), axis=1)



# 최종 메타 모델은 라쏘 모델 적용

meta_model_lasso = Lasso(alpha=0.0005)



# 개별 모델 예측값을 기반으로 새롭게 만들어진 학습/테스트 데이터로 메타 모델 예측 및 RMSE 측정

meta_model_lasso.fit(stack_final_x_train, y_train)

final = meta_model_lasso.predict(stack_final_x_test)

mse = mean_squared_error(y_test, final)

rmse = np.sqrt(mse)

print('스태킹 회귀 모델의 최종 RMSE 값은: ', rmse)