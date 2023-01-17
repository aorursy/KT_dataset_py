import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
house_df_org =  pd.read_csv('../input/train.csv')

house_df = house_df_org.copy()

house_df.head(3)
print('데이터 세트의 Shape:', house_df.shape)

print('\n전체 피처의  type \n', house_df.dtypes.value_counts())

isnull_series = house_df.isnull().sum()

print('\nNull 컬럼과 그 건수:\n', isnull_series[isnull_series>0].sort_values(ascending=False))
plt.title('Original SalePrice Histogram')

sns.distplot(house_df['SalePrice'])
plt.title('Log Transformed SalePrice Histogram')

log_SalePrice = np.log1p(house_df['SalePrice'])

sns.distplot(log_SalePrice)
#SalePrice 로그 변환

original_SalePrice = house_df['SalePrice']

house_df['SalePrice'] = np.log1p(house_df['SalePrice'])



#Null이 너무 많은 컬럼과 불필요한 컬럼 삭제

house_df.drop(['Id','PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1, inplace=True)



#드롭하지 않는 숫자형 Null 컬럼은 평균값으로 대체

house_df.fillna(house_df.mean(), inplace=True)



#Null 값이 있는 피처명과 타입을 추출

null_column_count = house_df.isnull().sum()[house_df.isnull().sum()>0]

print('## Null 피처의 Type :\n', house_df.dtypes[null_column_count.index])
print('get_dummies() 수행 전 데이터 Shape:', house_df.shape)

house_df_ohe = pd.get_dummies(house_df)

print('get_dummies() 수행 후 데이터 Shape:', house_df_ohe.shape)



null_column_count = house_df_ohe.isnull().sum()[house_df_ohe.isnull().sum()>0]

print('## Null 피처의 Type:\n', house_df_ohe.dtypes[null_column_count.index])
##End of Preprocessing
#Modeling

def get_rmse(model):

    pred = model.predict(X_test)

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
print(get_rmse(lr_reg))

print(get_rmse(ridge_reg))

print(get_rmse(lasso_reg))
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



y_target = house_df_ohe['SalePrice']

X_features = house_df_ohe.drop('SalePrice', axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)
#LinearRegression, Ridge, Lasso 학습, 예측, 평가

lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

ridge_reg = Ridge()

ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso()

lasso_reg.fit(X_train, y_train)
models=[lr_reg, ridge_reg, lasso_reg]

get_rmses(models)
def get_top_bottom_coef(model, n=10):

    coef = pd.Series(model.coef_, index=X_features.columns)

    

    coef_high = coef.sort_values(ascending=False).head(n)

    coef_low = coef.sort_values(ascending = False).tail(n)

    return coef_high, coef_low
def visualize_coef(models):

    fig, axs = plt.subplots(figsize=(24,10), nrows=1, ncols=3)

    fig.tight_layout()

    

    for i_num, model in enumerate(models):

        coef_high, coef_low = get_top_bottom_coef(model)

        coef_concat = pd.concat([coef_high, coef_low])

        axs[i_num].set_title(model.__class__.__name__+'Coeffiecents', size=25)

        axs[i_num].tick_params(axis='y', direction='in', pad=-120)

        

        for label in (axs[i_num].get_xticklabels()+axs[i_num].get_yticklabels()):

            label.set_fontsize(22)

        sns.barplot(x=coef_concat.values, y=coef_concat.index, ax=axs[i_num])



models = [lr_reg, ridge_reg, lasso_reg]

visualize_coef(models)
from sklearn.model_selection import cross_val_score



def get_avg_rmse_cv(models):

    

    for model in models:

        rmse_list = np.sqrt(-cross_val_score(model, X_features, y_target, scoring='neg_mean_squared_error', cv=5))

        rmse_avg = np.mean(rmse_list)

        print('\n{0} CV RMSE 값 리스트: {1}'.format(model.__class__.__name__, np.round(rmse_list, 3)))

        print('\n{0} CV 평균 RMSE 값:{1}'.format(model.__class__.__name__, np.round(rmse_avg, 3)))





models = [lr_reg, ridge_reg, lasso_reg]

get_avg_rmse_cv(models)
from sklearn.model_selection import GridSearchCV



def print_best_params(model, params):

    grid_model = GridSearchCV(model, param_grid = params,

                             scoring ='neg_mean_squared_error', cv=5)

    grid_model.fit(X_features, y_target)

    rmse = np.sqrt(-1 * grid_model.best_score_)

    print('{0} 5 CV 실행시 최적 평균 RMSE 값:{1}, 최적 알파{2}'.format(model.__class__.__name__,np.round(rmse, 4), grid_model.best_params_))



ridge_params = { 'alpha' : [0.05, 0.1, 1, 5, 8, 10, 12, 15, 20]}

lasso_params = { 'alpha' : [0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1, 5, 10]}

print_best_params(ridge_reg, ridge_params)

print_best_params(lasso_reg, lasso_params)
lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha=12)

ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha=0.001)

lasso_reg.fit(X_train, y_train)



models = [lr_reg, ridge_reg, lasso_reg]

get_rmses(models)



models = [lr_reg, ridge_reg, lasso_reg]

visualize_coef(models)
from scipy.stats import skew



# object가 아닌 숫자형 피쳐의 컬럼 index 객체 추출.

features_index = house_df.dtypes[house_df.dtypes != 'object'].index

# house_df에 컬럼 index를 [ ]로 입력하면 해당하는 컬럼 데이터 셋 반환. apply lambda로 skew( )호출 

skew_features = house_df[features_index].apply(lambda x : skew(x))

# skew 정도가 5 이상인 컬럼들만 추출. 

skew_features_top = skew_features[skew_features > 1]

print(skew_features_top.sort_values(ascending=False))
house_df[skew_features_top.index] = np.log1p(house_df[skew_features_top.index])
# Skew가 높은 피처들을 로그 변환 했으므로 다시 원-핫 인코딩 적용 및 피처/타겟 데이터 셋 생성,

house_df_ohe = pd.get_dummies(house_df)

y_target = house_df_ohe['SalePrice']

X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)



# 피처들을 로그 변환 후 다시 최적 하이퍼 파라미터와 RMSE 출력

ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }

lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }

best_ridge = print_best_params(ridge_reg, ridge_params)

best_lasso = print_best_params(lasso_reg, lasso_params)
# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 

lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha=10)

ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha=0.001)

lasso_reg.fit(X_train, y_train)



# 모든 모델의 RMSE 출력

models = [lr_reg, ridge_reg, lasso_reg]

get_rmses(models)



# 모든 모델의 회귀 계수 시각화 

models = [lr_reg, ridge_reg, lasso_reg]

visualize_coef(models)

plt.scatter(x = house_df_org['GrLivArea'], y = house_df_org['SalePrice'])

plt.ylabel('SalePrice', fontsize=15)

plt.xlabel('GrLivArea', fontsize=15)

plt.show()

# GrLivArea와 SalePrice 모두 로그 변환되었으므로 이를 반영한 조건 생성. 

cond1 = house_df_ohe['GrLivArea'] > np.log1p(4000)

cond2 = house_df_ohe['SalePrice'] < np.log1p(500000)

outlier_index = house_df_ohe[cond1 & cond2].index



print('아웃라이어 레코드 index :', outlier_index.values)

print('아웃라이어 삭제 전 house_df_ohe shape:', house_df_ohe.shape)

# DataFrame의 index를 이용하여 아웃라이어 레코드 삭제. 

house_df_ohe.drop(outlier_index , axis=0, inplace=True)

print('아웃라이어 삭제 후 house_df_ohe shape:', house_df_ohe.shape)

y_target = house_df_ohe['SalePrice']

X_features = house_df_ohe.drop('SalePrice',axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=156)



ridge_params = { 'alpha':[0.05, 0.1, 1, 5, 8, 10, 12, 15, 20] }

lasso_params = { 'alpha':[0.001, 0.005, 0.008, 0.05, 0.03, 0.1, 0.5, 1,5, 10] }

best_ridge = print_best_params(ridge_reg, ridge_params)

best_lasso = print_best_params(lasso_reg, lasso_params)

# 앞의 최적화 alpha값으로 학습데이터로 학습, 테스트 데이터로 예측 및 평가 수행. 

lr_reg = LinearRegression()

lr_reg.fit(X_train, y_train)

ridge_reg = Ridge(alpha=8)

ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha=0.001)

lasso_reg.fit(X_train, y_train)



# 모든 모델의 RMSE 출력

models = [lr_reg, ridge_reg, lasso_reg]

get_rmses(models)



# 모든 모델의 회귀 계수 시각화 

models = [lr_reg, ridge_reg, lasso_reg]

visualize_coef(models)

from xgboost import XGBRegressor



xgb_params = {'n_estimators':[1000]}

xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, 

                       colsample_bytree=0.5, subsample=0.8)

best_xgb = print_best_params(xgb_reg, xgb_params)

from lightgbm import LGBMRegressor



lgbm_params = {'n_estimators':[1000]}

lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, 

                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)

best_lgbm = print_best_params(lgbm_reg, lgbm_params)

# 모델의 중요도 상위 20개의 피처명과 그때의 중요도값을 Series로 반환.

def get_top_features(model):

    ftr_importances_values = model.feature_importances_

    ftr_importances = pd.Series(ftr_importances_values, index=X_features.columns  )

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



# 앞 예제에서 get_best_params( )가 반환한 GridSearchCV로 최적화된 모델의 피처 중요도 시각화    

models = [best_xgb, best_lgbm]

visualize_ftr_importances(models)

def get_rmse_pred(preds):

    for key in preds.keys():

        pred_value = preds[key]

        mse = mean_squared_error(y_test , pred_value)

        rmse = np.sqrt(mse)

        print('{0} 모델의 RMSE: {1}'.format(key, rmse))



# 개별 모델의 학습

ridge_reg = Ridge(alpha=8)

ridge_reg.fit(X_train, y_train)

lasso_reg = Lasso(alpha=0.001)

lasso_reg.fit(X_train, y_train)

# 개별 모델 예측

ridge_pred = ridge_reg.predict(X_test)

lasso_pred = lasso_reg.predict(X_test)



# 개별 모델 예측값 혼합으로 최종 예측값 도출

pred = 0.4 * ridge_pred + 0.6 * lasso_pred

preds = {'최종 혼합': pred,

         'Ridge': ridge_pred,

         'Lasso': lasso_pred}

#최종 혼합 모델, 개별모델의 RMSE 값 출력

get_rmse_pred(preds)

xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, 

                       colsample_bytree=0.5, subsample=0.8)

lgbm_reg = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=4, 

                         subsample=0.6, colsample_bytree=0.4, reg_lambda=10, n_jobs=-1)

xgb_reg.fit(X_train, y_train)

lgbm_reg.fit(X_train, y_train)

xgb_pred = xgb_reg.predict(X_test)

lgbm_pred = lgbm_reg.predict(X_test)



pred = 0.5 * xgb_pred + 0.5 * lgbm_pred

preds = {'최종 혼합': pred,

         'XGBM': xgb_pred,

         'LGBM': lgbm_pred}

        

get_rmse_pred(preds)

# 모델의 중요도 상위 20개의 피처명과 그때의 중요도값을 Series로 반환.

def get_top_features(model):

    ftr_importances_values = model.feature_importances_

    ftr_importances = pd.Series(ftr_importances_values, index=X_features.columns  )

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



# 앞 예제에서 get_best_params( )가 반환한 GridSearchCV로 최적화된 모델의 피처 중요도 시각화    

models = [best_xgb, best_lgbm]

visualize_ftr_importances(models)

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error



# 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수. 

def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds ):

    # 지정된 n_folds값으로 KFold 생성.

    kf = KFold(n_splits=n_folds, shuffle=False, random_state=0)

    #추후에 메타 모델이 사용할 학습 데이터 반환을 위한 넘파이 배열 초기화 

    train_fold_pred = np.zeros((X_train_n.shape[0] ,1 ))

    test_pred = np.zeros((X_test_n.shape[0],n_folds))

    print(model.__class__.__name__ , ' model 시작 ')

    

    for folder_counter , (train_index, valid_index) in enumerate(kf.split(X_train_n)):

        #입력된 학습 데이터에서 기반 모델이 학습/예측할 폴드 데이터 셋 추출 

        print('\t 폴드 세트: ',folder_counter,' 시작 ')

        X_tr = X_train_n[train_index] 

        y_tr = y_train_n[train_index] 

        X_te = X_train_n[valid_index]  

        

        #폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행.

        model.fit(X_tr , y_tr)       

        #폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장.

        train_fold_pred[valid_index, :] = model.predict(X_te).reshape(-1,1)

        #입력된 원본 테스트 데이터를 폴드 세트내 학습된 기반 모델에서 예측 후 데이터 저장. 

        test_pred[:, folder_counter] = model.predict(X_test_n)

            

    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성 

    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1,1)    

    

    #train_fold_pred는 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터

    return train_fold_pred , test_pred_mean

# get_stacking_base_datasets( )은 넘파이 ndarray를 인자로 사용하므로 DataFrame을 넘파이로 변환. 

X_train_n = X_train.values

X_test_n = X_test.values

y_train_n = y_train.values



# 각 개별 기반(Base)모델이 생성한 학습용/테스트용 데이터 반환. 

ridge_train, ridge_test = get_stacking_base_datasets(ridge_reg, X_train_n, y_train_n, X_test_n, 5)

lasso_train, lasso_test = get_stacking_base_datasets(lasso_reg, X_train_n, y_train_n, X_test_n, 5)

xgb_train, xgb_test = get_stacking_base_datasets(xgb_reg, X_train_n, y_train_n, X_test_n, 5)  

lgbm_train, lgbm_test = get_stacking_base_datasets(lgbm_reg, X_train_n, y_train_n, X_test_n, 5)

# 개별 모델이 반환한 학습 및 테스트용 데이터 세트를 Stacking 형태로 결합.  

Stack_final_X_train = np.concatenate((ridge_train, lasso_train, 

                                      xgb_train, lgbm_train), axis=1)

Stack_final_X_test = np.concatenate((ridge_test, lasso_test, 

                                     xgb_test, lgbm_test), axis=1)



# 최종 메타 모델은 라쏘 모델을 적용. 

meta_model_lasso = Lasso(alpha=0.0005)



#기반 모델의 예측값을 기반으로 새롭게 만들어진 학습 및 테스트용 데이터로 예측하고 RMSE 측정.

meta_model_lasso.fit(Stack_final_X_train, y_train)

final = meta_model_lasso.predict(Stack_final_X_test)

mse = mean_squared_error(y_test , final)

rmse = np.sqrt(mse)

print('스태킹 회귀 모델의 최종 RMSE 값은:', rmse)

submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
submission['SalePrice'] = pd.Series(np.expm1(final))
submission.head()
submission.to_csv('sample_submission.csv', index='Id')
a = pd.read_csv('sample_submission.csv')
b= a.drop('Unnamed: 0', axis=1)
b.to_csv('submission.csv')