# Loding packages

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import gc
# !kaggle competitions download -c 2019-2nd-ml-month-with-kakr
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print(df_train.shape)

print(df_test.shape)
df_train.head(20)
df_train.info()
# describe statistics summery

df_train['price'].describe()
# histogram

plt.figure(figsize=(8,6))

sns.distplot(a=df_train['price'])
# skewness and krutosis

print('Skewness:', df_train['price'].skew())

print('Kurtosis:', df_train['price'].kurt())
fig, axes = plt.subplots(nrows=1, ncols=2)

fig.set_size_inches(10,8)

stats.probplot(x=df_train['price'], plot=axes[0])

stats.probplot(x=np.log1p(df_train['price']), plot=axes[1])
df_train['price'] = np.log1p(df_train['price'])

# histogram

plt.figure(figsize=(8,6))

sns.distplot(a=df_train['price'])
# skewness and krutosis

print('Skewness:', df_train['price'].skew())

print('Kurtosis:', df_train['price'].kurt())
# saleprice correlation matrix

k = 15 # number of variables for heatmap

corrmat = abs(df_train.corr(method='spearman')) # correlation 전체 변수에 대해서 계산

cols = corrmat.nlargest(k, 'price').index # nlargest : return this many descending sorted values

cm = np.corrcoef(df_train[cols].values.T) # correlation 특정 칼럼에 대해서



fig, ax = plt.subplots(figsize=(18,8))

sns.heatmap(data=cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':8}, yticklabels=cols.values, xticklabels=cols.values, ax=ax)
# np.corrcoef(df_train[cols].values).shape
data = pd.concat([df_train['price'], df_train['grade']], axis=1)

f, ax = plt.subplots(figsize=(10,8))

sns.boxplot(data=data, x='grade', y='price', ax=ax)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

ax[0].set(title='Train set')

sns.countplot(x='grade', data=df_train, ax=ax[0], palette=sns.color_palette('pastel'))



ax[1].set(title='Test set')

sns.countplot(x='grade', data=df_test, ax=ax[1], palette=sns.color_palette('pastel'))
data = pd.concat([df_train['sqft_living'], df_train['price']], axis=1)

f, ax = plt.subplots(figsize=(10,8))

sns.regplot(x='sqft_living', y='price', data=data, ax=ax)
data = pd.concat([df_train['sqft_living15'], df_train['price']], axis=1)

f, ax = plt.subplots(figsize=(10,8))

sns.regplot(x='sqft_living15', y='price', data=data, ax=ax)
sns.distplot(a=df_train['sqft_living15'])
data = pd.concat([df_train['sqft_above'], df_train['price']], axis=1)

f, ax = plt.subplots(figsize=(10,8))

sns.regplot(x='sqft_above', y='price', data=data)
data = pd.concat([df_train['bathrooms'], df_train['price']], axis=1)

f, ac = plt.subplots(figsize=(15,8))

sns.boxplot(x='bathrooms', y='price', data=data, palette=sns.color_palette('pastel'))
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(20,8))

ax[0].set(title='Train set')

sns.countplot(x='bathrooms', data=df_train, ax=ax[0])



ax[1].set(title='Test set')

sns.countplot(x='bathrooms', data=df_test, ax=ax[1])
data = pd.concat([df_train['bedrooms'], df_train['price']], axis=1)

f, ac = plt.subplots(figsize=(12,8))

sns.boxplot(x='bedrooms', y='price', data=data)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

ax[0].set(title='Train set')

sns.countplot(x='bedrooms', data=df_train, ax=ax[0])



ax[1].set(title='Test set')

sns.countplot(x='bedrooms', data=df_test, ax=ax[1])
data = pd.concat([df_train['floors'], df_train['price']], axis=1)

f, ac = plt.subplots(figsize=(10,8))

sns.boxplot(x='floors', y='price', data=data, palette=sns.color_palette('pastel'))
data = pd.concat([df_train['view'], df_train['price']], axis=1)

f, ac = plt.subplots(figsize=(12,8))

sns.boxplot(x='view', y='price', data=data)
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,8))

ax[0].set(title='Train set')

sns.countplot(x='view', data=df_train, ax=ax[0])



ax[1].set(title='Test set')

sns.countplot(x='view', data=df_test, ax=ax[1])
data = pd.concat([df_train['waterfront'], df_train['price']], axis=1)

f, ac = plt.subplots(figsize=(12,8))

sns.boxplot(x='waterfront', y='price', data=data)
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,8))

ax[0].set(title='Train set')

sns.countplot(x='waterfront', data=df_train, ax=ax[0])



ax[1].set(title='Test set')

sns.countplot(x='waterfront', data=df_test, ax=ax[1])
import missingno as msno

msno.matrix(df_train)
msno.matrix(df_test)
import plotly as py

import plotly.graph_objs as go



py.offline.init_notebook_mode(connected=True)
## 유니크 갯수 계산

train_unique=[]

columns = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']



for i in columns:

    train_unique.append(len(df_train[i].unique()))

    

unique_train = pd.DataFrame()

unique_train['Columns'] = columns

unique_train['Unique_value'] = train_unique



data = [

    go.Bar(

        x = unique_train['Columns'],

        y = unique_train['Unique_value'],

        name = 'Unique value in features',

        textfont = dict(size=20),

        marker = dict(opacity=0.45)

    )

]



layout = go.Layout(

    title = 'Unique Value By Column',

    xaxis = dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

    yaxis = dict(title='Value Count', ticklen=5, gridwidth=2),

    showlegend = True

)



fig = go.Figure(data=data, layout=layout)

# py.offline.iplot(fig, filename='skin')

py.offline.iplot(fig)
data = pd.concat([df_train['sqft_living'], df_train['price']], axis=1)

f, ax = plt.subplots(figsize=(10,8))

sns.regplot(x='sqft_living', y='price', data=data)
df_train[df_train['sqft_living'] > 13000]
df_train = df_train.loc[df_train['id'] != 8912]
data = pd.concat([df_train['grade'], df_train['price']], axis=1)

f, ax = plt.subplots(figsize=(10,8))

sns.boxplot(x='grade', y='price', data=data)
df_train.loc[(df_train['price']>14.5) & (df_train['grade']==7)]
df_train.loc[(df_train['price']>14.7) & (df_train['grade']==8)]
df_train.loc[(df_train['price']>15.5) & (df_train['grade']==11)]
df_train = df_train.loc[df_train['id'] != 12346]

df_train = df_train.loc[df_train['id'] != 7173]

df_train = df_train.loc[df_train['id'] != 2775]
data = pd.concat([df_train['bedrooms'], df_train['price']], axis=1)

f, ax = plt.subplots(figsize=(10,8))

sns.boxplot(x='bedrooms', y='price', data=data)
df_train.loc[df_train['bedrooms']>=10]
df_test.loc[df_test['bedrooms']>=10]
skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']

skew_columns2 = ['sqft_living15', 'sqft_lot15'] # 1/3 제곱시켜 정규분포에 가깝게!



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)

    

for c in skew_columns2:

    df_train[c] = np.power(df_train[c].values, 1/3)

    df_test[c] = np.power(df_test[c].values, 1/3)
for df in [df_train, df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])
df_train.head()
for df in [df_train, df_test]:

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # 거실의 비율

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    # 총 면적

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    # 면적 대비 거실 비율

#     df['sqft_ratio_1'] = df['sqft_living'] / df['sqft_total_size']

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15']

#     df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_total15'] 



    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x==0 else 1)

    df['date'] = df['date'].astype('int')
df_train['per_price'] = df_train['price'] / df_train['sqft_total_size']

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean', 'var'}).reset_index()

df_train = pd.merge(df_train, zipcode_price, how='left', on='zipcode')

df_test = pd.merge(df_test, zipcode_price, how='left', on='zipcode')

del df_train['per_price']
from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline
y_reg = df_train['price']

del df_train['price']

del df_train['id']

test_id = df_test['id']

del df_test['id']
kfolds = KFold(n_splits=10, random_state=2019, shuffle=True)



def cv_rmse(model):

    rmse = np.sqrt(-cross_val_score(estimator=model, X=df_train, y=y_reg, scoring='neg_mean_squared_error', cv=kfolds))

    return rmse
train_columns = [c for c in df_train.columns if c not in ['id']]
import lightgbm as lgb

from sklearn.metrics import mean_squared_error



param = {

    'num_leaves' : 31,

    'min_data_in_leaf' : 10,

    'objective' : 'regression',

    'max_depth' : -1,

    'learning_rate' : 0.008,

    'min_child_samples' : 12,

    'boosting' : 'gbdt',

    'feature_fraction' : 0.3,

    'bagging_freq' : 1,

    'bagging_fraction' : 0.6,

    'bagging_seed' : 11,

    'metric' : 'rmse',

    'lambda_l2' : 8,

    'verbosity' : -1,

    'nthread' : -1,

    'random_state' : 2019

}



# prepare fit model with cross-validation

folds = KFold(n_splits=5, shuffle=True, random_state=2019)

oof1 = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



# run model

for fold_, (trn_index, val_index) in enumerate(folds.split(df_train)):

    trn_data = lgb.Dataset(df_train.iloc[trn_index][train_columns], label=y_reg.iloc[trn_index])

    val_data = lgb.Dataset(df_train.iloc[val_index][train_columns], label=y_reg.iloc[val_index])

    

    num_round = 10000

    clf = lgb.train(params=param, train_set=trn_data, num_boost_round=num_round, valid_sets=[trn_data, val_data], verbose_eval=500, early_stopping_rounds=500)

    oof1[val_index] = clf.predict(df_train.iloc[val_index][train_columns], num_iterations=clf.best_iteration) # 나중에 스태킹을 위한 train 데이터도 된다.

    # feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df['Feature'] = train_columns

    fold_importance_df['importance'] = clf.feature_importance()

    

    fold_importance_df['fold'] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    # predictions

    predictions += clf.predict(df_test[train_columns], num_iterations=clf.best_iteration) / folds.n_splits # 나중에 스태킹을 위한 test 데이터도 된다.

    

cv = np.sqrt(mean_squared_error(oof1, y_reg))

print(cv)
# plot the feature importance

cols = feature_importance_df[['Feature', 'importance']].groupby('Feature').mean().sort_values(by='importance',ascending=False)[:1000].index

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,26))

sns.barplot(x='importance', y='Feature', data=best_features.sort_values(by='importance',ascending=False))

plt.title('LightBGM Features(averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importaces.png')
import xgboost as xgb



params = {

    'booster' : 'gbtree',

    'max_depth': 7,

    'eta' : 0.008,

    'objective' : 'reg:linear',

    'eval_metric' : 'rmse',

    'n_thread' : -1,

    'silent' : 1,

    'lambda' : 8,

    'sub_sample' : 0.5,

    'random_state' : 2019

}



# prepare fit model with cross-validation

oof2 = np.zeros(len(df_train))

predictions2 = np.zeros((len(df_test)))

feature_importance_df2 = pd.DataFrame()



num_rounds = 10000

for folds_, (trn_index, val_index) in enumerate(folds.split(df_train)):

    trn_dmtx = xgb.DMatrix(data=df_train.iloc[trn_index][train_columns].values, label=y_reg.iloc[trn_index].values)

    val_dmtx = xgb.DMatrix(data=df_train.iloc[val_index][train_columns].values, label=y_reg.iloc[val_index].values)

    

    wlist = [(trn_dmtx, 'train'),(val_dmtx, 'eval')]

    clf = xgb.train(params=params, dtrain=trn_dmtx, num_boost_round=num_rounds, evals=wlist, verbose_eval=100, early_stopping_rounds=100)

    oof2[val_index] = clf.predict(data=val_dmtx) # 스태킹 모델링을 위해 남겨놓는다(스태킹 모델의 훈련 데이터)

    

    # 스테킹 모델의 테스트 데이터가 된다.

    predictions2 += clf.predict(data=xgb.DMatrix(data=df_test[train_columns].values)) / folds.n_splits

    

cv = np.sqrt(mean_squared_error(oof2, y_reg))

print(cv)
from sklearn.ensemble import RandomForestRegressor



rf_reg = RandomForestRegressor(n_estimators=300, random_state=2019, max_depth=15, n_jobs=-1).fit(df_train, y_reg)
print('RandomForest rmse:', cv_rmse(rf_reg).mean())
# plot the feature importance

feature_importance_df_rf = pd.Series(data=rf_reg.feature_importances_, index=df_train.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12,15), nrows=1, ncols=1)

sns.barplot(x=feature_importance_df_rf, y=feature_importance_df_rf.index, ax=ax)
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler



svr_reg = make_pipeline(StandardScaler(), SVR(C=3.5)).fit(df_train, y_reg)
print('Support Vector Regressor rmse :', cv_rmse(svr_reg).mean())
from sklearn.tree import ExtraTreeRegressor



extra_tree_reg = ExtraTreeRegressor(criterion='mse', max_depth=15, min_samples_leaf=10, min_samples_split=10, max_features='auto',random_state=2019).fit(df_train, y_reg)
print('ExtraTreeRegressor rmse :', cv_rmse(extra_tree_reg).mean())
from keras import models

from keras import layers

from keras import optimizers

from keras import regularizers

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import keras.backend as K
# 먼저 모델링 전 데이터들을 퍼셉트론이 가장 잘 학습할 수 있도록 정규화 해주도록 하자.

from sklearn.preprocessing import StandardScaler



x_scaler = StandardScaler().fit(df_train)

y_scaler = StandardScaler().fit(y_reg.values.reshape(-1,1))



x_train = x_scaler.transform(df_train)

x_test = x_scaler.transform(df_test)

y_train = y_scaler.transform(y_reg.values.reshape(-1,1))
print(x_train.shape[1])
np.random.seed(2019) # for reproduction
# root mean squared error를 평가 메트릭으로 사용하기위해 함수를 정의

def root_mean_squared_error(y_pred, y_true):

    squared_err = K.square(K.sum(y_true - y_pred, axis=0))

    mse = K.mean(squared_err)

    return K.sqrt(mse)
# 모델링

def build_model():

    model = models.Sequential()

    model.add(layers.Dense(12, activation='relu', input_dim=x_train.shape[1], kernel_initializer='he_uniform'))

    model.add(layers.Dense(6, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dense(3, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

    model.add(layers.Dense(1))

    

    optimizer = optimizers.Adam(lr=0.001)

    model.compile(optimizer=optimizer, loss='mse', metrics=[root_mean_squared_error])

    

    return model
# 하이퍼 파라미터 정의

epoch = 200

patient = 20

k = 5

folds = KFold(n_splits=k, shuffle=True, random_state=2019)
import os

model_path = './model'



if not os.path.exists(model_path):

    os.mkdir(model_path)

    

model_path1 = model_path + 'adam_model1.h5'
call_backs = [

    EarlyStopping(monitor='mse', patience=patient, mode='min', verbose=1),

    ModelCheckpoint(filepath=model_path1, monitor='mse', verbose=1, save_best_only=True, mode='min'),

    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patient/3, min_lr=0.000001, verbose=1, mode='min')

]
# 스태킹 모델링을 위해

oof_percep = np.zeros(len(x_train))

prediction_percep = np.zeros(len(x_test))



# 학습

for folds_, (trn_idx, val_idx) in enumerate(folds.split(x_train)):

    print('처리중인 폴드 #', folds_)

    # 학습 데이터

    partial_train_data = x_train[trn_idx]

    partial_train_targets = y_train[trn_idx]

    

    # 검증 데이터

    val_data = x_train[val_idx]

    val_targets = y_train[val_idx]

    

    model = build_model()

    history = model.fit(partial_train_data,

                       partial_train_targets,

                       validation_data=(val_data, val_targets),

                        epochs=epoch,

                        batch_size=16,

                        callbacks=call_backs

                       )

    # 가중치 저장

    if not os.path.exists('./weight_data'):

        os.mkdir('./weight_data')

#     model.save_weights('./weight_data/train_fold{}'.format(folds_))

        

    

    oof_percep[val_idx] = model.predict(val_data).squeeze()

    prediction_percep += model.predict(x_test).squeeze() / folds.n_splits
# plt.plot(history.history['root_mean_squared_error'])

# plt.plot(history.history['val_root_mean_squared_error'])

# plt.title('Root_mean_squared_error')

# plt.xlabel('Epochs')

# plt.ylabel('Score')

# plt.legend(['train', 'val'], loc='best')
# 예측결과를 저장해놓는다.

# percep_model_preds = model.predict(x_test)
# 두 종류의 모델이 필요하다.

# 하나는 지금까지 학습시킨 개별적인 기본모델들이고,

# 다른 하나는 개별 기반 모델의 예측 데이터를 학습데이터로 만들어서 학습하는 최종 메타 모델이다.



# 개별 기반 모델에서 최종 메타 모델이 사용할 학습 및 테스트용 데이터를 생성하기 위한 함수

def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n):

    # KFold는 위에서 만들어진것을 쓴다.

    # 추후에 메타 모델이 학습 데이터 반환을 위한 넘파이 배열 초기화

    train_fold_pred = np.zeros((df_train.shape[0], 1))

    test_pred = np.zeros((df_test.shape[0], folds.n_splits))

    print(model.__class__.__name__, ' model 시작')

    

    for folds_, (trn_index, val_index) in enumerate(folds.split(X_train_n)):

        # 입력된 학습 데이터에서 기반 모델이 학습/예측할 폴드 데이터 세트 추출

        print('\t폴드 세트: ', folds_, ' 시작')

        trn_data = X_train_n.iloc[trn_index]

        val_data = X_train_n.iloc[val_index]

        y_trn = y_train_n.iloc[trn_index]

        

        # 폴드 세트 내부에서 다시 만들어진 학습 데이터로 기반 모델의 학습 수행

        model.fit(trn_data, y_trn)

        # 폴드 세트 내부에서 다시 만들어진 검증 데이터로 기반 모델 예측 후 데이터 저장

        train_fold_pred[val_index,:] = model.predict(val_data).reshape(-1,1)

        # 입력된 원본 테스트 데이터를 폴드 세트 내 학습된 기반 모델에서 예측 후 데이터 저장

        test_pred[:, folds_] = model.predict(X_test_n)

        

    # 폴드 세트 내에서 원본 테스트 데이터를 예측한 데이터를 평균하여 테스트 데이터로 생성

    test_pred_mean = np.mean(test_pred, axis=1)

    

    # train_fold_predsms 최종 메타 모델이 사용하는 학습 데이터, test_pred_mean은 테스트 데이터

    return train_fold_pred, test_pred_mean
# xgboost와 lightgbm은 각각 oof1,oof2 | predictions, predictions2 로 만들어져 있으니 RandomForest와 SVR, ExtraTree 모델들만 구해본다.

rf_train, rf_test = get_stacking_base_datasets(rf_reg, df_train, y_reg, df_test)

svr_train, svr_test = get_stacking_base_datasets(svr_reg, df_train, y_reg, df_test)

extra_train, extra_test = get_stacking_base_datasets(extra_tree_reg, df_train, y_reg, df_test)



# xgboost와 lightgbm의 앙상블도 준비해보자.

# 비율은 0.4*xgb + 0.6*lgb

oof3 = 0.6 * oof1 + 0.4 * oof2

predictions3 = 0.6 * predictions + 0.4 * predictions2
oof_percep = y_scaler.inverse_transform(oof_percep)

prediction_percep = y_scaler.inverse_transform(prediction_percep)
from sklearn.linear_model import Ridge



# 개별 모델이 반환한 테스트용 데이터 세트를 스태킹 형태로 결합

## 스택모델1 : 랜덤포레스트, XGB, LGB, SVR, 엑스트라 랜더마이즈 트리

# Stack_final_X_train = np.concatenate([rf_train, oof1.reshape(-1,1), oof2.reshape(-1,1), svr_train, extra_train], axis=1)

# Stack_final_X_test = np.concatenate([rf_test[:,np.newaxis], predictions[:,np.newaxis], predictions2[:,np.newaxis],

#                                     svr_test[:,np.newaxis], extra_test[:,np.newaxis]], axis=1)



## 스택모델2 : 랜덤포레스트, XGB, LGB, 0.6*LGB+0.4*XGB

# Stack_final_X_train = np.concatenate([rf_train, oof1.reshape(-1,1), oof2.reshape(-1,1), oof3.reshape(-1,1)], axis=1)

# Stack_final_X_test = np.concatenate([rf_test[:,np.newaxis], predictions[:,np.newaxis], predictions2[:,np.newaxis], predictions3[:,np.newaxis]], axis=1)



## 스택모델3 : 다층 퍼셉트론, XGB, LGB, SVR, 엑스트라 랜더마이즈 트리

# Stack_final_X_train = np.concatenate([oof_percep.reshape(-1,1), rf_train, oof1.reshape(-1,1), oof2.reshape(-1,1), svr_train, extra_train], axis=1)

# Stack_final_X_test = np.concatenate([prediction_percep[:,np.newaxis], rf_test[:,np.newaxis], predictions[:,np.newaxis], predictions2[:,np.newaxis],

#                                     svr_test[:,np.newaxis], extra_test[:,np.newaxis]], axis=1)



## 스택모델4 : 다층 퍼셉트론, XGB, LGB, SVR, 엑스트라 랜더마이즈 트리

Stack_final_X_train = np.concatenate([oof_percep.reshape(-1,1), rf_train, oof1.reshape(-1,1), oof2.reshape(-1,1), svr_train, extra_train], axis=1)

Stack_final_X_test = np.concatenate([prediction_percep[:,np.newaxis], rf_test[:,np.newaxis], predictions[:,np.newaxis], predictions2[:,np.newaxis],

                                    svr_test[:,np.newaxis], extra_test[:,np.newaxis]], axis=1)





# 최종 메타 모델은 릿지 모델을 적용

meta_model_ridge = Ridge(alpha=0.1)



# 개별 모델 예측값을 기반으로 새롭게 만들어진 학습/테스트 데이터로 메타 모델 예측

meta_model_ridge.fit(Stack_final_X_train, y_reg)

final = meta_model_ridge.predict(Stack_final_X_test)
# test_ridge_preds = np.expm1(ridge_model2.predict(df_test))

# test_rf_preds = np.expm1(rf_reg.predict(df_test))

# test_lgb_preds = np.expm1(predictions)

# test_xgb_preds = np.expm1(predictions2)

# rest_ensemble_preds = 0.6*test_lgb_preds + 0.4*test_xgb_preds

# percep_inv_model_preds = np.expm1(y_scaler.inverse_transform(percep_model_preds))



stack_preds = np.expm1(final)
# submission0 = pd.DataFrame({'id' : test_id, 'price' : test_ridge_preds})

# submission0.to_csv('ridge.csv', index=False)
# submission0 = pd.DataFrame({'id' : test_id, 'price' : test_lgb_preds})

# submission0.to_csv('lgb.csv', index=False)
# submission0 = pd.DataFrame({'id' : test_id, 'price' : test_xgb_preds})

# submission0.to_csv('xgb.csv', index=False)
# submission0 = pd.DataFrame({'id' : test_id, 'price' : rest_ensemble_preds})

# submission0.to_csv('ensemble.csv', index=False)
# submission0 = pd.DataFrame({'id': test_id, 'price' : test_rf_preds})

# submission0.to_csv('rf.csv', index=False)
# submission0 = pd.DataFrame({'id': test_id, 'price' : percep_inv_model_preds.squeeze()})

# submission0.to_csv('percep_keras.csv', index=False)
# 스택모델 : 다층 퍼셉트론, 랜덤포레스트, XGB, LGB, ExtraTree

submission0 = pd.DataFrame({'id': test_id, 'price' : stack_preds})

submission0.to_csv('./submission.csv', index=False)

print('끝!')