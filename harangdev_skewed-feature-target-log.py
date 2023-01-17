import numpy as np

import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import LinearSVR, SVR

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/train.csv', index_col='id')
data.head()
data['date'] = pd.to_datetime(data['date'].astype('str').str[:8])

data['year'] = data['date'].dt.year

data['month'] = data['date'].dt.month

data['day'] = data['date'].dt.day

data = data.drop('date', axis=1)
data.head()
x_data = data.iloc[:, 1:]

y_data = data['price']
y_data.hist(bins=100);
y_data.skew()
log_y_data = np.log1p(y_data)
log_y_data.hist(bins=100);
log_y_data.skew()
n = 0

f, grid = plt.subplots(7, 3, figsize=(20, 50))

for row in grid:

    for ax in row:

        col = x_data.columns[n]

        ax.hist(x_data[col], bins=max(x_data[col].nunique()//20, 10))

        ax.set_title(col)

        n += 1
skewness = x_data.apply(lambda x: x.skew()).sort_values(ascending=False)

skewness
skew_feats = skewness[skewness>1].index

log_x_data = x_data.copy()

log_x_data[skew_feats] = np.log1p(log_x_data[skew_feats])
n = 0

f, grid = plt.subplots(7, 3, figsize=(20, 50))

for row in grid:

    for ax in row:

        col = log_x_data.columns[n]

        ax.hist(log_x_data[col], bins=max(log_x_data[col].nunique()//20, 10))

        ax.set_title(col)

        n += 1
log_x_data.apply(lambda x: x.skew()).sort_values(ascending=False)
x_data = StandardScaler().fit_transform(x_data)

log_x_data = StandardScaler().fit_transform(log_x_data)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, random_state=0)

log_x_train, log_x_val, log_y_train, log_y_val = train_test_split(log_x_data, log_y_data, random_state=0)
# target을 log 변환하지 않은 경우의 metric

def RMSE(y_pred, y_true):

    return -np.sqrt(np.mean(np.square(y_pred - y_true)))



# target을 log 변환한 경우의 metric

def RMSE_expm1(y_pred, y_true):

    return -np.sqrt(np.mean(np.square(np.expm1(y_pred) - np.expm1(y_true))))



# 모델을 학습시키고 최종 metric을 반환하는 함수

def train(x_train, y_train, x_val, y_val, algo, feval):

    

    kwargs = {}

    if algo.startswith('lgb'):

        params = {

            'num_iterations': 10**5

        }

        if algo == 'lgb_rf':

            params['boosting_type'] = 'rf'

            params['bagging_freq'] = 1

            params['bagging_fraction'] = 0.5

            model = LGBMRegressor(**params)

        else:

            model = LGBMRegressor(**params)

        kwargs['verbose'] = False

        kwargs['eval_set'] = [(x_val, y_val)]

        kwargs['early_stopping_rounds'] = 100

        def lgb_eval_metric(y_true, y_pred):

            return feval.__name__, feval(y_pred, y_true), True

        kwargs['eval_metric'] = lgb_eval_metric

    elif algo == 'lr':

        model = LinearRegression()

    elif algo == 'lin_svm':

        model = LinearSVR()

    elif algo == 'rbf_svm':

        model = SVR()

    elif algo == 'knn':

        model = KNeighborsRegressor()

        

    model.fit(x_train, y_train, **kwargs)

    pred = model.predict(x_val)

    score = feval(pred, y_val)

    

    return score



# 실험 결과를 저장할 데이터프레임

archive = pd.DataFrame(columns=['featureX_targetX', 'featureX_targetO', 'featureO_targetX', 'featureO_targetO'])
%%time

for algo in ['lgb', 'lgb_rf', 'lr', 'lin_svm', 'rbf_svm', 'knn']:

    score = train(x_train, y_train, x_val, y_val, algo, RMSE)

    archive.loc[algo, 'featureX_targetX'] = score
%%time

for algo in ['lgb', 'lgb_rf', 'lr', 'lin_svm', 'rbf_svm', 'knn']:

    score = train(x_train, log_y_train, x_val, log_y_val, algo, RMSE_expm1)

    archive.loc[algo, 'featureX_targetO'] = score
%time

for algo in ['lgb', 'lgb_rf', 'lr', 'lin_svm', 'rbf_svm', 'knn']:

    score = train(log_x_train, y_train, log_x_val, y_val, algo, RMSE)

    archive.loc[algo, 'featureO_targetX'] = score
%%time

for algo in ['lgb', 'lgb_rf', 'lr', 'lin_svm', 'rbf_svm', 'knn']:

    score = train(log_x_train, log_y_train, log_x_val, log_y_val, algo, RMSE_expm1)

    archive.loc[algo, 'featureO_targetO'] = score
archive = archive.astype(int)

archive
archive.T.plot(figsize=(20, 10), xticks=[0,1,2,3], title='Effect of Log Transformation on Skewed Feature and Target');
for algo in archive.index:

    print(algo, ':', archive.loc[algo].sort_values(ascending=False).index.tolist())