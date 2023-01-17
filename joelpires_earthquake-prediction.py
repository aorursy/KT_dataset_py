import numpy as np

import pandas as pd

from catboost import CatBoostRegressor, Pool

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.svm import NuSVR, SVR

from sklearn.kernel_ridge import KernelRidge

import matplotlib.pyplot as plt
train = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv', nrows=6000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train_ad_sample_df = train['acoustic_data'].values[::100]

train_ttf_sample_df = train['time_to_failure'].values[::100]
def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color='r')

    ax1.set_ylabel('acoustic data', color='r')

    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color='b')

    ax2.set_ylabel('time to failure', color='b')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    plt.grid(True)
plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

del train_ad_sample_df

del train_ttf_sample_df
def gen_features(X):

    strain = []

    strain.append(X.mean())

    strain.append(X.std())

    strain.append(X.min())

    strain.append(X.max())

    strain.append(X.kurtosis())

    strain.append(X.skew())

    strain.append(np.quantile(X,0.01))

    strain.append(np.quantile(X,0.05))

    strain.append(np.quantile(X,0.95))

    strain.append(np.quantile(X,0.99))

    strain.append(np.abs(X).max())

    strain.append(np.abs(X).mean())

    strain.append(np.abs(X).std())

    return pd.Series(strain)
train = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
X_train = pd.DataFrame()

y_train = pd.Series()

for df in train:

    ch = gen_features(df['acoustic_data'])

    X_train = X_train.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
X_train.describe()
train_pool = Pool(X_train, y_train)

m = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')

m.fit(X_train, y_train, silent=True)

m.best_score_
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.svm import NuSVR, SVR





scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)



parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

               'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]

               #'nu': [0.75, 0.8, 0.85, 0.9, 0.95, 0.97]}]



reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')

reg1.fit(X_train_scaled, y_train.values.flatten())

y_pred1 = reg1.predict(X_train_scaled)
print("Best CV score: {:.4f}".format(reg1.best_score_))

print(reg1.best_params_)
from sklearn.kernel_ridge import KernelRidge



parameters = [{'gamma': np.linspace(0.001, 0.1, 10),

               'alpha': [0.005, 0.01, 0.02, 0.05, 0.1]}]



reg2 = GridSearchCV(KernelRidge(kernel='rbf'), parameters, cv=5, scoring='neg_mean_absolute_error')

reg2.fit(X_train_scaled, y_train.values.flatten())

y_pred2 = reg2.predict(X_train_scaled)



print("Best CV score: {:.4f}".format(reg2.best_score_))

print(reg2.best_params_)
import lightgbm as lgb

from tqdm import tqdm_notebook as tqdm

import random





fixed_params = {

    'objective': 'regression_l1',

    'boosting': 'gbdt',

    'verbosity': -1,

    'random_seed': 19

}



param_grid = {

    'learning_rate': [0.1, 0.08, 0.05, 0.01],

    'num_leaves': [32, 46, 52, 58, 68, 72, 80, 92],

    'max_depth': [3, 4, 5, 6, 8, 12, 16, -1],

    'feature_fraction': [0.8, 0.85, 0.9, 0.95, 1],

    'subsample': [0.8, 0.85, 0.9, 0.95, 1],

    'lambda_l1': [0, 0.1, 0.2, 0.4, 0.6, 0.9],

    'lambda_l2': [0, 0.1, 0.2, 0.4, 0.6, 0.9],

    'min_data_in_leaf': [10, 20, 40, 60, 100],

    'min_gain_to_split': [0, 0.001, 0.01, 0.1],

}



best_score = 999

dataset = lgb.Dataset(X_train, label=y_train)  # no need to scale features



for i in tqdm(range(200)):

    params = {k: random.choice(v) for k, v in param_grid.items()}

    params.update(fixed_params)

    result = lgb.cv(params, dataset, nfold=5, early_stopping_rounds=50,

                    num_boost_round=20000, stratified=False)

    

    if result['l1-mean'][-1] < best_score:

        best_score = result['l1-mean'][-1]

        best_params = params

        best_nrounds = len(result['l1-mean'])
print("Best mean score: {:.4f}, num rounds: {}".format(best_score, best_nrounds))

print(best_params)

reg3 = lgb.train(best_params, dataset, best_nrounds)

y_pred3 = reg3.predict(X_train)
plt.tight_layout()

f = plt.figure(figsize=(12, 6))

f.add_subplot(1, 3, 1)

plt.scatter(y_train.values.flatten(), y_pred1)

plt.title('SVR', fontsize=16)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])



f.add_subplot(1, 3, 2)

plt.scatter(y_train.values.flatten(), y_pred2)

plt.title('Kernel Ridge', fontsize=16)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])



f.add_subplot(1, 3, 3)

plt.scatter(y_train.values.flatten(), y_pred3)

plt.title('Gradient boosting', fontsize=16)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show(block=True)
# second plot

plt.figure(figsize=(10, 5))

plt.plot(y_train.values.flatten(), color='blue', label='y_train')

plt.plot(y_pred1, color='orange', label='SVR')

plt.legend()

plt.title('SVR predictions vs actual')



# third plot

plt.figure(figsize=(10, 5))

plt.plot(y_train.values.flatten(), color='blue', label='y_train')

plt.plot(y_pred2, color='gray', label='KernelRidge')

plt.legend()

plt.title('Kernel Ridge predictions vs actual')



# fourth plot

plt.figure(figsize=(10, 5))

plt.plot(y_train.values.flatten(), color='blue', label='y_train')

plt.plot(y_pred3, color='green', label='Gradient boosting')

plt.legend()

plt.title('GBDT predictions vs actual')
# DISCLAIMER: THIS CODE IS A RESULT OF FOLLOWING THIS TUTORIAL: https://www.youtube.com/watch?v=TffGdSsWKlA