# Bread and butter

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-darkgrid')

%matplotlib inline



# ML modelling

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.metrics import mean_absolute_error

import xgboost as xgb



# Utility module: my user-defined functions | see: https://www.kaggle.com/slashie/lanl-udf

#import lanl_udf

from lanl_udf import *

#help(lanl_udf)
data_dir = '/kaggle/input/LANL-Earthquake-Prediction'

print(os.listdir(data_dir)) # let's see what's in the data directory!
n_obs = 150000 # number of obs to extract in one go (corresponds with how test data is set up - see test data section below)

n_skip = 4 * (10 ** 5) # number of rows to skip (i.e. to be able to look at different data sections, not just one!)

train_path = os.path.join(data_dir,'train.csv')

sample = pd.read_csv(train_path, 

                     nrows=n_obs, header=None, skiprows=n_skip) # header set to None, else values will be set as column names when skipping

sample.columns = ['acoustic_data','time_to_failure']

sample.head()
fig, ax = plt.subplots(nrows=1, ncols=2, facecolor='white', figsize=(14,7))

sample.acoustic_data.plot(linewidth=0.5, ax=ax[0])

ax[0].set_ylabel('Acoustic Data',fontsize=12)

sample.time_to_failure.plot(linewidth=1.5, ax=ax[1])

ax[1].set_ylabel('Time to Failure (seconds)',fontsize=12)

plt.show()
test_path = os.path.join(data_dir,'test')

test_files = os.listdir(test_path)

test_file_num = 28

test_sample = pd.read_csv(os.path.join(data_dir,'test',test_files[test_file_num]))

print("There are %d rows in each test file" %(test_sample.shape[0]))

display(test_sample.head())
X_names = ['mean','stdev','AC(1)','log(skew^2)','log(kurt)','AC(1)_diff',

           'mean(abs_dev)','gmean(abs_dev)','hmean(abs_dev)', 'frac_top500', 

           'frac_top25000', 'frac_dev>750', 'frac_eq_mode', 'wave_freq']

y_name = 'time_to_failure'

try:

    try:

        df_train = pd.read_csv('df_train.csv')

    except:

        df_train = pd.read_csv('/kaggle/input/lanl-review/df_train.csv')

    X_train = df_train[X_names].values

    y_train = df_train[y_name].values

except:

    X_train, y_train = gen_training_data(train_path)

    df_train = pd.DataFrame(X_train)

    df_train.columns = X_names

    df_train.loc[:,y_name] = y_train

df_train.to_csv('df_train.csv', index=False)

df_train.info()
corr_Xy = [np.corrcoef(y_train,X_train[:,i])[0,1] for i in range(X_train.shape[1])]

df = pd.DataFrame({'feature':X_names, 'corr_w_target':corr_Xy}).set_index('feature')

fig = plt.figure(facecolor='white', figsize=(14,7))

df['corr_w_target'].plot(kind='bar', fontsize=12)

plt.xlabel(None)

plt.xticks(rotation=45)

plt.show()
X_df = df_train[X_names]

fig = plt.figure(facecolor='white', figsize=(12,10))

hm = sns.heatmap(X_df.corr(), cmap='viridis')

hm.tick_params(labelsize=12)

plt.xticks(rotation=45)

plt.show()
for state in [7, 42, 88, 101]:

    print("Random state is %d"%state,"\n","-"*30)

    X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, test_size = 0.3, random_state=state)

    steps = [('scaler', StandardScaler()),

            ('reg', ElasticNet(alpha=0.01))]

    pipeline = Pipeline(steps)

    pipeline.fit(X_fit, y_fit)

    y_pred = pipeline.predict(X_eval)

    print("R^2: {}".format(pipeline.score(X_eval, y_eval)))

    MAE = mean_absolute_error(y_eval,y_pred)

    print("Mean Absolute Error: {}".format(MAE))

    print(pipeline.steps[1][1].coef_,'\n')
pipeline.fit(X_train, y_train)

print(pipeline.steps[1][1].coef_)
seg_id, X_test = gen_test_data(test_path)

print("Generated features for %d test segments"%len(seg_id))
elnet_pred = pipeline.predict(X_test)

elnet_pred[elnet_pred<0] = 0

elnet_submit_df = pd.DataFrame({'seg_id': seg_id, 'time_to_failure': elnet_pred})

#elnet_submit_df.to_csv('submission.csv', index=False)
xgb_train = xgb.DMatrix(data=X_train, label=y_train)

all_results = {'MAE-test':[], 'MAE-train':[], 'max_depth':[], 'eta':[], 'num_boost_round':[]}

for max_depth in [10, 12, 14]:

    for eta in [0.2, 0.25, 0.3]:

        for num_boost in [9, 11, 13]:

            params = {"objective":"reg:squarederror", "max_depth":max_depth, "eta":eta}

            cv_results = xgb.cv(dtrain=xgb_train, params=params, nfold=4, num_boost_round=num_boost, metrics="mae", seed=42)

            all_results['max_depth'].append(max_depth)

            all_results['eta'].append(eta)

            all_results['num_boost_round'].append(num_boost)

            all_results['MAE-test'].append(cv_results['test-mae-mean'].values[-1])

            all_results['MAE-train'].append(cv_results['train-mae-mean'].values[-1])

all_results_df = pd.DataFrame(all_results).sort_values(by='MAE-test')

all_results_df.head()
params = {"objective":"reg:squarederror", "max_depth":10, "eta":0.25}

xg_reg = xgb.train(params=params, dtrain=xgb_train, num_boost_round=11)

xgb_test = xgb.DMatrix(data=X_test, label=np.zeros([X_test.shape[0],]))

xgb_pred = xg_reg.predict(xgb_test)

xgb_submit_df = pd.DataFrame({'seg_id': seg_id, 'time_to_failure': xgb_pred})

xgb_submit_df.to_csv('submission.csv', index=False)
print("Correlation between XGBoost and ElasticNet predictions is %.3f"%np.corrcoef(xgb_pred, elnet_pred)[0,1])

fig = plt.figure(facecolor='white', figsize=(12,9))

plt.scatter(xgb_pred, elnet_pred)

plt.xlabel('XGBoost', fontsize=12)

plt.ylabel('ElasticNet', fontsize=12)

plt.show()