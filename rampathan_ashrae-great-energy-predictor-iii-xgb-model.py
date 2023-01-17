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

build_meta = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

sample_submission = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

weath_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
!pip install rfpimp
###########Import required packages###########

import pandas as pd

import numpy as np

import seaborn as sns

import csv as csv

import xgboost as xgb

from xgboost import plot_importance

from matplotlib import pyplot

from sklearn.model_selection import  train_test_split, RandomizedSearchCV

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_error

from sklearn.metrics import accuracy_score

from scipy.stats import uniform, randint

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from rfpimp import *

from sklearn.tree import export_graphviz

from subprocess import call

from IPython.display import Image

sns.set()

%matplotlib inline
###########Data Preprocessing###########

train.timestamp = pd.to_datetime(train.timestamp)

weath_train.timestamp = pd.to_datetime(weath_train.timestamp)



weath_train['month'] = weath_train['timestamp'].dt.month

weath_train['day'] = weath_train['timestamp'].dt.day

weath_train_togrp = weath_train.drop(['timestamp'], axis=1)

weath_train_daily = weath_train_togrp.groupby(['site_id','month','day']).mean().reset_index()



train['day_of_week'] = train['timestamp'].dt.dayofweek

train['month'] = train['timestamp'].dt.month

train['day'] = train['timestamp'].dt.day

train_togrp = train.drop(['timestamp'], axis=1)

train_daily = train_togrp.groupby(['building_id','meter','month','day']).mean().reset_index()



build_train_merged = pd.merge(build_meta, train_daily, on='building_id', how='inner')

data = pd.merge(build_train_merged, weath_train_daily, on=['site_id','month','day'], how='inner')
# To count the NULL/NaN values and drop columns 

len(data) - data.count().sort_values(ascending=True)

percent_missing = data.isnull().sum() * 100 / len(data)

print(percent_missing.sort_values(ascending=False))
########### Split Dataset into Train and Test###########

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['meter_reading']), 

                                                    data[['meter_reading']], 

                                                    test_size=0.25, 

                                                    random_state=42, shuffle=True)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
###########Data Visualization###########

features = ['site_id','building_id','square_feet','meter','month','air_temperature','dew_temperature',

            'wind_speed','cloud_coverage','wind_direction','day','precip_depth_1_hr']
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

#    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for train dataset', fontsize=15)

    plt.show()
plotCorrelationMatrix(X_train,6)
sns.pairplot(pd.concat([X_train,y_train], axis=1), x_vars=features[:4], y_vars='meter_reading')

sns.pairplot(pd.concat([X_train,y_train], axis=1), x_vars=features[4:8], y_vars='meter_reading')

sns.pairplot(pd.concat([X_train,y_train], axis=1), x_vars=features[8:], y_vars='meter_reading')
le = LabelEncoder()

X_train.primary_use = le.fit_transform(X_train['primary_use'])

X_test.primary_use = le.transform(X_test['primary_use'])

X_train.columns = [col.rstrip('_') for col in X_train.columns] 

X_test.columns = [col.rstrip('_') for col in X_test.columns]
def huber_approx_obj(train, preds):

    """

    Function returns gradient and hessein of the Pseudo-Huber function.

    """

    d = preds - train

    h = 1  ## constant

    scale = 1 + (d / h) ** 2

    scale_sqrt = np.sqrt(scale)

    grad = d / scale_sqrt

    hess = 1 / scale / scale_sqrt

    return grad, hess
## define huber loss - minimizing it means maximizing its negative

def huber_loss(preds, train):

    """Function returns the huber loss for h = 1"""

    d = preds - train

    h = 1

    return -1 * np.sum(np.sqrt(1 + (d/h)**2) - 1)
num_features_xgb = ['site_id', 'building_id', 'primary_use', 'square_feet', 'meter', 'month',

                'day', 'air_temperature', 'dew_temperature', 'wind_speed', 'wind_direction',

                'cloud_coverage', 'precip_depth_1_hr']



num_transformer_xgb = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])



preprocessor_xgb = ColumnTransformer(transformers=[('num', num_transformer_xgb, num_features_xgb)])



params_xgb = {

    "rcv_xgb__colsample_bytree": uniform(0.7, 0.1),

    "rcv_xgb__gamma": uniform(0, 0.2),

    "rcv_xgb__learning_rate": uniform(0.03, 0.12), 

    "rcv_xgb__subsample": uniform(0.8, 0.15),

    "rcv_xgb__booster": ['gbtree','dart']

}



pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor_xgb),

                           ('rcv_xgb', xgb.XGBRegressor(objective=huber_approx_obj, 

                                                        feval= huber_loss, max_depth=5,n_estimators=30))])

search_xgb = RandomizedSearchCV(pipeline_xgb, param_distributions=params_xgb, n_iter=10, 

                            scoring='neg_median_absolute_error', random_state=42, cv=5, 

                            verbose=1, n_jobs=4, return_train_score=True)
%%time

search_xgb.fit(X_train[num_features_xgb], y_train)
def report_best_scores(results, n_top=3):

    """Function gives hyperparameters for the top n models"""

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
report_best_scores(search_xgb.cv_results_, 3)
xgb_predictions = search_xgb.predict(X_test)
y_train.hist(bins = 100, range = [1,2000]);
y_train_pred_xgb = search_xgb.predict(X_train[num_features_xgb]) 

y_test_pred_xgb = search_xgb.predict(X_test[num_features_xgb])
train_medae_xgb = median_absolute_error(y_train, y_train_pred_xgb)

test_medae_xgb = median_absolute_error(y_test, y_test_pred_xgb)

print(f'MEDAE XGBoost: Train = {train_medae_xgb:.2f} , Test = {test_medae_xgb:.2f}')
train_rmse_xgb = np.sqrt(mean_squared_error(y_train, y_train_pred_xgb))

test_rmse_xgb = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))

print(f'RMSE XGBoost: Train = {train_rmse_xgb:.2f} , Test = {test_rmse_xgb:.2f}')
train_mae_xgb = mean_absolute_error(y_train, y_train_pred_xgb)

test_mae_xgb = mean_absolute_error(y_test, y_test_pred_xgb)

print(f'MAE XGBoost: Train = {train_mae_xgb:.2f} , Test = {test_mae_xgb:.2f}')
train_medae = huber_loss(y_train.values.ravel(), y_train_pred_xgb)

test_medae = huber_loss(y_test.values.ravel(), y_test_pred_xgb)

print(f'Huber Loss XGBoost: Train = {train_medae:.2f} , Test = {test_medae:.2f}')
I_xgb = importances(search_xgb.best_estimator_, X_test[num_features_xgb], y_test)

print(I_xgb)

plot_importances(I_xgb,title= 'Feature Importance',imp_range=(0, 0.05))
X_test.reset_index(inplace=True)

X_test['meter_reading'] = xgb_predictions[:,]

sample_submission = X_test[['index','meter_reading']]

sample_submission = sample_submission.rename(columns={"index": "row_id"})
########Predicted Results#########

print(sample_submission)
sample_submission.to_csv('sample_submission.csv')