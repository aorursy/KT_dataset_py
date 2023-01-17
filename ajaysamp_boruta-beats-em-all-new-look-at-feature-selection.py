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
# lets create a feature matrix

feature_names =    ['linear', 'nonlinear_square', 'nonlinear_sin',  'interaction_1', 'interaction_2',  'interaction_3',

   'noise_1', 'noise_2', 'noise_3', 'noise_4', 'noise_5', 'noise_6','noise_7', 'noise_8', 'noise_9','noise_10']
# define function for creating y

def yfromX(X):

    y = X['linear'] + X['nonlinear_square']**2 + np.sin(3 * X['nonlinear_sin']) + (X['interaction_1'] * X['interaction_2'] * X['interaction_3'])

    return y
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split



# create x and y

np.random.seed(0)



X = pd.DataFrame(np.random.normal(size = (20000, len(feature_names))), columns = feature_names)

y = yfromX(X)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



from sklearn.metrics import mean_absolute_error

from eli5.sklearn import PermutationImportance
# linear regression

lr = LinearRegression()

lr.fit(X_train, y_train)



lr_train_preds = lr.predict(X_train)

lr_test_preds = lr.predict(X_test)



lr_train_mae = mean_absolute_error(y_train, lr_train_preds)

lr_test_mae = mean_absolute_error(y_test, lr_test_preds)



lr_fi = PermutationImportance(lr, cv = 'prefit', n_iter = 3).fit(X_train, y_train).feature_importances_
# KNN

knn = KNeighborsRegressor(n_neighbors = int(np.sqrt(len(X_train))))

knn.fit(X_train, y_train)



knn_train_preds = knn.predict(X_train)

knn_test_preds = knn.predict(X_test)



knn_train_mae = mean_absolute_error(y_train, knn_train_preds)

knn_test_mae = mean_absolute_error(y_test, knn_test_preds)



knn_fi = PermutationImportance(knn).fit(X_train, y_train).feature_importances_
# Support Vector Regression 

svr = SVR(C = .1)

svr.fit(X_train, y_train)



svr_train_preds = svr.predict(X_train)

svr_test_preds = svr.predict(X_test)



svr_train_mae = mean_absolute_error(y_train, svr_train_preds)

svr_test_mae = mean_absolute_error(y_test, svr_test_preds)



svr_fi = PermutationImportance(svr).fit(X_train, y_train).feature_importances_
# Random Forest 

rf = RandomForestRegressor(max_depth=5)

rf.fit(X_train, y_train)



rf_train_preds = rf.predict(X_train)

rf_test_preds = rf.predict(X_test)



rf_train_mae = mean_absolute_error(y_train, rf_train_preds)

rf_test_mae = mean_absolute_error(y_test, rf_test_preds)



rf_fi = rf.feature_importances_
# XGBOOST 

xgb = XGBRegressor(max_depth=5)

xgb.fit(X_train, y_train)



xgb_train_preds = xgb.predict(X_train)

xgb_test_preds = xgb.predict(X_test)



xgb_train_mae = mean_absolute_error(y_train, xgb_train_preds)

xgb_test_mae = mean_absolute_error(y_test, xgb_test_preds)



xgb_fi = xgb.feature_importances_
# Light GBM

lgb = LGBMRegressor(max_depth=5)

lgb.fit(X_train, y_train)



lgb_train_preds = lgb.predict(X_train)

lgb_test_preds = lgb.predict(X_test)



lgb_train_mae = mean_absolute_error(y_train, lgb_train_preds)

lgb_test_mae = mean_absolute_error(y_test, lgb_test_preds)



lgb_fi = lgb.feature_importances_
# create a dataframe for feature importances and mean absolute error

mae_df = pd.DataFrame(columns=['Train','Test'])



# add mae's to the mae_df

mae_df.loc['Linear Regression','Train'] =  lr_train_mae

mae_df.loc['Linear Regression','Test'] =  lr_test_mae



mae_df.loc['KNN','Train'] =  knn_train_mae

mae_df.loc['KNN','Test'] =  knn_test_mae



mae_df.loc['Support Vector Regression','Train'] =  svr_train_mae

mae_df.loc['Support Vector Regression','Test'] =  svr_test_mae



mae_df.loc['Random Forest','Train'] =  rf_train_mae

mae_df.loc['Random Forest','Test'] =  rf_test_mae



mae_df.loc['XGBoost','Train'] =  xgb_train_mae

mae_df.loc['XGBoost','Test'] =  xgb_test_mae



mae_df.loc['Light GBM','Train'] =  lgb_train_mae

mae_df.loc['Light GBM','Test'] =  lgb_test_mae



mae_df['Model'] = mae_df.index

mae_df['Train'] = mae_df['Train'].astype(float)

mae_df['Test'] = mae_df['Test'].astype(float)



mae_df = pd.melt(mae_df, id_vars=['Model'], value_vars=['Train','Test'])
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.ticker import FuncFormatter 

import matplotlib.ticker as mtick
# plot mae 

fig,ax = plt.subplots(figsize=(10,4))

ax = sns.barplot(x='value',y='Model', hue='variable',data=mae_df.sort_values(by='value',ascending=False))

plt.xlabel('Mean Absolute Error')

plt.ylabel('')

plt.title('Mean Absolute Error Comparison')

plt.tight_layout()
# create faeature importance dataframe 

fi_df = pd.DataFrame(columns=['LR', 'KNN','SVR','RF','XGB','LGBM','Features'])



fi_df['Features'] = feature_names

fi_df['LR'] = lr_fi

fi_df['KNN'] = knn_fi

fi_df['SVR'] = svr_fi

fi_df['RF'] = rf_fi

fi_df['XGB'] = xgb_fi

fi_df['LGBM'] = lgb_fi/1000
fig = plt.figure(figsize=(18,8))



plt.subplot(2, 3, 1)

ax = sns.barplot(x='LR',y='Features',data=fi_df.sort_values(by='LR',ascending=False),color='b')

ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.xlabel('Feature Importance')

plt.ylabel('')

plt.title('Linear Regression')

plt.axvline(x=0.1, color='r', linestyle='dashed')

plt.tight_layout()



plt.subplot(2, 3, 2)

ax = sns.barplot(x='KNN',y='Features',data=fi_df.sort_values(by='KNN',ascending=False),color='b')

ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.xlabel('Feature Importance')

plt.ylabel('')

plt.title('K-Nearest Neighbors')

plt.axvline(x=0.1, color='r', linestyle='dashed')

plt.tight_layout()



plt.subplot(2, 3, 3)

ax = sns.barplot(x='SVR',y='Features',data=fi_df.sort_values(by='SVR',ascending=False),color='b')

ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.xlabel('Feature Importance')

plt.ylabel('')

plt.title('Support Vector Regression')

plt.axvline(x=0.1, color='r', linestyle='dashed')

plt.tight_layout()



plt.subplot(2, 3, 4)

ax = sns.barplot(x='RF',y='Features',data=fi_df.sort_values(by='RF',ascending=False),color='b')

ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.xlabel('Feature Importance')

plt.ylabel('')

plt.title('Random Forest')

plt.axvline(x=0.1, color='r', linestyle='dashed')

plt.tight_layout()



plt.subplot(2, 3, 5)

ax = sns.barplot(x='XGB',y='Features',data=fi_df.sort_values(by='XGB',ascending=False),color='b')

ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.xlabel('Feature Importance')

plt.ylabel('')

plt.title('XGBoost')

plt.axvline(x=0.1, color='r', linestyle='dashed')

plt.tight_layout()



plt.subplot(2, 3, 6)

ax = sns.barplot(x='LGBM',y='Features',data=fi_df.sort_values(by='LGBM',ascending=False),color='b')

ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

plt.xlabel('Feature Importance')

plt.ylabel('')

plt.title('Light GBM')

plt.axvline(x=0.1, color='r', linestyle='dashed')

plt.tight_layout()
from boruta import BorutaPy



new_rf = RandomForestRegressor(n_jobs = -1, max_depth = 5)



boruta_selector = BorutaPy(new_rf, n_estimators = 'auto', random_state = 0)

boruta_selector.fit(np.array(X_train), np.array(y_train))



boruta_ranking = boruta_selector.ranking_

selected_features = np.array(feature_names)[boruta_ranking <= 2]
boruta_ranking = pd.DataFrame(data=boruta_ranking, index=X_train.columns.values, columns=['values'])

boruta_ranking['Variable'] = boruta_ranking.index

boruta_ranking.sort_values(['values'], ascending=True, inplace=True)
fig,ax = plt.subplots(figsize=(8,4))

ax = sns.barplot(x='values',y='Variable',data=boruta_ranking, color='b')

plt.title('Boruta Feature Ranking')

plt.xlabel('')

plt.ylabel('')

plt.tight_layout()