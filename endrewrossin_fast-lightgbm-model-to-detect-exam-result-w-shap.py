#Loading libs

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import shap



import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold



import random



pd.set_option('display.max_rows', 500)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
SEED = 99

random.seed(SEED)

np.random.seed(SEED)
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx', sheet_name='All')
df.head()
df.info()
df.isnull().sum()
plt.figure()

plt.title('Exam result', fontsize=14)

sns.countplot('SARS-Cov-2 exam result',data=df)

plt.show()
df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].replace(['negative','positive'], [0,1])
corrmat = round(df.corr(method='pearson'),2)
plt.figure(figsize=(12,6))

sns.heatmap(corrmat.iloc[1:2,:25], vmax=1.0, vmin=-1.0, square=True, annot=True, cmap='RdYlBu')

plt.show()
plt.figure(figsize=(12,6))

sns.heatmap(corrmat.iloc[1:2,25:51], vmax=1.0, vmin=-1.0, square=True, annot=True, cmap='RdYlBu')

plt.show()
plt.figure(figsize=(12,6))

sns.heatmap(corrmat.iloc[1:2,51:], vmax=1.0, vmin=-1.0, square=True, annot=True, cmap='RdYlBu')

plt.show()
dfmodel = df.copy()



# read the "object" columns and use labelEncoder to transform to numeric

for col in dfmodel.columns[dfmodel.dtypes == 'object']:

    le = LabelEncoder()

    dfmodel[col] = dfmodel[col].astype(str)

    le.fit(dfmodel[col])

    dfmodel[col] = le.transform(dfmodel[col])
#change columns names to alphanumeric

dfmodel.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dfmodel.columns]
X = dfmodel.drop(['SARS_Cov_2_exam_result','Patient_ID'], axis = 1)

y = dfmodel['SARS_Cov_2_exam_result']
lgb_params = {

                    'objective':'binary',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.005,

                    'num_leaves': 20,

                    'max_depth':-1,

                    'subsample':0.9,

                    'n_estimators':2500,

                    'seed': SEED,

                    'early_stopping_rounds':100, 

                }
# choose the number of folds, and create a variable to store the auc values and the iteration values.

K = 5

folds = KFold(K, shuffle = True, random_state = SEED)

best_scorecv= 0

best_iteration=0



# Separate data in folds, create train and validation dataframes, train the model and cauculate the mean AUC.

for fold , (train_index,test_index) in enumerate(folds.split(X, y)):

    print('Fold:',fold+1)

          

    X_traincv, X_testcv = X.iloc[train_index], X.iloc[test_index]

    y_traincv, y_testcv = y.iloc[train_index], y.iloc[test_index]

    

    train_data = lgb.Dataset(X_traincv, y_traincv)

    val_data   = lgb.Dataset(X_testcv, y_testcv)

    

    LGBM = lgb.train(lgb_params, train_data, valid_sets=[train_data,val_data], verbose_eval=250)

    best_scorecv += LGBM.best_score['valid_1']['auc']

    best_iteration += LGBM.best_iteration



best_scorecv /= K

best_iteration /= K

print('\n Mean AUC score:', best_scorecv)

print('\n Mean best iteration:', best_iteration)
lgb_params = {

                    'objective':'binary',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.05,

                    'num_leaves': 20,

                    'max_depth':-1,

                    'subsample':0.9,

                    'n_estimators':round(best_iteration),

                    'seed': SEED,

                    'early_stopping_rounds':None, 

                }



train_data_final = lgb.Dataset(X, y)

LGBM = lgb.train(lgb_params, train_data)
print(LGBM)
# telling wich model to use

explainer = shap.TreeExplainer(LGBM)

# Calculating the Shap values of X features

shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[1], X, plot_type="bar")
shap.summary_plot(shap_values[1], X)