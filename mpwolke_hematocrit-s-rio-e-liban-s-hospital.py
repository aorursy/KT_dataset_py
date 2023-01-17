# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import shap

import random



import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')

df.head()
df.isnull().sum()
plt.style.use('dark_background')

sns.countplot(x="ICU",data=df,palette="GnBu_d",edgecolor="black")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

# changing the font size

sns.set(font_scale=1)
SEED = 99

random.seed(SEED)

np.random.seed(SEED)
corrmat = round(df.corr(method='pearson'),2)
plt.figure(figsize=(12,6))

sns.heatmap(corrmat.iloc[1:2,:25], vmax=1.0, vmin=-1.0, square=True, annot=True, cmap='summer')

plt.show()
plt.figure(figsize=(12,6))

sns.heatmap(corrmat.iloc[1:2,25:51], vmax=1.0, vmin=-1.0, square=True, annot=True, cmap='summer')

plt.show()
plt.figure(figsize=(12,6))

sns.heatmap(corrmat.iloc[1:2,51:], vmax=1.0, vmin=-1.0, square=True, annot=True, cmap='summer')

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
X = dfmodel.drop(['ICU','PATIENT_VISIT_IDENTIFIER'], axis = 1)

y = dfmodel['ICU']
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)