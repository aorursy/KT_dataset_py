import numpy as np

import pandas as pd

import os



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

%matplotlib inline



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go

from plotly.subplots import make_subplots



init_notebook_mode(connected=True)



from tqdm import tqdm_notebook

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit

from sklearn import metrics

from sklearn import linear_model

import gc

import seaborn as sns

import eli5

import shap

from IPython.display import HTML



import matplotlib.pyplot as plt

%matplotlib inline



%env JOBLIB_TEMP_FOLDER=/tmp
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

    



def missingData(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum())/df.isnull().count().sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

    return missing_data
folder_path = '../input/ieee-fraud-detection/'

train_identity = pd.read_csv(f'{folder_path}train_identity.csv')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')

test_identity = pd.read_csv(f'{folder_path}test_identity.csv')

test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
print(f'Train dataset has {train.shape[0]} examples and {train.shape[1]} features.')

print(f'Test dataset has {test.shape[0]} examples and {test.shape[1]} features.')
train_transaction.head(5)
train_identity.head(5)
del train_identity, train_transaction, test_identity, test_transaction;
import missingno as msno



missingdata_df = train.columns[train.isnull().any()].tolist()

msno.heatmap(train[missingdata_df], figsize=(20,20))
missing_data = missingData(train)

missing_data.head(35)
cols_to_drop = missing_data[missing_data['Percent'] > 0.5].index

train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)
missing_data = missingData(train)

null_cols = missing_data[missing_data['Percent']>0].index

missing_data.head(20)
for col in null_cols:

    #print('Data in column {} has format {}.'.format(col, str(train[col].dtype)))

    train[col] = train[col].replace(np.nan, train[col].mode()[0])

    test[col] = test[col].replace(np.nan, train[col].mode()[0])

    #print('Filled the null values of column {}'.format(col))

    #print('-----------------------------')
X = train.drop('isFraud', axis=1)

y = train['isFraud']



print('The design has shape {}'.format(X.shape))

print('The target has shape {}'.format(y.shape))
X_cat = X.select_dtypes(include='object')

X_num = X.select_dtypes(exclude='object')



cat_cols = X_cat.columns.values

num_cols = X_num.columns.values



print('Categorical columns: ', cat_cols)

print('Numerical Columns: ', num_cols)
fig = go.Figure()



fig.add_trace(

    go.Histogram(

        x = y,

        histnorm='probability')

)



fig.update_layout(height=450, width=400, title = 'Distribution of the target [isFraud]')



fig.show()
n_rows, n_cols = 2, 5

fig = make_subplots(rows=n_rows, cols=n_cols)



i=0

for r in range(1, n_rows+1):

    for c in range(1, n_cols+1):

        if i >= len(cat_cols): continue

        feature = cat_cols[i]

        fig.add_trace(

            go.Histogram(

                x=train[feature],

                histnorm='probability',

                name=feature

            ),            

            row= r, col = c

        )

        i+=1



fig.update_layout(height=900, width=1500, title='Distributions of categorical variables')

        

fig.show()
fig = go.Figure()



fig.add_trace(

    go.Histogram(

        x = train['TransactionDT'],

        histnorm='probability',

        name = 'Training set')

)



fig.add_trace(

    go.Histogram(

        x = test['TransactionDT'],

        histnorm='probability',

        name = 'Test set')

)



fig.update_layout(height=450, width=900, title = 'Distribution of transaction dates')



fig.show()
X.head()
cat_cols
encoder = OneHotEncoder(sparse=True)

encoder.fit(X_cat)

X_cat_e = encoder.transform(X_cat)

X_cat_e
from scipy.sparse import coo_matrix

X_num_sparse = coo_matrix(X_num)
X_num_sparse
import scipy

from scipy.sparse import hstack

X_sparse = scipy.sparse.hstack((X_cat_e, X_num_sparse))
scaler = RobustScaler(with_centering=False)

Xsc = scaler.fit_transform(X_sparse)

print(Xsc.shape)
feature_names = []

for arr in encoder.categories_:

    feature_names += list(arr)

feature_names += list(num_cols)

print(feature_names)
params = {'num_leaves': 491,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': 47

         }
# folds = TimeSeriesSplit(n_splits=5)



# aucs = list()

# feature_importances = pd.DataFrame()

# feature_importances['feature'] = feature_names



# training_start_time = time.time()

# for fold, (trn_idx, test_idx) in enumerate(folds.split(Xsc, y)):

#     start_time = time.time()

#     print('Training on fold {}'.format(fold + 1))

    

#     trn_data = lgb.Dataset(Xsc[trn_idx,:], label=y.iloc[trn_idx])

#     val_data = lgb.Dataset(Xsc[test_idx,:], label=y.iloc[test_idx])

#     clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)

    

#     feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()

#     aucs.append(clf.best_score['valid_1']['auc'])

    

#     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time.time() - start_time))))

# print('-' * 30)

# print('Training has finished.')

# print('Total training time is {}'.format(str(datetime.timedelta(seconds=time.time() - training_start_time))))

# print('Mean AUC:', np.mean(aucs))

# print('-' * 30)
# feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)

# feature_importances.to_csv('feature_importances.csv')



# plt.figure(figsize=(16, 16))

# sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

# plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));