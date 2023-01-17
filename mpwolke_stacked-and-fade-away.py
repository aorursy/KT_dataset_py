# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.offline as py

from plotly import tools, subplots

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/hackathon/task_2-owid_covid_data-21_June_2020.csv')

df.head()
# Data processing, metrics and modeling

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from datetime import datetime

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from sklearn import metrics

# Lgbm

import lightgbm as lgb

import catboost

from catboost import Pool

import xgboost as xgb



# Suppr warning

import warnings

warnings.filterwarnings("ignore")
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
%%time

missing_data(df)
import missingno as msno
#Nullity Matrix. The msno.matrix nullity matrix is a data-dense display which lets you quickly visually analyse data completion.

msno.matrix(df.head(10000))
%%time

a = msno.heatmap(df, sort='ascending')

a
%%time

a2 = msno.dendrogram(df)

a2
# Number of unique classes in each object column

df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Find correlations with the target and sort

correlations = df.corr()['total_cases'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
%%time

features = df.columns.values[2:112]

corrs_ = df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

corrs_ = corrs_[corrs_['level_0'] != corrs_['level_1']]

corrs_.head(10)
corrs_.head(10)
corrs_.tail(10)
%%time

corrs = df.corr()

plt.figure(figsize = (10, 6))

# Heatmap of correlations

sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = False, vmax = 0.8)

plt.title('Clustermap');
def plot_dist_col(column):

    pos__df = df[df['total_cases'] ==1]

    neg__df = df[df['total_cases'] ==0]



    '''plot dist curves for train and test weather data for the given column name'''

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.distplot(pos__df[column].dropna(), color='green', ax=ax).set_title(column, fontsize=16)

    sns.distplot(neg__df[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)

    plt.xlabel(column, fontsize=15)

    plt.legend(['total_cases', 'total_tests'])

    plt.show()

plot_dist_col('total_tests')
#fill in mean for floats

for c in df.columns:

    if df[c].dtype=='float16' or  df[c].dtype=='float32' or  df[c].dtype=='float64':

        df[c].fillna(df[c].mean())



#fill in -999 for categoricals

df = df.fillna(-999)

# Label Encoding

for f in df.columns:

    if df[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(df[f].values))

        df[f] = lbl.transform(list(df[f].values))

        

print('Labelling done.') 
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
from sklearn.kernel_ridge import KernelRidge

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df.values)

    rmse= np.sqrt(-cross_val_score(model, df.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
#ntrain = train.shape[0]

#ntest = test.shape[0]

y_train = df.total_cases.values

#all_data = pd.concat((train, test)).reset_index(drop=True)

a#ll_data.drop(['SalePrice'], axis=1, inplace=True)

#print("all_data size is : {}".format(all_data.shape))
#averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



#score = rmsle_cv(averaged_models)

#print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))