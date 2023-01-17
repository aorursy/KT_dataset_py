#Import Libraries

import sys 

import pandas as pd 

import matplotlib 

import numpy as np 

import scipy as sc

import IPython

from IPython import display 

import sklearn



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)



#Algorithms

from xgboost import XGBClassifier





from scipy.stats import skew





from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix





%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8





#Import Data

data_train = pd.read_csv('../input/train.csv')

data_test  = pd.read_csv('../input/test.csv')

data_submit = pd.read_csv('../input/sample_submission.csv')



data_train.head()

data_test.head()



#Add dfs

print('Shape Of train df:',data_train.shape)

print('Shape Of test df:',data_test.shape)

data_train_test = pd.concat([data_train, data_test]) #Remeber the index because we have to split data later (200000 in train )

print('Shape Of train test  df:',data_train_test.shape)

#Remove id

print('Shape Of old train test df:',data_train_test.shape)

data_train_test.drop(['ID_code'], axis=1, inplace = True)

print('Shape Of new train test df:',data_train_test.shape)

#Check null

nulls = np.sum(data_train_test.isnull())

nullcols = nulls.loc[(nulls != 0)]

dtypes = data_train_test.dtypes

dtypes2 = dtypes.loc[(nulls != 0)]

info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values")

print('Null in  data:',data_train_test.isnull().sum().sum())

print('Shape of df:', data_train_test.shape)
#PLot



sns.countplot(data_train['target'])
#Skewness in dataset

#As a general rule of thumb: If skewness is less than -1 or greater than 1, the distribution is highly skewed. 

#If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in data_train_test.columns:

    if data_train_test[i].dtype in numeric_dtypes: 

        numerics2.append(i)



skew_features = data_train_test[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

skews = pd.DataFrame({'skew':skew_features})

skews
#Function to plot Graph of each attribute vs target value

def PlotAttributeVsTarget(df,target):

    for y in df.columns:

        if y!= target:

            plt.figure(figsize=(20,5))

            grid = plt.GridSpec(1,2, wspace=0.4, hspace=0.3)

            plt.subplot(grid[0,0])

            sns.distplot(data_train[y])

            #plt.subplot(grid[0,0])

            #sns.scatterplot(x=data_train[y],y=data_train[target])

            plt.subplot(grid[0,1])

            sns.boxplot(x=data_train[target],y=data_train[y])

            

#PlotAttributeVsTarget(data_train.iloc[:,1:],'target') #Commented as this will take long time depending upon data and computing power. Change index to see plots of different variable



#Function to find CV of each variable

#CV is a measure of relative variation or dispersion 

def CoeffiecinetOfVariation(df):

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numerics2 = []

    for y in df.columns:

        if df[y].dtype in numeric_dtypes:

            numerics2.append(y)

    SD_features = df[numerics2].apply(lambda x: np.std(x))

    Mean_features = df[numerics2].apply(lambda x: np.mean(x))

    CV_features = df[numerics2].apply(lambda x: (np.std(x)/np.mean(x)))

    CV = pd.DataFrame({'SD':SD_features,'Mean':Mean_features,'CV':CV_features})

    CV = CV.sort_values(by='CV',ascending=False)

    return CV

    

#CoeffiecinetOfVariation(data_train) 
#Correlation

#correlation of each varibale with target

def Correlation(df):

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numerics2 = []

    crr = {}

    for y in df.columns:

        if df[y].dtype in numeric_dtypes:

            numerics2.append(y)

            for col in numerics2:

                if col != 'target':

                    crr[col] = np.corrcoef(df[col],df['target'])[1,0]

                    

    Crr_df = pd.DataFrame([crr])

    Crr_df = Crr_df.T

    Crr_df.columns = ['CorrCoef']

    Crr_df = Crr_df.sort_values(by='CorrCoef',ascending=False)

    return Crr_df

    

Correlation(data_train)

#Very weak correlations 


X_train = data_train.iloc[:,data_train.columns != 'target']

y_train = data_train.iloc[:,data_train.columns == 'target']

X_train.drop(['ID_code'], axis=1, inplace = True)

X_test = data_test.iloc[:,data_test.columns != 'target']

X_test.drop(['ID_code'], axis=1, inplace = True)

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }
#xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',silent=True, nthread=1)
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



"""

folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )

"""

# Here we go

#random_search.fit(X_train, y_train)
"""

print('\n All results:')

print(random_search.cv_results_)

print('\n Best estimator:')

print(random_search.best_estimator_)

print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))

print(random_search.best_score_ * 2 - 1)

print('\n Best hyperparameters:')

print(random_search.best_params_)

results = pd.DataFrame(random_search.cv_results_)

results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

"""
"""

xgb1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.8, gamma=1.5, learning_rate=0.02,

       max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,

       n_estimators=600, n_jobs=1, nthread=1, objective='binary:logistic',

       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

       seed=None, silent=True, subsample=0.6)

       

    

xgb1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.79, gamma=1.3, learning_rate=0.02,

       max_delta_step=0, max_depth=6, min_child_weight=3, missing=None,

       n_estimators=600, n_jobs=1, nthread=1, objective='binary:logistic',

       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

       seed=None, silent=True, subsample=0.6) """

    

xgb1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.79, gamma=1.3, learning_rate=0.02,

       max_delta_step=0, max_depth=7, min_child_weight=2, missing=None,

       n_estimators=1000, n_jobs=1, nthread=1,

       objective='binary:logistic', random_state=0, reg_alpha=0,

       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,

       subsample=0.6)



xgb1.fit(X_train,y_train.values.ravel())

y_pred = xgb1.predict_proba(X_test)

data_submit['target'] = y_pred

data_submit.to_csv("submit.csv", index=False)