# Data science libraries used by me

# import gc

# import dtale

# import model_evaluation_utils as meu

# import OrignalFunctionsVikram as ofv

import os

import re

import warnings

from collections import Counter

# from pathlib import Path



# import eli5

import featuretools as ft

import hyperopt as hp

import imblearn

# import knime

import lightgbm

# import lime

import matplotlib



# Matplotlib visualization

import matplotlib.pyplot as plt

import numpy as np



# Pandas and numpy for data manipulation

import pandas as pd

import pandas_profiling as pp

# import pydotplus

# import pylab

import scipy.stats as st



# Seaborn for visualization

import seaborn as sns

# import shap

import sklearn

# import statsmodels.api as sm

# import statsmodels.formula.api as smf

# import statsmodels.stats.api as sms

# import statsmodels.stats.stattools as stt

# import statsmodels.tsa.api as smt

import tpot

# from catboost import CatBoostClassifier

# from graphviz import Source

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# from imblearn.over_sampling import SMOTE



# # Internal ipython tool for setting figure size

# from IPython.core.pylabtools import figsize

# from IPython.display import SVG, Image

# from jupyterthemes import jtplot

# from lime import lime_tabular

# from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# from pyforest import *

# from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage

# from scipy.stats import expon as sp_expon

# from scipy.stats import randint as sp_randint

# from scipy.stats import ttest_1samp, ttest_ind, ttest_ind_from_stats

# from scipy.stats import uniform as sp_uniform

# from scipy.stats import wilcoxon

# from six import StringIO



# from skater.core.explanations import Interpretation

# from skater.model import InMemoryModel

# from skater.util.dataops import show_in_notebook

# from sklearn import metrics, tree

# from sklearn.cluster import KMeans

# from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.ensemble import (

    AdaBoostClassifier,

    GradientBoostingClassifier,

    GradientBoostingRegressor,

    RandomForestClassifier,

    RandomForestRegressor,

    VotingClassifier,

)

# from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.feature_selection import RFE

from sklearn.linear_model import (

    ElasticNet,

    ElasticNetCV,

    Lasso,

    LassoCV,

    LinearRegression,

    LogisticRegression,

    Ridge,

)

from sklearn.metrics import (

    accuracy_score,

    classification_report,

    confusion_matrix,

    r2_score,

    roc_auc_score,

    roc_curve,

)



# Splitting data into training and testing

from sklearn.model_selection import (

    GridSearchCV,

    KFold,

    RandomizedSearchCV,

    cross_val_score,

    train_test_split,

)

# from sklearn.multiclass import OneVsRestClassifier

# from sklearn.naive_bayes import BernoulliNB, GaussianNB

# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import (

#     LabelEncoder,

#     OneHotEncoder,

#     PolynomialFeatures,

#     StandardScaler,

#     minmax_scale,

# )

# from sklearn.svm import SVC, SVR

# from sklearn.tree import DecisionTreeClassifier, export_graphviz

# from skrules import SkopeRules

# from statsmodels.compat import lzip

# from statsmodels.formula.api import ols

# from statsmodels.graphics.gofplots import ProbPlot

# from statsmodels.stats.anova import anova_lm

# from statsmodels.stats.multicomp import pairwise_tukeyhsd

# from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# from statsmodels.stats.power import ttest_power

# from tpot import TPOTClassifier, TPOTRegressor



from xgboost import XGBClassifier,XGBRegressor



# %load_ext nb_black





# %load_ext autotime



# No warnings about setting value on copy of slice

pd.options.mode.chained_assignment = None



# Display up to 60 columns of a dataframe

pd.set_option("display.max_columns", 60)



%matplotlib inline



# Set default font size

plt.rcParams["font.size"] = 24



sns.set(font_scale=2)



%load_ext autoreload

%autoreload 2

%matplotlib inline

# os.environ["PATH"] = (

#     os.environ["PATH"] + ";" + os.environ["CONDA_PREFIX"] + r"\Library\bin\graphviz"

# )



# matplotlib.rcParams.update({"font.size": 12})

# # warnings.filterwarnings('ignore')

# %config InlineBackend.figure_format = 'retina'

# gc.collect()

# jtplot.style(theme="monokai", context="notebook", ticks=True, grid=False)
# NOTE : READING THE DATASETS

df2 = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

df1 = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
display(df1)

display(df2)
# NOTE : CHANGE TO PD.DATETIME

df1.Date = pd.to_datetime(df1.Date,infer_datetime_format=True)

df2.Date = pd.to_datetime(df2.Date,infer_datetime_format=True)
# NOTE : CONCISING THE TRAIN DATASET TO 18TH MARCH 2020.

MIN_TEST_DATE = df2.Date.min()

df1 = df1.loc[df1.Date < MIN_TEST_DATE, :]

# NOTE : RESETTING INDEX

df1.reset_index()
# FILLING MISSING VALUES

df1.fillna('',inplace=True)

df2.fillna('',inplace=True)
# NOTE : CREATING NEW REGION COLUMN

df1['Region'] = df1['Country_Region'] + df1['Province_State']

df2['Region'] = df2['Country_Region'] + df2['Province_State']
# NOTE : DROPPING COUNTRY REGION AND STATE

df1.drop(['Country_Region','Province_State'],axis=1,inplace=True)

df2.drop(['Country_Region','Province_State'],axis=1,inplace=True)

# NOTE : CONVERTING DATE COLUMN TO INTEGER

df1.loc[:, 'Date'] = df1.Date.dt.strftime("%m%d")

df2.loc[:, 'Date'] = df2.Date.dt.strftime("%m%d")

sns.lineplot(data=df1,x='Date',y='ConfirmedCases',hue='Region')

plt.show()
sns.lineplot(data=df1,x='Date',y='Fatalities',hue='Region')

plt.show()
# NOTE : CREATING X AND Y

X1 = df1.drop(['ConfirmedCases','Fatalities'],axis=1)

X2 = df1.drop(['ConfirmedCases','Fatalities'],axis=1)

y1 = df1['ConfirmedCases']

y2 = df1['Fatalities']
# NOTE : TEST 1 AND 2

test_1 = df2.copy()

test_2 = df2.copy()
# NOTE : FUNCTION FOR MEAN ENCODING

from sklearn.base import BaseEstimator

class MeanEncoding(BaseEstimator):





    """   In Mean Encoding we take the number

    of labels into account along with the target variable

    to encode the labels into machine comprehensible values    """



    def __init__(self, feature, C=0.1):

        self.C = C

        self.feature = feature



    def fit(self, X_train, y_train):



        df = pd.DataFrame({'feature': X_train[self.feature], 'target': y_train}).dropna()



        self.global_mean = df.target.mean()

        mean = df.groupby('feature').target.mean()

        size = df.groupby('feature').target.size()



        self.encoding = (self.global_mean * self.C + mean * size) / (self.C + size)



    def transform(self, X_test):



        X_test[self.feature] = X_test[self.feature].map(self.encoding).fillna(self.global_mean).values



        return X_test



    def fit_transform(self, X_train, y_train):



        df = pd.DataFrame({'feature': X_train[self.feature], 'target': y_train}).dropna()



        self.global_mean = df.target.mean()

        mean = df.groupby('feature').target.mean()

        size = df.groupby('feature').target.size()

        self.encoding = (self.global_mean * self.C + mean * size) / (self.C + size)



        X_train[self.feature] = X_train[self.feature].map(self.encoding).fillna(self.global_mean).values



        return X_train
for f2 in ['Region']:

    me2 = MeanEncoding(f2, C=0.01 * len(X2[f2].unique()))

    me2.fit(X2, y2)

    X2 = me2.transform(X2)

    test_2 = me2.transform(test_2)
for f1 in ['Region']:

    me1 = MeanEncoding(f1, C=0.01 * len(X1[f1].unique()))

    me1.fit(X1, y1)

    X1 = me1.transform(X1)

    test_1 = me1.transform(test_1)
test_1
test_2
# NOTE : FUNCTION FOR COMPARING DIFFERENT REGRESSORS

def algorithim_boxplot_comparison(X,

                                  y,

                                  algo_list=[],

                                  random_state=3,

                                  scoring='r2',

                                  n_splits=10):

    """To compare metric of different algorithims

       Paramters-

       algo_list : a list conataining algorithim models like random forest, decision trees etc.

       X : dataframe without Target variable

       y : dataframe with only Target variable

       random_state : The seed of randomness. Default is 3

       n_splits : Number of splits used. Default is 3

       ( Default changes from organization to organization)

       Returns-

       median accuracy and the standard deviation accuracy.

       Box Plot of Acuuracy"""

    import matplotlib.pyplot as plt

    from sklearn import model_selection

    import numpy as np

    results=[]

    names=[]

    for algo_name, algo_model in algo_list:

        kfold=model_selection.KFold(shuffle=True,

                                      n_splits=n_splits,

                                      random_state=random_state)

        cv_results=model_selection.cross_val_score(algo_model,

                                                     X,

                                                     y,

                                                     cv=kfold,

                                                     scoring=scoring)

        results.append(cv_results)

        names.append(algo_name)

        msg="%s: %s : (%f) %s : (%f) %s : (%f)" % (

            algo_name, 'median', np.median(cv_results), 'mean',

            np.mean(cv_results), 'variance', cv_results.var(ddof=1))

        print(msg)

    # boxplot algorithm comparison

    fig=plt.figure()

    fig.suptitle('Algorithm Comparison')

    ax=fig.add_subplot(111)

    plt.boxplot(results)

    ax.set_xticklabels(names)

    plt.show()
# NOTE : REGRESSORS

lr = LinearRegression(n_jobs=-1)

rfr = RandomForestRegressor(random_state=96,n_jobs=-1)

gbr = GradientBoostingRegressor(random_state=96)

xgbr = XGBRegressor()
# NOTE : APPENDING THE REGRESSORS IN A LIST

models = []

models.append(('lr',lr))

models.append(('rfr',rfr))

models.append(('gbr',gbr))

models.append(('xgbr',xgbr))

# NOTE : COMPARING DIFFERENT REGRESSORS

algorithim_boxplot_comparison(X1,y1,models,random_state=96,scoring='neg_root_mean_squared_error',n_splits=5)
# NOTE : HYPEROPT

# TODO : USE MORE ADVANCED HYPERPARAMTER TUNING METHODS LIKE OPTUNA, KEARS-TUNER, HPBANDSTER,TUNE



def auc_model(params):

    params = {'n_estimators': int(params['n_estimators']),

              'max_features': int(params['max_features']),

              'min_samples_leaf': int(params['min_samples_leaf']),

              'min_samples_split': int(params['min_samples_split'])}

    clf = RandomForestRegressor(**params,random_state=96,n_jobs=-1)

    return cross_val_score(clf, X1, y1, cv=3, scoring='neg_mean_squared_log_error').mean()





params_space = {'n_estimators': hp.quniform('n_estimators', 0, 300, 50),

                'max_features': hp.quniform('max_features', 1, 3, 1),

                'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 50, 1),

                'min_samples_split': hp.quniform('min_samples_split',1, 50, 1)}

best = 0





def f(params):

    global best

    auc = auc_model(params)

    if auc > best:

        print('New Best', best, params)

    return {'loss': -auc, 'status': STATUS_OK}





trials = Trials()

best = fmin(f, params_space, algo=tpe.suggest, max_evals=200, trials=trials)

print('best:\n',best)
# NOTE : HYPEROPT

# TODO : USE MORE ADVANCED HYPERPARAMTER TUNING METHODS LIKE OPTUNA, KEARS-TUNER, HPBANDSTER,TUNE



def auc_model(params):

    params = {'n_estimators': int(params['n_estimators']),

              'max_features': int(params['max_features']),

              'min_samples_leaf': int(params['min_samples_leaf']),

              'min_samples_split': int(params['min_samples_split'])}

    clf = RandomForestRegressor(**params,random_state=96,n_jobs=-1)

    return cross_val_score(clf, X2, y2, cv=3, scoring='neg_mean_squared_log_error').mean()





params_space = {'n_estimators': hp.quniform('n_estimators', 0, 300, 50),

                'max_features': hp.quniform('max_features', 1, 3, 1),

                'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 50, 1),

                'min_samples_split': hp.quniform('min_samples_split',1,50, 1)}

best = 0





def f(params):

    global best

    auc = auc_model(params)

    if auc > best:

        print('New Best', best, params)

    return {'loss': -auc, 'status': STATUS_OK}





trials = Trials()

best = fmin(f, params_space, algo=tpe.suggest, max_evals=200, trials=trials)

print('best:\n',best)
# NOTE : RANDOMFORESTREGRESSOR FOR CONFIRMEDCASUALTIES

rfr1 = RandomForestRegressor(max_features= 3, min_samples_leaf= 25, min_samples_split= 26, n_estimators= 250,random_state=96,n_jobs=-1)
# NOTE : RANDOMFORESTREGRESSOR FOR FATALITIES

rfr2= RandomForestRegressor(max_features= 3, min_samples_leaf= 17, min_samples_split= 23,n_estimators=100,random_state=96,n_jobs=-1)
# NOTE : FITTING RANDOMFORESTREGRESSOR FOR CONFIRMEDCASUALTIES

rfr1.fit(X1,y1)
# NOTE : FITTING RANDOMFORESTREGRESSOR FOR FATALITIES

rfr2.fit(X2,y2)
# NOTE : PREDICTING CONFIRMEDCASUALTIES

y_n_1 = rfr1.predict(test_1)
# NOTE : PREDICTING FATALITIES

y_n_2 = rfr2.predict(test_2)
# NOTE : SUBMISSION.CSV

df3 = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')
# NOTE : ADDING CONFIRMEDCASES

df3.ConfirmedCases = round(pd.DataFrame(y_n_1))
# NOTE : ADDING FATALITIES

df3.Fatalities = round(pd.DataFrame(y_n_2))
df3
df3.to_csv('submission.csv',index=False)