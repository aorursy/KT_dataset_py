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


import datetime

import warnings

warnings.filterwarnings('ignore')

import scipy as sp

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler , OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.linear_model.bayes import ARDRegression

from sklearn.ensemble.weight_boosting import AdaBoostRegressor

from sklearn.ensemble.bagging import BaggingRegressor

from sklearn.linear_model.bayes import BayesianRidge

from sklearn.cross_decomposition.cca_ import CCA

from sklearn.tree.tree import DecisionTreeRegressor

from sklearn.dummy import DummyRegressor

from sklearn.linear_model.coordinate_descent import ElasticNet , ElasticNetCV

from sklearn.tree.tree import ExtraTreeRegressor

from sklearn.ensemble.forest import ExtraTreesRegressor

from sklearn.gaussian_process.gpr import GaussianProcessRegressor

from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor

from sklearn.linear_model.huber import HuberRegressor

from sklearn.isotonic import IsotonicRegression

from sklearn.neighbors.regression import KNeighborsRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model.least_angle import Lars , LarsCV

from sklearn.linear_model.coordinate_descent import Lasso , LassoCV

from sklearn.linear_model.least_angle import LassoLars ,LassoLarsCV, LassoLarsIC

from sklearn.linear_model.base import LinearRegression

from sklearn.svm.classes import LinearSVR

from sklearn.neural_network.multilayer_perceptron import MLPRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model.coordinate_descent import MultiTaskElasticNet ,MultiTaskElasticNetCV ,MultiTaskLasso , MultiTaskLassoCV

from sklearn.svm.classes import NuSVR

from sklearn.linear_model.omp import OrthogonalMatchingPursuit ,OrthogonalMatchingPursuitCV

from sklearn.cross_decomposition.pls_ import PLSCanonical ,PLSRegression

from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor

from sklearn.linear_model.ransac import RANSACRegressor

from sklearn.neighbors.regression import RadiusNeighborsRegressor

from sklearn.ensemble.forest import RandomForestRegressor

from sklearn.multioutput import RegressorChain

from sklearn.linear_model.ridge import Ridge ,RidgeCV

from sklearn.linear_model.stochastic_gradient import SGDRegressor

from sklearn.svm.classes import SVR

from sklearn.linear_model.theil_sen import TheilSenRegressor

from sklearn.compose._target import TransformedTargetRegressor

from sklearn.ensemble.voting import VotingRegressor

from sklearn.calibration import _SigmoidCalibration

from scipy.sparse import dok_matrix

from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor



import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
train_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



train_df.head()

train_set=train_df.copy(deep=True)

test_set=test_df.copy(deep=True)



train_set.info()

for i in train_set.columns:

    if train_set[i].isnull().sum() >0:

        print(i +':' +str(train_set[i].isnull().sum()))
test_set.info()

for i in test_set.columns:

    if test_set[i].isnull().sum() >0:

        print(i +':' +str(test_set[i].isnull().sum()))
#-----------------data cleaning-----------------------

data_cleaner = [train_set, test_set]



for dataset in data_cleaner :

#combinig year and month in one column

    dataset['MoSold'] = dataset.MoSold.astype('str')

    dataset['YrSold'] = dataset.YrSold.astype('str')

    dataset['Sold_date'] = dataset['YrSold']+'-'+dataset['MoSold']



train_set['Sold_date']=pd.to_datetime(train_set['Sold_date'], format='%Y-%m').astype(str)

test_set['Sold_date']=pd.to_datetime(test_set['Sold_date'], format='%Y-%m').astype(str)





cols=['FireplaceQu','GarageType', 'GarageFinish','GarageQual','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1'] #,'BsmtFinType2','GarageCond'

for dataset in data_cleaner :

    dataset.drop(['MiscFeature','PoolQC','GarageYrBlt', 'Fence', 'Alley','LotFrontage','Id' ] , axis=1, inplace=True)  #,'YrSold','MoSold' Lotfrontage şimdilik silindi, daha sonra dahil edilerek denenebilir.

    dataset.drop(['MasVnrArea','BsmtFinType2','BsmtFinSF2', 'LowQualFinSF','2ndFlrSF' ,'GarageCond'] , axis=1, inplace=True)             #,'BsmtFinSF1'           #deneme amaçlı silinenler

    dataset.drop([], axis=1)

    for i in cols :

        dataset[i].fillna('no_feature',inplace=True)



train_set.dropna(axis=0 , inplace=True)

for i in train_set.columns:

    if train_set[i].isnull().sum() >0:

        print(i +':' +str(train_set[i].isnull().sum()))

        

print('-'*20)



for i in test_set.columns:

    if test_set[i].isnull().sum() >0:

        print(i +':' +str(test_set[i].isnull().sum()))
#---------------------------feature engineering------------------

#combining condition1 and condition2

for dataset in data_cleaner :

    conditions = pd.DataFrame()

    dummies_con1=pd.get_dummies(dataset['Condition1'])

    dummies_con2=pd.get_dummies(dataset['Condition2'])



    for i in dummies_con1.columns :

        if i in dummies_con2.columns :

            conditions[i]=dummies_con1[i]+dummies_con2[i]

        else :

            conditions[i] = dummies_con1[i]

        dataset[i]=conditions[i]



    dataset.drop(['Condition1','Condition2'],axis=1, inplace=True)

    

    

#combining Exterior1st:  and Exterior2nd:



for dataset in data_cleaner :

    Exterior= pd.DataFrame()

    dummies_Ext1=pd.get_dummies(dataset['Exterior1st'])

    dummies_Ext2=pd.get_dummies(dataset['Exterior2nd'])



    for i in (dummies_Ext1+dummies_Ext2).columns :

        if (i in dummies_Ext1.columns) and (i in dummies_Ext2.columns) :

            Exterior[i]=dummies_Ext1[i]+dummies_Ext2[i]

        elif (i in dummies_Ext1) and (i not in dummies_Ext2) :

            Exterior[i] = dummies_Ext1[i]

        else :

            Exterior[i] = dummies_Ext2[i]

        dataset[i]=Exterior[i]



    dataset.drop(['Exterior1st','Exterior2nd'],axis=1, inplace=True)

    dataset.info()







for i in (test_set + train_set).columns :

    if (i not in test_set.columns) and (i in train_set.columns) :

        print(i +' in train_set , but not in test_set')

    elif (i in test_set.columns) and (i not in train_set.columns) :

        print(i + ' in test_set , but not in train_set')



test_set['Other']=0





def correlation_heatmap(df):

    _, ax = plt.subplots(figsize=(60, 60))

    colormap = sns.diverging_palette(220, 10, as_cmap=True)     # renk paleti , bir renk listesi de olabilir.



    _ = sns.heatmap(

        train_set.corr(),

        cmap=colormap,                 #matplotlib colormap name or object, or list of colors, optional

        square=True,

        cbar_kws={'shrink': .9},       #dict of key, value mappings, optional

        ax=ax,                         #Axes in which to draw the plot, otherwise use the currently-active Axes.

        annot=True,                    #If True, write the data value in each cell. If an array-like with the same shape as data, then use this to annotate the heatmap instead of the raw data.

        linewidths=0.1, vmax=1.0, linecolor='white',

        annot_kws={'fontsize': 12}     #dict of key, value mappings, optional

    )



    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(train_set)    
for x in train_set.columns:

    if train_set[x].dtype != 'float64' or train_set[x].dtype != 'int64':

        print('Survival Correlation by:', x)

        print(train_set[[x, 'SalePrice']].groupby(x, as_index=False).mean())

        print('-'*10, '\n')

        



#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html

print(pd.crosstab(data1['Title'],data1[Target[0]]))
X=train_set.drop('SalePrice', axis=1)

y=train_set[['SalePrice']]





X_train , X_test, y_train , y_test =train_test_split(X, y,test_size=0.3 , random_state=20) #dok_matrix(X).toarray()





numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()      #ya da direkt sürun isimleri verilebilir.

categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()



numeric_feature_index = [i for i, col in enumerate(X.columns) if col in numeric_features]

categorical_feature_index = [i for i, col in enumerate(X.columns) if col in categorical_features]



numeric_transformer = Pipeline(steps=[

    ('imputation', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())

    ])



categorical_transformer = Pipeline(steps=[

    ('imputation', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])





preprocessor = ColumnTransformer(

    transformers=[

                    ('num', numeric_transformer, numeric_feature_index),

                    ('cat', categorical_transformer, categorical_feature_index)

                ], remainder='passthrough')







steps = [('preprocessor', preprocessor), ('regressor' , ElasticNet()) ]

pipeline=Pipeline(steps)


param_grid=[

    # {'regressor' : [ElasticNet()]},

    # {'regressor' : [DecisionTreeRegressor()]},

    # {'regressor': [GradientBoostingRegressor(random_state=25)],

    #  'regressor__loss' :['ls', 'lad' ,'huber', 'quantile'],

    #  'regressor__n_estimators' : [100,150,200,250],

    #  'regressor__subsample' :[ 0.8, 0.9 , 1  ],

    #  'regressor__max_depth' : [2, 3, 4 ,5],

    #  'regressor__max_features' : [None,'log2','sqrt']

    #  },

    {'regressor' : [XGBRegressor(random_state=25)],

     'regressor__booster' : ['gbtree', 'gblinear', 'dart'],

     'regressor__eta' : [0.1, 0.2, 0.3, 0.4, 0.5],

     'regressor__subsample' : [0.5 ,0.7, 1],

     'regressor__tree_method' : ['auto', 'hist' ,'approx'],

     'regressor__gamma' : [0 , 5 ,15 , 40, 80, 100],

     'regressor__max_depth' : [4, 5 ,6 ,8 ,10]



     }

    # {'regressor': [HistGradientBoostingRegressor()]} ,    #?????????????

    # {'regressor': [Lasso()]},

    # {'regressor': [RidgeCV()]},

    # {'regressor': [Ridge()]}

    # {'regressor': [RandomForestRegressor()]},

    # {'regressor': [VotingRegressor()]},

    #  {'regressor': [TheilSenRegressor()]}

    # {'regressor': [SVR()]}

    # {'regressor': [MultiTaskElasticNetCV()]},

    # {'regressor':[PassiveAggressiveRegressor()]}

    ]



reg = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=True )

reg.fit(X_train, y_train)





y_pred=reg.predict(X_test)


print("neg_mean_squared_error: {}".format(reg.score(X_test, y_test)))

# print('Best_estimator : {}'.format(reg.best_estimator_))

print('Best_parameters : {}'.format(reg.best_params_))