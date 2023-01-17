import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format(pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}". format(matplotlib.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



import sklearn #collection of machine learning algorithms

print("scikit-learn version: {}". format(sklearn.__version__))



#misc libraries

import timeit

import random

import time





#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier, XGBRegressor



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

# from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
print("data_train.shape: {}".format(data_train.shape))

print("data_test.shape: {}".format(data_test.shape))
X = data_train.copy(deep = True)

X_test = data_test.copy(deep = True)
X.drop(["SalePrice"], axis=1, inplace=True)
columns_too_less=["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscVal", "MiscFeature"]

X.drop(columns_too_less, axis=1, inplace=True)

X_test.drop(columns_too_less, axis=1, inplace=True)
#columns_drop_std = X.std()[X.std() < 3].index

#X.drop(columns_drop_std, axis=1, inplace = True)

#X_test.drop(columns_drop_std, axis=1, inplace = True)
#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(28, 24))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(X)
X = X.select_dtypes(exclude=["object"])

X_test = X_test.select_dtypes(exclude=["object"])
columns_drop_corr = X.corrwith(data_train["SalePrice"])[X.corrwith(data_train["SalePrice"]) < 0.1].index

X.drop(columns_drop_corr, axis=1, inplace = True)

X_test.drop(columns_drop_corr, axis=1, inplace = True)
columns = X.columns
from sklearn.preprocessing import Imputer

imputer = Imputer()

X_numerical = imputer.fit_transform(X)

X_test_numerical = imputer.transform(X_test)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_numerical = scaler.fit_transform(X_numerical)

X_test_numerical = scaler.transform(X_test_numerical)
X_numerical = pd.DataFrame(X_numerical, columns = columns)

X_test_numerical = pd.DataFrame(X_test_numerical, columns = columns)
print("X_numerical.shape: {}".format(X_numerical.shape))

print("X_test_numerical.shape: {}".format(X_test_numerical.shape))
X = data_train.select_dtypes(include=["object"]).copy(deep=True)

X_test = data_test.select_dtypes(include=["object"]).copy(deep=True)
# columns = X.count()[X.count() < 300].index

# columns
# X.drop(columns, axis=1, inplace=True)

# X_test.drop(columns, axis=1, inplace=True)
Catcols = X.columns
from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

le = LabelEncoder()

for col in Catcols:

    lbl_enc = LabelEncoder()

    im = SimpleImputer(strategy='most_frequent')

    X[col] = im.fit_transform(X[col].values.reshape(-1, 1))

    X_test[col] = im.transform(X_test[col].values.reshape(-1, 1))

    X[col] = lbl_enc.fit_transform(X[col].values)

    X_test[col] = lbl_enc.transform(X_test[col].values)
# X = pd.get_dummies(X)

# X_test = pd.get_dummies(X_test)
# columns = []

# for i in X.columns:

#     if i not in X_test.columns:

#         df = pd.DataFrame(np.zeros(len(X)-1),columns=[i])

#         X_test = pd.concat([X_test, df], axis=1)
#abs_median = np.abs(X.corrwith(data_train["SalePrice"]).median())
#columns = X.corrwith(data_train["SalePrice"])[np.abs(X.corrwith(data_train["SalePrice"]) < abs_median)].index

#X.drop(columns, axis=1, inplace = True)

#X_test.drop(columns, axis=1, inplace = True)
X_categorical = X.copy(deep=True)

X_test_categorical = X_test.copy(deep=True)
X_train = pd.concat([X_numerical, X_categorical], axis=1)

X_test = pd.concat([X_test_numerical, X_test_categorical], axis=1)

y_train = data_train["SalePrice"].copy(deep=True)
print("X_train.shape: {}".format(X_train.shape))

print("X_test.shape: {}".format(X_test.shape))

print("y_train.shape: {}".format(y_train.shape))
# train = pd.concat([X_train, y_train], axis=1)
# def threshold(column):

#     return train[column].std()*3
# columns = X_numerical.columns

# for i in columns:

#     train = train[np.abs(train.loc[:,i]- train[i].mean()) <= threshold(i)]

# train = train[np.abs(train.loc[:,'SalePrice']-train['SalePrice'].mean()) <= threshold('SalePrice')]
# y_train = train['SalePrice']

# X_train = train.drop(['SalePrice'], axis=1)
# #Machine Learning Algorithm (MLA) Selection and Initialization

# MLA = [

#     #Ensemble Methods

#     #ensemble.AdaBoostClassifier(),

#     #ensemble.BaggingClassifier(),

#     #ensemble.ExtraTreesClassifier(),

#     #ensemble.GradientBoostingClassifier(),

#     #ensemble.RandomForestClassifier(),

    

#     #Gaussian Processes

#     #gaussian_process.GaussianProcessClassifier(),

    

#     #GLM

#     #linear_model.LogisticRegressionCV(),

#     #linear_model.PassiveAggressiveClassifier(),

#     #linear_model.RidgeClassifierCV(),

#     #linear_model.SGDClassifier(),

#     #linear_model.Perceptron(),

    

#     #Navies Bayes

#     #naive_bayes.BernoulliNB(),

#     #naive_bayes.GaussianNB(),

    

#     #Nearest Neighbor

#     #neighbors.KNeighborsClassifier(),

    

#     #SVM

#     #svm.SVC(probability=True),

#     #svm.NuSVC(probability=True),

#     #svm.LinearSVC(),

    

#     #Trees    

#     #tree.DecisionTreeClassifier(),

#     #tree.ExtraTreeClassifier(),

    

#     #Discriminant Analysis

#     #discriminant_analysis.LinearDiscriminantAnalysis(),

#     #discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

#     #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

#     #XGBClassifier()

    

#     #linear_model.LinearRegression(),

#     linear_model.Ridge(),

#     linear_model.ElasticNet(),

#     linear_model.LassoLars(),

#     linear_model.BayesianRidge(),

#     #linear_model.ARDRegression(),

    

#     ensemble.RandomForestRegressor(),

#     ensemble.ExtraTreesRegressor(),

#     tree.DecisionTreeRegressor(),

    

#     neighbors.KNeighborsRegressor(),

    

#     XGBRegressor(learning_rate =0.01,

#                  n_estimators=5000,

#                  max_depth=4,

#                  min_child_weight=6,

#                  gamma=0,

#                  subsample=0.8,

#                  colsample_bytree=0.8,

#                  reg_alpha=0.01,

#                  objective= 'reg:linear',

#                  nthread=4,

#                  scale_pos_weight=1,

#                  seed=27)

#     ]
# cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0)
# #create table to compare MLA metrics

# MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']

# MLA_compare = pd.DataFrame(columns = MLA_columns)



# #create table to compare MLA predictions

# MLA_predict = pd.DataFrame()#y_train

# MLA_pred = pd.DataFrame()



# #index through MLA and save performance to table

# row_index = 0

# for alg in MLA:



#     #set name and parameters

#     MLA_name = alg.__class__.__name__

#     MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

#     MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

#     #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate

#     cv_results = model_selection.cross_validate(alg, X_train, y_train, cv = cv_split, scoring='r2')



#     MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()

#     MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()

#     MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   

#     #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets

#     MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!

    



#     #save MLA predictions - see section 6 for usage

#     alg.fit(X_train, y_train)

#     MLA_predict[MLA_name] = alg.predict(X_train)

#     MLA_pred[MLA_name] = alg.predict(X_test)

    

#     row_index+=1



    

# #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html

# MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

# MLA_compare

# #MLA_predict
# #barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html

# sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')



# #prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

# plt.title('Machine Learning Algorithm Accuracy Score \n')

# plt.xlabel('Accuracy Score (%)')

# plt.ylabel('Algorithm')
#predict = MLA[-1].predict(X_test)

#y_test = pd.read_csv("../input/sample_submission.csv")

#my_submission = pd.DataFrame({'Id': y_test.Id, 'SalePrice': predict.astype(int)})

# you could use any filename. We choose submission here

#my_submission.to_csv('submission.csv', index=False)
# import xgboost as xgb

# from sklearn import cross_validation, metrics

# from sklearn.grid_search import GridSearchCV



# import matplotlib.pylab as plt

# %matplotlib inline

# from matplotlib.pylab import rcParams



# rcParams['figure.figsize'] = 12, 4



# train = pd.concat([X_train, y_train], axis=1)

# target = 'SalePrice'

# IDcol = 'ID'
# def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

#     if useTrainCV:

#         xgb_param = alg.get_xgb_params()

#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

#             metrics= 'mae' , early_stopping_rounds=early_stopping_rounds)#, show_progress=False)

#         alg.set_params(n_estimators=cvresult.shape[0])

    

#     #Fit the algorithm on the data

#     alg.fit(dtrain[predictors], dtrain[target],eval_metric= 'mae' )#'mae')

        

#     #Predict training set:

#     dtrain_predictions = alg.predict(dtrain[predictors])

#     #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

        

#     #Print model report:

#     print ("\nModel Report")

#     print ("r2 score : %.4g" % metrics.r2_score(dtrain[target].values, dtrain_predictions))

#     #print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

                    

#     #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)

#     #feat_imp.plot(kind='bar', title='Feature Importances')

#     #plt.ylabel('Feature Importance Score')
# #uncomment it when u are tuning

# #Choose all predictors except target & IDcols

# predictors = [x for x in train.columns if x not in [target]]

# xgb1 = XGBRegressor(

#  learning_rate =0.1,

#  n_estimators=1000,

#  max_depth=5,

#  min_child_weight=1,

#  gamma=0,

#  subsample=0.8,

#  colsample_bytree=0.8,

#  objective= 'reg:linear',

#  nthread=4,

#  scale_pos_weight=1,

#  seed=27)

# modelfit(xgb1, train, predictors)
# param_test1 = {

#  'max_depth':list(range(3,10,2)),

#  'min_child_weight':list(range(1,6,2))

# }

# gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=5,

#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 

#  param_grid = param_test1, scoring='r2',n_jobs=4,iid=False, cv=5)

# gsearch1.fit(train[predictors],train[target])

# gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
# param_test2 = {

#  'max_depth':[3,4,5],

#  'min_child_weight':[4,5,6,7]

# }

# gsearch2 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=4,

#  min_child_weight=5, gamma=0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed=27), 

#  param_grid = param_test2, scoring='r2',n_jobs=4,iid=False, cv=5)

# gsearch2.fit(train[predictors],train[target])

# gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
# param_test3 = {

#  'gamma':[i/10.0 for i in range(0,5)]

# }

# gsearch3 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=4,

#  min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'reg:linear', nthread=4, scale_pos_weight=1,seed=27), 

#  param_grid = param_test3, scoring='r2',n_jobs=4,iid=False, cv=5)

# gsearch3.fit(train[predictors],train[target])

# gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
# xgb2 = XGBRegressor(

#  learning_rate =0.1,

#  n_estimators=1000,

#  max_depth=4,

#  min_child_weight=6,

#  gamma=0,

#  subsample=0.8,

#  colsample_bytree=0.8,

#  objective= 'reg:linear',

#  nthread=4,

#  scale_pos_weight=1,

#  seed=27)

# modelfit(xgb2, train, predictors)
# param_test4 = {

#  'subsample':[i/10.0 for i in range(6,10)],

#  'colsample_bytree':[i/10.0 for i in range(6,10)]

# }

# gsearch4 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=177, max_depth=4,

#  min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'reg:linear', nthread=4, scale_pos_weight=1,seed=27), 

#  param_grid = param_test4, scoring='r2',n_jobs=4,iid=False, cv=5)

# gsearch4.fit(train[predictors],train[target])

# gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
# param_test5 = {

#  'subsample':[i/100.0 for i in range(70,90,5)],

#  'colsample_bytree':[i/100.0 for i in range(70,90,5)]

# }

# gsearch5 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=177, max_depth=4,

#  min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'reg:linear', nthread=4, scale_pos_weight=1,seed=27), 

#  param_grid = param_test5, scoring='r2',n_jobs=4,iid=False, cv=5)

# gsearch5.fit(train[predictors],train[target])

# gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
# param_test6 = {

#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

# }

# gsearch6 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=177, max_depth=4,

#  min_child_weight=6, gamma=0.0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'reg:linear', nthread=4, scale_pos_weight=1,seed=27), 

#  param_grid = param_test6, scoring='r2',n_jobs=4,iid=False, cv=5)

# gsearch6.fit(train[predictors],train[target])

# gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
# param_test7 = {

#  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]

# }

# gsearch7 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=177, max_depth=4,

#  min_child_weight=6, gamma=0.0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'reg:linear', nthread=4, scale_pos_weight=1,seed=27), 

#  param_grid = param_test7, scoring='r2',n_jobs=4,iid=False, cv=5)

# gsearch7.fit(train[predictors],train[target])

# gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
# xgb3 = XGBRegressor(

#  learning_rate =0.01,

#  n_estimators=5000,

#  max_depth=4,

#  min_child_weight=6,

#  gamma=0,

#  subsample=0.8,

#  colsample_bytree=0.8,

#  reg_alpha=0.01,

#  objective= 'reg:linear',

#  nthread=4,

#  scale_pos_weight=1,

#  seed=27)

# modelfit(xgb3, train, predictors)
# predict = MLA[8].predict(X_test)
# y_test = pd.read_csv("../input/sample_submission.csv")

# my_submission = pd.DataFrame({'Id': y_test.Id, 'SalePrice': predict})

# # you could use any filename. We choose submission here

# my_submission.to_csv('submission.csv', index=False)
X = X_train.copy(deep=True)

y = y_train.copy(deep=True)
from sklearn.preprocessing import MinMaxScaler

target_scaler = MinMaxScaler()

y = target_scaler.fit_transform(y.values.reshape(-1, 1))
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, test_size=0.1)
features = X_train.columns

X_train = [X_train.loc[:, features].values[:, k].astype('float') for k in range(X_train.loc[:, features].values.shape[1])]

X_val = [X_val.loc[:, features].values[:, k].astype('float') for k in range(X_val.loc[:, features].values.shape[1])]

X_test = [X_test.loc[:, features].values[:, k].astype('float') for k in range(X_test.loc[:, features].values.shape[1])]
from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras.models import Model, load_model

from tensorflow.keras import callbacks

from tensorflow.keras import backend as K

from tensorflow.keras import utils

import tensorflow as tf

from tensorflow.python.ops import math_ops
def create_model(data, catcols):

    inputs = []

    outputs = []

    for c in data.columns:

        if c in catcols:

            num_unique_values = int(data[c].nunique())

#             embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

            embed_dim = 128

            inp = layers.Input(shape=(1,))

            out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)

            out = layers.SpatialDropout1D(0.3)(out)

            out = layers.Reshape(target_shape=(embed_dim, ))(out)

            inputs.append(inp)

            outputs.append(out)

        else:

            inp = layers.Input(shape=(1,))

            out = layers.Dense(128, activation="relu")(inp)

            inputs.append(inp)

            outputs.append(inp)

    

    x = layers.Concatenate()(outputs)

    x = layers.BatchNormalization()(x)

    

    x = layers.Dense(300, activation="relu")(x)

    x = layers.Dropout(0.5)(x)

    x = layers.BatchNormalization()(x)

    

    x = layers.Dense(150, activation="relu")(x)

    x = layers.Dropout(0.5)(x)

    x = layers.BatchNormalization()(x)

    

    x = layers.Dense(100, activation="relu")(x)

    x = layers.Dropout(0.5)(x)

    x = layers.BatchNormalization()(x)

    

    x = layers.Dense(50, activation="relu")(x)

    x = layers.Dropout(0.5)(x)

    x = layers.BatchNormalization()(x)

    

    y = layers.Dense(1, activation="sigmoid")(x)



    model = Model(inputs=inputs, outputs=y)

    return model
def rmsle(y_true, y_pred):

    ans = tf.math.sqrt(tf.keras.losses.MSE(tf.math.log1p(y_true), tf.math.log1p(y_pred)))

    return ans
x = [60., 80., 90., 750.]

y = [67., 78., 91., 102.]

rmsle(x,y) # 1.160
model = create_model(X, Catcols)

# model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=[rmsle])

model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')#, metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])
es = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=10,

                                 verbose=1, mode='auto', baseline=None, restore_best_weights=True)



rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,

                                  patience=5, min_lr=1e-10, mode='auto', verbose=1)
model.fit(X_train, y_train,

            validation_data=(X_val, y_val),

            verbose=1,

            batch_size=32,

            callbacks=[es, rlr],

            epochs=5000

            )
y_val_ = target_scaler.inverse_transform(y_val)
y_pred = model.predict(X_val)

y_pred = target_scaler.inverse_transform(y_pred)

y_pred
from sklearn.metrics import mean_squared_error

def rmsle(y_true, y_pred):

    return np.sqrt(mean_squared_error(np.log1p(y_true),np.log1p(y_pred)))

print(rmsle(y_val_, y_pred))
y_pred = model.predict(X_test)

y_pred = target_scaler.inverse_transform(y_pred)

y_pred = y_pred.reshape(-1)
y_test = pd.read_csv("../input/sample_submission.csv")

my_submission = pd.DataFrame({'Id': y_test.Id, 'SalePrice': y_pred})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
my_submission
# columns = ['BayesianRidge','XGBRegressor', 'ExtraTreesRegressor']

# model = linear_model.LinearRegression()

# model.fit(MLA_predict[columns], y_train)
# model.score(MLA_pred[columns], y_test[target])
# pred = model.predict(MLA_pred[columns])