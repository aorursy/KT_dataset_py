%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import numpy as np

import pandas as pd

import os

from __future__ import division 
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing
#filename_train = '/Users/manishrai/Desktop/DSMLAI/Kaggle/Zestimate/res/train_2016_v2.csv'



#filename_properties = '/Users/manishrai/Desktop/DSMLAI/Kaggle/Zestimate/res/properties_2016.csv'



#filename_sample = '/Users/manishrai/Desktop/DSMLAI/Kaggle/Zestimate/res/sample_submission.csv'



# importing the datasets

train_data_df = pd.read_csv('../input/train_2016_v2.csv').dropna(how='all')

properties_data_df = pd.read_csv('../input/properties_2016.csv').dropna(how='all')
df_f1 = pd.merge(train_data_df, properties_data_df, how='left', on=['parcelid'])
col_dtypes = df_f1.dtypes.reset_index()

col = ['colnames', 'datatypes'] 

col_dtypes.columns =col



col_dtypes_num = col_dtypes[(col_dtypes.datatypes == 'int64') | (col_dtypes.datatypes == 'float64')]

df2 = df_f1[list(col_dtypes_num.colnames)]

df2.head()
# Getting the percentage of missing values for each of the cloumns

df_null = df2.isnull().sum() /df2.index.max()

df_null = df_null.reset_index()

df_null.columns = ['colnames', 'pct_null']

df_null_sorted = df_null.sort_values('pct_null')

less_null_cols = list(df_null_sorted[df_null_sorted.pct_null<=0.8].colnames)





from sklearn import preprocessing



df3 = df2[less_null_cols]



imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)

imp.fit(df3)

missing_imputed = imp.transform(df3)

df_missing_imputed = pd.DataFrame(missing_imputed, columns=df3.columns)





df_corr = df_missing_imputed.corrwith(df_missing_imputed.logerror).reset_index()

df_corr.columns = ['colnames', 'correlation']

df_corr_sorted = df_corr.sort_values('correlation')

df_corr_sorted = df_corr_sorted.dropna(how='any')

#df_corr_sorted = df_corr_sorted[(df_corr_sorted['colnames'] != 'logerror')]



df4 = df_corr_sorted[(df_corr_sorted.correlation >= 0.01) | (df_corr_sorted.correlation <= -0.01)]



## Re-do the missing value imputation

#df3[list(df4.colnames)]

df_f = df_missing_imputed.drop(['parcelid'], axis=1)

df_f.head()
import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_friedman1

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import cross_validation, metrics





# Let's do the variable importance thest and randomtreeRegressor

X = df_f.drop(['logerror'], axis=1)

y = df_f.logerror



offset = int(X.shape[0] * 0.8)

X_train, y_train = X[:offset], y[:offset]

X_test, y_test = X[offset:], y[offset:]





# Fit regression model

params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 4,

          'learning_rate': 0.1, 'loss': 'ls'}

clf = GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

#Predict training set:

#dtrain_predictions = clf.predict(X_train)

#dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]





mse = mean_squared_error(y_test, clf.predict(X_test))



#Print model report:

print ("\nModel Report")

print ("Accuracy on the train dataset: %.4g" % clf.score(X_train, y_train))

print ("Accuracy on the test dataset: %.4g" % clf.score(X_test, y_test))



print("MSE: %.4f" % mse)
# #############################################################################

# Plot training deviance



# compute test set deviance

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



for i, y_pred in enumerate(clf.staged_predict(X_test)):

    test_score[i] = clf.loss_(y_test, y_pred)



plt.figure(figsize=(15, 6))

#plt.subplot(211)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',

         label='Training Loss')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Loss')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')

plt.grid()
# #############################################################################

# Plot feature importance

feature_importance = clf.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance) #Returns the indices that would sort an array.

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(15, 6))

#plt.subplot(212)

plt.bar(pos, feature_importance[sorted_idx], align='center')

plt.xticks(pos, X.columns, rotation='vertical')

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.grid()



plt.show()
from sklearn.decomposition import PCA, FactorAnalysis

from sklearn.covariance import ShrunkCovariance, LedoitWolf

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn import preprocessing



X1 = preprocessing.scale(X)
pca = PCA(n_components=31)

pca.fit(X1)

pca_val = pd.DataFrame(pca.explained_variance_)

pca_val.columns = ['PCA']

pca_val.PCA = pca_val.PCA.round(decimals = 2)

pca_val
pca = PCA(n_components=8)

pca.fit_transform(X1)

reduced_dim_df = pca.fit_transform(X1)
components_df = pd.DataFrame(pca.components_).T

components_df.columns = ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5', 'comp_6', 'comp_7', 'comp_8']

components_df
X_red_dim = pd.DataFrame(reduced_dim_df)

X_red_dim.columns = ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5', 'comp_6', 'comp_7', 'comp_8']

X_red_dim.head()
X = X_red_dim

offset = int(X.shape[0] * 0.8)

X_train, y_train = X[:offset], y[:offset]

X_test, y_test = X[offset:], y[offset:]





# Fit regression model

params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 4,

          'learning_rate': 0.1, 'loss': 'ls'}

clf = GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

#Predict training set:

#dtrain_predictions = clf.predict(X_train)

#dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]





mse = mean_squared_error(y_test, clf.predict(X_test))



#Print model report:

print ("\nModel Report")

print ("Accuracy on the train dataset: %.4g" % clf.score(X_train, y_train))

print ("Accuracy on the test dataset: %.4g" % clf.score(X_test, y_test))



print("MSE: %.4f" % mse)