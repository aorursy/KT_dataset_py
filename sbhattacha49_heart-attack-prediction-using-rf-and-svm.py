import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as stats
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
# Create Data audit Report for continuous variables

def continuous_var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  

                      x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),

                          x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), 

                              x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 

                  index = ['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1', 

                               'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])
# An utility function to create dummy variable

def create_dummies(df, colname):

    col_dummies = pd.get_dummies(df[colname], prefix = colname, drop_first = True)

    df = pd.concat([df, col_dummies], axis = 1)

    df.drop(colname, axis = 1, inplace = True )

    return df
#df_heart=pd.read_csv('F:/analytix_labs/extras/Health care Data set on Heart attack possibility/heart.csv')

df_heart=pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')
df_heart.info()
df_heart.apply(continuous_var_summary)
def cont_boxplot(x):

    sns.boxplot(x)

    plt.show()

def cont_distplot(x):

    sns.distplot(x)

    plt.show()
df_heart.apply(cont_boxplot)
df_heart.apply(cont_distplot)
df_heart_cont=df_heart[['age','trestbps','chol','thalach','oldpeak']]

df_heart_cat=df_heart.loc[:,df_heart.columns.difference(['age','trestbps','chol','thalach','oldpeak'])]
df_heart.fbs.value_counts()
target=df_heart_cat.target
df_heart_cat.drop(columns='target',inplace=True)
# for c_feature in categorical_features

for c_feature in list(df_heart_cat.columns):

    df_heart_cat[c_feature] = df_heart_cat[c_feature].astype('category')

    df_heart_cat = create_dummies(df_heart_cat, c_feature)
df_heart_cat
df_heart_cat=df_heart_cat.astype('category')
df_heart_cat.info()
df_heart_new=pd.concat([df_heart_cont,df_heart_cat],axis=1)
target.value_counts()
from sklearn.ensemble import RandomForestClassifier

import sklearn.tree as dt

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split





train_X, test_X, train_y, test_y = train_test_split(df_heart_new,

                                                  target,

                                                  test_size = 0.3,

                                                  random_state = 555 )
from imblearn.over_sampling import RandomOverSampler
#!pip install imblearn



ros = RandomOverSampler(random_state=123)



train_X_os, train_y_os = ros.fit_sample(train_X, train_y)



unique_elements, counts_elements = np.unique(train_y_os, return_counts=True)

print("Frequency of unique values of the said array:")

print(np.asarray((unique_elements, counts_elements)))
from sklearn.model_selection import GridSearchCV
pargrid_rf = {'n_estimators': np.arange(50,60,70),

                  'max_features': np.arange(3,8)}



#from sklearn.grid_search import GridSearchCV

gscv_rf = GridSearchCV(estimator=RandomForestClassifier(), 

                        param_grid=pargrid_rf, 

                        cv=5,

                        verbose=True, n_jobs=-1)



gscv_results = gscv_rf.fit(train_X_os, train_y_os)
gscv_results.best_params_
gscv_rf.best_score_
radm_clf = RandomForestClassifier(oob_score=True,n_estimators=50, max_features=3, n_jobs=-1)

#radm_clf.fit( train_X_os, train_y_os )

radm_clf.fit( train_X, train_y )
train_y_pred=radm_clf.predict(train_X)

test_y_pred=radm_clf.predict(test_X)
radm_test_pred = pd.DataFrame( { 'actual':  test_y,

                            'predicted': radm_clf.predict( test_X ) } )
print(metrics.accuracy_score( radm_test_pred.actual, radm_test_pred.predicted ))

print(metrics.roc_auc_score( radm_test_pred.actual, radm_test_pred.predicted ))
tree_cm = metrics.confusion_matrix( radm_test_pred.predicted,

                                 radm_test_pred.actual,

                                 [1,0] )

sns.heatmap(tree_cm, annot=True,

         fmt='.2f',

         xticklabels = ["Left", "No Left"] , yticklabels = ["Left", "No Left"] )



plt.ylabel('True label')

plt.xlabel('Predicted label')
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
clf = SVC(kernel='linear')

clf.fit(train_X_os, train_y_os)
radm_test_pred = pd.DataFrame( { 'actual':  test_y,

                            'predicted': clf.predict( test_X ) } )
print(metrics.accuracy_score( radm_test_pred.actual, radm_test_pred.predicted ))

print(metrics.roc_auc_score( radm_test_pred.actual, radm_test_pred.predicted ))
tree_cm = metrics.confusion_matrix( radm_test_pred.predicted,

                                 radm_test_pred.actual,

                                 [1,0] )

sns.heatmap(tree_cm, annot=True,

         fmt='.2f',

         xticklabels = ["Left", "No Left"] , yticklabels = ["Left", "No Left"] )



plt.ylabel('True label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(radm_test_pred.actual, radm_test_pred.predicted)) 
from sklearn.model_selection import GridSearchCV 

  

# defining parameter range 

param_grid = {'C': [0.1, 1, 10, 100, 1000],  

              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

              'kernel': ['linear']}  

  

grid = GridSearchCV(SVC(), param_grid, refit = True, n_jobs=-1) 

  

# fitting the model for grid search 

grid.fit(train_X_os, train_y_os) 
print(grid.best_params_) 
print(grid.best_estimator_) 
print(grid.best_score_)
clf = SVC(kernel='linear',C=100,gamma=1)

clf.fit(train_X_os, train_y_os)
radm_test_pred = pd.DataFrame( { 'actual':  test_y,

                            'predicted': clf.predict( test_X ) } )
print(metrics.accuracy_score( radm_test_pred.actual, radm_test_pred.predicted ))

print(metrics.roc_auc_score( radm_test_pred.actual, radm_test_pred.predicted ))