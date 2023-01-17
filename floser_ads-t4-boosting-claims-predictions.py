# Notebook settings

###################



# resetting variables

#get_ipython().magic('reset -sf') 



# formatting: cell width

#from IPython.core.display import display, HTML

#display(HTML("<style>.container { width:100% !important; }</style>"))



# plotting

%matplotlib inline
# importing xgboost REMARK: run this cell only if other imports failed. Delete it in case xgboost has been already imported

#import pip

#pip.main(['install', 'xgboost'])
# loading Python packages

#########################



# scientific packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import scipy



# boosting

import xgboost

from xgboost import XGBClassifier

from xgboost import plot_importance



# scipy

from scipy.stats import chi2



# pandas: selected modules and functions

from pandas.plotting import scatter_matrix



# sklearn: selected modules and functions

from sklearn import decomposition

from sklearn.decomposition import PCA



from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import scale



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier



from sklearn.utils import shuffle



from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.pipeline import Pipeline



from sklearn.metrics import roc_curve

from sklearn.metrics import auc 

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import classification_report

from sklearn.metrics import zero_one_loss



# utilities

from datetime import datetime
# data import: specify the path to the competition training data

#path = '...'

#data = pd.read_csv(path)

data = pd.read_csv('../input/train.csv')
print('Structure of imported data:', data.shape)
print('Memory usage of `data` (in bytes):', pd.DataFrame.memory_usage(data, index=True, deep=True).sum())
# data types

data.info()
# duplicates? None

data.drop_duplicates().shape
# imported data: first 10 entries

data.head(10).T
# imported data: statistical summaries with describe

data.describe().T
# missing values (encoded as -1)

feat_missing = []



for f in data.columns:

    missings = data[data[f] == -1][f].count()

    if missings > 0:

        feat_missing.append(f)

        missings_perc = missings/data.shape[0]

        

        # printing summary of missing values

        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))



# how many variables do present missing values?

print()

print('In total, there are {} variables with missing values'.format(len(feat_missing)))
# B. Carremans: recording meta-information for each column in train, following the official data description on the Kaggle Porto Seguro Challenge

info = []



for f in data.columns:



    # defining the role (target and id have to be separated from the other features)

    if f == 'target':

        role = 'target'

    elif f == 'id':

        role = 'id'

    else:

        role = 'input'

         

    # defining the levels    

    

    # _bin postfix = binary feature (target is binary as well)

    if 'bin' in f or f == 'target':

        level = 'binary'

    

    # _cat postfix = categorical feature

    elif 'cat' in f or f == 'id':

        level = 'categorical'

        

    # continuous or ordinal features: those which are neither _bin nor _cat    

    elif data[f].dtype == float:

        level = 'interval'

    else:

        level = 'ordinal'    

        

    # initialize 'keep' to True for all variables except for id

    keep = True

    if f == 'id':

        keep = False

    

    # defining the data type 

    dtype = data[f].dtype

    

    # creating a dictionary that contains all the metadata for the variable

    f_dict = {

        'varname': f,

        'role': role,

        'level': level,

        'keep': keep,

        'dtype': dtype

    }

    info.append(f_dict)



# collecting all meta-information into a meta dataframe

meta = pd.DataFrame(info, columns = ['varname', 'role', 'level', 'keep', 'dtype'])

meta.set_index('varname', inplace = True)
# showing meta-information data frame

print(meta.shape)

meta
# showing meta-information aggregated view

pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
# choose feature level 

level = 'binary'



# creating the dictionary with feature level counts 

ctabs = {}



for f in meta[(meta.level == level) & (meta.keep)].index:

    ctabs[f]=( pd.value_counts(data[f]) / data.shape[0] ) * 100

    

# printing the dictionary, with rounding of frequencies

for f in ctabs.keys():

    print(ctabs[f].round(2))

    print() 
# categorical variables: summary of distinct levels



# choose feature level 

level = 'categorical'



for f in meta[(meta.level == 'categorical') & (meta.keep)].index:

    dist_values = data[f].value_counts().shape[0]

    print('Variable {} has {} distinct values'.format(f, dist_values))
# categorical variables: tabulation



# choose feature level 

level = 'categorical'



# creating the dictionary with feature level counts 

ctabs = {}



for f in meta[(meta.level == level) & (meta.keep)].index:

    ctabs[f] = ( pd.value_counts(data[f]) / data.shape[0] ) * 100

    

# printing the dictionary, with rounding of frequencies    

for f in ctabs.keys():

    print(ctabs[f].round(2))

    print() 
# interval variables: tabulation 



# choose feature level 

level = 'interval'



# creating the dictionary with feature level counts (missing values are dropped)

ctabs = {}



for f in meta[(meta.level == level) & (meta.keep)].index:

    ctabs[f] = ( pd.value_counts(data[f]) / data.shape[0] ) * 100

    

# printing the dictionary, with rounding of frequencies    

for f in ctabs.keys():

    print(ctabs[f].round(2))

    print() 
# producing bar charts for selected variables

v = list(['ps_reg_01', 'ps_reg_02', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03'])

sns.set( rc = {'figure.figsize': (10, 10)})



for f in v:

    plt.figure()

    sns.countplot(x = f, data = data, linewidth=2, palette="Blues")

    plt.show()
# producing histograms for selected variables

v = list(['ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15'])

sns.set( rc = {'figure.figsize': (10, 10)})



for f in v:

    plt.figure()

    sns.distplot(data.loc[data[f] != -1, f], kde = False)

    plt.show()
squared = data['ps_car_15'] ** 2

squared.value_counts()
# ordinal features: visualization



# choose feature level 

level = 'ordinal'



# producing histograms

v = meta[(meta.level == level) & (meta.keep)].index

sns.set( rc = {'figure.figsize': (10, 10)})



for f in v:

    plt.figure()

     

    sns.distplot(data[f], kde = False)

    plt.show()
# ordinal features: tabulation



# creating the dictionary with feature level counts (missing values are dropped)

ctabs = {}

level = 'ordinal'



for f in meta[(meta.level == level) & (meta.keep)].index:

    ctabs[f] = ( pd.value_counts(data[f]) / data.shape[0] ) * 100

    

# printing the dictionary    

for f in ctabs.keys():

    print(ctabs[f].round(2))

    print() 
# levels for the target variable 

lev_target = ( pd.crosstab(index = data['target'], columns = 'Frequency') / data.shape[0] ) * 100

lev_target.round(2)
# Pearson correlation matrix: computation and visualization



# use method='pearson' resp. method='spearman' to compute Pearson resp. Spearman correlations

def corr_heatmap(v):

    correlations = data[v].corr(method='pearson')

    fig = plt.subplots(figsize=(10, 10))



    sns.heatmap(correlations,  center=0, fmt='.2f', cbar=False,

                square=True, linewidths=1, annot=True,  cmap="YlGnBu")

    plt.xticks(rotation=90) 

    plt.yticks(rotation=0) 

    plt.show()



# one applies the corr_heatmap function on the interval features    

v = meta[(meta.level == 'interval') & (meta.keep)].index

corr_heatmap(v)
# scatterplot high correlation interval variables

import seaborn

#due to runtime restrictions not executed:

#high = pd.Index(['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_15'])

#pd.plotting.scatter_matrix(data[high], alpha = 0.2, figsize = (40, 40), diagonal = 'kde')
# jitter plots: an example

sns.set( rc = {'figure.figsize': (10, 10)})

feat = list(['ps_reg_03', 'ps_car_12'])



for f in feat:

    plt.figure()

    sns.stripplot(y=f, x='target', data=data, jitter=True, palette="Blues")

    plt.show()
# ps_reg_01

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y='ps_reg_01', data=data, linewidth=2, palette="Blues")
# ps_reg_02

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x= "target", y ="ps_reg_02", data=data, linewidth=2, palette="Blues")
# ps_reg_03

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y ='ps_reg_03', data=data, linewidth=2, palette="Blues")
# ps_car_12

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y='ps_car_12', data=data, linewidth=2, palette="Blues")
# ps_car_13

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y='ps_car_13', data=data, linewidth=2, palette="Blues")
# ps_car_14

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y='ps_car_14', data=data, linewidth=2, palette="Blues")
# ps_car_15

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y='ps_car_15', data=data, linewidth=2, palette="Blues")
# ps_calc_01

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y='ps_calc_01', data=data, linewidth=2, palette="Blues")
# ps_calc_02

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y='ps_calc_02', data=data, linewidth=2, palette="Blues")
# ps_calc_03

sns.set( rc = {'figure.figsize': (10, 10)})

sns.boxplot(x='target', y='ps_calc_03', data=data,linewidth=2, palette="Blues")
# binary features

sns.set( rc = {'figure.figsize': (10, 10)})

feat = meta[(meta.level == 'binary') & (meta.keep)].index



for f in feat:

    plt.figure()

    sns.countplot(x=data[f], hue=data.target, data=data, palette="Blues")

    plt.show()
# for binary make a cross-tabulation rows : levels columns: 0/1 in %; sum of every row = 100%

v = meta[(meta.level == 'binary') & (meta.keep)].index.drop('target')



for f in v:

    crosstab = pd.crosstab(index=data[f], columns=data['target'], margins=True) 

    cross = pd.DataFrame(data=crosstab.div(crosstab['All'], axis=0).drop('All', 1))

    cross['All'] = crosstab.iloc[:,2]

    print(cross)

    print()    

    print()
 # for ordinal make a cross-tabulation rows : levels columns: 0/1 in %; sum of every row = 100%

v = meta[(meta.level == 'ordinal') & (meta.keep)].index



for f in v:

    crosstab = pd.crosstab(index=data[f], columns=data['target'], margins=True) 

    cross = pd.DataFrame(data=crosstab.div(crosstab['All'], axis=0).drop('All', 1))

    cross['All'] = crosstab.iloc[:,2]

    print(cross)

    print()    

    print()    
print('Structure of data before calc variable drop:', data.shape)
# dropping 'ps_calc_01',... 'ps_calc_14' variables and updating meta information

vars_to_drop = ['ps_calc_01', 'ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14']

data.drop(vars_to_drop, inplace = True, axis = 1)

meta.loc[(vars_to_drop), 'keep'] = False  
print('Structure of data after calc variable drop:', data.shape)
data.info()
# dropping 'ps_car_03_cat', 'ps_car_05_cat' and updating meta information

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

data.drop(vars_to_drop, inplace = True, axis = 1)

meta.loc[(vars_to_drop), 'keep'] = False  



# imputing with the mean or mode using Imputer from sklearn.preprocessing

from sklearn.preprocessing import Imputer



mean_imp = Imputer(missing_values = -1, strategy = 'mean', axis = 0)

mode_imp = Imputer(missing_values = -1, strategy = 'most_frequent', axis = 0)



data['ps_reg_03'] = mean_imp.fit_transform(data[['ps_reg_03']]).ravel()

data['ps_car_12'] = mean_imp.fit_transform(data[['ps_car_12']]).ravel()

data['ps_car_14'] = mean_imp.fit_transform(data[['ps_car_14']]).ravel()

data['ps_car_11'] = mode_imp.fit_transform(data[['ps_car_11']]).ravel()
print(data.shape)
# selecting categorical variables

v = meta[(meta.level == 'categorical') & (meta.keep)].index

print('Before dummification we have {} variables in train'.format(data.shape[1]))



# creating dummy variables

data = pd.get_dummies(data, columns = v, drop_first = True)

print('After dummification we have {} variables in data'.format(data.shape[1]))
print('Memory usage of `data` (in bytes):', pd.DataFrame.memory_usage(data,index=True, deep = True).sum())
#v = meta[(meta.level == 'categorical') & (meta.keep)].index

#data[v]
print(data.columns.values)
random_state = 123
# split 80-20% (no stratification)

X_train, X_test, y_train, y_test = train_test_split(data.drop(['id', 'target'], axis=1), 

                                                    data['target'], 

                                                    test_size=0.2,

                                                    random_state=random_state

                                                   )
# structural checks

print('Training dataset - dimensions:', X_train.shape)

print('Test dataset - dimensions:', X_test.shape)

print()

print('Random split check:', X_train.shape[0] + X_test.shape[0] == data.shape[0])

print()



# imbalancing: check

lev_target = ( pd.crosstab(index = data['target'], columns = 'count') / data.shape[0] ) * 100

lev_target_train = ( pd.crosstab(index = y_train, columns = 'count') / y_train.shape[0] ) * 100

lev_target_test = ( pd.crosstab(index = y_test, columns = 'count') / y_test.shape[0] ) * 100



print('target class imbalance data:')

print(lev_target)

print()

print('target class imbalance train:')

print(lev_target_train)

print()

print('target class imbalance test:')

print(lev_target_test)
del data
# insert paths to export training and test data sets

#X_train.to_csv('...')

#X_test.to_csv('...')
from sklearn.metrics import make_scorer



# Gini coefficient

def gini(actual, pred):

    

    # a structural check

    assert (len(actual) == len(pred))

    

    # introducing an array called all

    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)  #slicing along second axis

    

    # sorting the array along predicted probabilities (descending order) and along the index axis all[:, 2] in case of ties

    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]                             #



    # towards the Gini coefficient

    totalLosses = all[:, 0].sum()

    giniSum = all[:, 0].cumsum().sum() / totalLosses



    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)



# normalized Gini coefficient

def gini_normalized_score(actual, pred):

    return gini(actual, pred) / gini(actual, actual)



# score using the normalized Gini

score_gini = make_scorer(gini_normalized_score, greater_is_better=True, needs_threshold = True)
# fit model default hyper-parameters

xgb_default = XGBClassifier(random_state=random_state)

xgb_default.set_params()
# fitting the default XGBoostClassifier()

xgb_default.fit(X_train, y_train)
# default XGBoost classifier performance

y_pred_proba_xgb_default = xgb_default.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba_xgb_default[:, 1])

roc_auc = auc(fpr, tpr)



# AUC on test dataset

print('AUC:', roc_auc_score(y_test, y_pred_proba_xgb_default[:, 1]).round(3))
# GridSearch XGBoost optimization (change the hyperparameters according to the run under consideration)

#Mod: Grid changed from 'learning_rate': [0.001, 0.01, 0.1, 1], 'n_estimators': [100, 300, 500] due to runtime restrictions

param_grid = {'max_depth': [3],                                                            

              'learning_rate': [0.1],

              'n_estimators': [10, 300],

              'subsample': [0.5],

              'colsample_bytree': [0.75]

             } 



# cross-validation

cv = StratifiedKFold(n_splits=5, 

                     shuffle=False, 

                     random_state=random_state)



xgb = GridSearchCV(estimator=XGBClassifier(random_state=random_state), 

                   param_grid=param_grid, 

                   scoring=score_gini, 

                   cv=cv, 

                   verbose=10)



xgb.fit(X_train, y_train)
# best score on CV-data

print('Best Gini on CV-data:', xgb.best_score_)



# parameters for best model

print('Best set of hyperparameters:', xgb.best_params_)
# best xgb model

xgb_best = xgb.best_estimator_

xgb_best.fit(X_train, y_train)



# best xgb predictions on test data and AUC

y_pred_proba_xgb = xgb_best.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba_xgb[:, 1])

roc_auc = auc(fpr, tpr)



# plotting ROC and showing AUC (on test data)

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()    



# AUC on test dataset

print('AUC:', roc_auc_score(y_test, y_pred_proba_xgb[:, 1]).round(3))
# fitting best model (again, if not saved beforehand)

xgb_best_ov = XGBClassifier(max_depth=3,                                      

                            learning_rate=0.1,

                            n_estimators=300,

                            subsample=0.5,

                            colsample_bytree=0.75,

                            random_state=random_state)

xgb_best_ov.fit(X_train, y_train)
# plotting feature importance graph: top 10 features

plot_importance(xgb_best_ov, max_num_features=10)