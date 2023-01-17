### Imports
from sklearn import datasets

import pandas as pd
import numpy  as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# to evaluate the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import xgboost as xgb

RANDOM_STATE = 39
### Load iris dataset - data is sourced from sklearn datasets
(iris_data, iris_target) = datasets.load_iris(return_X_y = True)  # return tuble of data, target
# Lets get into pandas dataframe for further analysis

iris = pd.DataFrame(iris_data, columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
iris['target'] = iris_target
iris['target_names'] = iris['target'].map({0:'Setosa', 1:'Versicolour',2:'Virginica'})

iris.head()
# write into csv
iris.to_csv('iris.csv')
# Missing values?
iris.isnull().mean()

# No missing values
# data types?
iris.dtypes

#featurs is continuous numerical data and target is integer
# explore target
iris['target'].value_counts()

# classess in label are balanced.
# correlation with target?

iris.corr()['target'].sort_values(ascending = False)

# petal width, petal length is highly correlated with iris flower type
# lets have closer look of petal width and petal_length with target

g = sns.FacetGrid (data = iris, hue = 'target_names')
g = g.map(plt.scatter, 'petal_width', 'petal_length').add_legend()

# Setosa can easisly be separated from other flowers type linearly.
# Versicolour and virginica type flower can't be separated linearly
# Lets see the descriptive statistics

def descriptive_statistics(data):

    corr = data.describe().T
    # calculate iqr
    iqr = corr['75%'] - corr['25%']

    outlier_lower = corr['25%'] - 1.5 *iqr
    outlier_higher = corr['75%'] + 1.5 * iqr 
    
    # calculate whether outlier is present or not
    corr['outlier_lower?'] = np.where(corr['min'] < outlier_lower, 'yes', 'no')
    corr['outlier_higher?'] = np.where(corr['max'] < outlier_higher, 'yes', 'no') 
    
    return corr


descriptive_statistics(iris)

# petal length, width (which are in focus) suffer with outlier

data = iris.copy()

corr = data.corr().T
corr
# lets see the distribution and box plot
#fig, axes = plt.subplots(nrows =1, ncols = 2, sharex = True)


g = sns.FacetGrid(iris, hue="target_names", size = 4, aspect = 2)
g = g.map(sns.distplot,'petal_width', kde = False).add_legend()

g = sns.FacetGrid(iris, hue="target_names", size = 4, aspect = 2)
g = g.map(sns.distplot,'petal_length', kde = False).add_legend()

# distribution not gaussion like
# split train, test data
# drop target_names and target
target = iris.target
iris.drop(labels = ['target','target_names'], axis=1, inplace = True)

iris.columns
train_features, test_features, train_label, test_label = \
                                    train_test_split(iris, target,\
                                      test_size = 0.3, stratify = target, \
                                          random_state = RANDOM_STATE )
    
    
(train_features.shape, test_features.shape, train_label.shape, test_label.shape)
xgb_model = xgb.XGBClassifier(random_state = RANDOM_STATE)
xgb_model.fit(train_features, train_label, eval_metric ='auc',  verbose = False)
train_pred = xgb_model.predict(train_features)
print('xgb train accuracy: {}'.format(metrics.accuracy_score(train_label, train_pred)))

test_pred = xgb_model.predict(test_features)
print('xgb test accuracy: {}'.format(metrics.accuracy_score(test_label, test_pred)))
pd.Series(xgb_model.feature_importances_, train_features.columns)
# lets drop sepal_width

train_features = train_features.drop(['sepal_width'], axis=1)
test_features = test_features.drop(['sepal_width'], axis=1)
# train score is 1 and test score is 0.95..
# lets tune few hyperparameters to check whether improvement in variance
# lets reduce max_depth to see improvement in variance



xgb_model = xgb.XGBClassifier(random_state = RANDOM_STATE, max_depth = 2, n_estimators = 100)
xgb_model.fit(train_features, train_label, eval_metric ='auc',  verbose = False)
train_pred = xgb_model.predict(train_features)
print('xgb train accuracy: {}'.format(metrics.accuracy_score(train_label, train_pred)))

test_pred = xgb_model.predict(test_features)
print('xgb test accuracy: {}'.format(metrics.accuracy_score(test_label, test_pred)))
(test_label!=test_pred).sum()
xgb.plot_importance(xgb_model)

ax1 = xgb.plot_tree(xgb_model, num_trees =2)


