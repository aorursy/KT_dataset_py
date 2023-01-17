# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import scipy

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# classification
# Grid Search for tuning parameters
from sklearn.model_selection import GridSearchCV
# RandomizedSearch for tuning (possibly faster than GridSearch)
from sklearn.model_selection import RandomizedSearchCV
# Bayessian optimization supposedly faster than GridSearch
from bayes_opt import BayesianOptimization

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
testfile = '../input/data_set_ALL_AML_independent.csv'
trainfile = '../input/data_set_ALL_AML_train.csv'
labels = '../input/actual.csv'

X_train = pd.read_csv(trainfile)
X_test = pd.read_csv(testfile)
y = pd.read_csv(labels)

train_keepers = [col for col in X_train.columns if "call" not in col]
test_keepers = [col for col in X_test.columns if "call" not in col]

X_train = X_train[train_keepers]
X_test = X_test[test_keepers]

X_train = X_train.T
X_test = X_test.T

X_train.columns = X_train.iloc[1]
X_train = X_train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)
X_train.index = X_train.index.astype(int)
X_train = X_train.sort_index()

X_train_old = X_train.copy()
# X_train standardize
X_train = X_train.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

# Clean up the column names for training data
X_test.columns = X_test.iloc[1]
X_test = X_test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)
X_test.index = X_test.index.astype(int)
X_test = X_test.sort_index()

#X_test standardize
X_test = X_test.apply(lambda x: (x - np.mean(x)) / (np.std(x)))


# X_train.sample(n=100, axis=1).plot(kind="hist", normed=True, legend=None, bins=10, color='k')
# X_train.sample(n=100, axis=1).plot(kind="kde", legend=None);

# X_test.sample(n=100, axis=1).plot(kind="hist", normed=True, legend=None, bins=10, color='k')
# X_test.sample(n=100, axis=1).plot(kind="kde", legend=None);


# 4) Split into train and test 

y_train = y[y.patient <= 38].reset_index(drop=True)

cat_dic = {
    'ALL':1,
    'AML':-1
}

y_train.replace(cat_dic,inplace=True)
y_train_old = y_train.copy()

#Centralization y
# print(y_train)
y_train['cancer'] = preprocessing.scale(list(y_train['cancer']), axis=0, with_mean=True)
# print(y_train)
# Subet the rest for testing

y_test = y[y.patient > 38].reset_index(drop=True)
y_test.replace(cat_dic,inplace=True)

# CHERCHEZ FOR PARAMETERS
def cherchez(estimator, param_grid, search, X_train, y_train):
    """
    This is a helper function for tuning hyperparameters using teh two search methods.
    Methods must be GridSearchCV or RandomizedSearchCV.
    Inputs:
        estimator: Logistic regression, SVM, KNN, etc
        param_grid: Range of parameters to search
        search: Grid search or Randomized search
    Output:
        Returns the estimator instance, clf
    
    """   
    try:
        if search == "grid":
            clf = GridSearchCV(
                estimator=estimator, 
                param_grid=param_grid, 
                scoring=None,
                n_jobs=-1, 
                cv=10, 
                verbose=0,
                return_train_score=True
            )
        elif search == "random":           
            clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=10,
                n_jobs=-1,
                cv=10,
                verbose=0,
                random_state=1,
                return_train_score=True
            )
    except:
        print('Search argument has to be "grid" or "random"')
        sys.exit(0)
        
    # Fit the model
    clf.fit(X=X_train, y=y_train)
    
    return clf
# use svm
def classify_svm(X_train, y_train, X_test, y_test):
    # SVM
    svm_param = {
        "C": [.01, .1, 1, 5, 10, 100],
        "gamma": [0, .01, .1, 1, 5, 10, 100],
        "kernel": ["rbf"],
        "random_state": [1]
    }

    svm_dist = {
        "C": scipy.stats.expon(scale=.01),
        "gamma": scipy.stats.expon(scale=.01),
        "kernel": ["rbf"],
        "random_state": [1]
    }

    svm_grid = cherchez(SVC(), svm_param, "grid", X_train, y_train)
    acc = accuracy_score(y_true=y_test, y_pred=svm_grid.predict(X_test))
#     cfmatrix_grid = confusion_matrix(y_true=y_test, y_pred=svm_grid.predict(X_test))
    print("**Grid search results**")
    print("Best training accuracy:\t", svm_grid.best_score_)
    print("Test accuracy:\t", acc)
    


# 1. no feature selection
classify_svm(X_train, y_train_old['cancer'], X_test, y_test['cancer'])
# 2.feature select by k-split lasso, classification by svm
def LARS(X, Y):
    reg = linear_model.LassoLars(alpha=0.01, copy_X=True, normalize=False)
    lars = reg.fit(X, Y)
    return lars.coef_ != 0 

def k_split_lasso(X,Y,geneList,K):
    # split geneList
    geneSet = [geneList[i:i+K] for i in range(0, len(geneList), K)]
    first_select_features = []   
    for geneSetItem in geneSet:
        X_train_item = X[geneSetItem].copy()
#         selected_item_bool = LARS(preprocessing.scale(X_train_item, axis=0, with_mean=True, with_std=True), y_train['cancer'])
        selected_item_bool = LARS(X_train_item, y_train['cancer'])
        selected_item_feature = X_train_item.columns[selected_item_bool].tolist()
        first_select_features += selected_item_feature
    print(len(first_select_features))
    if len(first_select_features) > 0:
        first_select_data = X[first_select_features]
#         second_selected_bool = LARS(preprocessing.scale(first_select_data, axis=0, with_mean=True, with_std=True), y_train['cancer'])
        second_selected_bool = LARS(first_select_data, y_train['cancer'])
        print(sum(second_selected_bool))
        return first_select_data.columns[second_selected_bool].tolist()    
    else:
        return []
           
geneList = X_train.columns.tolist()
filter_features = k_split_lasso(X_train, y_train, geneList, 90)

if len(filter_features) > 0:
    #svm
    filter_X_train = X_train[filter_features]
    filter_X_test = X_test[filter_features]
    classify_svm(filter_X_train, y_train_old['cancer'], filter_X_test, y_test['cancer'])
else:
    print('false')
        


# 3.feature selection by SNR
def snr_filter(X,Y):
    # 对训练集进行最大最小规范化处理
    min_max_scaler = preprocessing.MinMaxScaler() 
    X[X.columns] = min_max_scaler.fit_transform(X)
    
    # 将X与Y合并
    new_X = pd.concat([X, Y], axis=1)
    # 计算训练集中每个基因的信噪比
    # 先对训练集分为正负两类
    positive_train = new_X.loc[new_X['cancer'] == 1].drop('cancer', axis=1)
    negative_train = new_X.loc[new_X['cancer'] == -1].drop('cancer', axis=1)
    
    snr_data = ((positive_train.mean() - negative_train.mean()).abs())/(positive_train.mean() + negative_train.mean())
    filter_features = snr_data.sort_values(ascending=False)[0:50].index
    print('snr_data', snr_data.sort_values(ascending=False)[0:50])
    print(filter_features)
    print(X_train[filter_features])
    classify_svm(X_train[filter_features], y_train_old['cancer'], X_test[filter_features], y_test['cancer'])
    

snr_filter(X_train_old.copy().reset_index(drop=True), y_train_old['cancer'].copy())

