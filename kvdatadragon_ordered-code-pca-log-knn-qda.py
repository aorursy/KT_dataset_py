# Nativos

import os

import sys



#calculo

import numpy as np

import pandas as pd



#modelamiento

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans



#warning ignore future

import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings("ignore")



import os



subfolder = "../input"

print(os.listdir(subfolder))
set_parameter_csv = {

    'sep': ',',

    'encoding': 'ISO-8859-1',

    'low_memory': False

}



train = pd.read_csv(subfolder + '/train.csv', **set_parameter_csv).round(2)

test = pd.read_csv(subfolder + '/test.csv', **set_parameter_csv).round(2)

train.shape, test.shape
train.head(2)
%%time

def get_memory_usage(data, deep=True):

    return '{} MB'.format(data.memory_usage(deep=deep).sum() / 1024 ** 2)



def reduce_size_data(df, default=''):

    print("INITIAL SIZE : DEEP", get_memory_usage(df), "REAL", get_memory_usage(df, deep=False))

 

    for col in df.select_dtypes(include=['int']).columns:

        df[col] = pd.to_numeric(arg=df[col], downcast=default or'integer')

    

    for col in df.select_dtypes(include=['float']).columns:

        df[col] = pd.to_numeric(arg=df[col], downcast=default or'float')

                

    print("FINAL SIZE : DEPP", get_memory_usage(df), "REAL", get_memory_usage(df, deep=False))               

    return df



train = reduce_size_data(train)

test = reduce_size_data(test)
train.isnull().sum().any(), test.isnull().sum().any()
for col in train.columns:

    unicos = train[col].unique().shape[0]

    if unicos < 1000:

        print(col, unicos)
SEED = 29082013

pd.np.random.seed(SEED)



col_wctm = 'wheezy-copper-turtle-magic'

col_target = 'target'

col_log = 'predict_log'

col_knn = 'predict_knn'

cols = [c for c in train.columns if c not in ['id', col_wctm, col_target]]



kfold_off = StratifiedKFold(

    n_splits=11, 

    shuffle=False, 

    random_state=SEED

)

param_grid_knn = {

    'n_neighbors': [7],

    'p': [2],

    'weights':['distance']

}

param_grid_gauss = {

    'priors': [[0.5, 0.5]],

    'reg_param': [0.3]

}



param_grid_log = {

    'solver': ['sag'],

    'penalty': ['l2'],

    'C': [0.001],

    'tol': [0.0001]

}



model_knn = KNeighborsClassifier()

model_gauss = QuadraticDiscriminantAnalysis()

model_log = LogisticRegression(random_state=SEED)

km = KMeans(n_clusters=5, n_init=5, init='k-means++', random_state=SEED, algorithm='elkan')

pca = PCA(svd_solver='full',n_components='mle')
def apply_pca(X_train, X_test):

    #PCA

    X_train = pd.DataFrame(pca.fit_transform(X_train))

    X_test = pd.DataFrame(pca.transform(X_test)) 

    return X_train, X_test



def apply_km(X_train, X_test):

    col_km = 'cluster_km'

    X_train[col_km] = km.fit_predict(X_train)

    X_test[col_km] = km.predict(X_test)

    return pd.get_dummies(X_train, columns=[col_km]), pd.get_dummies(X_test, columns=[col_km])



def apply_grid(X_train, y_train, X_test, model, param_grid, predict_train=True):

    grid = GridSearchCV(

        model, param_grid, cv=kfold_off, n_jobs=-1, scoring='roc_auc'

    )

    grid.fit(X_train, y_train)

    print(grid.best_score_, end=' / ')

    if predict_train:

        return grid.best_estimator_.predict_proba(X_train)[:,1], grid.best_estimator_.predict_proba(X_test)[:,1]

    else:

        return grid.best_estimator_.predict_proba(X_test)[:,1]
%%time

col_wctm = 'wheezy-copper-turtle-magic'

col_target = 'target'

cols = [c for c in train.columns if c not in ['id', col_wctm, col_target]]

result = []

scores = 0

scores2 = 0



for val in sorted(train[col_wctm].unique()):

    # Build X_train and y_train

    X_train = train[train[col_wctm] == val]

    y_train = X_train[col_target]

    X_test = test[test[col_wctm] == val]

    result_test = X_test[['id', col_wctm]]

    

    X_train = X_train[cols]

    X_test = X_test[cols]

    

    #PCA

    X_train, X_test = apply_pca(X_train, X_test)

    

    # ADD column prediction log or knn

    train_log, test_log = apply_grid(X_train, y_train, X_test, model_log, param_grid_log)

    train_knn, test_knn = apply_grid(X_train, y_train, X_test, model_knn, param_grid_knn)    

    

    X_train, X_test = apply_km(X_train, X_test)

    

    X_train[col_log] = train_log

    X_test[col_log] = test_log

    X_train[col_knn] = train_knn

    X_test[col_knn] = test_knn

    

    # TRAIN    

    result_test[col_target] = apply_grid(X_train, y_train, X_test, model_gauss, param_grid_gauss, predict_train=False)

    result.append(result_test[['id', col_target]])

    print("-"*100)
result = pd.concat(result).sort_index()

result.head(15)
"""

def fix_round(val):

    if val > 0.9:

        return 1

    elif val < 0.1:

        return 0

    else:

        return val

    

result[col_target] = result[col_target].apply(lambda _: fix_round(_))

"""
result.to_csv('oordered_log_kfold11_knn7_qda_38.csv', index=False)