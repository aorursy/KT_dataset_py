# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def load_data(file_type):

    return pd.read_csv('/kaggle/input/advertsuccess/' + file_type + '.csv')
test_data = load_data('Test')

train_data = load_data('Train')
train_data.head()
# test_data.head()
print('total train data: ' + str(train_data.shape[0]))

print('total test data: ' + str(test_data.shape[0]))
train_data.describe().T
train_data.info()
train_data.isnull().sum()
train_data.loc[train_data['netgain'] == False, ['netgain']] = 0

train_data.loc[train_data['netgain'] == True, ['netgain']] = 1
sns.set()

sns.countplot(train_data.netgain)
import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))

sns.countplot(train_data.realtionship_status)
plt.figure(figsize=(15,6))

sns.countplot(train_data.industry)
plt.figure(figsize=(15,6))

sns.countplot(train_data.genre)
sns.countplot(train_data.airtime)
X_train = train_data[:20000]

X_val = train_data[20000:]
# split train _ test

# from sklearn.model_selection import train_test_split

# y = (train_data["netgain"]).astype(np.int)

# X = train_data.drop(columns=["netgain"])

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

y_train = (X_train["netgain"]).astype(np.int)

y_val = (X_val["netgain"]).astype(np.int)

X_train.drop(columns=['netgain'], inplace=True)

X_val.drop(columns=['netgain'], inplace=True)
# make models trainer

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier   

sgd_clf = SGDClassifier(loss='hinge', penalty='l2')

nb_clf = GaussianNB()

models = [

        {

            "name": "SGD",

            "model": sgd_clf

        },

        {

            "name": "NB",

            "model": nb_clf 

        }

]    
from sklearn.metrics import classification_report, accuracy_score, make_scorer

def classification_report_with_accuracy_score(y_true, y_pred):

    return accuracy_score(y_true, y_pred)
from sklearn.model_selection import cross_val_score



def predict_linear(X, y):

    for mdl in models:

        print("******* model -> " + mdl['name'] + " *******")

        score = cross_val_score(mdl['model'], X= X, y = y, cv = 5,scoring = 'f1')

        print(score)
# ordinal encoding for text categorical

import category_encoders as ce

def make_ordinal_encoder_for_text_categorical(X, y, cols):

    ce_ord = ce.OrdinalEncoder(cols = cols)

    X = ce_ord.fit_transform(X, y)

    return X

# convet text categoricl data to ordinal encoding

X_ord_train = X_train

X_ord_val = X_val

# text categorical

text_categorical = ['airtime', 'realtionship_status', 'industry', 'genre', 'targeted_sex', 'expensive', 'money_back_guarantee', 'airlocation']

# 

X_ord_train = make_ordinal_encoder_for_text_categorical(X_ord_train, y_train, text_categorical)

X_ord_val = make_ordinal_encoder_for_text_categorical(X_ord_train, y_train, text_categorical)
# reshape y_train

# y_ord_train = y_train.values.reshape((y_train.values.shape[0], 1))

# y_train.shape

predict_linear(X_ord_train, y_train)
# svm on ordinal encoding data with GreadSearchCV

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
def find_best_param_for_svm(X, y, parameters):

    svc = svm.SVC()

    # find best parameter for svm

    svc_clf_pipe = Pipeline([

        ('scaler', StandardScaler()),

        ('svm_clf', GridSearchCV(svc, parameters))

    ])

    svc_clf_pipe.fit(X, y)

    # cross_val for svm with best param that we have fined

    best_params = svc_clf_pipe['svm_clf'].best_params_

    print("************************")

    print(best_params)

    return best_params
parameters = {'kernel': ('linear', 'rbf'), 'C': np.arange(0.5, 5, 0.5)}

best_params = find_best_param_for_svm(X_ord_train, y_train, parameters)
def svm_classifier(X, y, best_params):

    svc_clf_pipe = Pipeline([

        ('scaler', StandardScaler()),

        ('svm_clf', svm.SVC(kernel = best_params['kernel'], C = best_params['C'], gamma='auto'))

    ])

    score = cross_val_score(svc_clf_pipe, X = X, y = y, cv = 5, scoring='f1')

    print(score)
svm_classifier(X_ord_train, y_train, best_params)
# one hot encoding

def one_hot_encoding(X, columns):

    X_data =  X

    for col in columns:

        print(col)

        one_hot = pd.get_dummies(X[col])

        X_data = X.join(one_hot, how='left', lsuffix='_left', rsuffix='_right')

        X = X_data

    return X

        
X_one_hot_train = X_train

X_one_hot_val =  X_val

# we don't one_hot for airlocation beacuse its unique value is many and we use binary encoding for its

text_categorical = [ 'realtionship_status', 'industry', 'genre', 'targeted_sex', 'expensive', 'money_back_guarantee', 'airtime']

# X_one_hot_train = one_hot_encoding(X_one_hot_train, text_categorical)

X_one_hot_train = one_hot_encoding(X_one_hot_train, text_categorical)

X_one_hot_val = one_hot_encoding(X_one_hot_val, text_categorical)
# drop

X_one_hot_train.drop(columns=text_categorical, inplace=True)

X_one_hot_train.drop(columns=['id'], inplace = True)

X_one_hot_val.drop(columns=text_categorical, inplace=True)

X_one_hot_val.drop(columns=['id'], inplace = True)
X_one_hot_train.shape
# binary categorical

def make_binary_encoding(X, y, cols):

    ce_bin = ce.BinaryEncoder(cols = cols)

    return ce_bin.fit_transform(X, y)
col = ['airlocation']

X_one_hot_and_binary_train = make_binary_encoding(X_one_hot_train, y_train, col)

X_one_hot_and_binary_val = make_binary_encoding(X_one_hot_val, y_val, col)
# model 

predict_linear(X_one_hot_and_binary_train, y_train)
# show variance of feature for chosing number of principle component

from sklearn.pipeline import Pipeline

from sklearn import decomposition

def show_variance_of_feature(data):

    pca = decomposition.PCA(n_components = None)

    v = pca.fit_transform(data)

    explained_variance = pca.explained_variance_ratio_

    print(explained_variance)
show_variance_of_feature(X_one_hot_and_binary_train)

# choose 8 principle component
pca = decomposition.PCA(n_components= 10) 

principle_component_train = pca.fit_transform(X_one_hot_and_binary_train)

principle_component_val = pca.transform(X_one_hot_and_binary_val)
X_pca_train = pd.DataFrame(data = principle_component_train, columns=list(range(0, 10)))

X_pca_val = pd.DataFrame(data = principle_component_val, columns= list(range(0, 10)))
parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': np.arange(0.5, 5, 0.5)}

best_params = find_best_param_for_svm(X_pca_train, y_train, parameters)
svm_classifier(X_pca_train, y_train, best_params)
pca = decomposition.PCA(n_components= 15) 

principle_component_train_2 = pca.fit_transform(X_one_hot_and_binary_train)
X_pca_2_train = pd.DataFrame(data = principle_component_train, columns=list(range(0, 10)))
svm_classifier(X_pca_2_train, y_train, best_params)
svm_classifier(X_one_hot_and_binary_train, y_train, best_params)
param = {'C': 3.5, 'kernel': 'rbf'}

svm_classifier(X_pca_2_train, y_train, param)
# desition tree

from sklearn.tree import DecisionTreeRegressor

def tree_reg_classifire(X, y):

    tree_clf_pipe = Pipeline([

        ('scaler', StandardScaler()),

        ('tree_clf', DecisionTreeRegressor())

    ])

    score = cross_val_score(tree_clf_pipe, X, y, scoring='f1', cv = 5)

    print(score)
tree_reg_classifire(X_ord_train, y_train)
# tree_reg_classifire(X_one_hot_and_binary_train, y_train)