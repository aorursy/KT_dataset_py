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
# import the models and necessary libraries

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



# use for pipeline and encode features

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer



# models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans

import sklearn.metrics.cluster as smc



# validation of the models

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score
# import the datasets

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
# can uncomment the following to look at the info (type) of the features, shape of the datasets

#print(train.info())

#print(test.info())

train.head()
print(train.shape)

print(test.shape)
# Change boolean value to int so as to encode

train['bin_3'] = train['bin_3'].apply(lambda x: 1 if x=='T' else 0)

train['bin_4'] = train['bin_4'].apply(lambda x:1 if x =='Y' else 0)

test['bin_3'] = test['bin_3'].apply(lambda x:1 if x=='T' else 0)

test['bin_4'] = test['bin_4'].apply(lambda x:1 if x == 'Y' else 0)
# input the test_labels to validate the test sets later on

# drop the target column in train sets to separate the features and values we need to predict

test_labels = train['target']

train = train.drop(['target'],axis=1)

train.head()
# uncomment this if you want to test a new model using smaller datasets

# notice these are not randomly chosen, so be careful of overfitting

# X_train_part = X_train[:4200]

# y_train_part = y_train[:4200]

# X_test_part = X_test[:1800]

# y_test_part = y_test[:1800]

#train_part = train[:6000]

#test_labels2 = test_labels[:6000]
# pipelining the categorical features

# I chose to do one hot encoder on some of the categorical features since they are type-A/type-B features

# dropping some of the features out because they have high cardinalities, which would make the datasets have too many columns



%%time

from category_encoders.m_estimate import MEstimateEncoder

imputer1 = SimpleImputer(strategy="median")

imputer = SimpleImputer(strategy='most-frequent')

train_1=train

def Preparation(train,test_set=False):

    

    train_cat = train.drop(["id","nom_5","nom_6","nom_9"],axis=1)

    cat_pipeline = Pipeline([

                ('imputer2',SimpleImputer(strategy='most_frequent')),

                ('cat',OneHotEncoder(categories='auto')),

                #('cat',MEstimateEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True, handle_unknown='value', handle_missing='value', random_state=None, randomized=False, sigma=0.05, m=1.0)),

    ])

    train_cat_tr = cat_pipeline.fit_transform(train_cat)

    categorical_features = list(train_cat)

    

    full_pipeline = ColumnTransformer([

            #("num", num_pipeline, numerical_features),

            ("cat", cat_pipeline, categorical_features),

        ])



    train_prepared = full_pipeline.fit_transform(train)

    print(train_prepared.shape)

    return train_prepared

train_1 = Preparation(train_1) #train_1

#print(train_1)
# separate the datasets into 80% train sets and 20% test sets

# can also do K-fold validation



X_train,X_test,y_train,y_test = train_test_split(train_1,test_labels,random_state=42,test_size=0.2) #train_1,test_labels

#print(help(train_test_split))

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# Input the catboost models

# can use loops or grid-searchCV to tune parameters



%%time

from catboost import CatBoostClassifier

params = { #30,2000，0.15,5

    'bagging_temperature': 0.8,'l2_leaf_reg': 30,'iterations': 998,'learning_rate': 0.15,'depth': 5,

    'random_strength': 0.8,'loss_function': 'Logloss','eval_metric': 'AUC','verbose': False

}

catb = CatBoostClassifier(**params, nan_mode='Min').fit(X_train, y_train,verbose_eval=100, early_stopping_rounds=50,eval_set=(X_test, y_test),

                                                        use_best_model=False,

                                                        plot=True)

preds2 = catb.predict_proba(X_test)[:,1]



print("ROC AUC score is %.4f" %(roc_auc_score(y_test,preds2)))



print("Catboost Model Performance Results:\n")

plot_roc_curve(catb,X_test,y_test)

plt.title('ROC Curve')
# submission

test_id = test.index

test_sub = Preparation(test)

test_pred = catb.predict_proba(test_sub)[:,1]

submission = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

submission.target = test_pred

submission.to_csv('submission.csv', index=False)
