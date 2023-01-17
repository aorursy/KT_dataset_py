# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import warnings

warnings.filterwarnings('ignore')



from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv( '../input/train.csv' )

test_df = pd.read_csv( '../input/test.csv' )
# To verify the data types and data format

train_df.info()

train_df.shape
test_df.info()

test_df.shape
train_df.head( 10 )
test_df.head(10)
TrainMissingCount = train_df.isnull().sum()

print( TrainMissingCount )
TestMissingCount = test_df.isnull().sum()

print( TestMissingCount )
train_df.describe()
test_df.describe()
X_train = train_df[list(train_df.columns[2:])]
X_train.head(10)
# Visualizing the train - target variable

sns.countplot( train_df['target'] )
y_train = train_df['target']
y_train.head()
X_test = test_df[list(test_df.columns[1:])]

X_test.head()
train = train_df

test = test_df

train_correlations = train.drop(["target"], axis=1).corr()

train_correlations = train_correlations.values.flatten()

train_correlations = train_correlations[train_correlations != 1]



test_correlations = test.corr()

test_correlations = test_correlations.values.flatten()

test_correlations = test_correlations[test_correlations != 1]



plt.figure(figsize=(20,5))

sns.distplot(train_correlations, color="Red", label="train")

sns.distplot(test_correlations, color="Green", label="test")

plt.xlabel("Correlation values found in train")

plt.ylabel("Density")

plt.title("Correlations between features"); 

plt.legend();
learningRate = 0.1

xgbModel = xgb.XGBClassifier(   max_depth = 5,

                                objective = 'binary:logistic',

                                booster= 'gbtree',

                                learning_rate = learningRate,

                                base_score= 0.2,

                                alpha= 0.4,

                                n_estimators= 100,

                                seed= 1301,

                                silent= 1,

                                eta= 0.3,

                                gamma= 0,

                                min_child_weight= 1,

                                max_delta_step= 0,

                                subsample= .8,

                                colsample_bytree= .8,

                                reg_alpha=0,

                                reg_lambda= 1,

                                missing= None,

                                verbosity= 1,

                                nthread = 2 )
idx         = 1

kf          = StratifiedKFold( n_splits = 10, random_state = 44000, shuffle = False )



for train_index, val_index in kf.split( y_train, y_train ):

    print('\n{} of kfold {}'.format( idx, kf.n_splits ))

    trn_dat, trn_tgt = X_train.loc[ train_index ], y_train.loc[ train_index ]

    val_dat, val_tgt = X_train.loc[ val_index ], y_train.loc[ val_index ]

    xgbModel.fit( trn_dat, trn_tgt,

                  early_stopping_rounds = 20,

                  eval_set = [(trn_dat, trn_tgt), (val_dat, val_tgt)],

                  eval_metric = "auc", verbose = True )

    pred = xgbModel.predict( val_dat )

    print('accuracy_score', accuracy_score( val_tgt, pred ))

    idx += 1  

plt.figure( figsize=( 18, 14 ))

plt.bar(range(len(xgbModel.feature_importances_)), xgbModel.feature_importances_)

plt.show()
pred_prob_val = pd.DataFrame()

pred_prob_val['target'] = xgbModel.predict_proba( data=X_test )[:,1]

sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})

sub_df["target"] = xgbModel.predict( X_test )

sub_df.loc[pred_prob_val['target'] > 0.36, 'target'] = 1

sub_df.to_csv("submission.csv", index=False)