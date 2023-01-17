# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold, KFold

warnings.filterwarnings('ignore')
#load the training file

train_data = pd.read_csv('../input/train.csv')
train_data.head(10)
value_count = train_data['target'].value_counts()

print(f'THE VALUE COUNTS OF THE TARGET VARIABLE : \n{value_count}')
sns.set(style = 'darkgrid')

plt.figure(figsize = (12,10))

sns.countplot(y = train_data['target'])
#load in the testing file

test_data = pd.read_csv('../input/test.csv')
test_data.head(10)
#spearated the dataset into input features and labels

X = train_data.drop(['target', 'ID_code'], axis = 1)

y = train_data['target']
import keras

from keras.layers import Dense

from keras.layers import BatchNormalization

from keras.layers import MaxPooling2D

from keras.layers import Activation

from keras.layers import Dropout

from keras.models import Sequential

from keras import optimizers



from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
kernel_init = 'normal'

def SimpleFFNN(input_dim, activation, classes):

    model = Sequential()



    model.add(Dense(512, kernel_initializer = kernel_init, input_dim = input_dim))

    model.add(BatchNormalization())

    model.add(Activation(activation))

    model.add(Dense(512, kernel_initializer = kernel_init, input_dim = input_dim))

    model.add(BatchNormalization())

    model.add(Activation(activation))

    model.add(Dropout(0.5))



    model.add(Dense(256, kernel_initializer = kernel_init)) 

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dense(256, kernel_initializer = kernel_init)) 

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Dense(128, kernel_initializer = kernel_init))    

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dense(128, kernel_initializer = kernel_init))    

    model.add(Activation(activation))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Dense(classes, kernel_initializer = kernel_init))    

    model.add(Activation('sigmoid'))

    

    return model
#we will also flatten our output label

y_flatten = y.ravel()

print(f"THE SIZE OF THE OTUPUT LABELS : {y_flatten.shape}")
#let's scale the original training data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X)



X_scaled = sc.transform(X)
#scaling the testing data

test = test_data.drop('ID_code', axis = 1)

test_scaled = sc.transform(test)
input_dim = X_scaled.shape[1]

activation = 'relu'

classes = 1



history = dict() #dictionery to store the history of individual models for later visualization

prediction_scores = dict() #dictionery to store the predicted scores of individual models on the test dataset



#here we will be training the same model for a total of 10 times and will be considering the mean of the output values for predictions

for i in np.arange(0, 5):

    optim = optimizers.Adam(lr = 0.001)

    ensemble_model = SimpleFFNN(input_dim = input_dim, activation = activation, classes = classes)

    ensemble_model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])

    print('TRAINING MODEL NO : {}'.format(i))

    H = ensemble_model.fit(X_scaled, y_flatten,

                           batch_size = 128,

                           epochs = 200,

                           verbose = 1)

    history[i] = H

    

    ensemble_model.save('MODEL_{}.model'.format(i))

    

    predictions = ensemble_model.predict(test_scaled, verbose = 1, batch_size = 128)

    prediction_scores[i] = predictions
#we will considering all the features except 'ID_code' and 'target'

features = [value for value in train_data.columns if value not in ['ID_code', 'target']]
#defining the parameters

param = {

        'bagging_freq': 5,

        'bagging_fraction': 0.38,

        'boost_from_average':'false',

        'boost': 'gbdt',

        'feature_fraction': 0.045,

        'learning_rate': 0.0095,

        'max_depth': -1,  

        'metric':'auc',

        'min_data_in_leaf': 80,

        'min_sum_hessian_in_leaf': 10.0,

        'num_leaves': 13,

        'num_threads': 8,

        'tree_learner': 'serial',

        'objective': 'binary', 

        'verbosity': 1

    }
folds = StratifiedKFold(n_splits = 12, shuffle = False, random_state = 101)

train_mat = np.zeros(len(train_data))

predictions = np.zeros(len(test_data))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data.values, y.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train_data.iloc[trn_idx][features], label = y.iloc[trn_idx])

    val_data = lgb.Dataset(train_data.iloc[val_idx][features], label = y.iloc[val_idx])



    num_round = 500000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval = 1000, early_stopping_rounds = 500)

    train_mat[val_idx] = clf.predict(train_data.iloc[val_idx][features], num_iteration = clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis = 0)

    

    predictions += clf.predict(test_data[features], num_iteration = clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(y, train_mat)))
## Catboost : https://www.kaggle.com/wakamezake/starter-code-catboost-baseline

from catboost import Pool, CatBoostClassifier

model = CatBoostClassifier(loss_function = "Logloss", eval_metric = "AUC")

kf = KFold(n_splits = 5, random_state = 42, shuffle = True)



y_valid_pred = 0 * y

y_test_pred = 0



for idx, (train_index, valid_index) in enumerate(kf.split(train_data)):

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    X_train, X_valid = train_data[features].iloc[train_index,:], train_data[features].iloc[valid_index,:]

    _train = Pool(X_train, label = y_train)

    _valid = Pool(X_valid, label = y_valid)

    print( "\nFold ", idx)

    fit_model = model.fit(_train,

                          eval_set = _valid,

                          use_best_model = True,

                          verbose = 1

                         )

    pred = fit_model.predict_proba(X_valid)[:,1]

    print( "  auc = ", roc_auc_score(y_valid, pred) )

    y_valid_pred.iloc[valid_index] = pred

    y_test_pred += fit_model.predict_proba(test_data[features])[:,1]

y_test_pred /= 5
#submission for LGBM

df_lgbm = pd.DataFrame({"ID_code" : test_data["ID_code"].values})

df_lgbm["target"] = predictions

df_lgbm.to_csv("lgbm_submission.csv", index = False)
#submission for CAT

df_cat = pd.DataFrame({"ID_code": test_data["ID_code"].values})

df_cat["target"] = y_test_pred

df_cat.to_csv("cat_submission.csv", index = False)
#making predictions

prediction = np.hstack([p.reshape(-1,1) for p in prediction_scores.values()]) #taking the scores of all the trained models

predictions_ensemble = np.mean(prediction, axis = 1)



print(predictions_ensemble.shape)
#submission for ENSEMBLE

df_ensemble = pd.DataFrame({"ID_code" : test_data["ID_code"].values})

df_ensemble["target"] = predictions_ensemble

df_ensemble.to_csv("ensemble_submission.csv", index = False)
##submission of combined model

df_total = pd.DataFrame({"ID_code" : test_data["ID_code"].values})

df_total["target"] = 0.4*df_lgbm["target"] + 0.4*df_cat["target"] + 0.2*df_ensemble['target']

df_total.to_csv("lgbm_cat_ensemble_submission.csv", index = False)
df1 = pd.DataFrame({"ID_code" : test_data["ID_code"].values})

df1["target"] = 0.5*df_lgbm["target"] + 0.5*df_cat["target"]

df1.to_csv("lgbm_cat_submission.csv", index = False)
df2 = pd.DataFrame({"ID_code" : test_data["ID_code"].values})

df2["target"] = 0.5*df_lgbm["target"] + 0.5*df_ensemble["target"]

df2.to_csv("lgbm_ensemble_submission.csv", index = False)
df3 = pd.DataFrame({"ID_code" : test_data["ID_code"].values})

df3["target"] = 0.5*df_cat["target"] + 0.5*df_ensemble["target"]

df3.to_csv("cat_ensemble_submission.csv", index = False)
df4 = pd.DataFrame({"ID_code" : test_data["ID_code"].values})

df4["target"] = 0.2*df_cat["target"] + 0.4*df_ensemble["target"] + 0.4*df_lgbm["target"]

df4.to_csv("0.2cat_0.4ensemble_0.4lgbm_submission.csv", index = False)
df5 = pd.DataFrame({"ID_code" : test_data["ID_code"].values})

df5["target"] = 0.4*df_cat["target"] + 0.4*df_ensemble["target"] + 0.2*df_lgbm["target"]

df5.to_csv("0.4cat_0.4ensemble_0.2lgbm_submission.csv", index = False)