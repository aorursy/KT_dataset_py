from datetime import datetime



print("last update: {}".format(datetime.now())) 
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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,  GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier,  ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

import lightgbm as lgb

from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, roc_auc_score

from sklearn.svm import SVC 

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMModel,LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn import metrics

import numpy as np

np.random.seed(0)



# Read the data

X_original = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')

X_test_full = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')

X = X_original.copy()
X.head(1)
X_test_full.head(1)
Experiments = {"Algo":["RandomForestClassifier", "XGBClassifier", "MLPClassifier",  "ExtraTreesClassifier", 'LGBMClassifier'],

              "object": [lambda: RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt'),

                        lambda: XGBClassifier(learning_rate =0.05, n_estimators=1000, n_jobs=-1),

                        lambda: MLPClassifier(solver='adam', activation='relu', max_iter=3000, hidden_layer_sizes=(100,100), 

                                              random_state=1),

                        lambda: ExtraTreesClassifier(n_estimators = 1000),

                        lambda: LGBMClassifier(learning_rate =0.05, n_estimators=1000)],

               "F1_score": [],

               "prediction": [[] for _ in range(5)]}

def reverse_onehot(df, subset, reverse_name):

    df_new = pd.DataFrame()

    df_new = df.drop(subset, axis = 1)

    temp = df[subset]

    df_new[reverse_name] = temp.idxmax(axis = 1).astype(str)

    df_new[reverse_name] = df_new[reverse_name].apply(lambda x: int(str(x)[9:]))

    return df_new  



subset = ['Soil_Type'+ str(i) for i in range(1,41)]

Ds = reverse_onehot(X, subset, 'Soil_Type')

Ds.tail()
# scale before mlp

X_train, X_valid, y_train, y_valid = train_test_split(X.drop('Cover_Type', axis = 1), X['Cover_Type'], test_size = 0.2)

scale = StandardScaler()

preprocessor = ColumnTransformer(transformers = [('scaling numeric', scale, list(X.drop('Cover_Type', axis = 1).columns[0:10]))])

# Get F1_scores of differents models

for i, obj in enumerate(Experiments["object"]):

    if i == 2:

        model = obj()

        my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

        my_pipeline.fit(X_train, y_train)

        y_val_pred = my_pipeline.predict(X_valid)

        Experiments['prediction'].append(y_val_pred)

        Experiments['F1_score'].append(f1_score(y_valid, y_val_pred, average='weighted')) 

    else:

        model = obj()

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_valid)

        Experiments['prediction'].append(y_val_pred)

        Experiments['F1_score'].append(f1_score(y_valid, y_val_pred, average='weighted'))
# Print the F1_scores of the five selected mdels

Experiments['F1_score']
mlp =  MLPClassifier(solver='adam', activation='relu', max_iter=3000, hidden_layer_sizes=(50,50), random_state=1)

MLP_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', mlp)])
list_estimators = [RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt'), 

                   XGBClassifier(learning_rate =0.05, n_estimators=1000),

                   MLP_pipeline,

                   ExtraTreesClassifier(n_estimators = 1000, max_features = 'sqrt'), 

                   LGBMClassifier(learning_rate =0.05, n_estimators=1000)]

base_methods = list(zip(Experiments["Algo"], list_estimators))

base_methods 
# Hard voting decides according to vote number which is the majority wins

Voting_model_hard = VotingClassifier(estimators= base_methods, voting='hard')

Voting_model_hard.fit(X_train, y_train)

y_vp_val = Voting_model_hard.predict(X_valid)

y_vp_train = Voting_model_hard.predict(X_train)

print('f1_score', f1_score(y_valid, y_vp_val, average='weighted'))

print('acc_score_train', accuracy_score(y_train, y_vp_train))

print('acc_score_valid', accuracy_score(y_valid, y_vp_val))
# save the model to file and load it to make predictions on the unseen test set

import pickle

filename = 'Voting_model_hard.sav'

pickle.dump(Voting_model_hard, open(filename, 'wb'))
loaded_model_hard = pickle.load(open(filename, 'rb'))

result_hard = loaded_model_hard.score(X_valid, y_valid)

print(result_hard)
# In soft voting, we can set weight value to give more priorities to certain classifiers according to their performance

Voting_model_soft = VotingClassifier(estimators= base_methods, voting='soft', weights=[15, 9, 8, 14, 15])

Voting_model_soft.fit(X_train, y_train)

y_vp_val = Voting_model_soft.predict(X_valid)

y_vp_train = Voting_model_soft.predict(X_train)

print('f1_score', f1_score(y_valid, y_vp_val, average='weighted'))

print('acc_score_train', accuracy_score(y_train, y_vp_train))

print('acc_score_valid', accuracy_score(y_valid, y_vp_val))
#save the model to file and load it to make predictions on the unseen test set

filename_soft = 'Voting_model_soft.sav'

pickle.dump(Voting_model_soft, open(filename_soft, 'wb'))
loaded_model_soft = pickle.load(open(filename_soft, 'rb'))

result_soft = loaded_model_soft.score(X_valid, y_valid)

print(result_soft)
Ds_test = reverse_onehot(X_test_full, subset, 'Soil_Type')
Ds_test.head()
preds_test = Voting_model_soft.predict(X_test_full)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test_full.index,

                       'Cover_Type': preds_test})

output.to_csv('submission.csv', index=False)