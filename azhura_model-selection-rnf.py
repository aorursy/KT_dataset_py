from sklearn.ensemble import (

    RandomForestClassifier,

    ExtraTreesClassifier,

    AdaBoostClassifier,

)

from lightgbm import LGBMClassifier

from mlxtend.classifier import StackingCVClassifier

# cross validation

from sklearn.model_selection import KFold, cross_val_score

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plot

%matplotlib inline  

import os

import seaborn as sns # Plot

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler

from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.pipeline import make_pipeline

#Importing Confusion Matrix

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.classifier import StackingCVClassifier





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Load file

Path_train_ = "../input/learn-together/train.csv"

Path_test_ = "../input/learn-together/test.csv"

Path_train = "../input/comp-01/new_train.csv"

Path_test = "../input/comp-01/new_test.csv"



#Load dataset

ori_df_train_ = pd.read_csv(Path_train_,index_col='Id')# ->> "train.csv"

ori_df_test_ = pd.read_csv(Path_test_,index_col='Id')# ->>"test.csv"

df_train = pd.read_csv(Path_train)# ->> "train.csv"

df_test = pd.read_csv(Path_test)# ->>"test.csv"

original_data_train = ori_df_train_.copy()

original_data_test = ori_df_test_.copy()

data_train = df_train.copy()

data_test = df_test.copy()

print("original Train shape :",original_data_train.shape)

print("original Test shape :",original_data_test.shape)

print("New Train shape :",data_train.shape)

print("New Test shape :",data_test.shape)

y = data_train.Cover_Type.values

x = data_train.copy()

x.drop('Cover_Type', inplace=True, axis=1)

test = data_test.copy()

# Functions

def predict(model, filename, X=x, y=y, test=test):

    model.fit(X, y)

    predicts = model.predict(test)

    

    output = pd.DataFrame({'Id': test.index,

                       'Cover_Type': predicts})

    output.to_csv('best_submission.csv', index=False) 

    return predicts



def select(importances, edge):

    c = importances.Importances >= edge

    cols = importances[c].Features.values

    return cols



def feature_importances(clf, x, y, figsize=(30, 30)):

    clf = clf.fit(x, y)

    

    importances = pd.DataFrame({'Features': x.columns, 

                                'Importances': clf.feature_importances_})

    

    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



    fig = plt.figure(figsize=figsize)

    sns.barplot(x='Features', y='Importances', data=importances)

    plt.xticks(rotation='vertical')

    plt.show()

    return importances



def cross_val(models, X=x, y=y):

    r = dict()

    for name, model in models.items():

        cv_results = cross_val_score(model, x, y,

                             cv=cv, 

                             scoring='accuracy')

        r[name] = cv_results

        print(name, 'Accuracy Mean {0:.4f}, Std {1:.4f}'.format(

              cv_results.mean(), cv_results.std()))

    return r

    

def choose_best(results):

    errors = dict()



    for name, arr in results.items():

        errors[name] = arr.mean()



    best_model =  [m for m, e in errors.items() 

                   if e == max(errors.values())][0]

    return best_model



seed = 1



models = {

    'LGBM': LGBMClassifier(n_estimators=370,

                           metric='multi_logloss',

                           num_leaves=100,

                           verbosity=0,

                           random_state=seed,

                           n_jobs=-1), 

    'Random Forest': RandomForestClassifier(n_estimators=1000,

                                            n_jobs=-1,

                                            random_state=seed),

    'Extra Tree': ExtraTreesClassifier(

           max_depth=400, 

           n_estimators=450, n_jobs=-1,

           oob_score=False, random_state=seed, 

           warm_start=True)



}
clf = models['Random Forest']

importances = feature_importances(clf, x, y) 

col = select(importances, 0.0004)

x = x[col]

test = test[col]

# model selection functions

cv = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val(models)
best_model_name = choose_best(results)

model = models[best_model_name]

# Meta Classifier

meta_cls = XGBClassifier(learning_rate =0.1, n_estimators=500)

list_estimators = [RandomForestClassifier(n_estimators=500,random_state=1, n_jobs=-1),

                   MLPClassifier(batch_size='auto', 

                                 random_state=1,activation='relu',

                                 solver='adam',verbose=True,learning_rate='constant',

                                 alpha=0.004,hidden_layer_sizes=(100,100),max_iter=1000),

                   ExtraTreesClassifier(max_depth=400,

                                        criterion='entropy',

                                        n_estimators=898, 

                                        n_jobs=-1,

                                        oob_score=False, 

                                        random_state=seed, 

                                        warm_start=True)]

base_methods = ["RandomForestClassifier", "MLPClassifier","ExtraTreesClassifier"]

state = 1

stack = StackingCVClassifier(classifiers=list_estimators,

                             meta_classifier=meta_cls,

                             cv=5,

                             use_probas=True,

                             verbose=1, 

                             random_state=seed,

                             n_jobs=-1)
stack.fit(x, y)

print('Complete!')
preds_test = stack.predict(test)

# Save test predictions to file

output = pd.DataFrame({'Id': ori_df_test_.index,

                       'Cover_Type': preds_test})

output.to_csv('submission.csv', index=False)

output.head()