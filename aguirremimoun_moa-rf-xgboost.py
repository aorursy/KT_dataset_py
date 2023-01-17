# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from os import path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import log_loss, classification_report

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.multioutput import MultiOutputClassifier

from xgboost import XGBClassifier #faster than GradientBoostingClassifier??



import numpy as np # linear algebra

import pandas as pd 

import pickle

import time







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_target_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
def fit_and_save(method, X_train, y_train, filename):

    

    '''

    fit a dataset with the given method

    ------------------------------------------------------------------------------------------------------------

    Arguments

    - method = method to use to fit the data (e.g. RandomForest, XGBClassifier, MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist')))

    - X_train = Pandas dataframe, size (n,m) with n is the number of samples and m the number of features

    - y_train = pandas dataframe with the output of X_train, of size (n, p) with p the number of possible output

    - filename = a string, name of the file in which the model should be saved, if it does not exist yet

    ------------------------------------------------------------------------------------------------------------

    returns

    - a model

    

    '''

    

    if not path.exists(filename):

        print('fitting model')

        clf = method.fit(X_train, y_train)

        print(f'Model have been successfully fitted')

        pickle.dump(clf, open(filename, 'wb'))

        print(f'model have been successfully saved as {filename}')

    else:

        print('model has already been fitted previously')

        print('loading model from memory\n')

        clf = pickle.load(open(filename, 'rb'))

        print(f'Model named {filename} have been successfully loaded')



    return clf
def loss(model, X_val, y_val):

    '''

    arguments

    model = a sklearn model

    X_val = a pandas dataframe of data used to predict outcome y_pred

    y_val = a pands dataframe of the true outputs

    --------------------------------------------------------------------------------------------------

    returns

    log loss of the model with the inputed data

    '''

    print(f'computing the loss of {str(model)}... please wait')

    pred = model.predict(X_val)

    loss = log_loss(y_val, pred)

    print(f'the log_loss of the data given the model is {str(round(loss, 4))}')

    return loss

    
def submission_maker(model, test):

    '''

    given a fitted sklearn compatible model, return a submission file

    

    parameter:

    a sklearn model

    test = pandas dataframe of test set, that should have

    '''

    

    clf = model.predict(test) ## caution: if not all

    

    

    
#Split train and target into a train and validation set

X_train, X_val, y_train, y_val = train_test_split(train_features,train_targets_scored, test_size=0.25, shuffle=True, random_state=1)
#filename = '../input/moa-rf-xgboost/RF.sav'

#model = MultiOutputClassifier(RandomForestClassifier())

#clf = fit_and_save(model, X_train, y_train, filename)
#loss(clf, X_val, y_val)

# print(classification_report(y_val, pred))
# Building the Decision tree model using the training dataset with class weights

#DecTree = tree.DecisionTreeClassifier(class_weight={0:1, 1:1000})

#DecTree.fit(X_train, y_train_1)

#ypred = DecTree.predict(X_val)



#print(classification_report(ytrue_1, ypred))
start = time.time()

filename = 'MOA_XGB2.sav'

model = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))

clf = fit_and_save(model, X_train.iloc[:, 4:], y_train.iloc[:, 1:], filename)

print(f'model fitted in {time.time()-start} seconds')
loss = loss(clf, X_val.iloc[:, 4:], y_val.iloc[:, 1:])
pred = clf.predict(test_features.iloc[:, 4:])
mask_ctrl = test_features['cp_type'] == "ctl_vehicle"
idx = test_features.iloc[:, 0].copy()

pred_df = pd.DataFrame(pred, columns=train_targets_scored.columns[1:]) #make a df from the np array of predictions

pre_test = pd.concat([idx, pred_df], axis=1)

pre_test.loc[mask_ctrl, '5-alpha_reductase_inhibitor': 'wnt_inhibitor' ] = 0 #Change the result of ctrl to zero (no pathway activated) 



pre_test.to_csv('submission.csv', index=False, header=True)
