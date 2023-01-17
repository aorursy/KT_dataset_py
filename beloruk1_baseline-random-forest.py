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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize    
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')

def readFile(fn):
    file = '../input/train_sentiment/'+fn['PetID']+'.json'
    if os.path.exists(file):
        with open(file) as data_file:    
            data = json.load(data_file)  

        df = json_normalize(data)
        mag = df['documentSentiment.magnitude'].values[0]
        score = df['documentSentiment.score'].values[0]
        return pd.Series([mag,score],index=['mag','score']) 
    else:
        return pd.Series([0,0],index=['mag','score'])
    
def readTestFile(fn):
    file = '../input/test_sentiment/'+fn['PetID']+'.json'
    if os.path.exists(file):
        with open(file) as data_file:    
            data = json.load(data_file)  

        df = json_normalize(data)
        mag = df['documentSentiment.magnitude'].values[0]
        score = df['documentSentiment.score'].values[0]
        return pd.Series([mag,score],index=['mag','score']) 
    else:
        return pd.Series([0,0],index=['mag','score'])
train[['SentMagnitude', 'SentScore']] = train[['PetID']].apply(lambda x: readFile(x), axis=1)
test[['SentMagnitude', 'SentScore']] = test[['PetID']].apply(lambda x: readTestFile(x), axis=1)

train_X = train.drop(['Name', 'Description', 'RescuerID', 'PetID', 'AdoptionSpeed'], axis=1)
train_y = train['AdoptionSpeed']
test_X = test.drop(['Name', 'Description', 'RescuerID', 'PetID'], axis=1)

from sklearn.model_selection import GridSearchCV, StratifiedKFold

train_meta = np.zeros(train_y.shape)
test_meta = np.zeros(test_X.shape[0])

clf = RandomForestClassifier(bootstrap=True, criterion = 'gini', max_depth=80, max_features='auto', min_samples_leaf=5, min_samples_split=5, n_estimators=200)

splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=1812).split(train_X, train_y))
for idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = train_X.iloc[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X.iloc[valid_idx]
        y_val = train_y[valid_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        print(cohen_kappa_score(y_val, y_pred, weights='quadratic'))
        
        y_test = clf.predict(test_X)
        
        train_meta[valid_idx] = y_pred.reshape(-1)
        test_meta += y_test.reshape(-1) / len(splits)
sub = pd.read_csv('../input/test/sample_submission.csv')
sub['AdoptionSpeed'] = test_meta
sub['AdoptionSpeed'] = sub['AdoptionSpeed'].astype(int)
sub.to_csv("submission.csv", index=False)