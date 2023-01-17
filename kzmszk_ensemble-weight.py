# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import os

os.system("ls ../input/otto-group-product-classification-challenge")

# As for ensemble learning:
# https://qiita.com/CraveOwl/items/623fa9172d4a0a5fe018
# As for lightgbm and feature importance:
# https://qiita.com/studio_haneya/items/e70e835c26524d506e19
# As for minimize
# http://www.kamishima.net/mlmpyja/lr/optimization.html
# ensemble learning in general:
# https://qiita.com/TomHortons/items/2a05b72be180eb83a204
# https://mlwave.com/kaggle-ensembling-guide/
# slideshare by nishio
# https://www.slideshare.net/nishio/kaggle-otto

# what learned from otto
# https://medium.com/@chris_bour/6-tricks-i-learned-from-the-otto-kaggle-challenge-a9299378cd61
# 2nd winner blog
# https://medium.com/kaggle-blog/otto-product-classification-winners-interview-2nd-place-alexander-guschin-%E3%83%84-e9248c318f30
# slide 85th
# https://www.slideshare.net/eugeneyan/kaggle-otto-challenge-how-we-achieved-85th-out-of-3845-and-what-we
train = pd.read_csv("../input/otto-group-product-classification-challenge/train.csv")
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

labels = train['target']
train.drop(['target', 'id'], axis=1, inplace=True)

# describe() output basic static values like mean, min, max, std...
train.describe()
# get log
train_log = np.log(train + 1)
train_log.describe()
# get scale
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
train_scaled.describe()

### we need a test set that we didn't train on to find the best weights for combining the classifiers
sss = StratifiedShuffleSplit(1, test_size=0.05, random_state=1234)
for train_index, test_index in sss.split(train, labels):
    break

train_x, train_y = train.values[train_index], labels.values[train_index]
test_x, test_y = train.values[test_index], labels.values[test_index]

train_logx, train_logy = train_log.values[train_index], labels.values[train_index]
test_logx, test_logy = train_log.values[test_index], labels.values[test_index]

train_scaledx, train_scaley = train_scaled.values[train_index], labels.values[train_index]
test_scaledx, test_scaley = train_scaled.values[test_index], labels.values[test_index]
### building the classifiers
clfs = []

rfc = RandomForestClassifier(n_estimators=50, random_state=4141, n_jobs=-1)
rfc.fit(train_x, train_y)
print('RFC LogLoss {score}'.format(score=log_loss(test_y, rfc.predict_proba(test_x))))
clfs.append(rfc)
### usually you'd use xgboost and neural nets here

logreg = LogisticRegression()
logreg.fit(train_x, train_y)
print('LogisticRegression LogLoss {score}'.format(score=log_loss(test_y, logreg.predict_proba(test_x))))
clfs.append(logreg)

logreg2 = LogisticRegression()
logreg2.fit(train_scaledx, train_y)
print('LogisticRegression2 LogLoss {score}'.format(score=log_loss(test_y, logreg2.predict_proba(test_scaledx))))
clfs.append(logreg2)

logreg3 = LogisticRegression()
logreg3.fit(train_logx, train_y)
print('LogisticRegression2 LogLoss {score}'.format(score=log_loss(test_y, logreg3.predict_proba(test_logx))))
clfs.append(logreg3)

rfc2 = RandomForestClassifier(n_estimators=50, random_state=1337, n_jobs=-1)
rfc2.fit(train_x, train_y)
print('RFC2 LogLoss {score}'.format(score=log_loss(test_y, rfc2.predict_proba(test_x))))
clfs.append(rfc2)

rfc3 = RandomForestClassifier(n_estimators=50, random_state=1337, n_jobs=-1)
rfc3.fit(train_scaledx, train_y)
print('RFC3 LogLoss {score}'.format(score=log_loss(test_y, rfc3.predict_proba(test_scaledx))))
clfs.append(rfc3)

from lightgbm import LGBMClassifier
lgbm_params = {
    # 多値分類問題
    'objective': 'multiclass',
    'num_class': 9,
}
lgb = LGBMClassifier(objective='multiclass', num_class=9)
lgb.fit(train_x, train_y)
print('lgb LogLoss {score}'.format(score=log_loss(test_y, lgb.predict_proba(test_x))))
clfs.append(lgb)

lgb2 = LGBMClassifier(objective='multiclass', num_class=9)
lgb2.fit(train_logx, train_y)
print('lgb2 LogLoss {score}'.format(score=log_loss(test_y, lgb2.predict_proba(test_logx))))
#clfs.append(lgb2)

lgb3 = LGBMClassifier(objective='multiclass', num_class=9)
lgb3.fit(train_scaledx, train_y)
print('lgb3 LogLoss {score}'.format(score=log_loss(test_y, lgb3.predict_proba(test_scaledx))))
#clfs.append(lgb3)
### finding the optimum weights

predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(test_x))

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return log_loss(test_y, final_prediction)
    
#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
starting_values = [0.5]*len(predictions)

#adding constraints  and a different solver as suggested by user 16universe
#https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))
# randomforest x 2 + regression
# BEFORE: Ensamble Score: 0.5511264774379178
# add lightgbm
# add log(train + 1)
# add scale(train)
# Ensamble Score: 0.5141448040179412
# Ensamble Score: 0.5128691904172303
# other ensemble info:
# https://qiita.com/TomHortons/items/2a05b72be180eb83a204
# https://mlwave.com/kaggle-ensembling-guide/
res