import numpy as np 

import pandas as pd 

from sklearn.model_selection import KFold, train_test_split

from catboost import CatBoostClassifier, Pool

from sklearn.metrics import roc_auc_score as auc

import plotly.graph_objects as go
train_raw=pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test_raw=pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")



train = train_raw.drop(['id', 'target'], axis=1)

test = test_raw.drop(['id'], axis=1)

y = train_raw.target



cols = train.columns

train_length = train.shape[0]

test_length = test.shape[0]
# cbc = CatBoostClassifier(iterations = 300, learning_rate = 0.1, eval_metric = 'AUC', verbose = False)



# tr_x, val_x, tr_y, val_y = train_test_split(train, y, test_size = 0.2, shuffle = True, random_state = 10)



# cbc.fit(tr_x, tr_y, eval_set=(val_x, val_y), cat_features=cols)



# y_pred = cbc.predict_proba(test)[:, 1]



# submission = pd.DataFrame({'id': test_raw.id, 'target': y_pred})

# submission.to_csv('submission.csv', index=False)
n_splits = 5



kf = KFold(n_splits=n_splits, shuffle=True, random_state=10)



y_pred_ad = np.zeros(train.shape[0]) ### Initialize an array to record the predicted result from adversarial validation





### For all rows from train, no matter it is in tr_x or val_x, its taget value is setted as 1.

### For rows from test, its taget value is setted as 0.

for tr_range, val_range in kf.split(train):

    tr_x = train.loc[tr_range]

    val_x = train.loc[val_range]

    val_y = np.ones(val_x.shape[0]) 

    

    tr_x_combined = pd.concat([tr_x, test], axis = 0)

    tr_y_combined = np.append(np.ones(tr_x.shape[0]), np.zeros(test_length))

    

    cbc = CatBoostClassifier(iterations = 300, learning_rate = 0.1, eval_metric = 'AUC', verbose = False)

    cbc.fit(tr_x_combined, tr_y_combined, eval_set=(val_x, val_y), cat_features = cols)

    y_pred_ad[val_range] = cbc.predict_proba(val_x)[:, 1]  
lox = []

loy = []

for i in range(50, 60, 1): ### Proba from 0.5 to 0.6

    threshold = i / 100

    lox.append(threshold)

    train_ad = train[y_pred_ad < threshold]

    loy.append(train_ad.shape[0])

fig = go.Figure(data=go.Scatter(x=lox, y=loy))

fig.show()
train = train[y_pred_ad < 0.56]

y = y[y_pred_ad < 0.56]
cbc = CatBoostClassifier(iterations = 300, learning_rate = 0.1, eval_metric = 'AUC', verbose = False)



tr_x, val_x, tr_y, val_y = train_test_split(train, y, test_size = 0.2, shuffle = True, random_state = 10)



cbc.fit(tr_x, tr_y, eval_set=(val_x, val_y), cat_features=cols)



y_pred = cbc.predict_proba(test)[:, 1]



submission = pd.DataFrame({'id': test_raw.id, 'target': y_pred})

submission.to_csv('submission.csv', index=False)