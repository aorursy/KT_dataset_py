import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
train = pd.read_csv('../input/reducing-commercial-aviation-fatalities/train.csv')
train.sample(10)
print(train.shape)
test_1 = pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize = 5)

test_top = next(test_1)

test_top
pd.crosstab(train.experiment, train.event)
pd.crosstab(train.experiment, train.crew)
pd.crosstab(train.experiment,train.seat)
print(list(enumerate(train.columns)))
crew = 3

seat = 0

exp = 'DA'

ev = 'D'



sel = (train.crew == crew) & (train.experiment == exp) & (train.seat == seat)

pilot_info = train.loc[sel,:].sort_values(by = 'time')



plt.figure(figsize = [16,12])

for i in range (4,27):

    plt.subplot (6,4,i-3)

    plt.plot(pilot_info.time,

            pilot_info.iloc[:,i], zorder = 1)

    plt.scatter(pilot_info.loc[pilot_info.event == ev,:].time,

               pilot_info.loc[pilot_info.event == ev, :].iloc[:,i], c = 'red', zorder = 2, s=1)

    plt.title(pilot_info.columns[i])

    

plt.tight_layout()

plt.show()


train['f7-f8'] = train['eeg_f7'] - train['eeg_f8']

train['f3-f4'] = train['eeg_f3'] - train['eeg_f4']

train['t3-t4'] = train['eeg_t3'] - train['eeg_t4']

train['c3-c4'] = train['eeg_c3'] - train['eeg_c4']

train['p3-p4'] = train['eeg_p3'] - train['eeg_p4']

train['t5-t6'] = train['eeg_t5'] - train['eeg_t6']

train['o1-o2'] = train['eeg_o1'] - train['eeg_o2']



train_columns = ['crew','seat','f7-f8','f3-f4','t3-t4','c3-c4','p3-p4','t5-t6','o1-o2','ecg','r','gsr','event']

train = train.loc[:,train_columns]

train.sample(10)
test_top['f7-f8'] = test_top['eeg_f7'] - test_top['eeg_f8']

test_top['f3-f4'] = test_top['eeg_f3'] - test_top['eeg_f4']

test_top['t3-t4'] = test_top['eeg_t3'] - test_top['eeg_t4']

test_top['c3-c4'] = test_top['eeg_c3'] - test_top['eeg_c4']

test_top['p3-p4'] = test_top['eeg_p3'] - test_top['eeg_p4']

test_top['t5-t6'] = test_top['eeg_t5'] - test_top['eeg_t6']

test_top['o1-o2'] = test_top['eeg_o1'] - test_top['eeg_o2']



test_columns = ['id','crew','seat','f7-f8','f3-f4','t3-t4','c3-c4','p3-p4','t5-t6','o1-o2','ecg','r','gsr']

test_columns = test_top.loc[:,test_columns]

test_columns
y_train_full = train.event

X_train_full = train.iloc[:,:-1]

X_train_full.head()
pd.DataFrame({

    'min_val': X_train_full.min(axis=0).values,

    'max_val': X_train_full.max(axis=0).values

}, index = X_train_full.columns)
y_train_full.value_counts()
y_train_full.value_counts()/len(y_train_full)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size = 0.98, stratify = y_train_full, random_state = 1)

print(X_train.shape)
%%time



xbg_mod = XGBClassifier()

xbg_mod.fit(X_train, y_train)



xbg_mod.score(X_train, y_train)
from sklearn.metrics import log_loss

log_loss(y_train, xbg_mod.predict_proba(X_train))
log_loss(y_valid,xbg_mod.predict_proba(X_valid))
%%time

xgd_pipe = Pipeline(

    steps = [

        ('classifier', XGBClassifier(learning_rate=0.3, max_depth=6, alpha=1, n_estimators=50, subsample=0.5))

    ]

)



xgd_param_grid = {

    'classifier__learning_rate' : [0.1, 0.3, 0.5, 0.7, 0.9],

    'classifier__alpha' : [0, 1, 10, 100]

    #'classifier__max_depth': [8, 16, 32, 64, 128]

    

}





np.random.seed(1)

xgd_grid_search = GridSearchCV(xgd_pipe, xgd_param_grid, cv=5, refit='True')

xgd_grid_search.fit(X_train, y_train)



print(xgd_grid_search.best_score_)

print(xgd_grid_search.best_params_)
print(xbg_mod.predict_proba(test_columns.iloc[:,1:]))
%%time



cs = 1000000

i = 0





for test in pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize = cs):

    test['f7-f8'] = test['eeg_f7'] - test['eeg_f8']

    test['f3-f4'] = test['eeg_f3'] - test['eeg_f4']

    test['t3-t4'] = test['eeg_t3'] - test['eeg_t4']

    test['c3-c4'] = test['eeg_c3'] - test['eeg_c4']

    test['p3-p4'] = test['eeg_p3'] - test['eeg_p4']

    test['t5-t6'] = test['eeg_t5'] - test['eeg_t6']

    test['o1-o2'] = test['eeg_o1'] - test['eeg_o2']



    test_columns = ['id','crew','seat','f7-f8','f3-f4','t3-t4','c3-c4','p3-p4','t5-t6','o1-o2','ecg','r','gsr']

    test = test.loc[:,test_columns]

    print('--Iteration', i, 'is started')

    

    test_pred = xbg_mod.predict_proba(test.iloc[:,1:])

    

    partial_submission = pd.DataFrame({

        'id': test.id,

        'A' : test_pred[:,0],

        'B' : test_pred[:,1],

        'C' : test_pred[:,2],

        'D' : test_pred[:,3]

    })

    

    if i == 0:

        submission = partial_submission.copy()

    else:

        submission = submission.append(partial_submission, ignore_index=True)

        

    del test

    print('++Iteration', i, 'is done!')

    i +=1
submission.head()
submission.to_csv("submission.csv", index=False)