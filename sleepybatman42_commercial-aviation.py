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
train = pd.read_csv("../input/reducing-commercial-aviation-fatalities/train.csv")
train.sample(10)
print(train.shape)
test_iterator = pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize=5)

test_top = next(test_iterator)

test_top
sample_submission = pd.read_csv("../input/reducing-commercial-aviation-fatalities/sample_submission.csv")

sample_submission.sample(10)
pd.crosstab(train.experiment, train.event)
pd.crosstab(train.experiment, train.crew)
pd.crosstab(train.experiment, train.seat)
print(list(enumerate(train.columns)))
crew = 3

seat = 0

exp = 'DA'

ev = 'D'



sel = (train.crew == crew) & (train.experiment == exp) & (train.seat == seat)

pilot_info = train.loc[sel,:].sort_values(by='time')



plt.figure(figsize=[16,12])

for i in range(4, 27):

    plt.subplot(6,4,i-3)

    plt.plot(pilot_info.time,

            pilot_info.iloc[:,i], zorder=1)

    plt.scatter(pilot_info.loc[pilot_info.event == ev,:].time,

               pilot_info.loc[pilot_info.event == ev,:].iloc[:,i], c='red', zorder=2, s=1)

    plt.title(pilot_info.columns[i])

    

plt.tight_layout()

plt.show()
y_train_full = train.event

X_train_full = train.iloc[:,4:27]

X_train_full.head()
pd.DataFrame({

    'min_val':X_train_full.min(axis=0).values,

    'max_val':X_train_full.max(axis=0).values

}, index = X_train_full.columns

)
y_train_full.value_counts()
y_train_full.value_counts() / len(y_train_full)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.98, stratify=y_train_full, random_state=1)



print(X_train.shape)
%%time

lr_mod = LogisticRegression(solver='lbfgs', n_jobs=-1)

lr_mod.fit(X_train, y_train)



print('Training Accuracy: ', lr_mod.score(X_train, y_train))

print('Validation Accuracy:', lr_mod.score(X_valid, y_valid))
%%time



lr_pipe = Pipeline(

steps = [

    ('scaler', StandardScaler()),

    ('classifier', LogisticRegression(solver='lbfgs', n_jobs=-1))

]

)



lr_param_grid = {

    'classifier__C': [0.0001, 0.001, 0.1, 1.0],

}





np.random.seed(1)

grid_search = GridSearchCV(lr_pipe, lr_param_grid, cv=5, refit='True')

grid_search.fit(X_train, y_train)



print(grid_search.best_score_)

print(grid_search.best_params_)
%%time



rf_mod = RandomForestClassifier(n_estimators=10, max_depth=32, n_jobs=-1)

rf_mod.fit(X_train, y_train)



print('Training Accuracy: ', rf_mod.score(X_train, y_train))

print('Validation Accuracy:', rf_mod.score(X_valid, y_valid))
%%time

rf_pipe = Pipeline(

    steps = [

        ('scaler', StandardScaler()),

        ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))

    ]

)



lr_param_grid = {

    'classifier__max_depth': [8, 16, 32, 64, 128]

}





np.random.seed(1)

grid_search = GridSearchCV(rf_pipe, lr_param_grid, cv=5, refit='True')

grid_search.fit(X_train, y_train)



print(grid_search.best_score_)

print(grid_search.best_params_)
grid_search.cv_results_['mean_test_score']
%%time

rf_mod = RandomForestClassifier(n_estimators=100, max_depth=32, n_jobs=-1)

rf_mod.fit(X_train, y_train)



print('Training Accuracy: ', rf_mod.score(X_train, y_train))

print('Validation Accuracy:', rf_mod.score(X_valid, y_valid))
rf_mod.predict_proba(X_train)
from sklearn.metrics import log_loss



log_loss(y_train, rf_mod.predict_proba(X_train))
log_loss(y_valid, rf_mod.predict_proba(X_valid))
%%time



xbg_mod = XGBClassifier()

xbg_mod.fit(X_train, y_train)



xbg_mod.score(X_train, y_train)
xbg_mod.score(X_valid, y_valid)
log_loss(y_train, xbg_mod.predict_proba(X_train))
log_loss(y_valid, xbg_mod.predict_proba(X_valid))
%%time

xgd_pipe = Pipeline(

    steps = [

        ('classifier', XGBClassifier(learning_rate=0.3, max_depth=6, alpha=1, n_estimators=50, subsample=0.5))

    ]

)



xgd_param_grid = {

    'classifier__learning_rate' : [0.1, 0.3, 0.5, 0.7, 0.9],

    'classifier__alpha' : [0, 1, 10, 100]

    

}





np.random.seed(1)

xgd_grid_search = GridSearchCV(xgd_pipe, xgd_param_grid, cv=5, refit='True')

xgd_grid_search.fit(X_train, y_train)



print(xgd_grid_search.best_score_)

print(xgd_grid_search.best_params_)
test_iterator = pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize=5)

test_top = next(test_iterator)

test_top
print(xbg_mod.predict_proba(test_top.iloc[:,5:]))
%%time



cs = 1000000

i = 0



for test in pd.read_csv('../input/reducing-commercial-aviation-fatalities/test.csv', chunksize=cs):

    

    print('--Iteration',i, 'is started')

    

    test_pred = xbg_mod.predict_proba(test.iloc[:,5:])

    

    partial_submission = pd.DataFrame({

        'id':test.id,

        'A':test_pred[:,0],

        'B':test_pred[:,1],

        'C':test_pred[:,2],

        'D':test_pred[:,3]

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