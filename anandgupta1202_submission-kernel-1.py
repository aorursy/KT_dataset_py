import pandas as pd

import numpy as np

import os

import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split



from sklearn.dummy import DummyClassifier



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.linear_model import LogisticRegression



from xgboost.sklearn import XGBClassifier



from imblearn.combine import SMOTEENN

from imblearn.pipeline import Pipeline



from keras.models import Sequential

from keras.layers import Dense



from sklearn import metrics



import matplotlib.pyplot as plt

import seaborn as sns
#creditcard = pd.read_csv("creditcard.csv")

creditcard = pd.read_csv("../input/creditcard.csv")
creditcard.info()
creditcard.describe()
sns.distplot(creditcard["Time"].astype(float))
sns.countplot(data = creditcard, x = 'Class')
creditcard['Class'].unique()
len(creditcard[creditcard['Class']==1])
len(creditcard[creditcard['Class']==0])
492/(284315+492)*100
amount = creditcard[creditcard['Class']==1]['Amount']
sns.distplot(amount)
creditcard.hist(figsize=(30,30))
# RobustScaler is less prone to outliers.

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html



robust = RobustScaler()



creditcard['scaled_amount'] = robust.fit_transform(creditcard['Amount'].values.reshape(-1,1))

creditcard['scaled_time'] = robust.fit_transform(creditcard['Time'].values.reshape(-1,1))



creditcard.drop(['Time','Amount'], axis=1, inplace=True)
creditcard.info()
X = creditcard.drop('Class', axis=1)

y = creditcard['Class']
X.head()
y.head()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)



"""sss1 = StratifiedShuffleSplit(n_splits=5, random_state=10, test_size=0.2)

rest, test = sss1.split(X, y)

print("Train:", rest, "Test:", test)

#X_test, y_test = X.iloc[test], y.iloc[test]"""
sss = StratifiedShuffleSplit(n_splits=3, random_state=10, test_size=0.2)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



def plot_roc(y_test, y_pred, label):

    log_fpr, log_tpr, log_thresold = roc_curve(y_test, y_pred)

    plt.plot(log_fpr, log_tpr, label=label+'{:.4f}'.format(roc_auc_score(y_test, y_pred)))



    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.legend()
clf = DummyClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

plot_roc(y_test, y_pred, 'Dummy AUC ')
%%time

clf = LogisticRegression(random_state=10)

param_grid={'C': [0.01, 0.1, 1, 10, 100]}

scoring = 'f1_micro'



GS = GridSearchCV(clf, param_grid, scoring=scoring, cv=sss.split(x_train, y_train), n_jobs=-1)

LR_GS = GS.fit(x_train, y_train)





print(LR_GS.best_score_)

y_pred = LR_GS.best_estimator_.predict(x_test)
LR_GS.best_estimator_
plot_roc(y_test, y_pred, 'Logistic Regression AUC ')
%%time

clf = LogisticRegression(random_state=10, penalty='l1')

param_grid={'C': [0.01, 0.1, 1, 10, 100],

            'class_weight':[{0:.1,1:.9}, {0:.0017,1:0.9983}, {0:.3,1:.7}]}

scoring = 'roc_auc'



RS = RandomizedSearchCV(clf, param_grid, scoring=scoring, cv=sss.split(x_train, y_train), n_jobs=-1)

LR_RS = RS.fit(x_train, y_train)



print(LR_RS.best_score_)

y_pred = LR_RS.best_estimator_.predict(x_test)
LR_RS.best_estimator_
plot_roc(y_test, y_pred, 'Logistic Regression AUC ')
params = {

        'min_child_weight': [0.5, 1, 2],

        'gamma': [1, 1.5, 2, 5],

        'subsample': [0.5, 0.8, 1.0],

        'max_depth': [3, 5, 7],

        'scale_pos_weight':[10, 100, 300]

        }



xgb = XGBClassifier(learning_rate=0.02, n_estimators=500, objective='binary:logistic',

                    verbosity=2, nthread=-1)



#disable_default_eval_metric=1, eval_metric = 'auc',



rs_xgb = RandomizedSearchCV(xgb, param_distributions=params, 

                                   scoring='roc_auc', cv=3,

                                   verbose=2, n_jobs=-1,

                                   random_state=10)



#sss.split(x_train, y_train)
%%time

rs_xgb.fit(x_train, y_train)
rs_xgb.grid_scores_
print(rs_xgb.best_score_)

print(rs_xgb.best_params_)

y_pred = rs_xgb.best_estimator_.predict(x_test)
plot_roc(y_test, y_pred, 'XGBoost AUC ')
%%time

sm = SMOTEENN(random_state=10)

x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
print(len(x_train_res))

print(len(y_train_res))
np.unique(y_train_res)
x_train_res.shape
model = Sequential()

model.add(Dense(32, input_dim=30, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
%%time

model.fit(x_train_res, y_train_res, epochs=20, batch_size=5000)
y_pred = model.predict(x_test)
plot_roc(y_test, y_pred, 'Keras model_1 AUC ')
my_submission = pd.DataFrame(y_pred)

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
clf = LogisticRegression(random_state=10)

param_grid={'C': [0.01, 0.1, 1, 10, 100]}

scoring = 'f1_micro'



GS = GridSearchCV(clf, param_grid, scoring=scoring, cv=3, n_jobs=-1)

LR_GS_new = GS.fit(x_train_res, y_train_res)



print(LR_GS_new.best_score_)

y_pred = LR_GS_new.best_estimator_.predict(x_test)
plot_roc(y_test, y_pred, 'Log_Reg (sampled data) AUC ')