import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb # lightgbm 부스팅 알고리즘 사용
from lightgbm import LGBMClassifier

from sklearn import metrics
from sklearn.metrics import roc_auc_score
redwine = pd.read_csv('../input/wine-changed/wine2.csv')

redwine['quality'] = [1 if i == 'high rank' else 0 for i in redwine['quality'] ]
redwine.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(redwine[ redwine.columns[:-1] ], redwine['quality'], 
                                                    test_size = 0.3, random_state = 2)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred2 = y_pred.round(0)
y_pred2 = y_pred2.astype(int)
print('정확도 :', metrics.accuracy_score(y_test, y_pred2))
print('AUC    :', roc_auc_score(y_pred2, y_test))
model = RandomForestClassifier(random_state = 0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred2 = y_pred.round(0)
y_pred2 = y_pred2.astype(int)
print('정확도 :', model.score(X_test, y_test) )
print('AUC    :', roc_auc_score(y_pred2, y_test))
model = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, random_state = 0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred2 = y_pred.round(0)
y_pred2 = y_pred2.astype(int)
print('정확도 :', model.score(X_test, y_test) )
print('AUC    :', roc_auc_score(y_pred2, y_test))
lgbm = LGBMClassifier(random_state = 0)

lgbm_clf = lgbm.fit(X_train,y_train) #train the model on 100 epocs

y_pred = lgbm_clf.predict(X_test)
y_pred2 = y_pred.round(0)
y_pred2 = y_pred2.astype(int)

print('정확도 :', metrics.accuracy_score(y_test, y_pred2))
print('AUC    :', roc_auc_score(y_pred2, y_test))
d_train=lgb.Dataset(X_train, label=y_train)

lgbm_param = {'objective':'binary',
              "metric" : "auc",
              'boosting_type': 'gbdt',
              'random_state':42,
              'learning_rate':0.075,
            }

clf = lgb.train(lgbm_param,d_train,100) #train the model on 100 epocs

y_pred = clf.predict(X_test)
y_pred2 = y_pred.round(0)
y_pred2 = y_pred2.astype(int)

print('정확도 :', metrics.accuracy_score(y_test, y_pred2))
print('AUC    :', roc_auc_score(y_pred2, y_test))