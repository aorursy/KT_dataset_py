import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# to ignore warnings

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/preprocessed-titanic-dataset/train.csv")

test = pd.read_csv("/kaggle/input/preprocessed-titanic-dataset/test.csv")
df.head()
test.head()
from sklearn.model_selection import train_test_split

X = df.drop('Survived',axis=1)

y = df['Survived']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scal = scaler.fit_transform(X_train)

X_test_scal = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
model_log = LogisticRegression()

model_log.fit(X_train_scal,y_train)

pred = model_log.predict(X_test_scal)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
param_grid = {

    'penalty':['none', 'l1', 'l2', 'elasticnet'],

    'C':  [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,

    'solver':['ewton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

    'max_iter':[100,300,900],

}



grid = GridSearchCV( LogisticRegression(),param_grid=param_grid,n_jobs=-1)

grid.fit(X_train_scal,y_train)

pred = grid.predict(X_test_scal)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
model_svc = SVC(probability=True)

model_svc.fit(X_train_scal,y_train)

pred = model_svc.predict(X_test_scal)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train_scal,y_train)

    pred_i = knn.predict(X_test_scal)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
model_knn =  KNeighborsClassifier(n_neighbors=15)

model_knn.fit(X_train_scal,y_train)

pred = model_knn.predict(X_test_scal)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
model_dt = DecisionTreeClassifier()

model_dt.fit(X_train,y_train)

pred = model_dt.predict(X_test)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
from sklearn.tree import plot_tree

plt.figure(figsize=(100,100))

plot_tree(model_dt,feature_names=X_train.columns.values,  

                   class_names=["0","1"],

                   filled=True);
param_grid = {

    'criterion': ["gini", "entropy"],

    'max_depth' : [5, 8, 15, 25,None],

    'min_samples_split' : [2, 5, 10, 15],

    'max_features' :["auto", "sqrt", "log2"],

    'min_samples_leaf' : [1, 2, 5] 

}



grid = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,verbose=1,n_jobs=-1)

grid.fit(X_train,y_train)

pred = grid.predict(X_test)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
grid.best_estimator_
model_random = RandomForestClassifier(200,n_jobs=-1)

model_random.fit(X_train,y_train)

pred = model_random.predict(X_test)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
param_grid = {

    'criterion': ["gini", "entropy"],

    'max_depth' : [5, 8, 15, 25,None],

    'min_samples_split' : [2, 5, 10, 15],

    'bootstrap' : [True,False],

    'max_features' :["auto", "sqrt", "log2"],

    'min_samples_leaf' : [1, 2, 5] 

}



grid = GridSearchCV(RandomForestClassifier(200,random_state=1),param_grid=param_grid,verbose=1,n_jobs=-1)

grid.fit(X_train,y_train)

pred = grid.predict(X_test)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
model_xgb = xgb.XGBClassifier()

model_xgb.fit(X_train,y_train)

pred = model_xgb.predict(X_test)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
params = {'objective':"binary:logistic",

          'base_score':0.005,

          'eval_metric':'error',

          'n_jobs':-1

}

trainm = xgb.DMatrix(X_train,y_train)

validm = xgb.DMatrix(X_test,y_test)
from sklearn.metrics import accuracy_score

def accuracy(y_prob, dtrain):

    y_true = dtrain.get_label()

    best_mcc = accuracy_score(y_true, (y_prob>0.5)*1)

    return 'accuracy_score', best_mcc
watchlist = [(trainm, 'train'), (validm, 'val')]

xgb_model = xgb.train(params, trainm,

                num_boost_round=100,

                evals=watchlist,

                early_stopping_rounds=20,

                feval=accuracy,

                maximize=True

                )
xgb_model = xgb.train(params, trainm,

                num_boost_round=7,

                feval=accuracy

                )
pred = xgb_model.predict(validm)

print(classification_report(y_test,(pred>0.5)*1))

confusion_matrix(y_test,(pred>0.5)*1)
#feature importance graph of XGboost

xgb.plot_importance(model_xgb)
np.set_printoptions(suppress=True)

xgb_pred = xgb_model.predict(validm)

knn_pred = model_knn.predict_proba(X_test_scal)[:,0]

svc_pred = model_svc.predict_proba(X_test_scal)[:,0]
log_pred = model_log.predict_proba(X_test_scal)[:,0]

dt_pred = model_dt.predict_proba(X_test)[:,0]

random_pred = model_random.predict_proba(X_test)[:,0]
combined_pred = 1*((knn_pred + svc_pred+ log_pred)/3 < 0.5)



print(classification_report(y_test,combined_pred))

confusion_matrix(y_test,combined_pred)
combined_pred = 1*((knn_pred + svc_pred+ log_pred+ xgb_pred )/4 < 0.5)



print(classification_report(y_test,combined_pred))

confusion_matrix(y_test,combined_pred)
from sklearn.ensemble import VotingClassifier

vote = VotingClassifier(estimators=[('svc',SVC()),

                                    ('knn',KNeighborsClassifier(n_neighbors=15)),

                                    ('log',LogisticRegression()),

                                   ])

vote.fit(X_train_scal,y_train)

pred = vote.predict(X_test_scal)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
from sklearn.ensemble import VotingClassifier

vote = VotingClassifier(estimators=[('svc',SVC()),

                                    ('knn',KNeighborsClassifier(n_neighbors=15)),

                                    ('log',LogisticRegression()),

                                    ('Xgboost',xgb.XGBClassifier()),

                                    #('dtree',DecisionTreeClassifier(criterion='entropy', max_depth=5, max_features='sqrt',min_samples_leaf=2, min_samples_split=10)),

                                    ('randomf',RandomForestClassifier(200,n_jobs=-1))

                                   ])

vote.fit(X_train_scal,y_train)

pred = vote.predict(X_test_scal)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=15),n_estimators=7,n_jobs=-1,random_state=4)

bag.fit(X_train_scal,y_train)

pred = bag.predict(X_test_scal)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=200,n_jobs=-1),n_estimators=20)

boost.fit(X_train,y_train)

pred = boost.predict(X_test)

print(classification_report(y_test,pred))

confusion_matrix(y_test,pred)
scaler = StandardScaler()

scaler.fit(X)

test_scal = scaler.transform(test.drop(['PassengerId'],axis=1))
test['Survived'] = bag.predict(test_scal)
test[['PassengerId','Survived']].to_csv("submission.csv",index=False)