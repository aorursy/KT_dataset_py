# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from time import time
import datetime
from math import sqrt
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
from xgboost import plot_importance
import lightgbm as lgb
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop

t0 = time()
data = pd.read_csv('../input/creditcard.csv', sep=',', decimal='.')
data.info()
data.isnull().sum()
numericFeatures = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
scale_num = StandardScaler()
scale_num.fit(data[numericFeatures])
data[numericFeatures] = scale_num.transform(data[numericFeatures])
data.head()

plt.bar([0,1], height = data.Class.value_counts(), tick_label = ['No fraud','Fraud'])
y_data = data['Class']
X_data = data.drop('Class', 1)

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.33, random_state=29)

# pickle.dump((X_train, X_val, y_train, y_val), open(obj_save_path+'train_val_df.p', 'wb'))
#X_train, X_val, y_train, y_val = pickle.load(open(obj_save_path+'train_val_df.p', 'rb'))
print('Ready to start ML part !')
print('ML part. I : starting Random Forest !')

model_rf = RandomForestClassifier(n_estimators=50,
                                  max_depth=20,
                                  min_samples_split=5,
                                  min_samples_leaf=20,
                                  bootstrap=True, oob_score=True, criterion='gini',
                                  random_state=321, n_jobs=4, verbose=1)

model_rf.fit(X_train, y_train)

# pickle.dump(model_rf, open(obj_save_path+'model_rf.p', 'wb'))
#model_rf = pickle.load(open(obj_save_path+'model_rf.p', 'rb'))
def plot_imp_rf(model_rf, X):
    importances = model_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    names = X.columns[indices]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(str(f+1)+'. feature '+str(names[f])+' ('+str(importances[indices[f]])+')')
    # Plot the feature importances of the forest
    plt.figure(figsize=(10, 5))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), names, rotation=80)
    plt.xlim([-1, X.shape[1]])
    plt.show()

plot_imp_rf(model_rf, X_train)
# oob_error = 1 - model_rf.oob_score_
def verif_valid(model, X_val, y_val):
    if type(model) == Sequential:
        X_val = np.array(X_val)
    reality = y_val
    predictions = model.predict(X_val)
    if type(model) == lgb.basic.Booster:
        for i in range(len(predictions)):
            if predictions[i] >= 0.5:  # threshold = 0.5
               predictions[i] = 1
            else:
               predictions[i] = 0
    if len(predictions.shape) == 2:
        predictions = predictions[:, 0]
    print('Matrice de confusion :')
    print(confusion_matrix(reality, predictions))
    print('Métriques de précision associées :')
    print(classification_report(reality, predictions))
    print('Score AUC :')
    print(roc_auc_score(reality, predictions))

verif_valid(model_rf, X_val, y_val)

print('ML part. I : Random Forest, done !')
print('ML part. II : starting Gradient Boosting !')

model_gradb = GradientBoostingClassifier(loss='deviance',
                                        learning_rate=0.2,
                                        n_estimators=100,
                                        subsample=0.9,
                                        #min_samples_leaf=10,
                                        max_depth=6,
                                        random_state=321, verbose=0)

model_gradb.fit(X_train, y_train)

# pickle.dump(model_gradb, open(obj_save_path+'model_gradb.p', 'wb'))
#model_gradb = pickle.load(open(obj_save_path+'model_gradb.p', 'rb'))

verif_valid(model_gradb, X_val, y_val)

print('ML part. II : Gradient Boosting, done !')
print('ML part. III : starting XGBoost !')

model_xgb = xgb.XGBClassifier(base_score=0.5,
                              subsample=0.8,
                              max_delta_step=2,
                              max_depth=7,
                              min_child_weight=3,
                              learning_rate=0.1,
                              n_estimators=580,
                              objective='binary:logistic',
                              #booster='gbtree',
                              colsample_bytree=0.85,
                              gamma=0,
                              reg_alpha=0,
                              reg_lambda=1,
                              scale_pos_weight=1,
                              seed=321, silent=0)

model_xgb.fit(X_train, y_train)

print(model_xgb)

# pickle.dump(model_xgb, open(obj_save_path+'model_xgb.p', 'wb'))
#model_xgb = pickle.load(open(obj_save_path+'model_xgb.p', 'rb'))

plot_importance(model_xgb)
plt.show()

verif_valid(model_xgb, X_val, y_val)

print('ML part. III : XGBoost, done !')
print('ML part. IV : starting LightGBM !')

d_train = lgb.Dataset(X_train, label=y_train)
params = {}
params['learning_rate'] = 0.1
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
model_lgbm = lgb.train(params, d_train, 500)

# pickle.dump(model_lgbm, open(obj_save_path+'model_lgbm.p', 'wb'))
#model_lgbm = pickle.load(open(obj_save_path+'model_lgbm.p', 'rb'))

verif_valid(model_lgbm, X_val, y_val)

print('ML part. IV : LightGBM, done !')
print('ML part. V : starting Adaboost !')

model_adab = AdaBoostClassifier(#base_estimator=RandomForestClassifier(),
                               n_estimators=300,
                               learning_rate=0.28,
                               #loss='linear',
                               random_state=321)

model_adab.fit(X_train, y_train)

# pickle.dump(model_adab, open(obj_save_path+'model_adab.p', 'wb'))
#model_adab = pickle.load(open(obj_save_path+'model_adab.p', 'rb'))

verif_valid(model_adab, X_val, y_val)

print('ML part. V : Adaboost, done !')
