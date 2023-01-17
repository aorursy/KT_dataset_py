# import libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict,cross_val_score, KFold,StratifiedKFold
# load data
train_data = pd.read_csv("../input/train.csv", header=0)
test_data = pd.read_csv("../input/test.csv", header=0)
sub = pd.read_csv("../input/sample_submission.csv", header=0)

train_data = train_data.drop(['ID'], axis=1)
test_data = test_data.drop(['ID'], axis=1)
temp = pd.get_dummies(train_data['resting_electrocardiographic_results'],prefix='resting_electrocardiographic_results')

for col in temp.columns:
    train_data[col] = temp[col]
    
train_data = train_data.drop(['resting_electrocardiographic_results'], axis=1)

temp = pd.get_dummies(test_data['resting_electrocardiographic_results'],prefix='resting_electrocardiographic_results')

for col in temp.columns:
    test_data[col] = temp[col]
    
test_data = test_data.drop(['resting_electrocardiographic_results'], axis=1)
temp = pd.get_dummies(train_data['thal'],prefix='thal')

for col in temp.columns:
    train_data[col] = temp[col]
    
train_data = train_data.drop(['thal'], axis=1)

temp = pd.get_dummies(test_data['thal'],prefix='thal')

for col in temp.columns:
    test_data[col] = temp[col]
    
test_data = test_data.drop(['thal'], axis=1)
temp = pd.get_dummies(train_data['number_of_major_vessels'],prefix='number_of_major_vessels')

for col in temp.columns:
    train_data[col] = temp[col]
    
#train_data = train_data.drop(['number_of_major_vessels'], axis=1)

temp = pd.get_dummies(test_data['number_of_major_vessels'],prefix='number_of_major_vessels')

for col in temp.columns:
    test_data[col] = temp[col]
    
#test_data = test_data.drop(['number_of_major_vessels'], axis=1)
chest_bin = []

for v in train_data.chest.values:
    if v>3.5:
        chest_bin.append(4)
    elif v > 3:
        chest_bin.append(3.5)
    elif v > 2.5:
        chest_bin.append(3)
    elif v > 2:
        chest_bin.append(2.5)
    elif v > 1.5:
        chest_bin.append(2)
    elif v > 1:
        chest_bin.append(1.5)
    elif v > 0.5:
        chest_bin.append(1)
    elif v > 0:
        chest_bin.append(0.5)
    else:
        chest_bin.append(0)

train_data['chest_bin'] = chest_bin

chest_bin = []

for v in test_data.chest.values:
    if v>3.5:
        chest_bin.append(4)
    elif v > 3:
        chest_bin.append(3.5)
    elif v > 2.5:
        chest_bin.append(3)
    elif v > 2:
        chest_bin.append(2.5)
    elif v > 1.5:
        chest_bin.append(2)
    elif v > 1:
        chest_bin.append(1.5)
    elif v > 0.5:
        chest_bin.append(1)
    elif v > 0:
        chest_bin.append(0.5)
    else:
        chest_bin.append(0)

test_data['chest_bin'] = chest_bin
temp = pd.get_dummies(train_data['chest_bin'],prefix='chest_bin')

for col in temp.columns:
    train_data[col] = temp[col]
    
train_data = train_data.drop(['chest_bin'], axis=1)
train_data = train_data.drop(['chest'], axis=1)

temp = pd.get_dummies(test_data['chest_bin'],prefix='chest_bin')

for col in temp.columns:
    test_data[col] = temp[col]
    
test_data = test_data.drop(['chest_bin'], axis=1)
test_data = test_data.drop(['chest'], axis=1)
train_data.columns
plt.figure(figsize=(18,15))
sns.heatmap(train_data.corr())
# funtion to get accuracy
def get_score(y_temp_l):
    y_pred_l = []
    for i in y_temp_l:
        if i > 0.5:
            y_pred_l.append(1)
        else:
            y_pred_l.append(0)
    print(accuracy_score(y_pred_l,test_y))
feature_col = [col for col in train_data.columns if col != 'class']
X_train, X_test, train_y, test_y = train_test_split(train_data[feature_col],train_data['class'].values,test_size = 0.2, random_state=42)
%%time
# Fitting a simple Logistic Regression 
clf_log = LogisticRegression(C=1.0)
clf_log.fit(X_train, train_y)
predictions = clf_log.predict_proba(X_test)
predictions = [i[1] for i in predictions]
predictions_log = predictions
get_score(predictions)
#%%time
# Fitting a svm
#clf = SVC(C=1.0, probability=True)
#clf.fit(X_train, train_y)
#predictions = clf.predict_proba(X_test)
#predictions = [i[1] for i in predictions]
#predictions_sv = predictions
#get_score(predictions)
%%time
# Fitting a random forest
clf_r = RandomForestClassifier(n_estimators=200, max_depth=7,random_state=0)
clf_r.fit(X_train, train_y)
predictions = clf_r.predict_proba(X_test)
predictions = [i[1] for i in predictions]
predictions_r = predictions
get_score(predictions)
%%time
# Fitting xgboost
clf_x = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf_x.fit(X_train, train_y)
predictions = clf_x.predict_proba(X_test)
predictions = [i[1] for i in predictions]
predictions_x = predictions
get_score(predictions)
%%time
# Fitting lightgbm
clf_l = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective= 'binary',
          nthread= 4, # Updated from nthread
          metric = 'binary_error',
         seed  = 47,
        depth =  5)
clf_l.fit(X_train, train_y)
predictions = clf_l.predict_proba(X_test)
predictions = [i[1] for i in predictions]
predictions_l = predictions
get_score(predictions)
%%time
# Fitting catboost
clf_c = CatBoostClassifier(iterations=500, learning_rate=0.07, verbose=False, 
                           depth =  5,loss_function='Logloss', thread_count = 4,
                           eval_metric='Accuracy')

clf_c.fit(X_train, train_y)
predictions = clf_c.predict_proba(X_test)
predictions = [i[1] for i in predictions]
predictions_c = predictions
get_score(predictions)
# fitting neural network
# scale the data before any neural net:
scl = preprocessing.StandardScaler()
xtrain_scl = scl.fit_transform(X_train)
xvalid_scl = scl.transform(X_test)

model = Sequential()

model.add(Dense(29, input_dim=29, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(xtrain_scl, y=train_y, batch_size=64, 
          epochs=3, verbose=1, 
          validation_data=(xvalid_scl, test_y))

predictions = model.predict_proba(xvalid_scl)
predictions = [i for i in predictions]
predictions_n = predictions
get_score(predictions)
kfold = KFold(n_splits=5, random_state=7)
# fitting lightgbm
results = cross_val_score(lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective= 'binary',
          nthread= 4, # Updated from nthread
          metric = 'binary_error',
         seed  = 47), train_data[feature_col],train_data['class'].values, cv=kfold)

print(results)
print(np.mean(results))
print(np.std(results))
# fitting xgboost
results = cross_val_score(xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
                          , train_data[feature_col],train_data['class'].values, cv=kfold)

print(results)
print(np.mean(results))
print(np.std(results))
# fitting catboost
results = cross_val_score(CatBoostClassifier(iterations=500, learning_rate=0.1, verbose=False, depth =  8,loss_function='Logloss')
                          , train_data[feature_col],train_data['class'].values, cv=kfold)

print(results)
print(np.mean(results))
print(np.std(results))
predictions = (np.array(predictions_x) + np.array(predictions_l))/2
get_score(predictions)
predictions_x = clf_x.predict_proba(test_data)
predictions_x = [i[1] for i in predictions_x]
predictions_l = clf_l.predict_proba(test_data)
predictions_l = [i[1] for i in predictions_l]
predictions_c = clf_c.predict_proba(test_data)
predictions_c = [i[1] for i in predictions_c]
predictions = (np.array(predictions_x) + np.array(predictions_l) + np.array(predictions_c))/3

p = []

for i in predictions:
    if i > 0.5:
        p.append(1)
    else:
        p.append(0)
        
sub['class'] = p
sub.to_csv("submission_1.csv",index=False)
p = []

for i in predictions_c:
    if i > 0.5:
        p.append(1)
    else:
        p.append(0)
        
sub['class'] = p
sub.to_csv("submission_2.csv",index=False)
