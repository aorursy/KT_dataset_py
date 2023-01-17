import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))



from sklearn import preprocessing

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/noshowappointments-kagglev2-may-2016.csv')

data.head()
estimators=['Gender','Age','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received']



data['Gender']=data['Gender'].map({'F':0,'M':1})

data['No-show']=data['No-show'].map({'No':0,'Yes':1})



X = data[estimators]

y = data['No-show']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=666)

X_train.head()
from sklearn.ensemble import RandomForestClassifier

    

clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=69).fit(X_train,y_train)
rf_score = clf.score(X_test,y_test)

print('Random Forest score: {:.3f}'.format(rf_score))

feature_importances = clf.feature_importances_

print('Feature importances:')

for i,feature in enumerate(X.columns):

    print("\t{}: {:.2f}".format(feature,feature_importances[i]))
X_train.drop(columns=['Gender','Alcoholism'])

X_test.drop(columns=['Gender','Alcoholism'])

print('dropped unnecessary columns')
from keras.layers import *

from keras.regularizers import l2

from keras.models import Model

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam

from keras.backend import clear_session
clear_session()



model_entry = Input(shape=(len(X_train.columns),))

x = Dense(666,activation='relu',bias_initializer='glorot_normal',bias_regularizer=l2(0.002),kernel_regularizer=l2(0.002))(model_entry)

x = Dense(69,activation='relu',bias_initializer='glorot_normal',bias_regularizer=l2(0.001),kernel_regularizer=l2(0.001))(model_entry)

x = BatchNormalization()(x)

x = Dense(1,activation='sigmoid')(x)



model = Model(model_entry,x)

model.compile(optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss',patience=3,verbose=1)

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[early_stop],batch_size=256,epochs=25,shuffle=True,verbose=0)
metrics = model.metrics_names

dnn_scores = model.evaluate(X_test,y_test,verbose=0)

print("DNN accuracy: {:.3f}".format(dnn_scores[1]))
from sklearn.ensemble import BaggingClassifier



baggie = BaggingClassifier().fit(X_train,y_train)



bag_score = baggie.score(X_test,y_test)

print("Bagging score: {:.3f}".format(bag_score))
import xgboost as xgb

from sklearn.datasets import dump_svmlight_file

from sklearn.metrics import precision_score



dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)

dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)

dtrain_svm = xgb.DMatrix('dtrain.svm')

dtest_svm = xgb.DMatrix('dtest.svm')



params = {

    'max_depth': 5,  # the maximum depth of each tree

    'eta': 0.3,  # the training step for each iteration

    'silent': 1,  # logging mode - quiet

    'objective': 'multi:softprob',  # error evaluation for multiclass training

    'num_class': 2}  # the number of classes that exist in this datset

num_rounds = 15



boosted = xgb.train(params, dtrain_svm, num_rounds)
boost_preds = boosted.predict(dtest_svm)

best_preds = np.asarray([np.argmax(line) for line in boost_preds])

precision = precision_score(y_test, best_preds, average='macro')

print("XGBoost precision: {:.3f}".format(precision))
print('Random Forest score: {:.3f}'.format(rf_score))

print("DNN accuracy: {:.3f}".format(dnn_scores[1]))

print("Bagging score: {:.3f}".format(bag_score))

print("XGBoost precision: {:.3f}".format(precision))