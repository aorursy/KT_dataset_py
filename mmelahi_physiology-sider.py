# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.decomposition import KernelPCA

from sklearn.preprocessing import StandardScaler

from sklearn.multiclass import OneVsRestClassifier

from sklearn.utils import class_weight

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.utils import class_weight

from sklearn.calibration import CalibratedClassifierCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score

from sklearn.model_selection import train_test_split

from skmultilearn.model_selection import iterative_train_test_split

import pickle

from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input

from keras.models import Sequential, Model

from keras import optimizers, regularizers, initializers

from keras.callbacks import ModelCheckpoint, Callback

from keras import backend as K

from keras.optimizers import Adam

import tensorflow as tf

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
NCA1 = 100

NCA2 = 100

DROPRATE = 0.2

EP = 500

BATCH_SIZE = 128

VAL_RATIO = 0.1

TEST_RATIO = 0.1
sider_df= pd.read_csv('../input/sider_descriptors/sider_df_revised.csv')

print(sider_df.shape)

sider_df.head()
sider_df.drop(['smiles'], axis=1, inplace=True)

print(sider_df.shape)

sider_df.head()
# Get indices of NaN

#inds = pd.isnull(tox21_df).any(1).nonzero()[0]
# Drop NaN from the dataframe

#tox21_df.dropna(inplace=True)

#print(tox21_df.shape)

#tox21_df.head()
sider_df = sider_df.fillna(0)

sider_df.head()
sider_descriptors_df= pd.read_csv('../input/sider_descriptors/sider_descriptors_df.csv',low_memory=False)

print(sider_descriptors_df.shape)

sider_descriptors_df.head()
# function to coerce all data types to numeric



def coerce_to_numeric(df, column_list):

    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
coerce_to_numeric(sider_descriptors_df, sider_descriptors_df.columns)

sider_descriptors_df.head()
sider_descriptors_df = sider_descriptors_df.fillna(0)

sider_descriptors_df.head()
#tox21_descriptors_df.drop(tox21_descriptors_df.index[inds],inplace=True)

#tox21_descriptors_df.shape
sider_scaler1 = StandardScaler()

sider_scaler1.fit(sider_descriptors_df.values)

sider_descriptors_df = pd.DataFrame(sider_scaler1.transform(sider_descriptors_df.values),

                                    columns=sider_descriptors_df.columns)
nca = NCA1

cn = ['col'+str(x) for x in range(nca)]
sider_transformer1 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)

sider_transformer1.fit(sider_descriptors_df.values)

sider_descriptors_df = pd.DataFrame(sider_transformer1.transform(sider_descriptors_df.values),

                                    columns=cn)

print(sider_descriptors_df.shape)

sider_descriptors_df.head()
X_train, y_train, X_test, y_test = iterative_train_test_split(sider_descriptors_df.values,

                                                              sider_df.values, 

                                                              test_size=TEST_RATIO)
X_train, y_train, X_valid, y_valid = iterative_train_test_split(X_train, y_train, 

                                                                test_size=VAL_RATIO)
def Find_Optimal_Cutoff(target, predicted):

    """ Find the optimal probability cutoff point for a classification model related to event rate

    Parameters

    ----------

    target : Matrix with dependent or target data, where rows are observations



    predicted : Matrix with predicted data, where rows are observations



    Returns

    -------     

    list type, with optimal cutoff value



    """

    fpr, tpr, threshold = roc_curve(target, predicted)

    i = np.arange(len(tpr)) 

    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})

    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]



    return list(roc_t['threshold']) 
def Find_Optimal_threshold(target, predicted):

    

    rng = np.arange(0.0, 0.99, 0.001)

    f1s = np.zeros((rng.shape[0],predicted.shape[1]))

    for i in range(0,predicted.shape[1]):

        for j,t in enumerate(rng):

            p = np.array((predicted[:,i])>t, dtype=np.int8)

            scoref1 = f1_score(target[:,i], p, average='binary')

            f1s[j,i] = scoref1

            

    threshold = np.empty(predicted.shape[1])

    for i in range(predicted.shape[1]):

        threshold[i] = rng[int(np.where(f1s[:,i] == np.max(f1s[:,i]))[0][0])]

        

    return threshold
def ROC_AUC(target, predicted):

    score = np.zeros(np.shape(target)[-1])

    for i in range(len(score)):

        try:

            score[i] = roc_auc_score(target[:,i],predicted[:,i])

        except ValueError:

            score[i] = 0

    return np.mean(score)
parameters = {'estimator__class_weight':['balanced'],

              'estimator__kernel':['rbf'], 

              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['auto','scale']}

sider_svc = GridSearchCV(OneVsRestClassifier(SVC(probability=True,

                                                 random_state=23)), 

                         parameters, cv=2, scoring='roc_auc',n_jobs=-1)
result = sider_svc.fit(X_train, y_train)
pred = sider_svc.predict_proba(X_valid)

pred_svc_t = np.copy(pred)

ROC_AUC(y_valid,pred)
threshold = Find_Optimal_threshold(y_valid, pred)

print(threshold)
pred = sider_svc.predict(X_test)

f1_score(y_test,pred,average='macro')
pred = sider_svc.predict_proba(X_test)

pred_svc = np.copy(pred)

svc_roc = ROC_AUC(y_test,pred)

print(svc_roc)
pred[pred<=threshold] = 0

pred[pred>threshold] = 1

svc_score = f1_score(y_test,pred,average='macro')

print(svc_score)
y = X_test[23,:].reshape(1, -1)

result = sider_svc.predict(y)

prob = sider_svc.predict_proba(y)

pred = np.copy(prob)

pred[pred<=threshold] = 0

pred[pred>threshold] = 1

print(result)

print(prob)

print(pred)
sider_model = Sequential()

sider_model.add(Dense(128, input_dim=sider_descriptors_df.shape[1], 

                      kernel_initializer='he_uniform'))

sider_model.add(BatchNormalization())

sider_model.add(Activation('tanh'))

sider_model.add(Dropout(rate=DROPRATE))

sider_model.add(Dense(64,kernel_initializer='he_uniform'))

sider_model.add(BatchNormalization())

sider_model.add(Activation('tanh'))

sider_model.add(Dropout(rate=DROPRATE))

sider_model.add(Dense(32,kernel_initializer='he_uniform'))

sider_model.add(BatchNormalization())

sider_model.add(Activation('tanh'))

sider_model.add(Dropout(rate=DROPRATE))

sider_model.add(Dense(sider_df.shape[1],kernel_initializer='he_uniform',activation='sigmoid'))
sider_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
checkpoint = ModelCheckpoint('sider_model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
hist = sider_model.fit(X_train, y_train, 

                       validation_data=(X_valid,y_valid),epochs=EP, batch_size=BATCH_SIZE, 

                       callbacks=[checkpoint])
plt.ylim(0., 1.)

plt.plot(hist.epoch, hist.history["loss"], label="Train loss")

plt.plot(hist.epoch, hist.history["val_loss"], label="Valid loss")
sider_model.load_weights('sider_model.h5')
pred = sider_model.predict(X_valid)

pred_nn_t = np.copy(pred)
threshold = Find_Optimal_threshold(y_valid, pred)

#print(threshold)
pred = sider_model.predict(X_test)

pred_nn = np.copy(pred)

nn_roc = ROC_AUC(y_test,pred)

print(nn_roc)
pred[pred<=threshold] = 0

pred[pred>threshold] = 1

nn_score = f1_score(y_test,pred,average='macro')

print(nn_score)
prob = sider_model.predict(y)

prob[prob<=threshold] = 0

prob[prob>threshold] = 1

print(prob[0])
inp = sider_model.input

out = sider_model.layers[-2].output

sider_model_gb = Model(inp, out)
X_train = sider_model_gb.predict(X_train)

X_test = sider_model_gb.predict(X_test)

X_valid = sider_model_gb.predict(X_valid)
data = np.concatenate((X_train,X_test,X_valid),axis=0)
sider_scaler2 = StandardScaler()

sider_scaler2.fit(data)

X_train = sider_scaler2.transform(X_train)

X_test = sider_scaler2.transform(X_test)

X_valid = sider_scaler2.transform(X_valid)
data = np.concatenate((X_train,X_test,X_valid),axis=0)
nca = NCA2
sider_transformer2 = KernelPCA(n_components=nca, kernel='rbf', n_jobs=-1)

sider_transformer2.fit(data)

X_train = sider_transformer2.transform(X_train)

X_test = sider_transformer2.transform(X_test)

X_valid = sider_transformer2.transform(X_valid)
nca = X_train.shape[1]

parameters = {'estimator__class_weight':['balanced'],

              'estimator__kernel':['rbf'], 

              'estimator__C':[1,0.5,0.25], 'estimator__gamma':['scale','auto']}



sider_svc_gb = GridSearchCV(OneVsRestClassifier(SVC(probability=True,

                                                    random_state=23)), 

                            parameters, cv=2, scoring='roc_auc',n_jobs=-1)
result = sider_svc_gb.fit(X_train, y_train)
pred = sider_svc_gb.predict_proba(X_valid)

pred_svc_gb_t = np.copy(pred)

ROC_AUC(y_valid,pred)
threshold = Find_Optimal_threshold(y_valid, pred)

#print(threshold)
pred = sider_svc_gb.predict(X_test)

f1_score(y_test,pred,average='macro')
pred = sider_svc_gb.predict_proba(X_test)

pred_svc_gb = np.copy(pred)

svc_gb_roc = ROC_AUC(y_test,pred)

print(svc_gb_roc)
pred[pred<=threshold] = 0

pred[pred>threshold] = 1

svc_gb_score = f1_score(y_test,pred,average='macro')

print(svc_gb_score)
y = X_test[23,:].reshape(1, -1)

result = sider_svc_gb.predict(y)

prob = sider_svc_gb.predict_proba(y)

print(result)

print(prob)

prob[prob<=threshold] = 0

prob[prob>threshold] = 1

print(prob)
parameters = {'estimator__learning_rate':[0.05,0.1,0.15],'estimator__n_estimators':[75,100,125], 'estimator__max_depth':[3,5,7],

              'estimator__booster':['gbtree','dart'],'estimator__reg_alpha':[0.1,0.05],'estimator__reg_lambda':[0.5,1.]}



sider_xgb_gb = GridSearchCV(OneVsRestClassifier(XGBClassifier(random_state=32)), parameters, cv=2, scoring='roc_auc',n_jobs=-1)
result = sider_xgb_gb.fit(X_train, y_train)
pred = sider_xgb_gb.predict_proba(X_valid)

pred_xgb_gb_t = np.copy(pred)

ROC_AUC(y_valid,pred)
threshold = Find_Optimal_threshold(y_valid, pred)

#print(threshold)
pred = sider_xgb_gb.predict(X_test)

f1_score(y_test,pred,average='macro')
f1_score(y_test,pred,average=None)
pred = sider_xgb_gb.predict_proba(X_test)

pred_xgb_gb = np.copy(pred)

xgb_gb_roc = ROC_AUC(y_test,pred)

print(xgb_gb_roc)
pred[pred<=threshold] = 0

pred[pred>threshold] = 1

xgb_gb_score = f1_score(y_test,pred,average='macro')

print(xgb_gb_score)
result = sider_xgb_gb.predict(y)

prob = sider_xgb_gb.predict_proba(y)

print(result)

print(prob)

prob[prob<=threshold] = 0

prob[prob>threshold] = 1

print(prob)
pred = (pred_svc_t+pred_nn_t+pred_svc_gb_t+pred_xgb_gb_t)/4.
threshold = Find_Optimal_threshold(y_valid, pred)

print(threshold)
pred = (pred_svc+pred_nn+pred_svc_gb+pred_xgb_gb)/4.

ave_roc = ROC_AUC(y_test,pred)

print(ave_roc)

pred[pred<=threshold] = 0

pred[pred>threshold] = 1

ave_score = f1_score(y_test,pred,average='macro')

print(ave_score)
with open('sider_svc.pkl', 'wb') as fid:

    pickle.dump(sider_svc, fid)

with open('sider_transformer1.pkl', 'wb') as fid:

    pickle.dump(sider_transformer1, fid)

with open('sider_transformer2.pkl', 'wb') as fid:

    pickle.dump(sider_transformer2, fid)

with open('sider_scaler1.pkl', 'wb') as fid:

    pickle.dump(sider_scaler1, fid)

with open('sider_scaler2.pkl', 'wb') as fid:

    pickle.dump(sider_scaler2, fid)

with open('sider_svc_gb.pkl', 'wb') as fid:

    pickle.dump(sider_svc_gb, fid)

with open('sider_xgb_gb.pkl', 'wb') as fid:

    pickle.dump(sider_xgb_gb, fid)
sns.set(style="whitegrid")

ax = sns.barplot(x=[svc_roc,nn_roc,svc_gb_roc,xgb_gb_roc,ave_roc],

                 y=['SVC','NN','SVC_GB','XGB_GB','ave'])

ax.set(xlim=(0.55, None))
sns.set(style="whitegrid")

ax = sns.barplot(x=[svc_score,nn_score,svc_gb_score,xgb_gb_score,ave_score],

                 y=['SVC','NN','SVC_GB','XGB_GB','ave'])

ax.set(xlim=(0.55, None))