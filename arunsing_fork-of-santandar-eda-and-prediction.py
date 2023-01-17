# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# train_df.describe()
# test_df.describe()
# train_df.shape, test_df.shape
#list(train_df.columns)
def find_null(df):

    column_list = list(df.columns)

    for column in column_list:

        print (train_df[column].isnull().sum())
find_null(train_df)
train_df.head(10)
sns.countplot(train_df['target'])
train_df_1 = train_df.loc[train_df.target ==1]

train_df_0 = train_df.loc[train_df.target ==0]

print("Number of target value as 1 %d" %len(train_df_1))

print("Number of target value as 0 {}".format(len(train_df_0)))
# # let us try to find co-relation between features

# plt.figure(figsize=(30,30))

# corr = train_df.corr()

# sns.heatmap(corr)
# Lets work on creating a model, lets define X and Y

X_train = train_df.drop(['ID_code','target'], axis=1)

Y_train = train_df['target']

X_test = test_df.drop(['ID_code'],axis=1)

print(X_train.shape)

print(Y_train.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler



cv = list (StratifiedKFold(5,random_state=5756).split(X_train,Y_train))

scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)

# print(len(Y_train))

# lr = LogisticRegression()

# y_pred = cross_val_predict(lr, X_train_scaled, Y_train, cv=cv, method = 'predict_proba',verbose=2 )[:,1]

# #lr.fit(X_train_scaled, Y_train)

# print(len(y_pred))

# print(roc_auc_score(Y_train, y_pred))
# lr.fit(X_train_scaled,Y_train)

# #test_preds = lr.predict(X_test_scaled)

# preds=lr.predict(X_test_scaled)
from keras import layers

from keras import Sequential

from keras.layers import Dense, Flatten, Activation, Dropout, LeakyReLU

from keras.optimizers import Adam, Adadelta, RMSprop

from imblearn.over_sampling import SMOTE

from keras import regularizers
# print("Before OverSampling, counts of label '1': {}".format(sum(Y_train==1)))

# print("Before OverSampling, counts of label '0': {} \n".format(sum(Y_train==0)))



# sm = SMOTE(random_state=2)

# X_train_res, Y_train_res = sm.fit_sample(X_train_scaled, Y_train.ravel())



# print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

# print('After OverSampling, the shape of train_y: {} \n'.format(Y_train_res.shape))



# print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res==1)))

# print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res==0)))
from sklearn.model_selection import train_test_split



X_train_split,  X_test_split, Y_train_split, Y_test_split = train_test_split(X_train_scaled,Y_train, test_size = 0.20, random_state=143289)

print("Length of training data {}".format(len(X_train_split)))

print("Length of testing data {}".format(len(X_test_split)))
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(Y_train),

                                                 Y_train)
from keras.callbacks import EarlyStopping

from keras.layers.normalization import BatchNormalization

model = Sequential()

model.add(Dense(1024,input_dim=len(X_train.columns), activation='relu', kernel_initializer = 'he_uniform'))

model.add(BatchNormalization())

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform' ,kernel_regularizer=regularizers.l1(0.001)))

#model.add(Dropout(0.1))

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

#model.add(Dropout(0.02))

#model.add(Dense(256, activation='relu', kernel_initializer = 'he_uniform'))

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

#model.add(Dropout(0.1))

model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

#model.add(BatchNormalization())

#model.add(Dropout(0.025))

model.add(Dense(256, activation='relu', kernel_initializer = 'he_uniform'))

model.add(BatchNormalization())

#model.add(Dropout(0.05))

model.add(Dense(256, activation='relu', kernel_initializer = 'he_uniform'))

model.add(BatchNormalization())

#model.add(Dense(128, activation='relu', kernel_initializer = 'he_uniform',kernel_regularizer=regularizers.l1(0.0001)))

#model.add(Dropout(0.05))

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

# model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

# model.add(BatchNormalization())

# model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

# model.add(BatchNormalization())

# model.add(Dense(256, activation='relu', kernel_initializer = 'he_uniform'))

# model.add(Dense(256, activation='relu', kernel_initializer = 'he_uniform'))

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

# model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

# model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

model.add(Dense(256, activation='relu', kernel_initializer = 'he_uniform'))

model.add(BatchNormalization())

model.add(Dense(256, activation='relu', kernel_initializer = 'he_uniform'))

model.add(BatchNormalization())

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

#model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

# model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

# model.add(Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))

model.add(Dense(1,activation='sigmoid', kernel_initializer='he_uniform'))

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=.0001),metrics=['binary_accuracy'])

              

model.summary

model.fit(X_train_split, 

          Y_train_split, 

          batch_size=16, 

          epochs=10, 

          verbose=1, 

          class_weight=class_weights,

          callbacks=[EarlyStopping(monitor='binary_accuracy')],

          validation_data=(X_test_split, Y_test_split))





              



from sklearn.metrics import accuracy_score



preds=model.predict(X_test_split)

#preds = np.where(preds >= 0.5, 1, 0)

Test_Accuracy=roc_auc_score(Y_test_split, preds)

print(Test_Accuracy)
#from sklearn.metrics import roc_auc_score



train_predict = model.predict_proba(X_train_scaled)

train_roc = roc_auc_score(Y_train, train_predict)

print('Train AUC: {}'.format(train_roc))



val_predict = model.predict_proba(X_test_split)

val_roc = roc_auc_score(Y_test_split, val_predict )

print('Val AUC: {}'.format(val_roc))
print(preds[:,0])

preds=preds[:,0]

preds_final=model.predict(X_test_scaled)

#preds_final = np.where(preds_final >= 0.5, 1, 0)

preds_final=preds_final[:,0]
X_test_ID = test_df['ID_code']

submission = pd.DataFrame({ 'ID_code': X_test_ID,'target': preds_final})

submission.to_csv("submission_21.csv", index=False)