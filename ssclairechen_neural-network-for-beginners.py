import scipy

import pandas as pd

from scipy.io import arff

from sklearn.model_selection import train_test_split

from keras.layers import Dense

import numpy as np
DATA_DIR = '/kaggle/input/lish-moa/'
train = pd.read_csv(DATA_DIR + 'train_features.csv')

targets = pd.read_csv(DATA_DIR + 'train_targets_scored.csv')



test = pd.read_csv(DATA_DIR + 'test_features.csv')

sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')
X = train

y = targets

del X['sig_id']

del y['sig_id']
def preprocess(df):

    df.loc[:,'cp_type'] = df.loc[:,'cp_type'].map({'trt_cp':0,'ctl_vehicle':1})

    df.loc[:,'cp_dose'] = df.loc[:,'cp_dose'].map({'D1':0,'D2':1})
preprocess(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

y_train = y_train.astype(np.float64)

y_test = y_test.astype(np.float64)
import keras
model =  keras.Sequential()

def deep_model(feature_dim,label_dim):

    from keras.models import Sequential

    from keras.layers import Dense

    #model = Sequential()

    print("create model. feature_dim ={}, label_dim ={}".format(feature_dim, label_dim))

    model.add(Dense(600, activation='relu', input_dim=feature_dim))

    model.add(Dense(500, activation='relu'))

    model.add(Dense(400, activation='relu'))

    model.add(Dense(label_dim, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    return model
def train_deep(X_train,y_train,X_test,y_test):

    feature_dim = X_train.shape[1]

    label_dim = y_train.shape[1]

    model = deep_model(feature_dim,label_dim)

    model.summary()

    model.fit(X_train,y_train,batch_size=50, epochs=3,validation_data=(X_test,y_test))

    return model





model_ = train_deep(X_train,y_train,X_test,y_test)
preprocess(test)
test
del test['sig_id']
model_.predict(test)
pred = model_.predict(test)

pred = pd.DataFrame(pred)

control_mask = test['cp_type'] == 1

pred[control_mask] = 0

pred.columns = sub.columns[1:]

sub.iloc[:,1:] = pred
sub
sub.to_csv('submission.csv', index=False)