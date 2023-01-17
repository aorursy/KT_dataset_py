import numpy as np 
import pandas as pd
from sklearn import *
from sklearn.metrics import f1_score,make_scorer
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import xgboost as xgb
from catboost import Pool,CatBoostRegressor
import datetime
import gc
GROUP_BATCH_SIZE = 4000
WINDOWS = [10, 50]


BASE_PATH = '/kaggle/input/liverpool-ion-switching'
DATA_PATH = '/kaggle/input/data-without-drift'
RFC_DATA_PATH = '/kaggle/input/ion-shifted-rfc-proba'
MODELS_PATH = '/kaggle/input/ensemble-models'
%%time

def create_rolling_features(df):
    for window in WINDOWS:
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
        df["rolling_min_max_ratio_" + str(window)] = df["rolling_min_" + str(window)] / df["rolling_max_" + str(window)]
        df["rolling_min_max_diff_" + str(window)] = df["rolling_max_" + str(window)] - df["rolling_min_" + str(window)]

    df = df.replace([np.inf, -np.inf], np.nan)    
    df.fillna(0, inplace=True)
    return df


def create_features(df, batch_size):
    
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    for window in WINDOWS:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
        
    df['signal_2'] = df['signal'] ** 2
    return df   
## reading data
train = pd.read_csv(f'{DATA_PATH}/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
test  = pd.read_csv(f'{DATA_PATH}/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
sub  = pd.read_csv(f'{BASE_PATH}/sample_submission.csv', dtype={'time': np.float32})

# loading and adding shifted-rfc-proba features
y_train_proba = np.load(f"{RFC_DATA_PATH}/Y_train_proba.npy")
y_test_proba = np.load(f"{RFC_DATA_PATH}/Y_test_proba.npy")

for i in range(11):
    train[f"proba_{i}"] = y_train_proba[:, i]
    test[f"proba_{i}"] = y_test_proba[:, i]

    
train = create_rolling_features(train)
test = create_rolling_features(test)   
    
## normalizing features
train_mean = train.signal.mean()
train_std = train.signal.std()
train['signal'] = (train.signal - train_mean) / train_std
test['signal'] = (test.signal - train_mean) / train_std


print('Shape of train is ',train.shape)
print('Shape of test is ',test.shape)
## create features

batch_size = GROUP_BATCH_SIZE

train = create_features(train, batch_size)
test = create_features(test, batch_size)

cols_to_remove = ['time','signal','open_channels','batch','batch_index','batch_slices','batch_slices2', 'group']
cols = [c for c in train.columns if c not in cols_to_remove]
X_train = train[cols]
y = train['open_channels']
X_test = test[cols]

##from sklearn.model_selection import train_test_split
##X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state=101)
##converting to np arrays
X_train = X_train.values
y_train = y.values
X_test = X_test.values
del train
del test
gc.collect()
import keras
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Bidirectional
from keras.layers import Input
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Scaling and onehot encoding
from sklearn.preprocessing import MinMaxScaler
onh = OneHotEncoder(sparse=False)
sc = MinMaxScaler(feature_range = (0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
y_train = y_train.reshape(len(y_train),1)
y_train = onh.fit_transform(y_train)

print('Shape of X_train is ',X_train.shape)
print('Shape of y_train is ',y_train.shape)
print('Shape of X_test is ',X_test.shape)
##for converting input into 3D data
X_train= X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test= X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#build and compile model
def build_clf(optimizer):
    model = Sequential()
    model.add(Conv1D(128,16, strides=6, activation='relu', input_shape = (X_train.shape[1],X_train.shape[2])))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu',kernel_initializer='uniform'))
    model.add(Dense(128, activation='relu',kernel_initializer='uniform'))
    model.add(Dense(128, activation='relu',kernel_initializer='uniform'))
    model.add(Dense(units = 11, activation='softmax', kernel_initializer='uniform'))
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics =['accuracy'])
    model.summary()
    return model
#compile and fit model--96.25 rmsprop, 96.71--adadelta
model = build_clf('adam')
model.fit(X_train, y_train,epochs = 10, batch_size=256)                        
# Prediction and reversing One Hot Encoding
y_pred=model.predict(X_test)
y_pred =onh.inverse_transform(y_pred)
y_pred.max() #Should be 10
# making submission
sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
sub.iloc[:,1] = y_pred[:,0]
sub.to_csv('submission.csv',index=False,float_format='%.4f')
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model


##Tuning parameters to find best choice for model using Grid Search
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#scorer = make_scorer(f1_score, average = 'weighted')
#model = KerasClassifier(build_fn = build_clf)
#parameters = {'batch_size': [500,10000], 'epochs': [5, 200],'optimizer': ['adam', 'rmsprop','nadam','adadelta']}
#grid_search = GridSearchCV(estimator = model,param_grid = parameters,scoring = scorer,cv = 3, return_train_score= True)
#grid_search = grid_search.fit(X_train, y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_
#best_parameters

#Grid Search for multiclass classification failed