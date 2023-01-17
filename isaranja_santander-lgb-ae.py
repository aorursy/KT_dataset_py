import lightgbm as lgb



import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotting

import matplotlib

import matplotlib.pyplot as plt



# preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler # normalization



# keras packages

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten



from keras import optimizers



# model selection

from sklearn.model_selection import StratifiedKFold



# evaluation

from sklearn.metrics import confusion_matrix, accuracy_score , roc_auc_score



import tensorflow as tf
# focal loss 

def focal_loss(alpha=0.25,gamma=5.0):

    def focal_crossentropy(y_true, y_pred):

        bce = K.binary_crossentropy(y_true, y_pred)

        

        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())

        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

        alpha_t =  y_true*alpha + ((1-alpha)*(1-y_true))



        # compute the final loss and return

        return K.mean(alpha_t*K.pow((1-p_t), gamma)*bce, axis=-1)

    return focal_crossentropy



def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred,num_thresholds=10000)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")

submission = pd.read_csv("../input/sample_submission.csv")
train_df = train_df.assign(isTrain = True)

test_df = test_df.assign(isTrain=False)



full_df = pd.concat([train_df,test_df],sort=False)
scaler = MinMaxScaler()

X=scaler.fit_transform(full_df.loc[:,'var_0':'var_199'].values,range(-1,1))
x = X[full_df.isTrain]

y = full_df[full_df.isTrain].target

x_test = X[~full_df.isTrain]
input_data = Input(shape=(X.shape[1],)) 

encoded = Dense(128, activation='relu')(input_data) 

encoded = Dense(4, activation='relu')(encoded) 

encoded = Dense(128, activation='relu')(encoded) 

decoded = Dense(X.shape[1], activation='tanh')(encoded)

autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#Auto Encoder Training

autoencoder.fit(X,X, epochs=10, batch_size=512, shuffle=True)
x = np.insert(x,-1,np.mean(np.power(x - autoencoder.predict(x), 2), axis=1),axis=1)

x_test = np.insert(x_test,-1,np.mean(np.power(x_test - autoencoder.predict(x_test), 2), axis=1),axis=1)
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=42)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.33,

    'boost_from_average':'false',   

    'boost': 'gbdt',

    'feature_fraction': 0.04,

    'learning_rate': 0.01,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,     

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,            

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1

}
for fold_, (trn_idx, val_idx) in enumerate(folds.split(x, y)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(x[trn_idx], label=y[trn_idx])

    val_data = lgb.Dataset(x[val_idx], label=y[val_idx])

    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=2500, early_stopping_rounds = 100)

    oof[val_idx] = clf.predict(x[val_idx], num_iteration=clf.best_iteration)

    predictions += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(y, oof)))
submission = test_df.loc[:,['ID_code','target']]

submission['target'] = predictions

submission.to_csv("submission_16.csv", index=False)