# STEP-1

import category_encoders as ce

import pandas as pd

import numpy as np



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import MaxPool1D, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras import callbacks



from sklearn.metrics import accuracy_score, roc_auc_score, auc

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn import metrics, preprocessing

from tensorflow.keras import backend as K
sample_submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")
# STEP-2

def encoding(train, test, smooth):

    print('Target encoding...')

    train.sort_index(inplace=True)

    target = train['target']

    test_id = test['id']

    train.drop(['target', 'id'], axis=1, inplace=True)

    test.drop('id', axis=1, inplace=True)

    cat_feat_to_encode = train.columns.tolist()

    smoothing=smooth

    oof = pd.DataFrame([])

    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, target):

        ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

        ce_target_encoder.fit(train.iloc[tr_idx, :], target.iloc[tr_idx])

        oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(train, target)

    train = oof.sort_index()

    test = ce_target_encoder.transform(test)

    features = list(train)

    print('Target encoding done!')

    return train, test, test_id, features, target
# Encoding

train, test, test_id, features, target = encoding(train, test, 0.3)
train['target']=target

scaler=StandardScaler()



test_data=scaler.fit_transform(test[features])

test_data=test_data.reshape(test_data.shape[0],test_data.shape[1],1)
# STEP-3



def NN_model():

  # model

  model=Sequential()

  model.add(Flatten(input_shape=(23,1)))



  model.add(Dense(64, activation='relu'))

  model.add(BatchNormalization())

  model.add(Dropout(0.2))



  model.add(Dense(128, activation='relu'))

  model.add(BatchNormalization())

  model.add(Dropout(0.2))



  model.add(Dense(256, activation='relu'))

  model.add(Dropout(0.3))



  model.add(Dense(1,activation='sigmoid'))



  model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])



  return model
# STEP-4



oof_preds = np.zeros((len(train)))

test_preds = np.zeros((len(test)))



skf = StratifiedKFold(n_splits=10)

for train_index, test_index in skf.split(train, train.target.values):

    

    # Setting train and test dataset :-

    

    X_train, X_test = train.iloc[train_index, :], train.iloc[test_index, :] # for setting X_train and X_test every fold



    X_train = X_train.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)



    y_train, y_test = X_train.target, X_test.target # setting y_train and y_test





    #----------------------------------------------------------#

    # data preparation



    X_train=scaler.fit_transform(X_train[features])

    X_test=scaler.fit_transform(X_test[features])



    y_train=y_train.to_numpy()

    y_test=y_test.to_numpy()



    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)

    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)



    #----------------------------------------------------------#



    #set early stopping criteria

    es = callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5,verbose=1, mode='max', baseline=None, restore_best_weights=True)

    rlr = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,patience=3, min_lr=1e-6, mode='max', verbose=1)



    model = NN_model() #model creation

    class_weight = {0: 0.25, 1: 0.75}

    model.fit(X_train, y_train, epochs=25, batch_size=1024, callbacks=[es, rlr], verbose=1,validation_data=(X_test, y_test), class_weight=class_weight) 



    #----------------------------------------------------------#

    #validation

    

    valid_fold_preds = model.predict(X_test)

    test_fold_preds = model.predict(test_data)

    oof_preds[test_index] = valid_fold_preds.ravel()

    test_preds += test_fold_preds.ravel()

    print(metrics.roc_auc_score(y_test, valid_fold_preds))

    K.clear_session()
# STEP-5



test_preds /= 10

print("Saving submission file")

submission = pd.DataFrame.from_dict({

    'id': test_id,

    'target': test_preds

})

submission.to_csv("submission.csv", index=False)