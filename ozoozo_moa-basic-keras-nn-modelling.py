import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras import Sequential

from keras.layers import Dense, BatchNormalization,Dropout

from keras.callbacks import ModelCheckpoint, EarlyStopping

    

from sklearn.model_selection import KFold



path = "/kaggle/input/lish-moa/"

num_fold = 5
train_df = pd.read_csv(path + "train_features.csv", index_col = "sig_id")

test_df = pd.read_csv(path + "test_features.csv", index_col = "sig_id")

subm_df = pd.read_csv(path + "sample_submission.csv")

tr_scored = pd.read_csv(path + "train_targets_scored.csv", index_col = "sig_id")

tr_nonscored = pd.read_csv(path + "train_targets_nonscored.csv", index_col = "sig_id")
def make_numeric(df):

    df["cp_type"] = df["cp_type"].replace("trt_cp",0)

    df["cp_type"] = df["cp_type"].replace("ctl_vehicle",1)

    df["cp_dose"] = df["cp_dose"].replace("D1",0)

    df["cp_dose"] = df["cp_dose"].replace("D2",1)

    return df
train_df = make_numeric(train_df)

test_df = make_numeric(test_df)
# helps from 

# https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1

# https://www.kaggle.com/benfraser/deep-ann-tuning-and-submission

def get_keras_model(X_train, y_train, X_val, y_val, test):



    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)

    

    model = Sequential()

    model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(264, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(y_train.shape[1], activation='sigmoid', kernel_initializer='random_normal'))

    

    model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])



    

    best_model ="best_model.hdf5"



    cp = ModelCheckpoint(best_model, monitor='val_loss', verbose=2, save_best_only=True, mode='min')



    es = EarlyStopping(patience=10)



    cb = [cp, es]





    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=cb, verbose=2)

    

    X_val = sc.fit_transform(X_val)

    test = sc.fit_transform(test)

    

    return model, X_val, test
def get_pred(X, y, test):

    print("get_pred ")



    pred = np.zeros((len(test), y.shape[1]))

    pred_val = np.zeros((len(X), y.shape[1]))

            

    kf = KFold(n_splits=num_fold, random_state=None, shuffle=False)

    fold = 0

    score = 0

    for train_index, test_index in kf.split(X, y):

        fold += 1

        print("fold ", fold)

    

        X_train = X.iloc[train_index, :].values

        X_val = X.iloc[test_index, :].values

        y_train = y.iloc[train_index,:].values

        y_val = y.iloc[test_index,:].values

        

        model, X_val, test = get_keras_model(X_train, y_train, X_val, y_val, test)



        pred += model.predict(test)



        

        pred_val[test_index,:] = model.predict(X_val)            

    

    return pred/num_fold, pred_val
preds, pred_tr = get_pred(train_df, tr_scored, test_df)
subm_df.iloc[:,1:] = preds

subm_df.to_csv("submission.csv", index=False)
subm_df.to_csv("submission.csv", index=False)