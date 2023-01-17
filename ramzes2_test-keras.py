import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA, FastICA

from keras.models import Sequential

from keras.layers import Dense

import keras.backend as K

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import os



from sphinx.addnodes import highlightlang



#os.environ['PATH'] = os.environ['PATH'] + ';C:\\Users\\Roman\\Anaconda3\\Library\mingw-w64\\bin\\'

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
def prepareInputData(train_df, test_df):

    n = train_df.shape[1] - 2  # count of X columns in input data



    X = pd.concat([train_df[train_df.columns[-n:]], test_df[test_df.columns[-n:]]], ignore_index=True)

    X = pd.get_dummies(X)



    train_X = X.head(train_df.shape[0])

    test_X = X.tail(test_df.shape[0])



    # usable_columns = list(set(X.columns))

    #

    # for column in usable_columns:

    #     cardinality = len(np.unique(train_X[column]))

    #     if cardinality == 1:

    #         train_X.drop(column, axis=1)  # Column with only one value is useless so we drop it

    #         test_X.drop(column, axis=1)



    train_X = train_X.as_matrix()

    test_X = test_X.as_matrix()



    train_Y = train_df["y"].as_matrix()

    test_id = test_df["ID"].as_matrix()



    return train_X, train_Y, test_X, test_id
#def r2_keras(y_true, y_pred):

#    SS_res =  K.mean(K.square( y_true - y_pred ))

#    SS_tot = K.mean(K.square( y_true - K.mean(y_true) ))

#    return ( 1 - SS_res/(SS_tot) )



def r2_keras(y_true, y_pred):

    return K.sum(K.square(y_true - y_pred))



def r2_keras_neg(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true - y_pred ))

    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ))

    return -( 1 - SS_res/(SS_tot) )



def r2_my(y_true, y_pred):

    SS_res =  np.sum(np.square( y_true - y_pred ))

    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ))

    #return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    return ( 1 - SS_res/(SS_tot) )



def makeNnEnsemble(X, Y):

    models = []

    predicted_true = np.empty(0)

    predicted_y = np.empty(0)



    scaler_X = MinMaxScaler(feature_range=(-0.5, 0.5))

    scaler_Y = MinMaxScaler(feature_range=(-0.5, 0.5))

    X = scaler_X.fit_transform(X)

    Y = scaler_Y.fit_transform(Y.reshape(-1, 1)).reshape(len(Y))



    kf = KFold(n_splits=5, shuffle=False)

    for train_index, test_index in kf.split(X):

        x_train, x_valid = X[train_index], X[test_index]

        y_train, y_valid = Y[train_index], Y[test_index]



        x_valid = x_train

        y_valid = y_train



        model = Sequential()

        model.add(Dense(10, input_dim=X.shape[1], kernel_initializer='normal', activation='linear'))

        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        # Compile model

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2_keras])



        checkpoint = ModelCheckpoint('t.hdf5', save_best_only=True)

        s = model.fit(x_train, y_train, epochs=20, verbose=2, validation_data=(x_valid, y_valid), callbacks=[checkpoint], batch_size=64)

        model.load_weights("t.hdf5")

        ind = np.argmin(s.history["val_loss"])

        print("best validation loss=", s.history["val_loss"][ind], " val_r2_keras=", s.history["val_r2_keras"][ind],

              " train loss=", s.history["loss"][ind], " [", ind, "]")



        predicted_true = np.append(predicted_true, y_valid)

        a = model.predict(x_valid).reshape(len(y_valid))

        predicted_y = np.append(predicted_y, a)

        print("a.shape: " + str(a.shape))

        print("x_valid.shape: " + str(x_valid.shape))

        print("y_valid.shape: " + str(y_valid.shape))

        print("r2: " + str(r2_score(y_valid, a)))

        print("r2 my: " + str(r2_my(y_valid.reshape(len(y_valid)), a)))

        print("r2 keras: " + str(K.eval(r2_keras(K.variable(y_valid), K.variable(a)))) )

        print("mse: " + str(mean_squared_error(y_valid, a)) )





    print("validation r2: " + str(r2_score(predicted_true, predicted_y)))

    



train_df = pd.read_csv("../input/train.csv")

train_df.head()



test_df = pd.read_csv("../input/test.csv")

test_df.head()



cols = train_df.shape[1]



train_X, train_Y, test_X, test_id = prepareInputData(train_df, test_df)



print("rows before averaging: " + str(train_X.shape[0]))

# train_X, train_Y = averageEqualRows(train_X, train_Y)

print("rows after averaging: " + str(train_X.shape[0]))



makeNnEnsemble(train_X.astype(np.float32), train_Y)