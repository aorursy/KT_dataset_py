from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D, Concatenate, Multiply, BatchNormalization

from keras.models import Model, Sequential

from sklearn.preprocessing import LabelEncoder

from keras import regularizers

from sklearn.model_selection import KFold 

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from keras.callbacks import Callback

from sklearn.linear_model import LogisticRegression, BayesianRidge

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error

from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE

import lightgbm as lgb

import math

from sklearn.feature_selection import f_regression

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from scipy import stats

import pandas as pd 

import numpy as np

from sklearn.metrics import roc_curve, auc

import seaborn as sns

sns.set(style="whitegrid")

import matplotlib

from sklearn.naive_bayes import GaussianNB

np.random.seed(203)

import math

from scipy.stats import ks_2samp

from sklearn.ensemble import RandomForestRegressor

from keras.callbacks import ModelCheckpoint



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
targets = pd.read_csv("../input/y_train.csv")

df_train = pd.read_csv("../input/X_train.csv")

df_test = pd.read_csv("../input/X_test.csv")
targets = pd.get_dummies(targets, columns=["surface"])

Y = targets.drop(["series_id", "group_id"], axis=1).values

X = df_train.drop(["row_id", "series_id", "measurement_number"], axis=1).values

test = df_test.drop(["row_id", "series_id", "measurement_number"], axis=1).values

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = X.reshape((3810, 128, 10, 1))

X = X.transpose(0, 2, 1, 3)

test = scaler.transform(test)

test = test.reshape((3816, 128, 10, 1))

test = test.transpose(0, 2, 1, 3)
#Every column gets the smoothness features, except for the ones in the list below

for col in df_train.drop(["row_id", "series_id", "measurement_number"], axis=1).columns:

    df_train["diff_"+col] = df_train[col]-df_train[col].shift(1)

    df_test["diff_"+col] = df_test[col]-df_test[col].shift(1)

    df_train["ma_"+col] = np.square(abs(df_train[col]-df_train[col].rolling(8).mean()))

    df_test["ma_"+col] = np.square(abs(df_test[col]-df_test[col].rolling(8).mean()))



#The first 8 measurement (0-7) contain data from the previous series id (because of the rolling mean), so they have to be dropped

df_train = df_train[df_train["measurement_number"]>7]

df_test = df_test[df_test["measurement_number"]>7]



aggs = {

    "diff_orientation_X":"std",

    "diff_orientation_Y":"std",

    "diff_orientation_Z":"std",

    "diff_orientation_W":"std",

    "diff_angular_velocity_X":"std",

    "diff_angular_velocity_Y":"std",

    "diff_angular_velocity_Z":"std",

    "diff_linear_acceleration_X":"std",

    "diff_linear_acceleration_Y":"std",

    "diff_linear_acceleration_Z":"std",

    "ma_orientation_X":"sum",

    "ma_orientation_Y":"sum",

    "ma_orientation_Z":"sum",

    "ma_orientation_W":"sum",

    "ma_angular_velocity_X":"sum",

    "ma_angular_velocity_Y":"sum",

    "ma_angular_velocity_Z":"sum",

    "ma_linear_acceleration_X":"sum",

    "ma_linear_acceleration_Y":"sum",

    "ma_linear_acceleration_Z":"sum"

}



train_smooth = df_train.groupby("series_id").agg(aggs)

test_smooth = df_test.groupby("series_id").agg(aggs)

train_smooth.reset_index(inplace=True)

test_smooth.reset_index(inplace=True)
print(train_smooth.shape) #should be (3810, 20), because 10 columns got 2 additional features each

train_smooth.head()
X_smooth = train_smooth.drop(["series_id"], axis=1).values

test_smooth = test_smooth.drop(["series_id"], axis=1).values



scaler = StandardScaler()

X_smooth = scaler.fit_transform(X_smooth)

test_smooth = scaler.transform(test_smooth)
def init_model():

    

    #These features occur frequently throughout, so for easu of use, it's easier to change them up here.

    FIRST = 30 #20

    SECOND = 20 #10

    HEIGHT1 = 4 #4

    HEIGHT2 = 3 #4

    DROPOUT = 0.5

    STRIDES = None

    PS = 5

    

    input1 = Input(shape=(128, 1))

    a = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input1)

    a = BatchNormalization()(a)

    a = MaxPooling1D(strides=STRIDES, pool_size=PS)(a)

    a = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(a)

    a = BatchNormalization()(a)

    a = MaxPooling1D(strides=STRIDES, pool_size=PS)(a)

    a = Flatten()(a)

    a = Dropout(DROPOUT)(a)



    input2 = Input(shape=(128, 1))

    b = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input2)

    b = BatchNormalization()(b)

    b = MaxPooling1D(strides=STRIDES, pool_size=PS)(b)

    b = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(b)

    b = BatchNormalization()(b)

    b = MaxPooling1D(strides=STRIDES, pool_size=PS)(b)

    b = Flatten()(b)

    b = Dropout(DROPOUT)(b)



    input3 = Input(shape=(128, 1))

    c = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input3)

    c = BatchNormalization()(c)

    c = MaxPooling1D(strides=STRIDES, pool_size=PS)(c)

    c = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(c)

    c = BatchNormalization()(c)

    c = MaxPooling1D(strides=STRIDES, pool_size=PS)(c)

    c = Flatten()(c)

    c = Dropout(DROPOUT)(c)



    input4 = Input(shape=(128, 1))

    d = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input4)

    d = BatchNormalization()(d)

    d = MaxPooling1D(strides=STRIDES, pool_size=PS)(d)

    d = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(d)

    d = BatchNormalization()(d)

    d = MaxPooling1D(strides=STRIDES, pool_size=PS)(d)

    d = Flatten()(d)

    d = Dropout(DROPOUT)(d)



    input5 = Input(shape=(128, 1))

    e = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input5)

    e = BatchNormalization()(e)

    e = MaxPooling1D(strides=STRIDES, pool_size=PS)(e)

    e = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(e)

    e = BatchNormalization()(e)

    e = MaxPooling1D(strides=STRIDES, pool_size=PS)(e)

    e = Flatten()(e)

    e = Dropout(DROPOUT)(e)



    input6 = Input(shape=(128, 1))

    f = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input6)

    f = BatchNormalization()(f)

    f = MaxPooling1D(strides=STRIDES, pool_size=PS)(f)

    f = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(f)

    f = BatchNormalization()(f)

    f = MaxPooling1D(strides=STRIDES, pool_size=PS)(f)

    f = Flatten()(f)

    f = Dropout(DROPOUT)(f)



    input7 = Input(shape=(128, 1))

    g = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input7)

    g = BatchNormalization()(g)

    g = MaxPooling1D(strides=STRIDES, pool_size=PS)(g)

    g = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(g)

    g = BatchNormalization()(g)

    g = MaxPooling1D(strides=STRIDES, pool_size=PS)(g)

    g = Flatten()(g)

    g = Dropout(DROPOUT)(g)



    input8 = Input(shape=(128, 1))

    h = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input8)

    h = BatchNormalization()(h)

    h = MaxPooling1D(strides=STRIDES, pool_size=PS)(h)

    h = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(h)

    h = BatchNormalization()(h)

    h = MaxPooling1D(strides=STRIDES, pool_size=PS)(h)

    h = Flatten()(h)

    h = Dropout(DROPOUT)(h)



    input9 = Input(shape=(128, 1))

    i = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input9)

    i = BatchNormalization()(i)

    i = MaxPooling1D(strides=STRIDES, pool_size=PS)(i)

    i = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(i)

    i = BatchNormalization()(i)

    i = MaxPooling1D(strides=STRIDES, pool_size=PS)(i)

    i = Flatten()(i)

    i = Dropout(DROPOUT)(i)



    input10 = Input(shape=(128, 1))

    j = Conv1D(FIRST, HEIGHT1, activation="relu", kernel_initializer="uniform")(input10)

    j = BatchNormalization()(j)

    j = MaxPooling1D(strides=STRIDES, pool_size=PS)(j)

    j = Conv1D(SECOND, HEIGHT2, activation="relu", kernel_initializer="uniform")(j)

    j = BatchNormalization()(j)

    j = MaxPooling1D(strides=STRIDES, pool_size=PS)(j)

    j = Flatten()(j)

    j = Dropout(DROPOUT)(j)

    

    input11 = Input(shape=(20,))

    k = Dense(30, activation="relu", kernel_initializer="uniform")(input11)

    k = Dropout(0.25)(k)



    merged = Concatenate()([a, b])

    merged = Concatenate()([merged, c])

    merged = Concatenate()([merged, d])

    merged = Concatenate()([merged, e])

    merged = Concatenate()([merged, f])

    merged = Concatenate()([merged, g])

    merged = Concatenate()([merged, h])

    merged = Concatenate()([merged, i])

    merged = Concatenate()([merged, j])

    merged = Concatenate()([merged, k])

    merged = Dense(30, activation = "relu", kernel_initializer="uniform")(merged)

    merged = Dropout(0.25)(merged)

    

    output = Dense(9, activation="softmax", kernel_initializer="uniform")(merged)

    model = Model([input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11], output)

    return model
te = [test[:,i] for i in range(10)]

te.append(test_smooth)



folds = StratifiedKFold(n_splits=15, shuffle=True, random_state=0)

oof = np.zeros((len(X), 9))

test_predictions = np.zeros((len(test), 9))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X,np.argmax(Y, axis=1))):

    print(fold_)

    X_train, X_test = X[trn_idx], X[val_idx]

    y_train, y_test = Y[trn_idx], Y[val_idx]

    X_smooth_train, X_smooth_test = X_smooth[trn_idx], X_smooth[val_idx]

    

    #The model needs a list of inputs. The code beneath this creates a column-input for each unique column in the training set and appends it to the list. 

    X_tr = [X_train[:,i] for i in range(10)]

    X_tr.append(X_smooth_train)

    X_te = [X_test[:,i] for i in range(10)]

    X_te.append(X_smooth_test)

    

    model = init_model()

    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    model.fit(X_tr, y_train, validation_data = (X_te, y_test), epochs=100, shuffle=True, class_weight="balanced")

    pre = model.predict(X_te)

    oof[val_idx] = model.predict(X_te)

    test_predictions+= model.predict(te)/20
predictions = np.argmax(test_predictions, axis=1)

true = np.argmax(y_test, axis=1)



labels = {

    0:"carpet",

    1:"concrete",

    2:"fine_concrete",

    3:"hard_tiles",

    4:"hard_tiles_large_space",

    5:"soft_pvc",

    6:"soft_tiles",

    7:"tiled",

    8:"wood"

}



pre = []

for p in predictions:

    pre.append(labels[p])
sub = pd.read_csv("../input/sample_submission.csv")

sub["surface"] = np.array(pre)

sub.to_csv("submission.csv", index=False)