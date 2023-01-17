# Processing Stuff

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, accuracy_score

import numpy as np # linear algebra

np.random.seed(42)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# ML modules

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

# DL modules

from keras.layers import *

from keras.models import *

from keras.applications import *

from keras.optimizers import *

from keras.regularizers import *

from keras.applications.inception_v3 import preprocess_input

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

# check directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Using pretrained model from kaggle dataset

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

models_dir = join(cache_dir, 'models')

if not exists(models_dir):

    makedirs(models_dir)



!ls ../input/keras-pretrained-models/

!cp ../input/keras-pretrained-models/* ~/.keras/models/
#Load data

train = pd.read_json("../input/statoil-iceberg-classifier-challenge/train.json")

test = pd.read_json("../input/statoil-iceberg-classifier-challenge/test.json")

train.inc_angle = train.inc_angle.replace('na', 0)

train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

print("done!")

# Train data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

X_train = np.concatenate([x_band1[:, :, :, np.newaxis]

                          , x_band2[:, :, :, np.newaxis]

                         , ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)

X_angle_train = np.array(train.inc_angle)

y_train = np.array(train["is_iceberg"])



# Test data

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])

x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])

X_test = np.concatenate([x_band1[:, :, :, np.newaxis]

                          , x_band2[:, :, :, np.newaxis]

                         , ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)

X_angle_test = np.array(test.inc_angle)





X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train

                    , X_angle_train, y_train, random_state=123, train_size=0.7)
import matplotlib.pyplot as plt

%matplotlib inline

from skimage.transform import resize

from tqdm import tqdm



# Training data

width = 299

n = len(X_train)

X_train_resized = np.zeros((n, width, width, 3), dtype=np.float32)

for i in tqdm(range(n)):

    x = X_train[i]

    x = (x-x.min())/(x.max()-x.min()) # normalize for each pseudo pixel value

    X_train_resized[i] = resize(x, (299,299), mode='reflect')





# Validation data

width = 299

n = len(X_valid)

X_valid_resized = np.zeros((n, width, width, 3), dtype=np.float32)

for i in tqdm(range(n)):

    x = X_valid[i]

    x = (x-x.min())/(x.max()-x.min())  # normalize for each pseudo pixel value

    X_valid_resized[i] = resize(x, (299,299), mode='reflect')

    

# Test data

width = 299

n = len(X_test)

X_test_resized = np.zeros((n, width, width, 3), dtype=np.float32)

for i in tqdm(range(n)):

    x = X_test[i]

    x = (x-x.min())/(x.max()-x.min())  # normalize for each pseudo pixel value

    X_test_resized[i] = resize(x, (299,299), mode='reflect')



def get_features(MODEL, data=None):

    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')

    

    inputs = Input((width, width, 3))

    x = inputs

    x = Lambda(preprocess_input, name='preprocessing')(x)

    x = cnn_model(x)

    x = GlobalMaxPooling2D()(x)

    cnn_model = Model(inputs, x)



    features = cnn_model.predict(data, batch_size=4, verbose=1)

    return features

train_features = get_features(InceptionV3, X_train_resized)

valid_features = get_features(InceptionV3, X_valid_resized)

test_features = get_features(InceptionV3, X_test_resized)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, accuracy_score



# train

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=2017)

clf.fit(train_features, y_train)



# validate

y_probs = clf.predict_proba(valid_features)

print('Validation Inception-V3 LogLoss {}'.format(log_loss(y_valid, y_probs)))

print('Validation Inception-V3 Accuracy {}'.format(accuracy_score(y_valid, y_preds)))



# predict

logreg_preds = clf.predict(test_features)
import xgboost as xgb





d_train =  xgb.DMatrix(X_train_resized,label=y_train)

d_valid =  xgb.DMatrix(X_valid_resized,label=y_valid)

d_test =  xgb.DMatrix(X_test_resized,label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]





params = {

        'objective': 'binary:logistic',

        'n_estimators':1000,

        'max_depth': 8,

        'subsample': 0.9,

        'colsample_bytree': 0.9 ,

        'eta': 0.01,

        'eval_metric': 'logloss'

        }



# train

clf =xgb.train(params, d_train, 1600, eval_set=watchlist, early_stopping_rounds=70,  verbose_eval=100)



# validate

y_probs = clf.predict_proba(d_valid)

print('Validation Inception-V3 LogLoss {}'.format(log_loss(y_valid, y_probs)))

print('Validation Inception-V3 Accuracy {}'.format(accuracy_score(y_valid, y_preds)))



# predict

xgb_preds = clf.predict(d_test)
def get_model():

    bn_model = 0

    p_activation = "elu"

    input_layer = Input(shape=(4096,), name="X_1")

    dense_layer = Dropout(0.2) (BatchNormalization(momentum=bn_model) ( Dense(12, activation=p_activation)(input_layer) ))

    dense_layer = Dropout(0.2) (BatchNormalization(momentum=bn_model) ( Dense(92, activation=p_activation)(dense_layer) ))

    

    output = Dense(1, activation="sigmoid")(dense_layer)

    

    model = Model(input_layer,  output)

    optimizer = Adam(lr=0.005, epsilon=1e-08)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model
# train

model = get_model()

model.fit(train_features, y_train, epochs=2, validation_data=(valid_features, y_valid))



# validate

print("Validation: ")

print(model.evaluate(valid_features, y_valid, verbose=1, batch_size=200))



# predict

mlp_preds = model.predict(test_features)
logreg_sub = pd.DataFrame({'id': test["id"], 'is_iceberg': logreg_preds})

logreg_sub.to_csv("./logreg_sub.csv", index=False)



xgb_sub = pd.DataFrame({'id': test["id"], 'is_iceberg': xgb_preds})

xgb_sub.to_csv("./xgb_sub.csv", index=False)



mlp_sub = pd.DataFrame({'id': test["id"], 'is_iceberg': mlp_preds.reshape((mlp_preds.shape[0]))})

mlp_sub.to_csv("./mlp_sub.csv", index=False)



# ensemble with harmonic mean
