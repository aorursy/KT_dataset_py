# Get Apple stock price data from https://www.macrotrends.net/stocks/charts/AAPL/apple/stock-price-history

import pandas as pd

import wandb



# Read in dataset

apple = pd.read_csv("../input/kernel-files/apple.csv")

apple = apple[-1000:]

wandb.init(project="visualize-models", name="a_metric")
%%wandb

# Log the metric

for price in apple['close']:

    wandb.log({"Stock Price": price})
# Get the dataset from UCI

!wget https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data -qq



# modified from https://github.com/dmlc/xgboost/blob/master/demo/multiclass_classification/train.py

# Import wandb

import wandb

import numpy as np

import xgboost as xgb



wandb.init(project="visualize-models", name="xgboost")



# label need to be 0 to num_class -1

data = np.loadtxt('./dermatology.data', delimiter=',',

        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x) - 1})

sz = data.shape



train = data[:int(sz[0] * 0.7), :]

test = data[int(sz[0] * 0.7):, :]



train_X = train[:, :33]

train_Y = train[:, 34]



test_X = test[:, :33]

test_Y = test[:, 34]



xg_train = xgb.DMatrix(train_X, label=train_Y)

xg_test = xgb.DMatrix(test_X, label=test_Y)

# setup parameters for xgboost

param = {}

# use softmax multi-class classification

param['objective'] = 'multi:softmax'

# scale weight of positive examples

param['eta'] = 0.1

param['max_depth'] = 6

param['silent'] = 1

param['nthread'] = 4

param['num_class'] = 6

wandb.config.update(param)



watchlist = [(xg_train, 'train'), (xg_test, 'test')]

num_round = 5
%%wandb

# Add the wandb xgboost callback

bst = xgb.train(param, xg_train, num_round, watchlist, callbacks=[wandb.xgboost.wandb_callback()])

# get prediction

pred = bst.predict(xg_test)

error_rate = np.sum(pred != test_Y) / test_Y.shape[0]

print('Test error using softmax = {}'.format(error_rate))



wandb.summary['Error Rate'] = error_rate
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

import pandas as pd

import wandb

# wandb.init(anonymous='allow', project="sklearn")

wandb.init(project="visualize-models", name="sklearn")



# Load data

boston = load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)

y = boston.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Train model, get predictions

reg = Ridge()

reg.fit(X, y)

y_pred = reg.predict(X_test)



# Visualize model performance

wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, 'Ridge')
# WandB – Import the W&B library

import wandb

from wandb.keras import WandbCallback



from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from keras.utils import np_utils

from keras.optimizers import SGD

from keras.callbacks import TensorBoard
# Default values for hyper-parameters

defaults=dict(

    dropout = 0.2,

    hidden_layer_size = 32,

    layer_1_size = 32,

    learn_rate = 0.01,

    decay = 1e-6,

    momentum = 0.9,

    epochs = 5,

    )



# Initialize a new wandb run and pass in the config object

# wandb.init(anonymous='allow', project="kaggle", config=defaults)

wandb.init(project="visualize-models", config=defaults, name="neural_network")

config = wandb.config



(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

labels=["T-shirt/top","Trouser","Pullover","Dress","Coat",

        "Sandal","Shirt","Sneaker","Bag","Ankle boot"]



img_width=28

img_height=28



X_train = X_train.astype('float32')

X_train /= 255.

X_test = X_test.astype('float32')

X_test /= 255.



#reshape input data

X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)[:10000]

X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)[:10000]



# one hot encode outputs

y_train = np_utils.to_categorical(y_train)[:10000]

y_test = np_utils.to_categorical(y_test)[:10000]

num_classes = y_test.shape[1]



# build model

model = Sequential()

model.add(Conv2D(config.layer_1_size, (5, 5), activation='relu',

                            input_shape=(img_width, img_height,1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(config.dropout))

model.add(Flatten())

model.add(Dense(num_classes, activation='softmax'))



sgd = SGD(lr=config.learn_rate, decay=config.decay, momentum=config.momentum, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
%%wandb

# Add WandbCallback() to the fit function

model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.epochs,

    callbacks=[WandbCallback(data_type="image", labels=labels)])