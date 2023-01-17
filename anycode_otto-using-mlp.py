!pip install uc==2.2.0
import os

from uc.mlp import MLP 

from uc.plotting import plot_importance 



import pandas as pd

import numpy as np

from sklearn import preprocessing



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample = pd.read_csv('../input/sampleSubmission.csv')



dict_mapping = {"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}

labels = train[["target"]]['target'].apply(lambda x: dict_mapping[x]-1)



train = train.drop(["id", "target"], axis=1)

test = test.drop("id", axis=1)



feature_names = list(train.columns[0:])



train_x = np.array(train, np.float64)

train_y = np.array(labels, np.float64)

test_x = np.array(test, np.float64)



feature_range=(-1, 1)

scaler = preprocessing.MinMaxScaler(feature_range=feature_range, copy=False).fit(train_x)



scaler.transform(train_x)

scaler.transform(test_x)

np.clip(test_x, out=test_x, a_min=feature_range[0], a_max=feature_range[1])



print("train_x", train_x.shape, train_x.min(), train_x.max())

print("train_y", train_y.shape, train_y.min(), train_y.max())

print("test_x", test_x.shape, test_x.min(), test_x.max())



epoch_decay = 4

epoch_train = epoch_decay * 16



mlp = MLP(

    layer_size = [train_x.shape[1], 100, 100, 100, 100, 100, 100, 9],

    activation = 'am2',

    op='fc',



    rate_init = 0.05, 

    rate_decay = 0.8, 

    bias_rate = [], 

    regularization = 0,

    

    loss_type = "softmax",

    verbose=1, 

    importance_out=True, 

    importance_mul = 0.0001, 



    epoch_train = epoch_train, 

    epoch_decay = epoch_decay,

    exf = [

        {'Tins': [0], 'Touts': [1]},

        {'Tins': [1], 'Touts': [2]},



        {'Tins': [0], 'Touts': [3]},

        {'Tins': [3], 'Touts': [4]},

        {'Tins': [4], 'Touts': [5], 'Op': 'sigmoid'},



        {'Tins': [2, 5], 'Touts': [6], 'Op': 'mul'},

        {'Tins': [6], 'Touts': [7]},

    ]

)



mlp.fit(train_x, train_y)



plot_importance(mlp.feature_importances_, 20, feature_names=feature_names)



pred = mlp.predict_proba(test_x)



submission = pd.DataFrame(pred, index=sample.id.values, columns=sample.columns[1:])

submission.to_csv('submission.csv', index_label='id')