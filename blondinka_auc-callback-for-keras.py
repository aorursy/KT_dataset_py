# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

"""

An example to check the AUC score on a validation set for each N epochs.

I hope it will be helpful for optimizing number of epochs.

"""

import logging

from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback

class AUCCallback(Callback):

    """

    Calculates AUC for train and val sets and passes them as lists every n epochs

    

    Args: 

        interval: how often to validate. Default = 1

    """

    def __init__(self, train_data: tuple =(), validation_data: tuple=(), interval: int=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.X_train, self.y_train = train_data

        self.X_val, self.y_val = validation_data

        self.auc = []

        self.auc_train = []



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            #auc for train data

            y_pred_train = self.model.predict(self.X_train, verbose=0)

            score_train = roc_auc_score(self.y_train, y_pred_train)

            self.auc_train.append(score_train)

            #auc for validation data

            y_pred = self.model.predict(self.X_val, verbose=0)

            score = roc_auc_score(self.y_val, y_pred)

            self.auc.append(score)

            print("epoch: {:d} - roc_auc_score train: {:.6f}".format(epoch, score_train))

            print("epoch: {:d} - roc_auc_score val: {:.6f}".format(epoch, score))

            logging.info("epoch: {:d} - score: {:.6f}".format(epoch, score))
