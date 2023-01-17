# I will share datasets with you. Click on Add data on the right, then filter by "my datasets" and pick both
# the scratch one is small and nice for playing around.
# this is all dummy data, so no privacy issues etc..

# full dataset is at: www.kaggle.com/dataset/8e2c553021e61f41ba40e06f215f2c8346aa7d9aaa52ac3b0a01a10ccea6510d
# scratch around dataset is at: www.kaggle.com/dataset/5c7ca7c6ade4fb31fe2328f64d872fc35aaf3fd74afd90ccaf2ba056e18454d9
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#pip install tensorflow==1.15 keras==2.2.4

# on kaggle the versions are:
# K '2.3.1'
# T '2.1.0'

from time import gmtime, strftime

import numpy as np
import pandas as pd
import sys
import logging
logging.basicConfig(level=logging.INFO)

import numpy  # for np_utils import error, import before keras
import random
import matplotlib.pyplot as plt
from seaborn import distplot
from datetime import datetime

from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Concatenate, Input, Dense, Multiply, Average
from keras.layers import GRU
from keras.layers import Conv1D
from keras.layers import Activation
from keras.layers.noise import AlphaDropout
from keras.layers import Dropout
from keras.regularizers import l2, l1, l1_l2
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.optimizers import Adam

dfp = pd.read_csv("/kaggle/input/payment_data_scratch1.csv.bz2")
dfc = pd.read_csv("/kaggle/input/customer_data_scratch1.csv.bz2")


def build_model():
    outputs = 1
    num_years = 3
    features = 52
    dense_unit_count = 1
    payments_input = Input(batch_shape=(None, num_years, features), name="payments-input-layer")

    layer_output = Dense(dense_unit_count, activation='sigmoid', name="dense_layer")(payments_input)
    layer_output = Dense(outputs, activation='sigmoid', name="final_layer")(layer_output)

    model = Model(inputs=payments_input, outputs=layer_output)

    learning_rate=0.01
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    return model

# build datasets to make it easy to save and load repeatedly.
# it's easiest to save them using numpy format, or hickle or pyarrow, or whatever format works for you.
# hint: you might want to build more than these two datasets.
# e.g.
#    np.save("/no_backup/payments", payments)
#    np.save("/no_backup/target", target)


payments = np.load("/no_backup/payments.npy")
target = np.load("/no_backup/target.npy")


test_payments = np.load("/no_backup/test_payments.npy")
test_target = np.load("/no_backup/test_target.npy")

def train_and_test(model):
    num_epochs = 10
    checkpoint_epochs = 2
    val_loss_epoch_patience = 4000 # quit after N epochs if no improvement

    train_y = target
    train_x = payments

    data_size = len(target)

    checkpoint_name = "checkpoint"
    checkpoint_filename = checkpoint_name + '.epoch_{epoch:06d}.val_loss_{val_loss:06f}.hdf5'

    model_filename = "last_model.hdf5"

    callbacks = [
        ModelCheckpoint(
            checkpoint_filename,
            monitor='val_loss',  # save validation loss
            verbose=1,
            period=checkpoint_epochs,
            mode='auto',
            save_best_only=True),

        # if you are running this with tensorboard, you can view gradients and error gradients graphically.
        # TensorBoard(job_dir, write_graph=False, write_images=False, write_grads=True, histogram_freq=checkpoint_epochs,
        #             batch_size=train_batch_size),

        EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=val_loss_epoch_patience)
    ]

    val_size = int(data_size * 0.10)

    x = train_x[:-val_size]
    y = train_y[:-val_size]
    val_x = train_x[-val_size:]
    val_y = train_y[-val_size:]

    history = model.fit(
        x=x,
        y=y,
        validation_data=(val_x, val_y),
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=2,
        shuffle=True
    )

    logging.info("Done training.")

    model.save(model_filename)
    logging.info("Final Model saved.")

    validation_losses = history.history["val_loss"]
    val_loss = np.min(validation_losses)

    logging.info("Min Val Loss {}".format(val_loss))

    steps = len(history.history["loss"])
    logging.info("DONE after {} steps".format(steps))

    test_supplemental = np.column_stack((test_blacklisted_week, test_mccs))

    test_x = {'payments-input-layer': test_payments, 'supplemental-input-layer': test_supplemental}

    y_hat = model.predict(test_x)

    # we could have predicted more than one output, take the last one as it's the yearly prediction we picked.
    y_hat_target = y_hat[:, -1]
    y_target = test_target[:, -1]

    error_percent = (y_target - y_hat_target) / y_target  # i.e. perfect prediction is 0% error

    distplot(error_percent)
    plt.show()



model = build_model()

print(model.summary())

# optionally save a graph-like image,
# !pip install pydot
# from keras.utils import plot_model
# plot_model(model, show_shapes=True, to_file='model.png')


train_and_test(model)
