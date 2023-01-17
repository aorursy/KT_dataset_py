import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



# tf.keras stuff

from tensorflow.keras import layers

from tensorflow.keras import callbacks

from tensorflow.keras import optimizers

from tensorflow.keras import backend

from tensorflow.keras import models
# loading data

train_raw = pd.read_csv("../input/digit-recognizer/train.csv")

test_raw = pd.read_csv("../input/digit-recognizer/test.csv")



print("Training dataset has", train_raw.shape[0], "rows/instances and", train_raw.shape[1], "columns/features.")

print("Testing dataset has", test_raw.shape[0], "rows/instances and", test_raw.shape[1], "columns/features.")
# splitting target variable from training data

X_train_full = train_raw.drop('label', axis=1)

y_train_full = train_raw['label']
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.

y_valid, y_train = y_train_full[:5000], y_train_full[5000:]



print("Training dataset has", X_train.shape[0], "rows/instances and", X_train.shape[1], "columns/features.")

print("Validation dataset has", X_valid.shape[0], "rows/instances and", X_valid.shape[1], "columns/features.")
class ExponentialLearningRate(callbacks.Callback):

    def __init__(self, factor):

        self.factor = factor

        self.rates = []

        self.losses = []

    

    def on_batch_end(self, batch, logs):

        self.rates.append(backend.get_value(self.model.optimizer.lr))

        self.losses.append(logs["loss"])

        backend.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)
# list of layers to be used in model

layers_list = [

    layers.InputLayer(input_shape=[784]),

    layers.Dense(300, activation="relu"),

    layers.Dense(100, activation="relu"),

    layers.Dense(10, activation="softmax")

]



model = models.Sequential(layers_list)
# compiling model

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3), metrics=["accuracy"])



exp_lr = ExponentialLearningRate(factor=1.005)
history = model.fit(X_train, y_train, epochs=1, 

          validation_data=(X_valid, y_valid), 

          callbacks=[exp_lr])
# plotting loss as function of learning rate

plt.plot(exp_lr.rates, exp_lr.losses)

plt.gca().set_xscale('log')

plt.hlines(min(exp_lr.losses), min(exp_lr.rates), max(exp_lr.rates))

plt.axis([min(exp_lr.rates), max(exp_lr.rates), 0, exp_lr.losses[0]])

plt.xlabel("Learning rate")

plt.ylabel("Loss")
backend.clear_session()
model = models.Sequential(layers_list)



model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizers.SGD(lr=2e-1), metrics=["accuracy"])
early_stopping_cb = callbacks.EarlyStopping(patience=20)    # stopping training early if validation loss isn't improving

checkpoint_cb = callbacks.ModelCheckpoint("mnist_model.h5", save_best_only=True)    # save best model only



history = model.fit(X_train, y_train, epochs=100, 

          validation_data=(X_valid, y_valid), 

          callbacks=[early_stopping_cb, checkpoint_cb])
# rollback to best model

model = models.load_model("mnist_model.h5")
# feature scaling test data

X_test = test_raw / 255.
final_pred = model.predict_classes(X_test)
# create submission file

my_submission = pd.DataFrame({'ImageId': np.array(range(1,28001)), 'Label': final_pred})

my_submission.to_csv("submission.csv", index=False)