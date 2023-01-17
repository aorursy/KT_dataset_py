from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam
from keras.callbacks import Callback
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys
from termcolor import colored as coloured
from sklearn.datasets import make_blobs
class RocCallback(Callback):
    def __init__(self, training_data, test_data, max_epochs):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_test = test_data[0]
        self.y_test = test_data[1]
        self.max_roc_auc = 0  # For use in finding max ROC AUC and saving the model
        global train_roc_auc_glob
        global test_roc_auc_glob
        train_roc_auc_glob = []
        test_roc_auc_glob = []
        self.epoch_count = 0
        self.best_epoch_classvar = 0
        self.max_epochs = max_epochs

    def on_train_begin(self, logs={}):
        # Get time the training started for use in calculating time left
        self.fitting_time_init = timeit.default_timer()
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count = self.epoch_count + 1
        # Get start time of current epoch for use in calcaulting time left
        temp_time_since_fitting_began = timeit.default_timer() - self.fitting_time_init
        # How long per epoch so far, multiplied but amount left will give a time left
        temp_time_left_until_end = (temp_time_since_fitting_began/self.epoch_count) * (self.max_epochs-self.epoch_count)
        percent_done = int(self.epoch_count/self.max_epochs*100)

        # Calculate metrics at this epoch in training
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        train_roc_auc_glob.append(roc)
        y_pred_test = self.model.predict(self.x_test)
        roc_test = roc_auc_score(self.y_test, y_pred_test)
        test_roc_auc_glob.append(roc_test)

        # Updating output:
        # Format string to pad 0s so progress output doesn't move
        sys.stdout.write('\rTrain ROC AUC: {0} || Val ROC AUC: {1} - Best Val ROCAUC: {2} at Epoch {3}'.format(
            coloured('{:.4f}'.format(np.round(roc, 4)), 'green', attrs=['bold']),
            coloured('{:.4f}'.format(round(roc_test, 4)), 'green', attrs=['bold']),
            coloured('{:.4f}'.format(np.round(self.max_roc_auc, 4)), 'cyan', attrs=['bold']),
            coloured(str(self.best_epoch_classvar), 'cyan', attrs=['bold'])))

        sys.stdout.write(coloured('          {0:2d}% Done ({1}/{2})'.format(int(percent_done), int(self.epoch_count),
                                                                         int(self.max_epochs)),
                                  'magenta', attrs=['bold']))
        # If more than an hour, don't display seconds
        if temp_time_left_until_end > 3600:
            sys.stdout.write(coloured('          {0:.0f}h {1:2.0f}m left'.format(np.floor(temp_time_left_until_end / 3600),
                                                                                 temp_time_left_until_end % 60),
                                      'yellow', attrs=['bold']))

        # If less than a minute, don't display minutes, just seconds
        elif temp_time_left_until_end < 60:
            sys.stdout.write(coloured('             {0:2.0f}s left'.format(temp_time_left_until_end),
                                       'yellow', attrs=['bold']))
        # Else display minutes and seconds
        else:
            sys.stdout.write(coloured('          {0:2.0f}m {1:2.0f}s left'.format(np.floor(temp_time_left_until_end/60),
                                                                                temp_time_left_until_end % 60),
                                     'yellow', attrs=['bold']))

        sys.stdout.flush()

        # Save weights and epoch number of model with best test ROC AUC

        if roc_test > self.max_roc_auc:  # Save weights of model with best test ROC AUC
            self.max_roc_auc = roc_test
            self.best_epoch_classvar = epoch + 1
            global best_epoch
            best_epoch = epoch + 1
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
epochs = 1000
train_prop = 0.8
samples = 250
# Make data
x, y = make_blobs(n_samples=samples, n_features=2, centers=2, cluster_std=6, random_state = 123)

# Plot
df = pd.DataFrame(dict(x=x[:, 0], y=x[:, 1], label=y))

groups = df.groupby('label')

fig, ax = plt.subplots()
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
ax.legend()
plt.show()
# Split into test and train and minmax scale:
train_samples = int(round(x.shape[0] * train_prop, 0))
sequence = np.linspace(0, x.shape[0] - 1, num=x.shape[0], dtype=int)
np.random.shuffle(sequence)
train_ind = sequence[0:train_samples]
val_ind = sequence[train_samples:]

x_train = df.iloc[train_ind, [0, 1]]
y_train = df.iloc[train_ind, 2]

x_val = df.iloc[val_ind, [0, 1]]
y_val = df.iloc[val_ind, 2]

x_train = (x_train-x_train.min())/(x_train.max()-x_train.min())
x_val = (x_val-x_val.min())/(x_val.max()-x_val.min())
# Construct layers then compile
model = Sequential()
model.add(Dense(20, input_dim=x_train.shape[1]))
model.add(LeakyReLU())
model.add(Dropout(0.1))

model.add(Dense(10))
model.add(LeakyReLU())
model.add(Dropout(0.1))


model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=adam(lr=0.0001), metrics=['accuracy'])

model_hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=32,
                       verbose=0, callbacks=[RocCallback(training_data=(x_train, y_train),
                                                                test_data=(x_val, y_val),
                                                                max_epochs=epochs)])
# PLOT - Accuracy and loss combined
# Only plot first 10% of data to avoid axes being stretched
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(model_hist.history['acc'], label="Train", linewidth=0.75)
axarr[0].plot(model_hist.history['val_acc'], label="Test", linewidth=0.75)

axarr[0].legend(loc='best')
axarr[0].set_title('Model Accuracy')
axarr[1].set_title('Model Loss')
axarr[0].set_ylabel("Accuracy")
axarr[1].set_xlabel("Epoch")
axarr[1].set_ylabel("Loss")

axarr[1].plot(model_hist.history['loss'], label="Train", linewidth=0.75)
axarr[1].plot(model_hist.history['val_loss'], label="Test", linewidth=0.75)
axarr[1].legend(loc='best')
axarr[0].grid(True)
axarr[1].grid(True)
plt.show()
# PLOT - Accuracy and loss combined - MA
# Only plot where MA is calculated correctly

# MA, has int(epochs/100)+1 as cannot convolve with N = 0
# Only use last 90% of data as in other plots
MA_Var = int(epochs / 30) + 1
acc_ma = np.convolve(np.asarray(model_hist.history['acc']),
                     np.ones((MA_Var,)) / MA_Var,
                     mode='valid')
val_acc_ma = np.convolve(np.asarray(model_hist.history['val_acc']),
                         np.ones((MA_Var,)) / MA_Var,
                         mode='valid')
loss_ma = np.convolve(np.asarray(model_hist.history['loss']),
                      np.ones((MA_Var,)) / MA_Var,
                      mode='valid')
val_loss_ma = np.convolve(np.asarray(model_hist.history['val_loss']),
                          np.ones((MA_Var,)) / MA_Var, mode='valid')


f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(list(range(int(MA_Var), epochs+1)),
              acc_ma, label="Train", linewidth=0.75)
axarr[0].plot(list(range(int(MA_Var), epochs+1)),
              val_acc_ma, label="Test", linewidth=0.75)

axarr[0].legend(loc='best')
axarr[0].set_title('Model Accuracy - Moving Average')
axarr[0].set_ylabel("Accuracy")
axarr[1].set_xlabel("Epoch")
axarr[1].set_ylabel("Loss")
axarr[1].set_title('Model Loss - Moving Average')
axarr[1].plot(list(range(int(MA_Var), epochs+1)),
              loss_ma, label="Train", linewidth=0.75)
axarr[1].plot(list(range(int(MA_Var), epochs+1)),
              val_loss_ma, label="Test", linewidth=0.75)
axarr[1].legend(loc='best')
axarr[0].grid(True)
axarr[1].grid(True)
plt.show()

f, ax = plt.subplots()
ax.plot(train_roc_auc_glob, label="Train", linewidth=0.75)
ax.plot(test_roc_auc_glob, label="Test", linewidth=0.75)
best_iter = np.argmax(test_roc_auc_glob)
plt.axvline(x=best_iter, ls='--', color='black', label='Best Test Value', linewidth=0.75)
ax.set_xlabel("Epoch")
ax.set_ylabel("Area under ROC")
ax.legend(loc='best')
ax.grid(True)
# ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="0.3") - Plots diagonal line
ax.set_title("ROC AUC by Epoch")
plt.show()