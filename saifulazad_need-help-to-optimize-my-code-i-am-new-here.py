
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

train_data = pd.DataFrame(df_train, columns=['GrLivArea', 'TotalBsmtSF','OverallQual', 'FullBath', 'TotRmsAbvGrd','SalePrice'])

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500


train_label = train_data['SalePrice']
del train_data['SalePrice']

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 500000])


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_label, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

test_data = pd.DataFrame(df_test, columns=['GrLivArea', 'TotalBsmtSF','OverallQual', 'FullBath', 'TotRmsAbvGrd'])
test_predictions = model.predict(test_data).flatten()


val = pd.concat([df_test['Id'], pd.Series(test_predictions)], axis=1)

val.to_csv('t.csv')

