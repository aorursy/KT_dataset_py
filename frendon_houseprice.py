import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train_csv = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_csv = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

c_test = test_csv.copy()
c_train = train_csv.copy()
c_test_labels_sample = sample_submission.copy()

df_train = pd.DataFrame(c_train)
df_test = pd.DataFrame(c_test)
df_test_labels_sample = pd.DataFrame(c_test_labels_sample)

# Target variable 'Sale Price'
train_labels = df_train.pop('SalePrice')
test_labels_sample = df_test_labels_sample.pop('SalePrice')

# Pop Id column
df_train_id = df_train.pop('Id')
df_test_id = df_test.pop('Id')

# One-hot variables
one_hot_encoded_df_train = pd.get_dummies(df_train)
one_hot_encoded_df_test = pd.get_dummies(df_test)

# Align
train, test = one_hot_encoded_df_train.align(one_hot_encoded_df_test, join='left', axis=1)

# Nan Values
train[test.keys()] = train[train.keys()].replace(np.nan, 0)
test[test.keys()] = test[test.keys()].replace(np.nan, 0)
sns.pairplot(df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']], height = 2.5)
train_stats = train.describe()
train_stats = train_stats.transpose()
train_stats
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train = norm(train)
normed_test = norm(test)
normed_train_v2 = normed_train[['YrSold', 'MoSold', 'LotArea', 'BedroomAbvGr']]
normed_test_v2 = normed_test[['YrSold', 'MoSold', 'LotArea', 'BedroomAbvGr']]
def build_model():
    
    model = keras.Sequential([
        layers.Dense(300, activation='relu', input_shape=[len(normed_train.keys())]),
        layers.Dense(int(300/2), activation='relu'),
        layers.Dense(int(300/4), activation='relu'),
        layers.Dense(int(300/8), activation='relu'),
        layers.Dense(1)])
    
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse'])
    return model
def build_model_v2():
    
    model_v2 = keras.Sequential([
        layers.Dense(300, activation='relu', input_shape=[len(normed_train_v2.keys())]),
        layers.Dense(int(300/2), activation='relu'),
        layers.Dense(int(300/4), activation='relu'),
        layers.Dense(int(300/8), activation='relu'),
        layers.Dense(1)])
    
    model_v2.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'mse'])
    return model_v2
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

model = build_model()
model.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train, train_labels, epochs=1000, validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])


plt.title('MAE')
plt.ylabel('Mean Abs Error')
plt.xlabel('Epoch')
plt.plot(history.history['mae'], label='Train Error')
plt.plot(history.history['val_mae'], label='Val Error')
plt.legend()
plt.show()

plt.title('MSE')
plt.ylabel('Mean Square Error')
plt.xlabel('Epoch')
plt.plot(history.history['mse'], label='Train Error')
plt.plot(history.history['val_mse'], label='Val Error')
plt.legend()
plt.show()
model_v2 = build_model_v2()
model_v2.summary()
early_stop_v2 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history_v2 = model_v2.fit(normed_train_v2, train_labels, epochs=1000, validation_split = 0.2, verbose=0, callbacks=[early_stop_v2, PrintDot()])


plt.title('MAE')
plt.ylabel('Mean Abs Error')
plt.xlabel('Epoch')
plt.plot(history_v2.history['mae'], label='Train Error')
plt.plot(history_v2.history['val_mae'], label='Val Error')
plt.legend()
plt.show()

plt.title('MSE')
plt.ylabel('Mean Square Error')
plt.xlabel('Epoch')
plt.plot(history_v2.history['mse'], label='Train Error')
plt.plot(history_v2.history['val_mse'], label='Val Error')
plt.legend()
plt.show()
test_predictions = model.predict(normed_test).flatten()

predictions = pd.DataFrame()
predictions['Id'] = df_test_id
predictions['Sale Price'] = test_predictions
predictions
test_predictions_v2 = model_v2.predict(normed_test_v2).flatten()

predictions_v2 = pd.DataFrame()
predictions_v2['Id'] = df_test_id
predictions_v2['Sale Price'] = test_predictions_v2
predictions_v2
predictions.to_csv('predictions.csv')
predictions_v2.to_csv('predictions_v2.csv')
os.listdir('/kaggle/working/')
loss, mae, mse = model.evaluate(normed_test, test_labels_sample, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} Sale Price".format(mae))
loss_v2, mae_v2, mse_v2 = model_v2.evaluate(normed_test_v2, test_labels_sample, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} Sale Price".format(mae))
plt.scatter(test_labels_sample, test_predictions)
plt.xlabel('True Values [Sale Price]')
plt.ylabel('Predictions [Sale Price]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.scatter(test_labels_sample, test_predictions_v2)
plt.xlabel('True Values [Sale Price]')
plt.ylabel('Predictions [Sale Price]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - test_labels_sample
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Sale Price]")
_ = plt.ylabel("Count")
error = test_predictions_v2 - test_labels_sample
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Sale Price]")
_ = plt.ylabel("Count")