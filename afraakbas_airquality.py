# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy, CategoricalCrossentropy, mean_squared_error, MeanSquaredError
from tensorflow.keras.backend import sign
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
data = pd.read_excel('/kaggle/input/AirQualityUCI.xlsx')
data.shape
data.head()
data.tail()
data.drop(data[data['Date'] == '2004-03-10'].index, inplace=True)
data.drop(data[data['Date'] == '2005-04-04'].index, inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
data.shape
data.isna().sum()
# plot each column
groups = data.columns
i = 0
values = data.values
for group in groups:
    plt.figure(figsize=(10, 8))
    plt.subplot(len(groups), 1, (i+1))
    plt.plot(values[:, i])
    plt.title(data.columns[i], y=1, loc='center')
    i += 1
plt.show()
def normalize_data(df):
    scaler = StandardScaler()
    for i in df.columns:
        df[i] = scaler.fit_transform(np.array(df[i]).reshape(-1, 1))
    return df

data_norm = np.array(normalize_data(data))
data_norm.max(), data_norm.min(), data_norm.shape
y_values = int(data_norm.shape[0] / 24)
n_features = data_norm.shape[1]
x_values = int(data_norm.shape[0] - y_values)
print(x_values, y_values, n_features)
def data_preparation(dataset):
    test = np.zeros((y_values, n_features))
    train = np.zeros((x_values, n_features))
    for i in range(y_values):
        test[i] = dataset[i*24]
    j = 0
    for i in range((len(dataset))):
        if i%24 != 0:
            train[j] = dataset[i]
            j+=1
    
    return train, test
X, y = data_preparation(data_norm)
X.shape, y.shape
X = X.reshape((389, 23, 13))
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
lstm_model = Sequential()

lstm_model.add(LSTM(128, input_shape=X_train.shape[-2:], return_sequences=True))
lstm_model.add(LSTM(64))
lstm_model.add(Dropout(0.15))
lstm_model.add(Dense(13))

lstm_model.summary()
# train LSTM model
adam_modified = optimizers.Adam(learning_rate=0.001, beta_1=0.7, beta_2=0.9, amsgrad=False)

lstm_model.compile(optimizer=adam_modified,loss="MAE", metrics=["accuracy"])
lstm_model.fit(X_train, y_train, epochs=25, validation_split=0.2)
lstm_predictions = lstm_model.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_score = sqrt(mean_squared_error(y_test, lstm_predictions))
r2 = r2_score(y_test, lstm_predictions)
print("RMSE Score of LSTM model = ",rmse_score)
print("R2 score = ", r2)
y_test[0], lstm_predictions[0] #aslında burda benzer değerler olduğunu farkettim belki de r^2 score hatalı?
def adversarial_pattern(dataset, label):
    dataset = tf.cast(dataset, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(dataset)
        prediction = lstm_model(dataset)
        
        loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(label, prediction)
    
    gradient = tape.gradient(loss, dataset)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad

signed = adversarial_pattern(X_test, y_test)
perturbed_data = X_test + 0.2 * signed #0.2
X_perturbed = perturbed_data.numpy()
lstm_adv_predictions = lstm_model.predict(X_perturbed)
rmse_adv_score = sqrt(mean_squared_error(y_test, lstm_adv_predictions))
r2_adv = r2_score(y_test, lstm_adv_predictions)
print("RMSE Score of attacked LSTM model = ",rmse_adv_score)
print("R2 score = ", r2_adv)
def itrAdvAttacks(itr, epsilon, alfa, dataset, label):
    adversarial = dataset
    for i in range(itr):
        n = alfa * adversarial_pattern(adversarial, label)
        adversarial += n
        maxValues = np.maximum((dataset-epsilon), np.array(adversarial)) 
        adversarial = np.minimum((dataset+epsilon), maxValues)
        
        itrPrediction = lstm_model.predict(adversarial)
        print("iteration ", (i+1) , sqrt(mean_squared_error(label, itrPrediction))) 
        
    return adversarial, itrPrediction

iterativeAdv, itrPrediction = itrAdvAttacks(5, 0.2, 0.05, X_test, y_test)
import matplotlib.cm as cm
def convertImage(dataset, adversarial_dataset):
    diff = dataset - adversarial_dataset
    
    fig, ax = plt.subplots(figsize=(20,30))
    im = ax.imshow(diff, cmap=cm.gray)

    # Loop over data dimensions and create text annotations.
    for i in range(23):
        for j in range(13):
            text = ax.text(j, i, diff[i, j]*100,
                           ha="center", va="center", color="r")

    ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    plt.show()
    
#     plt.figure(figsize=(10, 8))
#     plt.imshow(dataset, cmap=cm.gray)
    
#     plt.figure(figsize=(10, 8))
#     plt.imshow(adversarial_dataset, cmap=cm.gray)
    
#     plt.figure(figsize=(10, 8))
#     plt.imshow(diff, cmap=cm.gray)
    
convertImage(X_test[0], iterativeAdv[0])
def labelImage(y_values, adv_values):
    diff = y_values - adv_values
    diff = np.where(diff < 0.011, 0, diff)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_values, adv_values)
labelImage(y_test, itrPrediction)
