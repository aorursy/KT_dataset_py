import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_error

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn import metrics
# train=pd.read_csv('../input/train.csv')

# test=pd.read_csv('../input/test.csv')

df_test = pd.read_csv('../input/test.csv')

df_train = pd.read_csv('../input/train.csv')
df_train.shape
df_train.head(3)
df_test.head(3)
#--Feature selection

features = [x for x in df_train.columns.values.tolist() if x.startswith("var_")]

#features = df_train.columns.values[2:202]
fig, ax = plt.subplots()

fig.set_size_inches(10, 6)

sns.distplot(df_test[features].mean(axis=0),color="green", kde=True, label='test', ax=ax)

sns.distplot(df_train[features].mean(axis=0),color="yellow", kde=True,bins=120, label='train', ax=ax)

plt.legend()

plt.show()
#--Scaling data and store scaling values

scaler = StandardScaler().fit(df_train[features].values)

X = scaler.transform(df_train[features].values)

y = df_train['target'].values
#X_train and y_train

X = X.astype(float)

y = y.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.20, )

X_train.shape,  X_test.shape, y_train.shape, y_test.shape
from keras.models import load_model

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical

import collections, numpy

from sklearn.metrics import confusion_matrix

from keras.regularizers import l2

from tensorflow import keras

from keras.layers import Dropout

import matplotlib.patches as mpatches

from keras.callbacks import EarlyStopping

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot

from keras.layers.normalization import BatchNormalization
n_cols = X_train.shape[1]

n_cols
targets = y_train.shape[0]

targets
#create model

model_0 = Sequential()



#add model layers

model_0.add(Dense(220, activation='relu', input_shape=(n_cols,))) # ‘n_cols,’. There is nothing after the comma which  

model_0.add(BatchNormalization())                                                                  #  indicates that there can be any amount of rows.

model_0.add(Dropout(0.25))



model_0.add(Dense(420, activation='relu'))

model_0.add(BatchNormalization())

model_0.add(Dropout(0.25))



model_0.add(Dense(420, activation='relu'))

model_0.add(BatchNormalization())

model_0.add(Dropout(0.25))



model_0.add(Dense(1, activation='linear'))





#compile model using accuracy to measure model performance

model_0.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])



early_stopping_monitor = EarlyStopping(patience=9)



#train model

tensor_baseline = model_0.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=100, 

                              callbacks=[early_stopping_monitor])

tensor_baseline
val_predictions = model_0.predict(X_test)

mse = mean_squared_error(val_predictions, y_test)

rmse = np.sqrt(mean_squared_error(val_predictions, y_test))



print('Model validation metrics')

print('MSE: %.2f' % mse)

print('RMSE: %.2f' % rmse)
def plot_metrics(loss, val_loss):

    fig, (ax1) = plt.subplots(1, 1, sharex='col', figsize=(20,7))

    ax1.plot(loss, label='Train loss')

    ax1.plot(val_loss, label='Validation loss')

    ax1.legend(loc='best')

    ax1.set_title('Loss')

    plt.xlabel('Epochs')

    

plot_metrics(tensor_baseline.history['loss'], tensor_baseline.history['val_loss'])
X_valid = scaler.transform(df_test[features].values)

y_valid = df_test.ID_code.values

prediction = model_0.predict(X_valid)

prediction
result = pd.DataFrame({"ID_code": y_valid})

result["target"] = prediction

result.to_csv("submission.csv", index=False)

model_0.save('./my_model.h5')