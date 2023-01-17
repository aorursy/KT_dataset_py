# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualisation

import tensorflow as tf





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/year_prediction.csv")

data = data.rename(index=str, columns={"label":"year"})
nsongs = {}

for y in range(1922,2012):

    nsongs[y] = len(data[data.year==y])

yrs = range(1922,2011)

values = [nsongs[y] for y in yrs]

plt.bar(yrs, values, align='center')

plt.xlabel("Year")

plt.ylabel("Number of songs")
# separate input attributes and output into different dataframes

X = data.iloc[:,1:]

Y = data.iloc[:,0]



# Train set

X_train = X.iloc[0:463715,:]

y_train = Y.iloc[0:463715]



# Validation set

X_test = X.iloc[463715:,:]

y_test = Y.iloc[463715:]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



# Fit on training set only.

scaler.fit(X_train)

# Apply transform to both the train set and the test set.

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)



X_train = pd.DataFrame(X_train_scaled,columns=X_train.columns)

X_test = pd.DataFrame(X_test_scaled,columns=X_train.columns)
X_train.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training set only.

scaler.fit(X_train)

# Apply transform to both the train set and the test set.

X_train_std = scaler.transform(X_train)

X_test_std = scaler.transform(X_test)
X_train_std = pd.DataFrame(X_train_std,columns=X_train.columns)

X_train_std.describe()
from sklearn.decomposition import PCA

# Make an instance of the Model

pca = PCA(.90)



# We fit to only our training set

pca.fit(X_train_std)

# Print number of components generated

pca.n_components_
X_train_proc = pca.transform(X_train_std)

X_test_proc = pca.transform(X_test_std)
y_train_proc = y_train - min(y_train)

y_test_proc = y_test - min(y_test)

# y_train_proc
from tensorflow.python.keras import Sequential

from tensorflow.python.keras.layers import Dense, Lambda, Dropout

# from tensorflow.python.keras.initializers import Initializer

from tensorflow.python.keras.utils import to_categorical

from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
y_train_hot = to_categorical(y_train_proc, 90)

y_test_hot = to_categorical(y_test_proc, 90)
print(X_train_proc.shape)

print(y_test_hot.shape)
def plot(history):

    epochs = range(1, len(history.history['loss']) + 1)

    plt.plot(epochs, history.history['mean_absolute_error'], label='train');

    plt.plot(epochs, history.history['val_mean_absolute_error'], label='val');

    plt.xlabel('epoch');

    plt.ylabel('mae');

    plt.legend();

    plt.show();
model1 = Sequential()

model1.add(Dense(55, input_shape=(55,)))

model1.add(Dense(110))

model1.add(Dense(90, activation='softmax'))
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=4, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)
model1.compile(optimizer='adam'

             , loss='categorical_crossentropy'

             , metrics=['accuracy'])
fit1 = model1.fit(x=X_train_proc, y=y_train_hot

          , epochs=5

          , batch_size=64

          , validation_data=(X_test_proc, y_test_hot)

          , callbacks=[learning_rate_reduction])
# plot(fit1)
model1.summary()
preds = model1.predict_classes(X_test_proc)
print(np.array(y_test_proc))

print(preds)

np.mean(np.absolute((preds-np.array(y_test_proc))))
model2 = Sequential()

model2.add(Dense(55, input_shape=(55,), activation='relu'))

model2.add(Dense(1))
learning_rate_reduction1 = ReduceLROnPlateau(monitor='mean_absolute_error', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)
model2.compile(optimizer='adam'

             , loss='mse'

             , metrics=['mae'])
fit2 = model2.fit(x=X_train_proc, y=y_train_proc

          , epochs=10

          , batch_size=64

          , validation_data=(X_test_proc, y_test_proc)

          , callbacks=[learning_rate_reduction1])
preds_model_rms = model2.predict(X_test_proc)

np.mean(np.absolute(preds_model_rms.T-np.array(y_test_proc)))
plot(fit2)
# from sklearn.metrics import mean_squared_error, r2_score

# mean_squared_error(predictions_linearRegr, np.array(y_test_proc))

es = EarlyStopping(monitor='val_mean_absolute_error', patience=2, restore_best_weights=True)
model3 = Sequential()

model3.add(Dense(55, input_shape=(55,), activation='relu'))

model3.add(Dense(110, activation='relu'))

model3.add(Dropout(0.2))

model3.add(Dense(1))



model3.compile(optimizer='adam'

             , loss='mse'

             , metrics=['mae'])
fit3 = model3.fit(x=X_train_proc, y=y_train_proc

          , epochs=10

          , batch_size=64

          , validation_data=(X_test_proc, y_test_proc)

          , callbacks=[learning_rate_reduction1, es])
preds_model_rms = model3.predict(X_test_proc)

np.mean(np.absolute(preds_model_rms.T-np.array(y_test_proc)))
plot(fit3)
model3.compile(optimizer='adam'

             , loss='mse'

             , metrics=['mae'])

fit3 = model3.fit(x=X_train_proc, y=y_train_proc

          , epochs=10

          , batch_size=128

          , validation_data=(X_test_proc, y_test_proc)

          , callbacks=[learning_rate_reduction1, es])
preds_model_rms = model3.predict(X_test_proc)

np.mean(np.absolute(preds_model_rms.T-np.array(y_test_proc)))
plot(fit3)
from keras.optimizers import RMSprop

# adam = optimizers.Adam()
model3.compile(optimizer='RMSprop'

             , loss='mse'

             , metrics=['mae'])

fit4 = model3.fit(x=X_train_proc, y=y_train_proc

          , epochs=10

          , batch_size=64

          , validation_data=(X_test_proc, y_test_proc)

          , callbacks=[learning_rate_reduction1, es])
preds_model_rms = model3.predict(X_test_proc)

np.mean(np.absolute(preds_model_rms.T-np.array(y_test_proc)))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.round_(np.array(preds_model_rms), decimals=-1), np.round_(np.array(y_test_proc), decimals=-1))
import seaborn as sns

ind = list(range(1920,2030,10))

df_heat = pd.DataFrame(cm, index=ind, columns=ind)

len(ind)

# lab = pd.unique(df_heat[0])

sns.heatmap(df_heat)

df_heat

# df_plot.transpose().corr()

# cm.shape