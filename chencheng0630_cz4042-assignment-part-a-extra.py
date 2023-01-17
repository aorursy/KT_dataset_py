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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import time
# Set up random seed
seed = 17
np.random.seed(seed)
tf.random.set_seed(seed)
# Read raw csv data
data = pd.read_csv("../input/ctg-data/ctg_data_cleaned.csv")
data.head(5)
data['NSP'].value_counts()
data.describe()
NUM_CLASS = 3
# Convert the target variable to one-hot vector
def one_hot_encoding(dataframe):
    encoder = OneHotEncoder(sparse=False)
    dataframe = dataframe.values.reshape(-1, 1)
    data = encoder.fit_transform(dataframe)
    return encoder, data
# Scale the input data
def scale(dataframe):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(dataframe)
    return scaler, data
# Split the data for X and y. Drop unnecessary columns.
X_dataframe = data.iloc[:, :21]
y_dataframe = data.iloc[:, -1]
scaler, X_dataframe = scale(X_dataframe)
encoder, y_dataframe = one_hot_encoding(y_dataframe)
data["NSP"].head(5)
# Checking the result of one-hot encoding
y_dataframe[:5]
# Split the data into 70:30 training set and test set. 
# Preserve class distribution through stratified splitting. 
X_train, X_test, y_train, y_test = train_test_split(X_dataframe, y_dataframe, test_size=0.3, stratify=y_dataframe, random_state = 17)
def fit_baseline_model(X_train, y_train, X_test, y_test, batch_size, verbose, hidden_neurons, decay_rate, epochs):

    model = models.Sequential()

    model.add(layers.Dense(hidden_neurons, activation='relu', 
                           input_shape=(X_train.shape[1],),
                           kernel_regularizer=keras.regularizers.l2(decay_rate)))
    
    model.add(layers.Dense(NUM_CLASS, activation='softmax',
                          kernel_regularizer=keras.regularizers.l2(decay_rate)))

#     opt = keras.optimizers.SGD(learning_rate=0.01)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    
    model.compile(optimizer=opt,
                 loss= keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])

#     model.summary()

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        shuffle=True,
                        validation_data=(X_test, y_test))
    
    return model, history
    
model, history = fit_baseline_model(X_train, y_train, X_test, y_test, 
                             batch_size=32, 
                             epochs=400, 
                             verbose=0,
                             hidden_neurons=25,
                             decay_rate=1e-6)
plt.plot(history.history['accuracy'], label='Train_Accuracy')
plt.plot(history.history['val_accuracy'], label='Test_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epochs')
plt.title('Training Accuracy and Test Accuracy')
plt.legend()
plt.savefig('AQ1a')
plt.plot(history.history['loss'], label='Train_Error')
plt.plot(history.history['val_loss'], label='Test_Error')
plt.ylabel('Error')
plt.xlabel('No. of Epochs')
plt.title('Training Error and Test Error')
plt.legend()
plt.savefig('AQ1b')
y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
plt.savefig("Extra-1")
# Split the data for X and y. Drop unnecessary columns.
X_dataframe = data.iloc[:, :21]
y_dataframe = data.iloc[:, -1]
scaler, X_dataframe = scale(X_dataframe)
# encoder, y_dataframe = one_hot_encoding(y_dataframe)
y_dataframe = y_dataframe.values - 1
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_dataframe),
                                                 y_dataframe)
X_train, X_test, y_train, y_test = train_test_split(X_dataframe, y_dataframe, test_size=0.3, stratify=y_dataframe, random_state = 17)
class_weights_dict = dict(enumerate(class_weights))
decay_rate = 1e-6
hidden_neurons = 25
batch_size = 32
epochs = 400

model = models.Sequential()

model.add(layers.Dense(hidden_neurons, activation='relu', 
                       input_shape=(X_train.shape[1],),
                       kernel_regularizer=keras.regularizers.l2(decay_rate)))

model.add(layers.Dense(NUM_CLASS, activation='softmax',
                      kernel_regularizer=keras.regularizers.l2(decay_rate)))

#     opt = keras.optimizers.SGD(learning_rate=0.01)
opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt,
             loss= keras.losses.SparseCategoricalCrossentropy(),
             metrics=['accuracy'])

#     model.summary()

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    shuffle=True,
                    class_weight = class_weights_dict,
                    validation_data=(X_test, y_test))

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

y_true = y_test
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
plt.savefig("Extra-2")
np.array(np.unique(y_dataframe, return_counts=True)).T
import imblearn
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X_dataframe, y_dataframe, 
                                                    test_size=0.3, 
                                                    stratify=y_dataframe, 
                                                    random_state = 17)

smote = SMOTE(sampling_strategy='not majority')
X_sm, y_sm = smote.fit_sample(X_train, y_train)
np.array(np.unique(y_sm, return_counts=True)).T
decay_rate = 1e-6
hidden_neurons = 25
batch_size = 32
epochs = 2000

model = models.Sequential()

model.add(layers.Dense(hidden_neurons, activation='relu', 
                       input_shape=(X_train.shape[1],),
                       kernel_regularizer=keras.regularizers.l2(decay_rate)))

model.add(layers.Dense(NUM_CLASS, activation='softmax',
                      kernel_regularizer=keras.regularizers.l2(decay_rate)))

#     opt = keras.optimizers.SGD(learning_rate=0.01)
opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt,
             loss= keras.losses.SparseCategoricalCrossentropy(),
             metrics=['accuracy'])

#     model.summary()

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data = (X_test, y_test))
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

y_true = y_test
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
plt.savefig("Extra-3")
