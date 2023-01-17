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
from tensorflow.keras import layers
import matplotlib.pyplot as plt
print(tf.__version__)
!pip install git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
df = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', header=0)
msk = np.random.rand(len(df)) < 0.8
train_data = df[msk]
test_data = df[~msk]
true_salesprice = test_data['SalePrice']
test_data['SalePrice'] = np.NaN
data = pd.concat([train_data,test_data])
data.tail()
numeric_cols = data.dtypes[data.dtypes != "object"].index  # Getiing non-categorical variables
numeric_cols = [col for col in numeric_cols if col not in ['Id'] ]
mean_cols = data[numeric_cols].mean()
std_cols = data[numeric_cols].std()
#Normalized the numeric Data
numeric_cols = data.dtypes[data.dtypes != "object"].index  # Getiing non-categorical variables
numeric_cols = [col for col in numeric_cols if col not in ['Id'] ]
data[numeric_cols]=(data[numeric_cols]-data[numeric_cols].mean())/data[numeric_cols].std()
data.head()
#Remove columns having 'na' Values
SalePrice = data.SalePrice
data.dropna(axis=1,inplace=True)
data['SalePrice'] = SalePrice
# Creating dummies for categoricol variables
categorical_features = [col for col in data.columns if col not in numeric_cols and col not in ['Id','SalePrice'] ]
data = pd.get_dummies(data,columns =categorical_features)
train_set = data.loc[data.SalePrice.notna()]
test_set = data.loc[data.SalePrice.isna()]
test_set.head()
train_X = train_set[train_set.columns[train_set.columns.values!='SalePrice']]
train_y = train_set.SalePrice
train_X.drop('Id',axis=1,inplace=True)
test_X = test_set[test_set.columns[test_set.columns.values!='SalePrice']]
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='tanh', input_shape=[len(train_X.keys())]),
    layers.Dense(64, activation='tanh'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()
model.summary()
EPOCHS = 200

history = model.fit(
  train_X, train_y,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [salesprice]')
submission=pd.DataFrame(columns=['Id','SalePrice'])
submission['Id']=test_set['Id']
test_X.drop('Id',axis=1,inplace=True)
submission['SalePrice']=model.predict(test_X)
submission['SalePrice'] = (submission['SalePrice'] * std_cols['SalePrice']) + mean_cols['SalePrice']
submission.head()
train_y = (train_y * std_cols['SalePrice']) + mean_cols['SalePrice'] 
plt.plot(true_salesprice)
plt.plot(submission['SalePrice'])
