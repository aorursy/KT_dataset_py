# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 
# Any results you write to the current directory are saved as output.
data = pd.read_csv(r'../input/predicting-a-pulsar-star/pulsar_stars.csv')
data.head()
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
print('the shape of the data is:', data.shape)
data.describe()
data.head()
data.isna().sum()
data_correlate = data.corr()
plt.figure(figsize = (12, 9))
sns.heatmap(data_correlate, linecolor = 'black', linewidth = 1, annot = True)
plt.title('correlation of the data')
plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(copy = True)
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled)
x_data = data_scaled[data_scaled.columns[range(0,len(data.columns)-1)]]
y_data = data_scaled[data_scaled.columns[len(data.columns)-1]]
from sklearn.feature_selection import SelectKBest, chi2
select_k_best = SelectKBest(chi2, 5)
select_k_best.fit(x_data, y_data)
print(select_k_best.scores_)
x_data = x_data[x_data.columns[select_k_best.scores_>200]]
data_final = x_data.join(y_data)
data_final.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 42)
from keras import models
from keras import layers
#building the model
model = models.Sequential()
model.add(layers.Dense(7, activation = 'relu', input_shape = (5,)))
model.add(layers.Dense(1, activation = 'sigmoid'))
from keras import optimizers
from keras import losses
from keras import metrics
model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
#fitting the model
history = model.fit(x_train, y_train, epochs = 50, validation_split = 0.5)
hist_dict = history.history
loss_values = hist_dict['loss']
val_loss_values = hist_dict['val_loss']
epochs = range(1, 51)
plt.figure(figsize = (10, 4))
plt.plot(epochs, loss_values, 'bo', label = 'training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
y_predict = model.predict(x_test)
y_predict
y_test = np.array(y_test).reshape(-1,1)
from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_test, y_predict)
print('the mse obtained is:',error)
y_test = y_test.astype(float)
y_predict = y_predict.astype(float)
from keras import metrics
bin_acc = metrics.binary_accuracy
acc = bin_acc(y_test.reshape(1,-1), y_predict.reshape(1,-1))
from keras.utils import plot_model
import pydot
plot_model(model, to_file ='model_plot.png', show_shapes=True, show_layer_names=True)
