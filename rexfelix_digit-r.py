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
# preparing data

train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df= pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
sample_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train_df.shape, test_df.shape, sample_df.shape

x_train = train_df.iloc[:,1:]
y_train = train_df.iloc[:,:1]

x_train=np.array(x_train)
y_train=np.array(y_train)
x_test = np.array(test_df)


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

print(np.min(x_train[:,:]), np.max(x_train[:,:]), np.mean(x_train[:,:]), np.std(x_train[:,:]))

from scipy.stats import levene

levene(np.mean(x_train, axis=1), np.mean(x_test, axis=1))


train_mean = np.mean(x_train[:,:],axis=(0,1))
train_std = np.std(x_train[:,:], axis=(0,1))

x_train = (x_train - train_mean)/train_std
x_test = (x_test - train_mean)/train_std
x_val = (x_val - train_mean)/train_std

print(np.mean(x_train[:,:], axis=(0,1)).round(), np.std(x_train[:,:],axis=(0,1)).round())
print(np.mean(x_test[:,:], axis=(0,1)).round(), np.std(x_test[:,:],axis=(0,1)).round())

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# building model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64,input_shape=x_train.shape[1:], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

model.summary()

#from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

#to_TB = TensorBoard(log_dir='logs', histogram_freq=1)
#es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, 
          validation_data=(x_val,y_val))


# check result

import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline

hist_dict = hist.history
hist_df = pd.DataFrame(hist_dict)

plt.style.use('ggplot')

plt.figure(figsize=(5,5))
sns.lineplot(data=hist_df['loss'])
sns.lineplot(data=hist_df['val_loss'])

plt.figure(figsize=(5,5))
sns.lineplot(data=hist_df['acc'])
sns.lineplot(data=hist_df['val_acc'])


# predict & save result

pred = model.predict(x_test)

res = np.argmax(pred, axis=1)
print(res)

res = pd.DataFrame({'ImageId':range(1,len(res)+1), 'label': res})
res

res.to_csv('submission.csv', index=False)
