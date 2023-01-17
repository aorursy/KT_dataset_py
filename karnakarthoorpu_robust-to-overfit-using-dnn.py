# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.set_printoptions(suppress=True)



from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/dont-overfit-ii/train.csv')

test_df = pd.read_csv('/kaggle/input/dont-overfit-ii/test.csv')

sample_df = pd.read_csv('/kaggle/input/dont-overfit-ii/sample_submission.csv')

train_df.drop(['id'],inplace = True,axis = 1)

print('train = {0},test = {1}, sample = {2}'.format(train_df.shape,test_df.shape,sample_df.shape))
train_df.head()
train_df.target.unique()
test1_df = test_df.copy()

test1_df['target'] = -1.0
full_df = train_df.append(test1_df,sort = False)
full_df.shape
correlation_mat = full_df.corr()

corr_np = correlation_mat.to_numpy()

np.fill_diagonal(corr_np,0)

np.where(corr_np > 0.7)
train_df.target.value_counts()
train_df.target.value_counts().plot(kind = 'bar')
X_train, X_test, y_train, y_test = train_test_split(train_df.loc[:, train_df.columns != 'target'], train_df.loc[:,'target'])
def normalize(array):

    return (array - array.mean())/array.std()
train_df_norm = normalize(train_df)
X_train_norm = normalize(X_train)

y_train_norm = normalize(y_train)
(X_train_norm.shape)
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import RMSprop

class mycallbacks(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if logs.get('accuracy') and logs.get('accuracy') >= 0.998:

            print('Reached 99% accuracy so cancelling training!')

            self.model.stop_training = True

callbacks = mycallbacks()

model = keras.Sequential([

                    keras.layers.Dense(units = 8 ,input_shape = [X_train.shape[1]], activation = 'relu'),

                    keras.layers.Dropout(0.1),

                    keras.layers.Dense(units = 16 ,input_shape = [X_train.shape[1]], activation = 'relu'),

                    keras.layers.Dropout(0.1),

                    keras.layers.Dense(units = 32 ,input_shape = [X_train.shape[1]], activation = 'relu'),

                    keras.layers.Dropout(0.1),

                    keras.layers.Dense(units = 64 ,input_shape = [X_train.shape[1]], activation = 'relu'),

                    keras.layers.Dropout(0.1),

                    keras.layers.Dense(units = 1, activation = 'sigmoid')])

model.compile(optimizer = RMSprop(lr = 0.01),loss = 'binary_crossentropy',metrics=['accuracy'])

model.summary()
history = model.fit(train_df.loc[:, train_df.columns != 'target'],train_df.loc[:,'target'],validation_split= 0.4,epochs = 150)
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
predictions = model.predict(test_df.loc[:,test_df.columns != 'id'])

prediction = pd.DataFrame(predictions, columns=['target'])

pd.concat([test_df[['id']].astype('int'),prediction[['target']].astype('int')], axis=1).to_csv('results.csv',header = True,index = False)