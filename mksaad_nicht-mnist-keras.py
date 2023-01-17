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

tf.__version__
train_data = pd.read_csv('../input/nicht-mnist/train.csv',header=None, index_col=0)
train_data
train_data[1] = pd.Categorical(train_data[1])

train_data[1] = train_data[1].cat.codes

train_data[1]
df_test = train_data.sample(frac=0.3, random_state=7)

df_train = train_data.drop(df_test.index)
y_train = df_train.iloc[:,0]

x_train = df_train.iloc[:,1:]

y_val = df_train.iloc[:,0]

x_val = df_train.iloc[:,1:]
x_train
y_train.value_counts()
len(y_train.value_counts())
x_train = x_train / 255.0 

x_val = x_val / 255.0 
x_train
x_train.shape
x_train_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28) for i, r in x_train.iterrows()] ] )
x_val_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28) for i, r in x_val.iterrows()] ] )
x_train_np
x_train_np.shape
model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(28, 28)),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10)

])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',

              loss=loss_fn,

              metrics=['accuracy'])
model.fit(x_train_np, y_train, epochs=10)
model.evaluate(x_val,  y_val, verbose=2)
test_data = pd.read_csv('../input/nicht-mnist/test.csv',header=None, index_col=0)

test_data
test_data = test_data / 255.0

test_data
x_test_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28) for i, r in test_data.iterrows()] ] )
model.predict(x_test_np)
preds = np.argmax(model.predict(x_test_np), axis=1).tolist()
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

pred_labes = pd.Series([class_labels[p] for p in preds])

pred_labes
out_df = pd.DataFrame({

    'Id': test_data.index,

    'target': pred_labes

})

out_df
out_df.to_csv('my_submission.csv', index=False)