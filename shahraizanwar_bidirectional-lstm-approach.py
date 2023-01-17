# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/conways-reverse-game-of-life-2020/train.csv')
df.head()
x = df.iloc[:,1].value_counts()
sns.barplot(x.index, x.values)
plt.title('Delta distribution')
plt.show()

def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(625,1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','mae'])
    
    return model

"""
def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(25,25,1)),
    tf.keras.layers.Conv2D(625,3,padding='same'),
    tf.keras.layers.GlobalMaxPooling2D(),
    #tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=2)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','mae'])
    
    return model
"""


def create_dataset(df):
    
    X = df.iloc[:,627:]
    y = df.iloc[:,2:627]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00000001, random_state=42)
    
    X_train = np.expand_dims(X_train,axis=2)
    #X_train = X_train.values.reshape(X_train.values.shape[0],25,25,1)
    
    BUFFER_SIZE = len(X_train)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    return dataset

x = create_model()
x.summary()
## Creating Separate Model for each Delta

BATCH_SIZE = 64
models = []
histories = []
EPOCHS = 15

for delta in range(5):
    
    data = df[df.delta==delta+1]
    dataset = create_dataset(data)
    model = create_model()
    
    print("For Delta: {}".format(delta+1))
    print("______________________________")
    
    history = model.fit(dataset,epochs=EPOCHS)
    
    histories.append(history)
    models.append(model)
df_test = pd.read_csv('../input/conways-reverse-game-of-life-2020/test.csv')

submission_cols = df.iloc[:,2:627].columns
threshold = 0.5
df_created = False

for delta in range(5):
    
    print("Predicting output for Delta: {}".format(delta+1))
    
    data = df_test[df_test.delta==delta+1]
    x = np.expand_dims(data.iloc[:,2:].values,axis=2)
    
    pred = models[delta].predict(x)[:, :, 0]
    pred = np.where(pred>threshold,1,0)
    
    ## Creating Dataframe of Predictions
    if not df_created:
        result = pd.DataFrame(pred,columns=submission_cols)
        id_col = data['id'].reset_index(drop=True)
        result.insert(0,"id",list(id_col.values))
        df_created = True
    else:
        temp = pd.DataFrame(pred,columns=submission_cols)
        id_col = data['id'].reset_index(drop=True)
        temp.insert(0,"id",list(id_col.values))
        result = pd.concat([result,temp], axis=0,ignore_index=True)
result = result.sort_values(by='id')
result.to_csv('submission.csv',index=False)

result
