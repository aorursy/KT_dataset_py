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

from tensorflow.keras.layers import Dense,Input,Dropout,BatchNormalization
df_x_raw = pd.read_csv(r"/kaggle/input/lish-moa/train_features.csv")

df_y_raw = pd.read_csv(r"/kaggle/input/lish-moa/train_targets_scored.csv")
df_x_raw.head()
## one-hot encoding

df_x = pd.get_dummies(df_x_raw,columns = ['cp_type', 'cp_time', 'cp_dose'],drop_first = True)

df_y = df_y_raw.copy()
def df_to_array(df):

    return np.array(df.values)[:,1:]
x_array = df_to_array(df_x)

y_array = df_to_array(df_y)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_array,y_array,test_size=0.7, random_state=42)



n_features = x_train.shape[1]

n_labels = y_train.shape[1]
model = tf.keras.Sequential([

    Dense(512,input_shape = [n_features], activation = "relu" ),

    Dropout(0.25),

    BatchNormalization(),

    Dense(512,activation = "relu"),

    Dropout(0.25),

    BatchNormalization(),

    Dense(n_labels,activation = "sigmoid")

])
model.summary()
model.compile(

    loss = "binary_crossentropy",

    optimizer = "adam"

)
x_train = x_train.astype('float64')

y_train = y_train.astype('float64')

x_test = x_test.astype('float64')

y_test = y_test.astype('float64')
history = model.fit(x_train,y_train,epochs = 10,batch_size = 32,validation_data=(x_test,y_test))
x_submission_raw = pd.read_csv(r"/kaggle/input/lish-moa/test_features.csv")

x_submission = pd.get_dummies(x_submission_raw,columns = ['cp_type', 'cp_time', 'cp_dose'],drop_first = True)
pred = model.predict(x_submission.iloc[:,1:].values)



y_submission = pd.read_csv(r"/kaggle/input/lish-moa/sample_submission.csv")



y_submission.iloc[:,1:] = pred



y_submission.to_csv(r"submission.csv",index = False)
for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))