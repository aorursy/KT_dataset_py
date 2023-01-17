# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_train.shape
y = df_train.label
X = df_train.drop(columns="label")
X.head()
y.head()
plt.imshow(np.reshape(X.values[0], newshape=[28,28]) );
X = X/255.   # V.V.I: Normalization
X.values[1].shape
from keras.models import Model, Input
from keras.layers import LSTM, Bidirectional, Dense, Concatenate, GlobalMaxPooling1D
ip_tensor = Input(shape=(28,28))    # Instantiate Keras tensor
t = Bidirectional(LSTM(units=15, activation="relu", return_sequences=True))(ip_tensor)
t = GlobalMaxPooling1D()(t)
ip_tensor_T = Input(shape=(28,28))
t_T = Bidirectional(LSTM(15, activation="relu", return_sequences=True))(ip_tensor_T)
t_T = GlobalMaxPooling1D()(t_T)
t_T.shape
t_concat = Concatenate(axis=-1)([t,t_T])
t_concat.shape
L1 = Dense(units=10, activation="softmax")(t_concat)
my_model = Model([ip_tensor, ip_tensor_T], L1)
from keras.optimizers import Adam
my_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
my_model.summary()
X_reshaped = X.values.reshape([-1,28,28])
#plt.imshow(X_reshaped[20,:,:])
plt.imshow(np.rot90(X_reshaped, axes=(1,2))[20,:,:])
trace = my_model.fit([X_reshaped, np.rot90(X_reshaped, axes=(1,2))], y, batch_size=300,
                     epochs=50,validation_split=0.3)
df_sample_submission = pd.read_csv('../input/sample_submission.csv')
df_test = pd.read_csv('../input/test.csv')
df_sample_submission
df_test = df_test/255.
test_reshaped = df_test.values.reshape([-1,28,28])
pred = my_model.predict([test_reshaped, np.rot90(test_reshaped, axes=(1,2))])
df_test_pred = pd.DataFrame({"ImageId":range(1,pred.shape[0]+1),  "Label":pred.argmax(axis=1)})
df_test_pred
df_test_pred.to_csv("submission.csv")