# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn import preprocessing

import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
src='/kaggle/input/airplane-accidents-severity-dataset/'

print(src)

dest=os.getcwd()

dest
train_df=pd.read_csv(src+"train.csv")

test_df=pd.read_csv(src+"test.csv")

sample=pd.read_csv(src+"sample_submission.csv")
df=train_df.copy()

df_normalize=train_df.copy()
train_df.head()
test_df.head()
print(df.shape, test_df.shape)
df.info()
df.describe()
df.isna().sum()
test_df.info()
x=df.drop(['Severity', 'Accident_ID'], axis=1)

x_test=test_df.drop(['Accident_ID'],axis=1)

y_train=df['Severity']
y_train.unique()
x_n=df_normalize.drop(['Severity','Accident_ID'],axis=1)

y_n=df_normalize['Severity']
x_n.describe()
y_n.value_counts()
#x_no=x_n[:].values.astype("float64")

#x_no=x_n
#x_test=x_test[:].values.astype("float64")
#x_test=preprocessing.normalize(x_test)
x_m = x.values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x_m)

X_train_final = pd.DataFrame(x_scaled)

X_test_final=pd.DataFrame(min_max_scaler.fit_transform(x_test.values))
x_no = X_train_final

class_map = {

    'Minor_Damage_And_Injuries': 0,

    'Significant_Damage_And_Fatalities': 1,

    'Significant_Damage_And_Serious_Injuries': 2,

    'Highly_Fatal_And_Damaging': 3

}

inverse_class_map = {

    0: 'Minor_Damage_And_Injuries',

    1: 'Significant_Damage_And_Fatalities',

    2: 'Significant_Damage_And_Serious_Injuries',

    3: 'Highly_Fatal_And_Damaging'

}
y_no = y_n.map(class_map)

print(x_no.shape,y_no.shape)
from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

# integer encode

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(y_no)

# binary encode

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)
y_no=onehot_encoded

print(y_no.shape)
print(x_no.shape)
#random shuffle

x_r=np.random.shuffle(x_no)
#splitting into train and val

x_train=x_no[:6000]

x_val=x_no[6000:]

print(x_train.shape,x_val.shape)
y_train=y_no[:6000]

y_val=y_no[6000:]


# One-Hot Encode

#y_no=keras.utils.to_categorical(y_no,4)

#y_no
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten
model=Sequential()

model.add(Dense(12,activation='relu',input_dim=10))

model.add(Dense(8,activation='relu'))

model.add(Dense(4,activation='sigmoid'))

print(model.summary())
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#X=X_train_final
history=model.fit(x_train,y_train,epochs=50,batch_size=10, validation_data=(x_val, y_val))
history_dict=history.history

print(history_dict.keys())
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
pred_test=np.argmax(model.predict(X_test_final),axis=1)

pred_test
submission = pd.DataFrame([test_df['Accident_ID'], np.vectorize(inverse_class_map.get)(pred_test)], index=['Accident_ID', 'Severity']).T

submission.to_csv('/kaggle/working/submission_keras1.csv', index=False)
