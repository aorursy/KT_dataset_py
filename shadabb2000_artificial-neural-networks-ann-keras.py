import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv("../input/Churn_Modelling.csv", index_col='RowNumber')

df.head()

df_drop = df.drop(['CustomerId','Surname'],axis=1) ## Removing surname as onhot encoding will cause issues for each one of them

df_drop.info()
## Label Encoding of all the columns

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()



# Categorical boolean mask

categorical_feature_mask = df_drop.dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = df_drop.columns[categorical_feature_mask].tolist()

df_drop[categorical_cols] = df_drop[categorical_cols].apply(lambda col: le.fit_transform(col))

print(df_drop.info())
df_drop.head()
from scipy.stats import zscore

df_scaled = df_drop.apply(zscore)

X_columns =  df_scaled.columns.tolist()[1:10]

Y_Columns = df_drop.columns.tolist()[-1:]



X = df_scaled[X_columns].values # Credit Score through Estimated Salary

Y = np.array(df_drop['Exited']) # Exited



print(Y)

print(X)
from sklearn.model_selection import train_test_split

# print(Y)



# Y = Y.astype('bool_')

# print(Y.dtype)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=8)
from tensorflow.keras.utils import to_categorical

#Encoding the output class label (One-Hot Encoding)

y_train=to_categorical(y_train,2)

y_test=to_categorical(y_test,2)
import tensorflow as tf

from tensorflow.keras import models

from tensorflow.keras.layers import Dense

#Initialize Sequential Graph (model)

model = tf.keras.Sequential()
model.add(Dense(18, activation='relu', input_shape=(9,)))

model.add(Dense(20, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=25, validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test,verbose=1)



print(score)