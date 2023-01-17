import pandas as pd;

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import BatchNormalization

from keras.optimizers import Adam, SGD, Adagrad

from keras.layers.core import Activation

from keras.callbacks import LambdaCallback

from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split  

from sklearn import metrics

from keras.callbacks import ModelCheckpoint,EarlyStopping
pd.set_option('display.max_columns', 500)
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_test.head()
temp = pd.DataFrame(df_train.isnull().sum()).reset_index()

temp.columns = ["variable", "null_count"]

temp[temp["null_count"]>0]

list_nan_variables = temp[temp["null_count"]>0]["variable"]

for c in list_nan_variables:

    df_train[c].fillna(df_train[c].mean(), inplace=True)
data_train, data_test = train_test_split(df_train, test_size=0.20, random_state=21)
listofcolumns = data_train.columns
columns_move_train = list(listofcolumns)

columns_move_train.remove('target')



y_train = data_train["target"].values

X_train = data_train[columns_move_train].values

y_test = data_test["target"].values

X_test = data_test[columns_move_train].values
scaler = MinMaxScaler(feature_range=(0, 1))

rescaledX_train = scaler.fit_transform(X_train[:,1:])

rescaledX_test = scaler.transform(X_test[:,1:])
model = Sequential()

model.add(Dense(2000, input_dim=200, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(1600, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(400, activation='relu'))

model.add(Dense(20, activation='relu'))

model.add(Dense(1, activation='sigmoid'))





opti = Adam(lr=0.0001)

model.compile(optimizer=opti, loss='binary_crossentropy')



model.summary()



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)



checkpointer = ModelCheckpoint(

    filepath='model.hdf5'

    , monitor = "val_loss"

    , verbose=1

    , save_best_only=True

    , mode = 'auto'

)



score = model.fit(rescaledX_train, y_train, epochs=1000, verbose=2, batch_size=1000, validation_data=(rescaledX_test, y_test)

                  , callbacks=[es, checkpointer])
from keras.models import load_model

model = load_model("model.hdf5") 
losses = score.history['loss']

val_losses = score.history['val_loss']

plt.figure(figsize=(10,5))

plt.plot(losses, label="trainset")

plt.plot(val_losses, label="testset")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
y_pred = model.predict(rescaledX_test)
y_test = y_test.reshape(len(y_test),1)

y_pred = y_pred.reshape(len(y_pred),1)

resultarray = np.append(X_test, y_test, axis=1)

resultarray = np.append(resultarray, y_pred, axis=1)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

metrics.auc(fpr, tpr)
columns_result_analiz = list(columns_move_train)

columns_result_analiz.append("y_test")

columns_result_analiz.append("y_pred")



resultanaliz = pd.DataFrame(resultarray, columns=columns_result_analiz)
resultanaliz.sort_values(by="y_test", ascending=False).head(10)
a = list(resultanaliz["y_test"])

b = list(resultanaliz["y_pred"])
plt.figure(figsize=(14,7))

sns.distplot(a);

sns.distplot(b);

plt.grid(axis='y')

plt.xlabel('Value')

plt.ylabel('Frequency')

plt.title('Actuals & Predictions')
X_submit = df_test[columns_move_train].values

rescaledX_submit = scaler.transform(X_submit[:,1:])

y_sub_pred = model.predict(rescaledX_submit, verbose=2)

y_sub_pred = y_sub_pred.reshape(len(y_sub_pred),1)

resultarray = np.append(X_submit, y_sub_pred, axis=1)
columns_submit = list(columns_move_train)

columns_submit.append("target")



resultdf = pd.DataFrame(resultarray, columns=columns_submit)
columns_to_be_dropped = list(columns_submit)

columns_to_be_dropped.remove("ID_code")

columns_to_be_dropped.remove("target")



resultdf = resultdf.drop(columns_to_be_dropped, axis=1)
resultdf.head()
resultdf.to_csv("ann_v3.csv", index=False)