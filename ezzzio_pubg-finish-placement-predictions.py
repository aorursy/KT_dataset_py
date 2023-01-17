import pandas as pd

import numpy as np



dataset_train = pd.read_csv('../input/train_V2.csv')

dataset_test = pd.read_csv('../input/test_V2.csv')
from IPython.display import display



dataset_train = dataset_train.dropna()

dataset_train = dataset_train.drop(labels=['Id','groupId','matchId'],axis=1)
dataset_train = pd.get_dummies(dataset_train,drop_first=True)

dataset_train_y = dataset_train.loc[:,'winPlacePerc']

dataset_train_x = dataset_train.drop(labels='winPlacePerc',axis=1)

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



dataset_train_x = scaler.fit_transform(dataset_train_x)


display(dataset_train_y.head())
from keras.models import Sequential

from keras.layers import Dense,InputLayer,Dropout

from keras.callbacks import ReduceLROnPlateau

from keras.utils import normalize,to_categorical



reduceLr = ReduceLROnPlateau(monitor='loss', factor=0.01,patience=5, min_lr=0.00001)







input_layer = InputLayer(input_shape = (39,))

model = Sequential()

model.add(input_layer)

model.add(Dense(128,activation = "relu"))

model.add(Dense(256,activation = "relu"))

model.add(Dense(128,activation = "relu"))

model.add(Dense(256,activation = "relu"))

model.add(Dense(128,activation = "relu"))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer = "adadelta",loss="binary_crossentropy",metrics=["mae","accuracy"])

model.summary()

history = model.fit(dataset_train_x,dataset_train_y,epochs=10,batch_size= 5000,callbacks=[reduceLr])
import seaborn as sns

sns.lineplot(history.epoch,history.history['loss'])
dataset_test_id = dataset_test.loc[:,"Id"]

dataset_test = dataset_test.dropna()

dataset_test = dataset_test.drop(labels=['Id','groupId','matchId'],axis=1)

dataset_test = pd.get_dummies(dataset_test,drop_first=True)

display(dataset_test.head())

scaler = StandardScaler()

dataset_test = scaler.fit_transform(dataset_test)
output = model.predict(dataset_test)



data_file = pd.DataFrame({"Id":dataset_test_id,"winPlacePerc":pd.Series(output.reshape(-1,))})
display(output)
data_file.to_csv("pubg_predictions.csv",index = False)