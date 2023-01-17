import os

import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from keras import backend as K

from keras import layers as L

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

from keras.regularizers import l1_l2

import matplotlib.pyplot as plt
data = pd.read_csv('../input/machine-learning-for-diabetes-with-python/diabetes_data.csv')

scaler = MinMaxScaler()



y = data.pop('Outcome')

X = pd.DataFrame(scaler.fit_transform(data))



X_test = X[-80:]

y_test = y[-80:]



X = X.drop(X.index[-80:])

y = y.drop(y.index[-80:])



display(len(X))

display(X.head())
rf_clf = RandomForestClassifier(n_estimators = 200, max_depth=16)

rf_clf.fit(data[data.columns[:-1]], data[data.columns[-1]])

pd.Series(rf_clf.feature_importances_, index = data.columns[:-1]).nlargest(12).plot(kind='barh',figsize=(10,10),title = 'Feature importance').invert_yaxis()
K.clear_session()

reduce_lr = ReduceLROnPlateau(patience=5,verbose=False)

model_ckpt = ModelCheckpoint('DiabetesNet.h5',save_best_only=True,verbose=False)

early_stop = EarlyStopping(patience=8,verbose=False)



entry = L.Input(shape=(len(X.columns),))

x = L.GaussianNoise(0.2)(entry)

x = L.Dense(69,activation='linear')(x)

x = L.LeakyReLU(0.4)(x)

x = L.Dense(42,activation='linear')(x)

x = L.LeakyReLU(0.4)(x)

x = L.Dense(9,activation='linear',kernel_regularizer=l1_l2(2e-4))(x)

x = L.LeakyReLU(0.3)(x)

x = L.Dense(1,activation='hard_sigmoid')(x)



model = Model(entry,x)

model.compile(loss='mse',optimizer=Adam(lr=1e-4),metrics=['accuracy'])

history = model.fit(X,y,epochs=666,verbose=0,callbacks=[reduce_lr,model_ckpt,early_stop],steps_per_epoch=200,validation_steps=50,validation_split=0.3)

print('roc/auc: {}'.format(roc_auc_score(y_test,model.predict(X_test))))
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='lower right')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()