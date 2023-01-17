import numpy as np
import pandas as pd

from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/diabetes.csv')
df.head()
x_all = df.drop(['Outcome'], axis=1)

labels = df['Outcome'].values
y_all = np_utils.to_categorical(labels)
class_weights = compute_class_weight('balanced', np.unique(labels), labels)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.33, random_state=42)
dropout = 0.2
epochs = 10000
batch_size = 8

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(dropout))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(dropout))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
history = model.fit(x_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    class_weight=class_weights,
                    validation_split=0.1,
                    callbacks=[checkpoint, reduce_lr])
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Model score: ', score)