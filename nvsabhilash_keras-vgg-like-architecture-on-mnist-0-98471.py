import numpy as np
import pandas as pd

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D 
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

import matplotlib.pyplot as plt
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
df_train.head()
y_train_temp = to_categorical(df_train.iloc[:, 0].values, num_classes=10)
x_train_temp = df_train.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
permut = np.random.permutation(x_train_temp.shape[0])

x_train = x_train_temp[permut]
y_train = y_train_temp[permut]
print("shape of X: {}, shape of y: {}".format(x_train.shape, y_train.shape))
# Print the mean image
mean_image = np.mean(x_train, axis=0).astype(np.uint8).reshape(28, 28)
plt.imshow(mean_image, cmap='gray')
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='sigmoid'))

model.summary()
adam = Adam(lr=1e-4, decay=1e-6)
model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])
!mkdir models
!mkdir logs
tensorboard = TensorBoard(write_grads=True, write_images=True)
chkpoint = ModelCheckpoint("models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard, chkpoint], validation_split=0.2)
adam = Adam(lr=1e-5, decay=1e-6)
model.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, callbacks=[tensorboard, chkpoint], validation_split=0.2)
# best_model = load_model('models/weights.01-0.05.hdf5') (Skipping for now as this is a mannual process)
best_model = model
x_test = df_test.values.reshape(-1, 28, 28, 1)
print('Test set: {}'.format(x_test.shape))
probs = best_model.predict(x_test, verbose=1)
preds = np.argmax(probs, axis=1)

print("Predictions: {}".format(preds.shape))
submission = pd.DataFrame({'ImageId': np.arange(1, len(preds)+1), 'Label': preds})
submission.to_csv('submission_0.05.csv', index=False)
