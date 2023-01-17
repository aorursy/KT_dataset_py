import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb              # convenient plotting functionality
data_train = pd.read_json('../input/train.json')
data_test  = pd.read_json('../input/test.json')
print("Number of training samples: \t", data_train.shape[0])
print("Number of test samples: \t", data_test.shape[0])
data_train.head()
sb.countplot(data_train['is_turkey'])
# Above you can see the audio_embedding field of the dataframe has the data
# For example,
image = np.array(data_train['audio_embedding'][0])
plt.imshow(image)
plt.colorbar()
image = np.array(data_train['audio_embedding'][100])
plt.imshow(image)
plt.colorbar()
# So what we're really being provided here are spectrograms.
# Let's make sure all the data is the same size, in terms of length and width.

data_train['length'] = data_train['audio_embedding'].apply(len)
plt.yscale('log')
sb.countplot('length', hue='is_turkey', data=data_train)
plt.show()
new_length   = 10
feature_size = 128

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
X = pad_sequences(data_train['audio_embedding'], maxlen=new_length, padding='post')
X_test = pad_sequences(data_test['audio_embedding'], maxlen=new_length, padding='post')
plt.imshow(X_test[0])
# define the target variable
y = data_train['is_turkey'].values
X.shape
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Reshape, GlobalMaxPooling1D, GlobalAveragePooling1D, Input, concatenate, BatchNormalization, Dense, Conv2D, MaxPooling2D, Flatten,Activation,Embedding
from keras.optimizers import Adam
model = Sequential()

model.add(Conv2D(128*2,(3,3), input_shape=(10,128,1) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))


model.add(Dropout(0.1))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
# reshape the training data for the 2d Conv Net
n_images = X.shape[0]
X_reshaped = X.reshape(n_images, 10, 128, 1)

n_images_test = X_test.shape[0]
X_test_reshaped = X_test.reshape(n_images_test, 10, 128, 1)
model.summary()
# save the randomly initialized weights
model.save_weights('model.h5')
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.1, patience=2, verbose=1, min_lr=1e-8)
model.fit(X_reshaped, y, epochs=20, batch_size=256, verbose=2, callbacks=[reduce_lr])
y_test = model.predict(X_test_reshaped, verbose=1)
submission = pd.DataFrame({'vid_id': data_test['vid_id'].values,
                           'is_turkey (pred)': list(y_test.flatten()) })
submission.head()
submission.to_csv("submission.csv", index=False)
from sklearn.model_selection import train_test_split 
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y, test_size=0.2)
# load the original model weights
model.load_weights('model.h5')
# train the model
history = model.fit(X_train, y_train, batch_size=256, epochs=20, validation_data=[X_val, y_val], callbacks=[reduce_lr], verbose=2)
from sklearn.metrics import accuracy_score

y_pred_val = model.evaluate(X_val, y_val, verbose=1)
print("Train accuracy : ", y_pred_val[1])
