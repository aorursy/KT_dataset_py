import numpy as np

import pandas as pd 

import seaborn as sns

import os

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print(data.shape)
data
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(test_data.shape)
test_data
sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

print(sample_submission.shape)
sample_submission
train_data = data[:]

val_data = data[40000:]

train_label = np.float32(train_data.label)

val_label = np.float32(val_data.label)

train_image = np.float32(train_data[train_data.columns[1:]])

val_image = np.float32(val_data[val_data.columns[1:]])

test_image = np.float32(test_data[test_data.columns])

print('train shape: %s'%str(train_data.shape))

print('val shape: %s'%str(val_data.shape))

print('train_label shape: %s'%str(train_label.shape))

print('val_label shape: %s'%str(val_label.shape))

print('train_image shape: %s'%str(train_image.shape))

print('val_image shape: %s'%str(val_image.shape))

print('test_image shape: %s'%str(test_image.shape))
g = sns.countplot(train_label)

g
plt.imshow(train_image[13].reshape(28,28))

plt.show()

print(train_image[13].shape)



train_image = train_image/255.0

val_image = val_image/255.0

test_image = test_image/255.0



train_image = train_image.reshape(train_image.shape[0],28,28,1)

val_image = val_image.reshape(val_image.shape[0],28,28,1)

test_image = test_image.reshape(test_image.shape[0],28,28,1)

print('train_image shape: %s'%str(train_image.shape))



print('train_image shape: %s'%str(train_image.shape))

print('val_image shape: %s'%str(val_image.shape))



train_label1 = train_label

val_label1 = val_label

from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder(sparse=False,categories='auto')

yy = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]

encoder.fit(yy)

# transform

train_label = train_label.reshape(-1,1)

val_label = val_label.reshape(-1,1)





train_label = encoder.transform(train_label)

val_label = encoder.transform(val_label)







print('train_label shape: %s'%str(train_label.shape))

print('val_label shape: %s'%str(val_label.shape))




import numpy as np

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import SGD

from keras.layers.normalization import BatchNormalization

from keras.layers import LeakyReLU



model = Sequential()

# input: 28x28 images with 1 channels -> (28, 28, 1) tensors.

# this applies 32 convolution filters of size 3x3 each.

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))





model.add(Conv2D(128, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(128, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))





model.add(Conv2D(256, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(256, kernel_size=5, activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))





model.add(Flatten())







#model = keras.applications.inception_v3.InceptionV3(weights= None, include_top=False, input_shape= (28,28,1))

model.add(Dense(256, activation='relu', name='my_dense'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(loss='categorical_crossentropy', optimizer=sgd)



model.summary()
from keras.models import Model

layer_name='my_dense'

intermediate_layer_model = Model(inputs=model.input,

                                 outputs=model.get_layer(layer_name).output)



intermediate_layer_model.summary()
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range = 15,

    horizontal_flip = False,

    zoom_range = 0.20)
from keras.optimizers import Adam, Adadelta, RMSprop



model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])



datagen.fit(train_image)



# training

history = model.fit_generator(datagen.flow(train_image,train_label, batch_size=32),

                              epochs = 75,

                              shuffle=True,

                              validation_data = (val_image,val_label),

                              verbose = 1,

                              steps_per_epoch=train_image.shape[0] // 32)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()





plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
intermediate_output = intermediate_layer_model.predict(train_image) 

intermediate_output = pd.DataFrame(data=intermediate_output)
val_data = intermediate_output[40000:]
submission_cnn = model.predict(test_image)
intermediate_test_output = intermediate_layer_model.predict(test_image)

intermediate_test_output = pd.DataFrame(data=intermediate_test_output)
from xgboost import XGBClassifier



xgbmodel = XGBClassifier(objective='multi:softprob', 

                      num_class= 10)

xgbmodel.fit(intermediate_output, train_label1)

xgbmodel.score(val_data, val_label1)
submission_xgb = xgbmodel.predict(intermediate_test_output)
submission_cnn = submission_cnn.astype(int)

submission_xgb = submission_xgb.astype(int)

submission_cnn

label = np.argmax(submission_cnn,1)

id_ = np.arange(0,label.shape[0])

label
#if (xgbmodel.score(val_data, val_label1) > )

final_sub = submission_xgb
save = pd.DataFrame({'ImageId':sample_submission.ImageId,'label':final_sub})

print(save.head(10))

save.to_csv('submission.csv',index=False)