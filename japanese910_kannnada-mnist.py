import numpy as np

import pandas as pd

import datetime



from sklearn.model_selection import train_test_split



from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Lambda, LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

Dig = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

train_data = train.append(Dig)

y_train = np.array(train_data["label"])

y_train = to_categorical(y_train)

x_train = train_data.drop("label",axis=1).values

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_train.shape, y_train.shape
X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.10, random_state=0) 



train_datagen = ImageDataGenerator(rescale = 1./255.,  #0~1内に

                                   rotation_range = 10,  #回転範囲

                                   width_shift_range = 0.25,  #水平シフト

                                   height_shift_range = 0.25,  #垂直シフト

                                   shear_range = 0.1,  #反時計回りのシアー角度

                                   zoom_range = 0.25,  #ランダムにズームする範囲

                                   horizontal_flip = False)  #水平方向に入力をランダムに反転





valid_datagen = ImageDataGenerator(rescale=1./255) 
model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)))

model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(64,  (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))

          

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))

          

model.add(Conv2D(128, (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

model.add(Conv2D(128,  (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))       

          

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))  

          

model.add(Conv2D(256, (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

          

model.add(Conv2D(128, (3,3), padding='same'))

model.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))

          

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.2))

          

model.add(Flatten())        

model.add(Dense(256,activation='relu',name='dense1'))

model.add(LeakyReLU(alpha=0.1))          

model.add(BatchNormalization())

model.add(Dense(10,activation='softmax'))
"""initial_learningrate=1e-3



def lr_decay(epoch):

    if epoch < 5:

        return initial_learningrate

    else:

        return initial_learningrate * 0.99 ** epoch"""



initial_learningrate=2e-3



def lr_decay(epoch):

    return initial_learningrate * 0.99 ** epoch

    

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=initial_learningrate), metrics=["acc"])

model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=2048),

                    steps_per_epoch=300,

                    epochs=10,

                    callbacks=[LearningRateScheduler(lr_decay,verbose=1)],

                    validation_data=valid_datagen.flow(X_valid,Y_valid),

                    validation_steps=50,  

                    verbose=1)


X_test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

X_test = X_test.drop("id", axis=1).values

submission = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")



X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_test = X_test.astype(np.float32)/ 255





predictions = model.predict_classes(X_test)





submissions=pd.DataFrame({"id": submission["id"], "label": predictions})

submissions.to_csv("sUbmission.csv", index=False)



print("FINISH")