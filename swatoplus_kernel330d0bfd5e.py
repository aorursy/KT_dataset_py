import pandas as pd

import numpy as  np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop,Adam



from keras.callbacks import ReduceLROnPlateau, EarlyStopping



train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')



def load_dataset(filename, limit=None):

    csv_data = pd.read_csv(filename, nrows=limit)

    

    x_data = csv_data.drop('label',axis=1)

    y_data = csv_data.label

    

    x_data = x_data.values.astype('float32') / 255   

    x_data = x_data.reshape(-1, 28, 28,1)

    

    y_data = to_categorical(y_data)

    

    return x_data, y_data



# X_train_with_val, y_train_with_val = load_dataset('/kaggle/input/Kannada-MNIST/train.csv', limit=100)

X_train_with_val, y_train_with_val = load_dataset('/kaggle/input/Kannada-MNIST/train.csv')

X_train, X_val, y_train, y_val = train_test_split(X_train_with_val, y_train_with_val, random_state=1, test_size=0.15)



def build_lenet_model():

    model = Sequential()



    model.add(layers.Conv2D(filters=7, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))

    model.add(layers.AveragePooling2D(strides=2))



    model.add(layers.Conv2D(filters=19, kernel_size=(5, 5), activation='relu', padding='same'))

    model.add(layers.AveragePooling2D(strides=2))



    model.add(layers.Conv2D(filters=199, kernel_size=(5, 5), activation='relu'))

    model.add(layers.Flatten())



    model.add(layers.Dense(units=84, activation='relu'))



    model.add(layers.Dense(units=10, activation = 'softmax'))



    return model



model = build_lenet_model()

  

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                patience=3, 

                                verbose=1, 

                                factor=0.2, 

                                min_lr=1e-6)

    

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)

    

model.fit(X_train, y_train, batch_size=32,

                    epochs=60,

                    validation_data=(X_val, y_val),

                    callbacks=[learning_rate_reduction, es],

                    verbose=2)



test = test.drop('id',axis=1)

X_test = test.values.astype('float32') / 255

X_test = X_test.reshape(-1, 28, 28,1)



sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

sub = model.predict(X_test)

sub = np.argmax(sub, axis=1)



sample_sub['label']= sub

sample_sub.to_csv('submission.csv',index=False)