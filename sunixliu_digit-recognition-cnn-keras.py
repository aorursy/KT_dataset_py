%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns;sns.set()

import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
train_data=pd.read_csv("../input/train.csv")

test_data=pd.read_csv('../input/test.csv')
y_label=train_data['label']
img_rows, img_cols = 28, 28

num_classes = 10
def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes) # 将整型的类别标签转为onehot编码



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y
train_size =len(train_data)
x,y = data_prep(train_data)
# With data augmentation to prevent overfitting

# Randomly rotate some training images by 10 degrees

# Randomly Zoom by 10% some training images

# Randomly shift images horizontally by 10% of the width

# Randomly shift images vertically by 10% of the height



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.18, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x)
X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size = 0.1, random_state=2)
class myCallback(keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('acc')>0.997):

            print("\nReached 99.7% accuracy so cancelling training")

            self.model.stop_training=True
model = Sequential()
# model.add(

#     Conv2D(

#         32,

#         kernel_size=(3, 3),

#         activation='relu',

#         input_shape=(img_rows, img_cols, 1)))

# model.add(Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

# model.add(Flatten())

# model.add(Dense(256, activation='relu'))

# model.add(Dropout(0.5))

# model.add(Dense(num_classes, activation='softmax'))



# Set the CNN model 



model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.2))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

# model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

#                  activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(512, activation = "relu"))

# model.add(Dropout(0.2))

model.add(Dense(10, activation = "softmax"))

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(

    loss=keras.losses.categorical_crossentropy,

    optimizer= adam,

    metrics=['accuracy'])
model.summary()
# history = model.fit(x, y, batch_size=128, epochs=12, validation_split=0.2)

# Fit the model

callbacks=myCallback()

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=128),

                              epochs = 80, validation_data = (X_val,Y_val), callbacks=[callbacks])
def v_train_history(trainhist, train_metrics, valid_metrics):

    plt.plot(trainhist.history[train_metrics])

    plt.plot(trainhist.history[valid_metrics])

    plt.title('Training metrics')

    plt.ylabel(train_metrics)

    plt.xlabel('Epochs')

    plt.legend(['train','validation'],loc='upper left')

    plt.show()
v_train_history(history,'loss','val_loss')
v_train_history(history,'acc','val_acc')
num_images=test_data.shape[0]

test_as_array = test_data.values[:,:]

test_shaped_array = test_as_array.reshape(num_images, img_rows, img_cols, 1)

out_test= test_shaped_array / 255

y_pred=model.predict_classes(out_test)
submission = pd.DataFrame({"ImageId": np.arange(1, len(test_data)+1), "Label": y_pred})

submission.to_csv('submission.csv', index=False)