#import includes

import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd



from sklearn.model_selection import train_test_split
#read input data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv") 
y_train = train["label"]

x_train = train.drop(labels = ["label"], axis = 1)

x_eval = x_train[:4200]
x_eval.describe()
x_train.describe()
y_train.describe()
#data normalization  in range 0..1

x_train /=255

test /=255

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

x_eval = x_train[:4200]

x_train = x_train[4200:]

y_eval = y_train[:4200]

y_train = y_train[4200:]
aug_data = keras.preprocessing.image.ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images)

batches = aug_data.flow(x_train, y_train, batch_size = 86)
model = keras.Sequential([

    keras.layers.Conv2D(256, (3,3),padding = 'Same', activation=tf.nn.relu, input_shape=(28,28,1)),

    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(128, (3,3),padding = 'Same', activation=tf.nn.relu),

    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Flatten(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)    

]) 



learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5)
model.compile(optimizer=tf.train.AdamOptimizer(),

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])



#model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=200, validation_split = 0.1)

#callbacks=[learning_rate_reduction],

history = model.fit_generator(generator=batches,steps_per_epoch=batches.n,  epochs=200)
plt.plot(history.history['loss'], color='r', label="Train Loss")

plt.title("Train Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
plt.plot(history.history['acc'], color='g', label='Train Accuracy')

plt.title('Train Accuracy')

plt.xlabel('Number of Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()


test_loss, test_acc = model.evaluate(x_eval, y_eval)

print('accuracy:', test_acc)
output = model.predict_classes(test, verbose=1)


file = pd.DataFrame({"ImageId":list(range(1,len(output)+1)),"Label":output})



file.to_csv('submission.csv',index=False,header=True)


