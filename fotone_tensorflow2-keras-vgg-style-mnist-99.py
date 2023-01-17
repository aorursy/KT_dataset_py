import tensorflow as tf

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from tensorflow import keras
train_pd = pd.read_csv('../input/digit-recognizer/train.csv')

test_pd = pd.read_csv('../input/digit-recognizer/test.csv')



train_x = train_pd.drop('label', axis=1).to_numpy() / 255.0

train_x = train_x.reshape(-1, 28, 28, 1)





train_y = train_pd['label'].to_numpy()

print(train_x.shape)

print(train_y.shape)



plt.imshow(train_x[765].reshape(28,28))

plt.show()

print(train_y[765])



test_x = test_pd.to_numpy() / 255.0

test_x = test_x.reshape(-1, 28, 28, 1)
model = keras.Sequential([

    keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=32, padding='same', activation='relu', name="conv1_1"),

    keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu', name="conv1_2"),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.3),

    keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'),

    keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.3),

    keras.layers.Flatten(),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dropout(0.3),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.Dropout(0.3),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])



model.summary()

hist = model.fit(train_x, train_y, batch_size=6720, epochs=30, validation_split=0.2)

plt.plot(hist.history['loss'], 'b-', label='loss')

plt.plot(hist.history['val_loss'], 'r--', label='val_loss')

plt.show()



plt.plot(hist.history['acc'], 'b-', label='acc')

plt.plot(hist.history['val_acc'], 'r--', label='val_acc')

plt.show()
results = model.predict(test_x)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("vgg_mnist_submission.csv",index=False)