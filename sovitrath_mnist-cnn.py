import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

import pandas as pd

import os



from sklearn.model_selection import train_test_split



print(os.listdir("../input"))
np.random.seed(42)
# load the train and test csv files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print('TRAIN CSV FILES DATA: \n', train.head(10))
# for labels 

train_y = train['label']

print(train_y.head(10))
# for pixel values

train_x = train.drop(labels=['label'], axis=1)

print(train_x.head(10))
# total train instances

print(len(train_x))
# total instances for each digit

print(train_y.value_counts())
# normalize the data

train_x = train_x / 255.0

test = test / 255.0
print(train_x.head(10))
train_x = train_x.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
plt.figure(figsize=(8, 4))

plt.imshow(train_x[9][:, :, 0], cmap='gray')
train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)

print(train_y)
train_X, val_X, train_Y, val_Y = train_test_split(train_x, train_y, test_size = 0.2, random_state=42)
input_shape = (28, 28, 1)
model = tf.keras.Sequential()



model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', 

            activation='relu', input_shape=input_shape))    

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.2))

        

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', 

                                 activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.2))

 

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', 

                                 activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Dropout(0.2))

 

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1024, activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1024, activation='relu'))

model.add(tf.keras.layers.Dropout(0.4))

 

model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(

    optimizer = tf.keras.optimizers.RMSprop(),

    loss = tf.keras.backend.categorical_crossentropy, 

    metrics = ['accuracy']

)
history = model.fit(train_X, train_Y, 

          batch_size=32,

          validation_data=(val_X, val_Y),

          epochs=30)
num_epochs = np.arange(0, 30)

plt.figure(dpi=300)

plt.plot(num_epochs, history.history['loss'], label='train_loss', c='red')

plt.plot(num_epochs, history.history['val_loss'], 

    label='val_loss', c='orange')

plt.plot(num_epochs, history.history['acc'], label='train_acc', c='green')

plt.plot(num_epochs, history.history['val_acc'], 

    label='val_acc', c='blue')

plt.title('Training Loss and Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Loss/Accuracy')

plt.legend()

plt.savefig('plot.png')

# predict results

results = model.predict(test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results], axis = 1)



submission.to_csv("submission.csv",index=False)