import tensorflow as tf

from tensorflow import keras

from datetime import datetime

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical





(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()



X_learn, X_val, Y_learn, Y_val = train_test_split(X_train, Y_train, test_size = 0.333)



X_learn_3d = X_learn.reshape(-1,28,28,1)

X_val_3d = X_val.reshape(-1,28,28,1)



from keras.utils import to_categorical

Y_learn_bin = to_categorical(Y_learn)

Y_val_bin = to_categorical(Y_val)



print("X_learn shape:",X_learn.shape)

print("X_val shape:",X_val.shape)

print("Y_learn shape:",Y_learn.shape)

print("X_learn_3d shape:",X_learn_3d.shape)

print("Y_learn_bin shape:",Y_learn_bin.shape)

del X_train

del Y_train
import matplotlib.pyplot as plt



plt.figure(figsize=(20,5))

for i in range(20):  

    plt.subplot(2, 10, i+1)

    plt.imshow(X_learn[i].reshape(28,28), cmap='gray')

    plt.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()
import tensorflow as tf

from tensorflow import keras

from datetime import datetime

import numpy as np

print("TensorFlow version: ", tf.__version__)
model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(10, activation='sigmoid'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])
experiments_1 = {}

EPOCHS = 10

BATCH_SIZE = 100

def show_experiment(experiments):

    lstyles = ['-', '--', '-.', ':']

    plt.figure(figsize=(15,5))

    acc_and_experiments = [(np.mean(e[1].history['val_accuracy'][-3:]),e[0],e[1]) for e in experiments.items()]

    acc_and_experiments.sort(key = lambda x: x[0], reverse = True) 

    titles = []

    for acc,model_name,history in acc_and_experiments:

        titles.append(f"acc:%.4f %s" % (acc,model_name))

        stl = lstyles[hash(model_name) % len(lstyles)]

        plt.plot(history.history['val_accuracy'],linestyle=stl)

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(titles, loc='best')

    axes = plt.gca()

    plt.show()
model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(10, activation='sigmoid'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



history = model.fit(X_learn,Y_learn_bin,

                    validation_data = (X_val,Y_val_bin),

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS

                   )



experiments_1['Dummy sigmoid NN'] = history

show_experiment(experiments_1)
# tanh

model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(10, activation='tanh'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



history = model.fit(X_learn,Y_learn_bin,

                    validation_data = (X_val,Y_val_bin),

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS

                   )



experiments_1['Dummy tanh NN'] = history



# relu

model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(10, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



history = model.fit(X_learn,Y_learn_bin,

                    validation_data = (X_val,Y_val_bin),

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS

                   )



experiments_1['Dummy relu NN'] = history



# 

show_experiment(experiments_1)
experiments_2 = {}

experiments_2["Baseline"] = experiments_1['Dummy sigmoid NN']

EPOCHS = 10

BATCH_SIZE = 100
model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



history = model.fit(X_learn,Y_learn_bin,

                    validation_data = (X_val,Y_val_bin),

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS

                   )



experiments_2['Wide NN'] = history

show_experiment(experiments_2)
model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



history = model.fit(X_learn,Y_learn_bin,

                    validation_data = (X_val,Y_val_bin),

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS

                   )



experiments_2['Tall NN'] = history

show_experiment(experiments_2)
%%time

model = keras.models.Sequential([

    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=(28,28,1)),

    keras.layers.Flatten(),

    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



history = model.fit(X_learn_3d,Y_learn_bin,

                    validation_data = (X_val_3d,Y_val_bin),

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS

                   )



experiments_2['Dummy CNN'] = history

show_experiment(experiments_2)
%%time

model = keras.models.Sequential([

    keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)),

    keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'),

    keras.layers.MaxPool2D((2, 2)),

    keras.layers.Dropout(0.20),

    keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),

    keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),

    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'),

    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



history = model.fit(X_learn_3d,Y_learn_bin,

                    validation_data = (X_val_3d,Y_val_bin),

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS

                   )



experiments_2['Cool CNN'] = history

show_experiment(experiments_2)
EPOCHS = 100

experiments_3 = {}



model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



small_X,small_Y = X_learn[:100],Y_learn_bin[:100]



history = model.fit(small_X,small_Y,

                    validation_data = (X_val,Y_val_bin),

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS

                   )



experiments_3['Small data NN'] = history

show_experiment(experiments_3)
from keras.preprocessing.image import ImageDataGenerator



X_sample = X_learn_3d[:1]

Y_sample = Y_learn[:1]



datagen = ImageDataGenerator(

        rotation_range=5,  

        zoom_range = 0.05,  

        width_shift_range=0.1, 

        height_shift_range=0.1)



plt.figure(figsize=(20,5))

lines = 3

columns = 10

for i in range(lines):  

    for j in range(columns):

        X_train2, Y_train2 = datagen.flow(X_sample,Y_sample).next()

        plt.subplot(lines, columns, i*columns + j+1)

        plt.imshow(X_train2[0].reshape(28,28),cmap='gray')

        plt.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()
from keras.preprocessing.image import ImageDataGenerator



model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28, 1)),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy'])



small_X,small_Y = X_learn_3d[:100],Y_learn_bin[:100]



datagen = ImageDataGenerator(

        rotation_range=5,  

        zoom_range = 0.05,  

        width_shift_range=0.1, 

        height_shift_range=0.1)



history = model.fit(datagen.flow(small_X, small_Y, batch_size=BATCH_SIZE*10),

                              validation_data = (X_val_3d,Y_val_bin),

                              epochs=EPOCHS)



experiments_3['Big data NN'] = history

show_experiment(experiments_3)
import numpy as np



import tensorflow as tf

!pip install -q tensorflow_datasets

# !pip install -q tensorflow-hub

# !pip install -q tfds-nightly

import tensorflow_hub as hub

import tensorflow_datasets as tfds



print("Version: ", tf.__version__)

print("Eager mode: ", tf.executing_eagerly())

print("Hub version: ", hub.__version__)

print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
train_data, validation_data, test_data = tfds.load(

    name="imdb_reviews", 

    split=('train[:60%]', 'train[60%:]', 'test'),

    as_supervised=True)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(4)))

train_examples_batch
train_labels_batch
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
from scipy.spatial import distance



embed = hub.load("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1")

embeddings = embed(["film", "movie"])

distance.cosine(embeddings[0],embeddings[1])
embeddings = embed(["film", "move"])

distance.cosine(embeddings[0],embeddings[1])
experiments_4 = {}
hub_layer = hub.KerasLayer(embedding, input_shape=[], 

                           dtype=tf.string, trainable=False)

hub_layer(train_examples_batch)



model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1))



model.summary()
%%time

model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])



history = model.fit(train_data.shuffle(10000).batch(512),

                    epochs=20,

                    validation_data=validation_data.batch(512),

                    verbose=1)



experiments_4['Pretrained (freezed) NN'] = history

show_experiment(experiments_4)
hub_layer = hub.KerasLayer(embedding, input_shape=[], 

                           dtype=tf.string, trainable=True)



model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(16, activation='relu'))

model.add(tf.keras.layers.Dense(1))



model.summary()
%%time

model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])



history = model.fit(train_data.shuffle(10000).batch(512),

                    epochs=20,

                    validation_data=validation_data.batch(512),

                    verbose=1)



experiments_4['Pretrained NN'] = history

show_experiment(experiments_4)