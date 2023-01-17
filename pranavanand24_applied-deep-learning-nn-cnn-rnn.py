import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
train_labels
test_images.shape
#If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255

plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()
train_images = train_images / 255.0



test_images = test_images / 255.0
#let's display the first 25 images from the training set and display the class name below each image

plt.figure(figsize=(12,12))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



print('\nTest accuracy:', test_acc)


from keras.datasets import imdb



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data[0]
train_labels[0]
#Since we restricted ourselves to the top 10,000 most frequent words, no word index will exceed 10,000

max([max(sequence) for sequence in train_data])


import numpy as np



def vectorize_sequences(sequences, dimension=10000):

    # Create an all-zero matrix of shape (len(sequences), dimension)

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1  #set specific indices of results[i] to 1s

    return results



# Our vectorized training data

x_train = vectorize_sequences(train_data)

# Our vectorized test data

x_test = vectorize_sequences(test_data)
x_train[0]
# Our vectorized labels

y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])
from keras import optimizers



model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics=['accuracy'])
from keras import losses

from keras import metrics



model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss=losses.binary_crossentropy,

              metrics=[metrics.binary_accuracy])
x_val = x_train[:10000]

partial_x_train = x_train[10000:]



y_val = y_train[:10000]

partial_y_train = y_train[10000:]
history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=20,

                    batch_size=512,

                    validation_data=(x_val, y_val))
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt



acc = history.history['loss']

val_acc = history.history['binary_accuracy']

loss = history.history['val_loss']

val_loss = history.history['val_binary_accuracy']



epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()   # clear figure

acc_values = history_dict['binary_accuracy']

val_acc_values = history_dict['val_binary_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_test, y_test)
results
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import os

import numpy as np

import matplotlib.pyplot as plt
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'



path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)



PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
#After extracting its contents, assign variables with the proper file path for the training and validation set

train_dir = os.path.join(PATH, 'train')

validation_dir = os.path.join(PATH, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures

train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures

validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures

validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures
num_cats_tr = len(os.listdir(train_cats_dir))

num_dogs_tr = len(os.listdir(train_dogs_dir))



num_cats_val = len(os.listdir(validation_cats_dir))

num_dogs_val = len(os.listdir(validation_dogs_dir))



total_train = num_cats_tr + num_dogs_tr

total_val = num_cats_val + num_dogs_val
print('total training cat images:', num_cats_tr)

print('total training dog images:', num_dogs_tr)



print('total validation cat images:', num_cats_val)

print('total validation dog images:', num_dogs_val)

print("--")

print("Total training images:", total_train)

print("Total validation images:", total_val)
#Setting up variables to use while pre-processing the dataset and training the network

batch_size = 128

epochs = 15

IMG_HEIGHT = 150

IMG_WIDTH = 150
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory=train_dir,

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,

                                                              directory=validation_dir,

                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                              class_mode='binary')
model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.summary()
history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=total_val // batch_size

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,

                                               directory=train_dir,

                                               shuffle=True,

                                               target_size=(IMG_HEIGHT, IMG_WIDTH))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,

                                               directory=train_dir,

                                               shuffle=True,

                                               target_size=(IMG_HEIGHT, IMG_WIDTH))



augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# zoom_range from 0 - 1 where 1 = 100%.

image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5) 
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,

                                               directory=train_dir,

                                               shuffle=True,

                                               target_size=(IMG_HEIGHT, IMG_WIDTH))



augmented_images = [train_data_gen[0][0][0] for i in range(5)]
image_gen_train = ImageDataGenerator(

                    rescale=1./255,

                    rotation_range=45,

                    width_shift_range=.15,

                    height_shift_range=.15,

                    horizontal_flip=True,

                    zoom_range=0.5

                    )
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,

                                                     directory=train_dir,

                                                     shuffle=True,

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                     class_mode='binary')
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,

                                                 directory=validation_dir,

                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                 class_mode='binary')
model_new = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', 

           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Dropout(0.2),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Dropout(0.2),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])
model_new.compile(optimizer='adam',

                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

                  metrics=['accuracy'])



model_new.summary()
history = model_new.fit_generator(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=total_val // batch_size

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
import tensorflow as tf



import numpy as np

import os

import time
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# length of text is the number of characters in it

print ('Length of text: {} characters'.format(len(text)))
# Take a look at the first 250 characters in text

print(text[:250])
# The unique characters in the file

vocab = sorted(set(text))

print ('{} unique characters'.format(len(vocab)))
# Creating a mapping from unique characters to indices

char2idx = {u:i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)



text_as_int = np.array([char2idx[c] for c in text])
print('{')

for char,_ in zip(char2idx, range(20)):

    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))

print('  ...\n}')
# Show how the first 13 characters from the text are mapped to integers

print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
# The maximum length sentence we want for a single input in characters

seq_length = 100

examples_per_epoch = len(text)//(seq_length+1)



# Create training examples / targets

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)



for i in char_dataset.take(5):

  print(idx2char[i.numpy()])
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)



for item in sequences.take(5):

  print(repr(''.join(idx2char[item.numpy()])))
def split_input_target(chunk):

    input_text = chunk[:-1]

    target_text = chunk[1:]

    return input_text, target_text



dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):

  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))

  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):

    print("Step {:4d}".format(i))

    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))

    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
# Batch size

BATCH_SIZE = 64



# Buffer size to shuffle the dataset

# (TF data is designed to work with possibly infinite sequences,

# so it doesn't attempt to shuffle the entire sequence in memory. Instead,

# it maintains a buffer in which it shuffles elements).

BUFFER_SIZE = 10000



dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



dataset
# Length of the vocabulary in chars

vocab_size = len(vocab)



# The embedding dimension

embedding_dim = 256



# Number of RNN units

rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):

  model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim,

                              batch_input_shape=[batch_size, None]),

    tf.keras.layers.GRU(rnn_units,

                        return_sequences=True,

                        stateful=True,

                        recurrent_initializer='glorot_uniform'),

    tf.keras.layers.Dense(vocab_size)

  ])

  return model
model = build_model(

  vocab_size = len(vocab),

  embedding_dim=embedding_dim,

  rnn_units=rnn_units,

  batch_size=BATCH_SIZE)
for input_example_batch, target_example_batch in dataset.take(1):

  example_batch_predictions = model(input_example_batch)

  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
model.summary()
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)

sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))

print()

print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))
def loss(labels, logits):

  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)



example_batch_loss  = loss(target_example_batch, example_batch_predictions)

print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")

print("scalar_loss:      ", example_batch_loss.numpy().mean())
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved

checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")



checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
EPOCHS=10



history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)



model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))



model.build(tf.TensorShape([1, None]))
model.summary()
def generate_text(model, start_string):

  # Evaluation step (generating text using the learned model)



  # Number of characters to generate

  num_generate = 1000



  # Converting our start string to numbers (vectorizing)

  input_eval = [char2idx[s] for s in start_string]

  input_eval = tf.expand_dims(input_eval, 0)



  # Empty string to store our results

  text_generated = []



  # Low temperatures results in more predictable text.

  # Higher temperatures results in more surprising text.

  # Experiment to find the best setting.

  temperature = 1.0



  # Here batch size == 1

  model.reset_states()

  for i in range(num_generate):

      predictions = model(input_eval)

      # remove the batch dimension

      predictions = tf.squeeze(predictions, 0)



      # using a categorical distribution to predict the character returned by the model

      predictions = predictions / temperature

      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()



      # We pass the predicted character as the next input to the model

      # along with the previous hidden state

      input_eval = tf.expand_dims([predicted_id], 0)



      text_generated.append(idx2char[predicted_id])



  return (start_string + ''.join(text_generated))
print(generate_text(model, start_string=u"ROMEO: "))