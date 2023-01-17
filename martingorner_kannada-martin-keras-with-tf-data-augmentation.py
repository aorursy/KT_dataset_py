import tensorflow as tf

import os

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import PIL.Image, PIL.ImageFont, PIL.ImageDraw

#import tensorflow_addons as tfa

AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Tensorflow version " + tf.__version__)
BATCH_SIZE = 64



TOTAL_DIGITS = 60000

VALIDATION_DIGITS = 8000 # validation digits taken out of train.csv
"""

This cell contains helper functions used for visualization

and downloads only. You can skip reading it. There is very

little useful Keras/Tensorflow code here.

"""



# Matplotlib config

plt.rc('image', cmap='gray_r')

plt.rc('grid', linewidth=0)

plt.rc('xtick', top=False, bottom=False, labelsize='large')

plt.rc('ytick', left=False, right=False, labelsize='large')

plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')

plt.rc('text', color='a8151a')

plt.rc('figure', facecolor='F0F0F0')# Matplotlib fonts

#MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")



def dataset_to_numpy_util(dataset, N):

    dataset = dataset.unbatch().batch(N)

  

    for digits, labels in dataset:

        digits = digits.numpy()

        labels = np.argmax(labels.numpy(), axis=1) # these were one-hot encoded in the dataset

        break

    

    return digits, labels



# create digits from local fonts for testing

def create_digits_from_local_fonts(n):

    font_labels = []

    img = PIL.Image.new('LA', (28*n, 28), color = (0,255)) # format 'LA': black in channel 0, alpha in channel 1

    font1 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'DejaVuSansMono.ttf'), 21)

    font2 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'STIXGeneral.ttf'), 21)

    d = PIL.ImageDraw.Draw(img)

    for i in range(n):

        font_labels.append(i%10)

        d.text((7+i*28,0 if i<10 else -4), chr(ord(u'೦')+i%10), fill=(255,255), font=font1 if i<10 else font2)

        #d.text((7+i*28,0 if i<10 else -4), chr(ord(u'0')+i%10), fill=(255,255), font=font1 if i<10 else font2)

    font_digits = np.array(img.getdata(), np.float32)[:,0] / 255.0 # black in channel 0, alpha in channel 1 (discarded)

    font_digits = np.reshape(np.stack(np.split(np.reshape(font_digits, [28, 28*n]), n, axis=1), axis=0), [n, 28*28])

    return font_digits, font_labels



# utility to display a row of digits with their predictions

def display_digits(digits, predictions, labels, title, digits_per_row, row, n_rows):

    if row==0: # set up the subplots on the first call

        plt.subplots(figsize=(14,1.1*n_rows))

        plt.tight_layout()

    ax = plt.subplot(n_rows, 1, row+1) # index is 1-based

    digits = np.reshape(digits, [digits_per_row, 28, 28])

    digits = np.swapaxes(digits, 0, 1)

    digits = np.reshape(digits, [28, 28*digits_per_row])

    ax.set_yticks([])

    ax.set_xticks([28*x+14 for x in range(digits_per_row)])

    ax.set_xticklabels(predictions)

    for i,t in enumerate(ax.get_xticklabels()):

        if predictions[i] != labels[i]: t.set_color('red') # bad predictions in red

    ax.imshow(digits)

    ax.grid(None)

    ax.set_title(title)



def display_rows_of_digits(digits, predictions, labels, title, n_rows, digits_per_row):

    digits = np.reshape(digits[:n_rows*digits_per_row], [n_rows, digits_per_row, 28, 28])

    predictions = np.reshape(predictions[:n_rows*digits_per_row], [n_rows, digits_per_row])

    labels = np.reshape(labels[:n_rows*digits_per_row], [n_rows, digits_per_row])

    for row in range(n_rows):

        display_digits(digits[row], predictions[row], labels[row],

                       title if row==0 else "", digits_per_row, row, n_rows)

  

# utility to display multiple rows of digits, sorted by unrecognized/recognized status

def display_top_unrecognized(digits, predictions, labels, n_rows, digits_per_row):

    idx = np.argsort(predictions==labels) # sort order: unrecognized first

    for row in range(n_rows):

        display_digits(digits[idx][row*digits_per_row:(row+1)*digits_per_row],

                       predictions[idx][row*digits_per_row:(row+1)*digits_per_row],

                       labels[idx][row*digits_per_row:(row+1)*digits_per_row],

                       "{} sample validation digits out of {} with bad predictions in red and sorted first".format(digits_per_row*n_rows,

                                                                                                                   len(digits)) if row==0 else "",

                       digits_per_row, row, n_rows)

    

# utility to display training and validation curves

def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.grid(linewidth=1, color='white')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])

    

def display_lr_schedule(lr_fn, epochs):

    X = range(epochs)

    Y = np.array([lr_fn(x) for x in X])

    min_lr = min(Y)

    max_lr = max(Y)

    title = "Learning rate schedule\n"

    title += "max lr: {0:.4g}".format(max_lr) + "\n"

    title += "min lr: {0:.4g}".format(min_lr)

    plt.title(title)

    plt.plot(X, Y)

    plt.show()
column_defaults = [tf.constant(0, dtype=tf.int32) for i in range(28*28+1)] # format: label, 28*28=784 pixel columns, all ints

def decode_csv_line(line, has_labels):    

    data = tf.io.decode_csv(line, column_defaults)

    first_col = data[0]

    if has_labels:

        label = tf.one_hot(first_col, 10)

    else:

        label = first_col # in fact just an index

    pixels = data[1:]

    image = tf.reshape(pixels, [28,28])

    image = tf.cast(image, tf.float32) / 255.0

    return image, label



#def augment_rotate(image, label):

#    # this one requires tensorflow_addons (not yet available on Kaggle)

#    angles = tf.random.normal([tf.shape(image)[0]], stddev=0.4)

#    image = tfa.image.rotate(tf.expand_dims(image, axis=-1), angles, interpolation='BILINEAR')

#    return tf.squeeze(image, axis=-1), label



def augment(image, label):

    std = 0.1

    image = tf.expand_dims(image, axis=-1)

    batch = tf.shape(image)[0]

    dy1 = tf.random.normal([batch], stddev=std)

    dx1 = tf.random.normal([batch], stddev=std)

    dy2 = tf.random.normal([batch], stddev=std)

    dx2 = tf.random.normal([batch], stddev=std)

    y1 = tf.zeros([batch]) + dy1

    x1 = tf.zeros([batch]) + dx1

    y2 = tf.ones([batch]) + dy2

    x2 = tf.ones([batch]) + dx2

    box = tf.stack([y1, x1, y2, x2], axis=-1)

    image = tf.image.crop_and_resize(image, box, tf.range(batch), [28, 28])

    return tf.squeeze(image, axis=-1), label



def blur(image, label):

    image = tf.expand_dims(image, axis=-1)

    image = tf.image.resize(image, [56,56], method=tf.image.ResizeMethod.BICUBIC, antialias=True)

    image = tf.image.resize(image, [28,28], method=tf.image.ResizeMethod.BICUBIC, antialias=True)

    image = tf.squeeze(image, axis=-1)

    return (image, label)



def load_dataset(filename, has_labels):

    dataset = tf.data.TextLineDataset(filename)

    dataset = dataset.skip(1) # header line

    dataset = dataset.map(lambda line: decode_csv_line(line, has_labels), num_parallel_calls=AUTOTUNE)

    return dataset 



def get_training_dataset(filename, batch_size):

    dataset = load_dataset(filename, has_labels=True).skip(VALIDATION_DIGITS)

    dataset = dataset.cache()  # this small dataset can be entirely cached in RAM

    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)

    dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    

    dataset = dataset.map(blur)

    dataset = dataset.map(augment)

    

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset



def get_validation_dataset(filename, batch_size):

    dataset = load_dataset(filename, has_labels=True).take(VALIDATION_DIGITS)

    dataset = dataset.cache()

    dataset = dataset.batch(batch_size)

    

    dataset = dataset.map(blur)

    

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset



def get_submission_dataset(filename, batch_size):

    dataset = load_dataset(filename, has_labels=False)

    dataset = dataset.cache()

    dataset = dataset.batch(batch_size)

    

    dataset = dataset.map(blur)

    

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
train_dataset = get_training_dataset("/kaggle/input/Kannada-MNIST/train.csv", batch_size=BATCH_SIZE)

valid_dataset = get_validation_dataset("/kaggle/input/Kannada-MNIST/train.csv", batch_size=VALIDATION_DIGITS) # one batch with everything

submit_dataset = get_submission_dataset("/kaggle/input/Kannada-MNIST/test.csv", batch_size=1000)

steps_per_epoch = (TOTAL_DIGITS - VALIDATION_DIGITS) // BATCH_SIZE
PER_ROW = 24

N = PER_ROW*4

train_digits, train_labels = dataset_to_numpy_util(train_dataset, N)

valid_digits, valid_labels = dataset_to_numpy_util(valid_dataset, N)

display_rows_of_digits(train_digits, train_labels, train_labels, "training digits and their labels",   N//PER_ROW, PER_ROW)

display_rows_of_digits(valid_digits, valid_labels, valid_labels, "validation digits and their labels", N//PER_ROW, PER_ROW)
def make_model():

    model = tf.keras.Sequential(

      [

        tf.keras.layers.Reshape(input_shape=(28,28), target_shape=(28, 28, 1), name="image"),

        

        tf.keras.layers.SeparableConv2D(filters=12, kernel_size=3, padding='same', use_bias=False, depth_multiplier=2),

        #tf.keras.layers.Conv2D(filters=12, kernel_size=3, padding='same', use_bias=False), # no bias necessary before batch norm

        tf.keras.layers.BatchNormalization(scale=False, center=True), # no batch norm scaling necessary before "relu"

        tf.keras.layers.Activation('relu'), # activation after batch norm

          

        tf.keras.layers.SeparableConv2D(filters=16, kernel_size=3, padding='same', use_bias=False, depth_multiplier=2),

        #tf.keras.layers.Conv2D(filters=12, kernel_size=3, padding='same', use_bias=False), # no bias necessary before batch norm

        tf.keras.layers.BatchNormalization(scale=False, center=True), # no batch norm scaling necessary before "relu"

        tf.keras.layers.Activation('relu'), # activation after batch norm



        tf.keras.layers.SeparableConv2D(filters=20, kernel_size=6, padding='same', use_bias=False, strides=2, depth_multiplier=2), 

        #tf.keras.layers.Conv2D(filters=24, kernel_size=6, padding='same', use_bias=False, strides=2, name="XXX"),

        tf.keras.layers.BatchNormalization(scale=False, center=True),

        tf.keras.layers.Activation('relu'),

          

        tf.keras.layers.SeparableConv2D(filters=24, kernel_size=3, padding='same', use_bias=False, depth_multiplier=2), 

        #tf.keras.layers.Conv2D(filters=24, kernel_size=6, padding='same', use_bias=False, strides=2, name="XXX"),

        tf.keras.layers.BatchNormalization(scale=False, center=True),

        tf.keras.layers.Activation('relu'),

          

        tf.keras.layers.SeparableConv2D(filters=28, kernel_size=3, padding='same', use_bias=False, depth_multiplier=2), 

        #tf.keras.layers.Conv2D(filters=24, kernel_size=6, padding='same', use_bias=False, strides=2, name="XXX"),

        tf.keras.layers.BatchNormalization(scale=False, center=True),

        tf.keras.layers.Activation('relu'),



        tf.keras.layers.SeparableConv2D(filters=32, kernel_size=6, padding='same', use_bias=False, strides=2, depth_multiplier=2),

        #tf.keras.layers.Conv2D(filters=32, kernel_size=6, padding='same', use_bias=False, strides=2),

        tf.keras.layers.BatchNormalization(scale=False, center=True),

        tf.keras.layers.Activation('relu'),

          

        tf.keras.layers.SeparableConv2D(filters=36, kernel_size=3, padding='same', use_bias=False, depth_multiplier=2), 

        #tf.keras.layers.Conv2D(filters=24, kernel_size=6, padding='same', use_bias=False, strides=2, name="XXX"),

        tf.keras.layers.BatchNormalization(scale=False, center=True),

        tf.keras.layers.Activation('relu'),



        #tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(100, use_bias=False),

        tf.keras.layers.BatchNormalization(scale=False, center=True),

        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dropout(0.4), # Dropout on dense layer only



        tf.keras.layers.Dense(10, activation='softmax')

      ])



    model.compile(optimizer='adam', # learning rate will be set by LearningRateScheduler

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model

    

#with strategy.scope():

model = make_model()



# print model layers

model.summary()



# set up learning rate decay

def lr_fn(epoch):

    return LEARNING_RATE * LEARNING_RATE_EXP_DECAY**epoch

lr_decay = tf.keras.callbacks.LearningRateScheduler(lr_fn, verbose=True)
EPOCHS = 60

LEARNING_RATE = 0.01

LEARNING_RATE_EXP_DECAY = 0.95

display_lr_schedule(lr_fn, EPOCHS)
history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, validation_data=valid_dataset, validation_steps=1, epochs=EPOCHS, callbacks=[lr_decay])
print("Final validation accuracy:", history.history["val_accuracy"][-1])

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
# recognize validation digits

digits, labels = dataset_to_numpy_util(valid_dataset, VALIDATION_DIGITS)

probabilities = model.predict(digits, steps=1)

predicted_labels = np.argmax(probabilities, axis=1)

display_top_unrecognized(digits, predicted_labels, labels, n_rows=12, digits_per_row=24)
results = None

for images, indices in submit_dataset:

    predictions = model.predict(images)

    df = pd.DataFrame({'id':indices.numpy(), 'label':np.argmax(predictions, axis=-1)})

    results = results.append(df) if results is not None else df

results.to_csv("submission.csv", index=False)

!ls -al "submission.csv"

!wc -l "submission.csv"

!head "submission.csv" -n 15

!echo "..."

!tail "submission.csv" -n 15