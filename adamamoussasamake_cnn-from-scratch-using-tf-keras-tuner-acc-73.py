!pip install -U keras-tuner
import itertools
import os
import math
from IPython.display import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np

from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow import keras
species = {0:'mantled_howler', 
           1:'patas_monkey', 
           2: 'bald_uakari', 
           3: 'japanese_macaque', 
           4: 'pygmy_marmoset', 
           5: 'white_headed_capuchin',
           6: 'silvery_marmoset',
           7: 'common_squirrel_monkey',
           8: 'black_headed_night_monkey' ,
           9: 'nilgiri_langur'
           }

def prepare_labels (dataset):
  label = []
  for key, specie in species.items():
    path = f'../input/10-monkey-species/{dataset}/n{key}'
    species_list_length = len(os.listdir(path))
    for j in range (1, species_list_length + 1):
      label.append(specie)
  return label
def prepare_images_paths (dataset):
  images = []
  for i in range (0, 10):
    path = f'../input/10-monkey-species/{dataset}/n{i}'
    for image in os.listdir(path):
      images.append(f'{path}/{image}')
  return images
X_train = prepare_images_paths('training/training')
y_train = prepare_labels('training/training')
X_valid =  prepare_images_paths('validation/validation')
y_valid = prepare_labels('validation/validation')
# Let's display species and their number 
pd.Series(y_train).value_counts()
# Let's transform each element in labels into an array of boolean
def label_transformation (labels):
  return [label == pd.Series(labels).unique() for label in labels]
y_train = label_transformation(y_train)
y_valid = label_transformation(y_valid)
size = [224, 224]
def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    
    image = tf.image.decode_jpeg(image_string, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)

    resized_image = tf.image.resize(image, size=size)
    return resized_image, label
def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)
    
    image = tf.image.random_flip_left_right(image)
    
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label
class MiniBatch ():
  def __init__ (self, batch_size):
    self.batch_size = batch_size
  
  def training_batch (self, X, y):
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    dataset_shuffled = dataset.shuffle(buffer_size=len(X))
    dataset = dataset_shuffled.map(parse_function, num_parallel_calls=4)
    dataset = dataset.map(train_preprocess, num_parallel_calls=4).batch(self.batch_size)
    dataset = dataset.prefetch(1)

    return dataset
  
  def validation_batch (self, X, y):
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
    dataset = dataset.map(parse_function,  num_parallel_calls=4).batch(self.batch_size)
    dataset = dataset.prefetch(1)
    return dataset
mini_batch = MiniBatch(32)
train_data = mini_batch.training_batch(X_train, y_train)
valid_data = mini_batch.validation_batch(X_valid, y_valid)
def build_model (hp):
 
  i = Input(shape=[224, 224, 3])
  x = Conv2D(hp.Int(f'conv_1', min_value=64, max_value=128, step=32),
             kernel_size=hp.Choice('conv_1_kernel', values = [3,5, 7]),
             activation='relu',  padding='same')(i) #kernel_initializer='he_uniform',

  x = BatchNormalization()(x)

  x = Conv2D(hp.Int(f'conv_2', min_value=64, max_value=128, step=32),
             kernel_size=hp.Choice('conv_2_kernel', values = [3,5, 7]),
             activation='relu', padding='same')(x)

  x = BatchNormalization()(x)
  x = MaxPool2D(pool_size=(2, 2))(x)
  x = Dropout(0.3)(x)

  for layer in range(hp.Int('n_layers', 1, 6)):   
    x = Conv2D(hp.Int(f'firs_conv_{layer}_units', min_value=64, max_value=256, step=32),
               kernel_size=hp.Choice(f'firs_conv_{layer}_kernel', values = [3,5]),
               activation='relu', padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv2D(hp.Int(f'second_conv_{layer}_units', min_value=64, max_value=256, step=32),
               kernel_size=hp.Choice(f'second_conv_{layer}_kernel', values = [3,5]),
               activation='relu', padding='same')(x)

    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
  
  x=Flatten()(x)
  x = Dropout(0.3)(x)
  x=Dense(hp.Int(f'Dense_units', min_value=700, max_value=1024, step=100), activation='relu')(x)
  x = Dropout(0.4)(x)
  x=Dense(10, activation='softmax')(x)
  model = keras.Model(inputs=[i], outputs=[x])
  model.compile(optimizer= keras.optimizers.Adam(learning_rate=hp.Choice(f'lsr', values=[2.5e-04, 1.25e-04])), loss='categorical_crossentropy', metrics=['accuracy'])
  return model
tuner_search = RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=5,
                          directory='output',
                          project_name="Monkey species2",
                          executions_per_trial=2,
                          )
tuner_search.search(x=train_data, validation_data=valid_data, epochs=50)
best_model = tuner_search.get_best_models()[0]
best_model.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',  patience=10)
history = best_model.fit(x=train_data, validation_data=valid_data, epochs=100, callbacks=[early_stop], initial_epoch=50)
predictions = best_model.predict(valid_data, verbose=1)
species_list = pd.Series(prepare_labels('validation/validation')).unique()
# Create a function to unbatch a batched dataset
def unbatch_data(data):
  """
  Takes a batched dataset of (image, label) Tensors and returns separate arrays
  of images and labels.
  """
  images = []
  labels = []
  # Loop through unbatched data
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(species_list[np.argmax(label)])
  return images, labels
# Unbatchify the validation data
valid_images, valid_labels = unbatch_data(valid_data)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return species_list[np.argmax(prediction_probabilities)]
def plot_pred(prediction_probabilities, labels, images, n=1):
 
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
  
  pred_label = get_pred_label(pred_prob)
  
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  # Change the color of the title depending on if the prediction is right or wrong
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  plt.title(f"{pred_label} {np.round(np.max(pred_prob)*100, 2)}% ({true_label})", color=color)
plot_pred(prediction_probabilities=predictions, labels=valid_labels, images=valid_images, n= 50)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix - training', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure(figsize=(8, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm = confusion_matrix(np.array(y_valid).astype(int).argmax(axis=1), predictions.argmax(axis=1))
plot_confusion_matrix(cm, species_list, normalize=False)
def show_report():
  print(classification_report(np.array(y_valid).astype(int).argmax(axis=1), predictions.argmax(axis=1)))
show_report()
