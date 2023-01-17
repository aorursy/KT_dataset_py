!pip install tensorflow==2.2.0rc3
import tensorflow as tf

print(tf.__version__)
from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3



image_size = 300

pre_trained_model = InceptionV3(

  input_shape = (image_size, image_size, 3),

  include_top = False,

  weights = 'imagenet')



for layer in pre_trained_model.layers:

  layer.trainable = False



last_layer = pre_trained_model.get_layer('mixed7')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC



# Flatten the output layer to 1 dimension

x = layers.Flatten()(last_output)

# Add a fully connected layer with 1,024 hidden units and ReLU activation

x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.4

x = layers.Dropout(0.4)(x)

# Add a final sigmoid layer for classification

x = layers.Dense(1, activation='sigmoid')(x)



model = Model(pre_trained_model.input, x)



threshold = 0.5

METRICS = [

  TruePositives(name='tp', thresholds = threshold),

  FalsePositives(name='fp', thresholds = threshold),

  TrueNegatives(name='tn', thresholds = threshold),

  FalseNegatives(name='fn', thresholds = threshold),

  BinaryAccuracy(name='accuracy', threshold = threshold),

  Precision(name='precision', thresholds = threshold),

  Recall(name='recall', thresholds = threshold),

  AUC(name='auc')

]



model.compile(

  optimizer = Adam(lr=1e-3), 

  loss = 'binary_crossentropy', 

  metrics = METRICS)
import os

import numpy as np

import matplotlib.pyplot as plt



# Define our example directories and files

base_dir = '../input/chest-xray-pneumonia/chest_xray'



train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'val')

test_dir = os.path.join( base_dir, 'test')





train_normal_dir = os.path.join(train_dir, 'NORMAL')

train_pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')

validation_normal_dir = os.path.join(validation_dir, 'NORMAL')

validation_pneumonia_dir = os.path.join(validation_dir, 'PNEUMONIA')

test_normal_dir = os.path.join(test_dir, 'NORMAL')

test_pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')



train_normal_fnames = os.listdir(train_normal_dir)

train_pneumonia_fnames = os.listdir(train_pneumonia_dir)



# Show some examples

plt.figure(figsize=(12,8))

for n in range(3):

  plt.subplot(2,3,n+1)

  path = os.path.join(train_normal_dir, train_normal_fnames[n])

  img = plt.imread(path)

  plt.imshow(img, cmap='gray')

  plt.title(train_normal_fnames[n])

  plt.axis('off')

  plt.subplot(2,3,n+4)

  path = os.path.join(train_pneumonia_dir, train_pneumonia_fnames[n])

  img = plt.imread(path)

  plt.imshow(img, cmap='gray')

  plt.title(train_pneumonia_fnames[n])

  plt.axis('off')

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Add our data-augmentation parameters to ImageDataGenerator

train_datagen = ImageDataGenerator(

  rescale = 1./255.,

  rotation_range = 10,

  width_shift_range = 0.2,

  height_shift_range = 0.2,

  shear_range = 0,

  zoom_range = 0.2,

  horizontal_flip = False)



# Note that the validation data should not be augmented!

validation_datagen = ImageDataGenerator(rescale = 1./255.)



train_generator = train_datagen.flow_from_directory(

  train_dir,

  batch_size = 20,

  class_mode = 'binary',

  target_size = (image_size, image_size))



print(train_generator.class_indices)

classes = np.array(train_generator.classes)

(_ , occurences) = np.unique(classes, return_counts=True)

neg = occurences[0]

pos = occurences[1]

tot = neg + pos

print('Training normal cases: {}'.format(neg))

print('Training pneumonia cases: {}'.format(pos))

print('Training total cases: {}'.format(tot))



validation_generator = validation_datagen.flow_from_directory(

  validation_dir,

  batch_size = 20,

  class_mode = 'binary',

  target_size = (image_size, image_size))
weight_for_0 = tot/neg/2.0 

weight_for_1 = tot/pos/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))
history = model.fit(

  train_generator,

  validation_data = validation_generator,

  steps_per_epoch = len(train_generator),

  validation_steps = len(validation_generator),

  epochs = 10,

  class_weight = class_weight,

  verbose = 2)
import matplotlib.pyplot as plt



acc = history.history['accuracy']

loss = history.history['loss']

recall = history.history['recall']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'g', label='Validation loss')

plt.title('Training')

plt.legend(loc=0)

plt.axis([1, len(acc), 0, 1])

plt.grid(True)

plt.figure()

plt.show()
test_datagen = ImageDataGenerator(rescale = 1./255.)

test_generator = test_datagen.flow_from_directory(

  test_dir,

  batch_size = 20,

  class_mode = 'binary',

  shuffle = False,

  target_size = (image_size, image_size))



print(test_generator.class_indices)

(_ , occurences) = np.unique(test_generator.classes, return_counts=True)

test_neg = occurences[0]

test_pos = occurences[1]

test_tot = test_neg + test_pos

print('Test normal cases: {}'.format(test_neg))

print('Test pneumonia cases: {}'.format(test_pos))

print('Test total cases: {}'.format(test_tot))
results = model.evaluate(test_generator, steps=len(test_generator))

for name, value in zip(model.metrics_names, results):

  print(name, ': ', value)
predictions = []

labels = []

for i in range(len(test_generator)):

  x, y = next(test_generator)

  predictions.append(model.predict(x))

  labels.append(y)

predictions = np.concatenate(predictions)

labels = np.concatenate(labels)
import seaborn as sns

from sklearn.metrics import confusion_matrix



def plot_cm(labels, predictions, p=0.5):

  cm = confusion_matrix(labels, predictions > p)

  plt.figure(figsize=(5,5))

  sns.heatmap(cm, annot=True, fmt="d")

  plt.title('Confusion matrix @{:.2f}'.format(p))

  plt.ylabel('Actual label')

  plt.xlabel('Predicted label')



plot_cm(labels, predictions)
import sklearn.metrics



def plot_roc(name, labels, predictions, **kwargs):

  fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predictions)

  plt.figure(figsize=(5,5))

  plt.plot(fpr, tpr, label=name, linewidth=2, **kwargs)

  plt.xlabel('False positives rate')

  plt.ylabel('True positives rate')

  plt.grid(True)

  # Youden's J statistic

  J = tpr - fpr

  ix = np.argmax(J)

  best_threshold = thresholds[ix]

  print('Best Threshold : {:.2f}'.format(best_threshold))

  plt.plot([0,1], [0,1], linestyle='--', color='k', label='No Skill')

  plt.scatter(fpr[ix], tpr[ix], marker='o', color='r', label='Best')

  ax = plt.gca()

  ax.set_aspect('equal')

  return best_threshold



best_threshold = plot_roc(

  "Test",

  labels,

  predictions,

  color='b')

plt.legend(loc='lower right')
plot_cm(labels, predictions, p=best_threshold)
NEW_METRICS = [

  TruePositives(name='tp', thresholds = best_threshold),

  FalsePositives(name='fp', thresholds = best_threshold),

  TrueNegatives(name='tn', thresholds = best_threshold),

  FalseNegatives(name='fn', thresholds = best_threshold),

  BinaryAccuracy(name='accuracy', threshold = best_threshold),

  Precision(name='precision', thresholds = best_threshold),

  Recall(name='recall', thresholds = best_threshold),

  AUC(name='auc')

]

for m in NEW_METRICS:

  m.update_state(labels, predictions) 

  print (m.name, ': ', m.result().numpy())
import numpy as np

from keras.preprocessing import image



test_normal_fnames = os.listdir(test_normal_dir)

test_pneumonia_fnames = os.listdir(test_pneumonia_dir)



true_negatives = 0

false_positives = 0

for fn in test_normal_fnames:

  path = os.path.join(test_normal_dir, fn)

  img = image.load_img(path, target_size=(image_size, image_size))

  x = image.img_to_array(img)

  x /= 255.0

  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])

  out = model.predict(images)

  if out <= best_threshold:

    true_negatives += 1

  else:

    false_positives += 1



true_positives = 0

false_negatives = 0

for fn in test_pneumonia_fnames:

  path = os.path.join(test_pneumonia_dir, fn)

  img = image.load_img(path, target_size=(image_size, image_size))

  x = image.img_to_array(img)

  x /= 255.0

  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])

  out = model.predict(images)

  if out <= best_threshold:

    false_negatives += 1

  else:

    true_positives += 1



print('Best threshold: ', best_threshold)

print('True negatives: ', true_negatives)

print('False positives: ', false_positives)

print('False negatives: ', false_negatives)

print('True positives: ', true_positives)