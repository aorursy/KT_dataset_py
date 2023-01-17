import pandas as pd

root_path = '../input/'
root_folder = ''

train_file = 'fashion-mnist_train.csv'
test_file = 'fashion-mnist_test.csv'

train = pd.read_csv(root_path+root_folder+train_file)
test = pd.read_csv(root_path+root_folder+test_file)
train['label'].value_counts()
image_slice = slice(1,None)
image_dims = (28, 28)
input_size = image_dims+(1,)

classes = ['T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot']
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_image(image, title=None):
  plt.axis('off')
  plt.imshow(image, interpolation='nearest')
  if(title is not None):
    plt.title(title)

def get_class_name(logit, classes=classes):
  if(logit < len(classes) and logit >= 0):
    return classes[logit]
  else:
    return 'Unknown'
  
def get_image(df, index=None, 
              image_slice=image_slice, 
              image_dimensions=image_dims):
  if(index is None):
    index = random.randint(0, len(df))
  image = df.iloc[index,image_slice]
  return np.resize(image, image_dimensions), df.iloc[index, 0]

def plot_random_samples(df, grid_shape=(8,8),
                     plot=plot_image, 
                     get_image=get_image,
                     get_class_name=get_class_name):
  plt.figure(figsize=(8,10))
  for index in range(grid_shape[0] * grid_shape[1]):
    plt.subplot(grid_shape[0], grid_shape[1], index + 1)
    image, label = get_image(df)
    plot(image, get_class_name(label))
plot_random_samples(train)
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def process_labels(df, label_index=0):
  return to_categorical(np.array(df.iloc[:,label_index]))

def prepare_image_feature(df, image_slice=image_slice):
  return np.array(df.iloc[:, image_slice])

def reshape_normalize_image(X, image_shape=image_dims):
  return (X.reshape(len(X), image_shape[0], image_shape[1], 1).astype('float32')) / 255

y = process_labels(train)
X = prepare_image_feature(train)
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                    stratify=y, 
                                                    test_size=0.10, 
                                                    random_state=4422)

y_test = process_labels(test)
X_test = prepare_image_feature(test)

X_train = reshape_normalize_image(X_train)
X_test = reshape_normalize_image(X_test)
X_val = reshape_normalize_image(X_val)

from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    featurewise_center=True,
    samplewise_center=True)

datagen_train.fit(X_train)

datagen_val = ImageDataGenerator(
    rotation_range=3,
    horizontal_flip=True,
    featurewise_center=True,
    samplewise_center=True)

datagen_val.fit(X_val)
# Let's plot the first augmented examples
subplot_dims = (8,8)
index = 0

max_examples = subplot_dims[0]*subplot_dims[1]
for x_batch, y_batch in datagen_train.flow(X_train, y_train, batch_size=max_examples):
  plt.figure(figsize=(8,8))
  for index in range(max_examples):
    plt.subplot(subplot_dims[0], subplot_dims[1], index + 1)
    plot_image(x_batch[index, :,:,0])
  break
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

num_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_size))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 256
epochs = 150

train_step_per_epoch = len(X_train) / batch_size
val_step_per_epoch = len(X_val) / batch_size

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.01,
                              patience=10,
                              mode='auto')

history = model.fit_generator(generator=datagen_train.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    steps_per_epoch=train_step_per_epoch,
                    validation_data=datagen_val.flow(X_val, y_val, batch_size=batch_size),
                    validation_steps=val_step_per_epoch,
                    callbacks=[early_stopping],
                    initial_epoch=0,
                    workers=10, 
                    use_multiprocessing=True,
                    verbose=2)
def plot_history(keras_history,
                 metrics=['acc'], 
                 plot_validation=True, 
                 figure_size=(15,10)):
  plt.figure(figsize=figure_size)
  
  for index, metric in enumerate(metrics):
    ax = plt.subplot(2, 2, index + 1)
    ax.plot(history.history[metric], label=metric)
    if(plot_validation):
      val = 'val_{0}'.format(metric)
      ax.plot(history.history[val], label=val)
    
    ax.legend(framealpha=0.8, fancybox=True)
    max_epoch = len(history.history[metric])
    step = int(max_epoch/15) if (max_epoch > 30) else 2

    plt.xticks(np.arange(0, len(history.history[metric]), step=step))
    ax.set(xlabel='Epoch')
    
plot_history(keras_history=history, metrics=['acc', 'loss'])

stopped_at_epoch = len(history.history['acc'])
evaluations =  model.evaluate(x=X_val, y=y_val, batch_size=batch_size)
evaluations
from sklearn.metrics import classification_report

y_pred_test = to_categorical(model.predict(X_test).argmax(axis=1))

print(classification_report(y_test, y_pred_test, target_names=classes))
history_sup = model.fit(X_train, y_train, 
                     initial_epoch=stopped_at_epoch, 
                     verbose=0,
                     validation_data=(X_val, y_val),
                     epochs=stopped_at_epoch+15)

def concatenate_history(history, history2, metrics=['acc', 'loss'], validation=True):
  for metric in metrics:
    history.history[metric] = history.history[metric] + history_sup.history[metric]
    if(validation):
      history.history['val_'+metric] = history.history['val_'+metric] + history_sup.history['val_'+metric]
     
  return history

history = concatenate_history(history, history_sup)
plot_history(keras_history=history, metrics=['acc', 'loss'])
from sklearn.metrics import classification_report

y_pred_test = to_categorical(model.predict(X_test).argmax(axis=1))

print(classification_report(y_test, y_pred_test, target_names=classes))
