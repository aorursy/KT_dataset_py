import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import os
import shutil
import matplotlib.pyplot as plt
train = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')

target = train[['healthy', 'multiple_diseases', 'rust', 'scab']]
test_ids = test['image_id']

train_len = train.shape[0]
test_len = test.shape[0]

train.describe()
print("Shape of train data: " + str(train.shape))
print("Shape of test data: " + str(test.shape))

train_len = train.shape[0]
test_len = test.shape[0]
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tqdm.notebook import tqdm

path = '../input/plant-pathology-2020-fgvc7/images/'
size = 224

train_images = np.ndarray(shape=(train_len, size, size, 3))
for i in tqdm(range(train_len)):
  img = load_img(path + f'Train_{i}.jpg', target_size=(size, size))
  train_images[i] = np.uint8(img_to_array(img))

test_images = np.ndarray(shape=(test_len, size, size, 3))
for i in tqdm(range(test_len)):
  img = load_img(path + f'Test_{i}.jpg', target_size=(size, size))
  test_images[i] = np.uint8(img_to_array(img))

train_images.shape, test_images.shape
for i in range(4):
	plt.subplot(220 + 1 + i)
	plt.title(train['image_id'][i])
	plt.imshow(np.uint8(train_images[i]))
plt.show()
for i in range(4):
	plt.subplot(220 + 1 + i)
	plt.title(test['image_id'][i])
	plt.imshow(np.uint8(test_images[i]))
plt.show()
plt.savefig('test_images.png')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_images, target.to_numpy(), test_size=0.1, random_state=289) 

x_train.shape, x_test.shape, y_train.shape, y_test.shape
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=289)

x_train, y_train = ros.fit_resample(x_train.reshape((-1, size * size * 3)), y_train)
x_train = x_train.reshape((-1, size, size, 3))
x_train.shape, y_train.shape
import gc

del train_images
gc.collect()
from keras_preprocessing.image import ImageDataGenerator

batch_size = 8

train_datagen = ImageDataGenerator(samplewise_center = True,
                                   samplewise_std_normalization = True,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   rotation_range=20)

train_generator = train_datagen.flow(
    x = x_train, 
    y = y_train,
    batch_size = batch_size)

validation_datagen = ImageDataGenerator(samplewise_center = True,
                                        samplewise_std_normalization = True)

validation_generator = validation_datagen.flow(
    x = x_test, 
    y = y_test,
    batch_size = batch_size)
idx = np.random.randint(8)
x, y = train_generator.__getitem__(idx)
plt.title(y[idx])
plt.imshow(x[idx])
plt.savefig('processed_img.png')
base_model = tf.keras.applications.ResNet50(include_top = False, weights='imagenet', input_shape=(size, size, 3))

def create_model():
    model = tf.keras.Sequential([
      base_model,
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(4, activation='softmax')
      ])
    model.compile(
        loss = 'kullback_leibler_divergence', 
        optimizer = 'adam', 
        metrics = ['accuracy'])
    return model

model = create_model()

model.summary()
epochs = 100
steps_per_epoch = x_train.shape[0] // batch_size
validation_steps = x_test.shape[0] // batch_size
print(steps_per_epoch)
es = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1)
mc = tf.keras.callbacks.ModelCheckpoint('model.hdf5', save_best_only=True, verbose=0)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1)

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005
rampup_epochs = 20
sustain_epochs = 15
exp_decay = .8

def lrfn(epoch):
  if epoch < rampup_epochs:
    return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
  elif epoch < rampup_epochs + sustain_epochs:
    return max_lr
  else:
    return min_lr
    
lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

rang = np.arange(epochs)
y = [lrfn(x) for x in rang]
plt.plot(rang, y)
print('Learning rate per epoch:')
history = model.fit(
    x = train_generator,  
    validation_data = validation_generator,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    verbose=1,
    callbacks=[es, lr, mc, rlr])
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
train_err = (1-history.history['accuracy'][-1])*100
validation_err = (1-history.history['val_accuracy'][-1])*100
print("Train set error " + str(train_err))
print("Validation set error " + str(validation_err))
test_datagen = ImageDataGenerator(samplewise_center = True,
                                 samplewise_std_normalization = True)

test_generator = test_datagen.flow(
    x = test_images,
    shuffle = False)
probabilities = model.predict(test_generator, steps = len(test_generator))
print(probabilities[:,0].mean()*100)
print(probabilities[:,1].mean()*100)
print(probabilities[:,2].mean()*100)
print(probabilities[:,3].mean()*100)
base_model.trainable = True
model.summary()
model.compile(
        loss = 'kullback_leibler_divergence', 
        optimizer = tf.keras.optimizers.Adam(1e-5), 
        metrics = ['accuracy'])
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True)
epochs = 10
history = model.fit(
    x = train_generator,  
    validation_data = validation_generator,
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    verbose=1,
    callbacks = [es])
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
train_err = (1-history.history['accuracy'][-1])*100
validation_err = (1-history.history['val_accuracy'][-1])*100
print("Train set error " + str(train_err))
print("Validation set error " + str(validation_err))