!pip install tensorflow==2.2.0
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tf.__version__
PATH = "../input/chest-xray-pneumonia/chest_xray"
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')
test_dir = os.path.join(PATH, 'test')

train_normal = os.path.join(train_dir ,'NORMAL')
train_pneu   = os.path.join(train_dir ,'PNEUMONIA')
validation_normal = os.path.join(validation_dir , 'NORMAL')
validation_pnem = os.path.join(validation_dir , 'PNEUMONIA')
test_normal = os.path.join(test_dir ,'NORMAL')
test_pnem   = os.path.join(test_dir ,'PNEUMONIA')
num_normal_tr = len(os.listdir(train_normal))
num_pnem_tr = len(os.listdir(train_pneu))

num_normal_val = len(os.listdir(validation_normal))
num_pnem_val = len(os.listdir(validation_pnem))

num_normal_test = len(os.listdir(test_normal))
num_pnem_test = len(os.listdir(test_pnem))

total_train = num_normal_tr + num_pnem_tr
total_val = num_normal_val + num_pnem_val
total_test = num_normal_test + num_pnem_test

print('total training normal images:', num_normal_tr)
print('total training pnemonia images:', num_pnem_tr)

print('total validation normal images:', num_normal_val)
print('total validation pnemonia images:', num_pnem_val)

print('total test normal images:', num_normal_test)
print('total test pnemonia images:', num_pnem_test)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
print("Total test images:", total_test)
list_train_normal_ds = tf.data.Dataset.list_files(str(train_dir + '*/NORMAL/*'))
list_train_pneu_ds = tf.data.Dataset.list_files(str(train_dir + '*/PNEUMONIA/*'))
list_val_normal_ds = tf.data.Dataset.list_files(str(validation_dir + '*/NORMAL/*'))
list_val_pneu_ds = tf.data.Dataset.list_files(str(validation_dir + '*/PNEUMONIA/*'))
list_combined_normal = list_train_normal_ds.concatenate(list_val_normal_ds)
list_combined_pnem = list_train_pneu_ds.concatenate(list_val_pneu_ds)
total_val_train_combined_normal = len(list(list_combined_normal.as_numpy_iterator()))
total_val_train_combined_pnem = len(list(list_combined_pnem.as_numpy_iterator()))
print("Total normal images after combining train and validation:",total_val_train_combined_normal)
print("Total pneumonia images after combining train and validation:",total_val_train_combined_pnem)
val_normal_size = int(0.15 * total_val_train_combined_normal)
val_pnem_size = int(0.10 * total_val_train_combined_pnem)

list_combined_normal_ds = list_combined_normal.shuffle(1000)
list_combined_normal = list(list_combined_normal_ds.as_numpy_iterator())

list_val_normal = list_combined_normal[:val_normal_size]
list_train_normal = list_combined_normal[val_normal_size:]

list_val_normal_dataset = tf.data.Dataset.from_tensor_slices(list_val_normal)
list_train_normal_dataset = tf.data.Dataset.from_tensor_slices(list_train_normal)
len(set(list(list_val_normal_dataset.as_numpy_iterator())) & set(list(list_train_normal_dataset.as_numpy_iterator())))
list_combined_pnem_ds = list_combined_pnem.shuffle(4000)
list_combined_pnem = list(list_combined_pnem_ds.as_numpy_iterator())

list_val_pnem = list_combined_pnem[:val_pnem_size]
list_train_pnem = list_combined_pnem[val_pnem_size:]

list_val_pnem_dataset = tf.data.Dataset.from_tensor_slices(list_val_pnem)
list_train_pnem_dataset = tf.data.Dataset.from_tensor_slices(list_train_pnem)
len(set(list(list_val_pnem_dataset.as_numpy_iterator())) & set(list(list_train_pnem_dataset.as_numpy_iterator())))
list_val_ds = list_val_normal_dataset.concatenate(list_val_pnem_dataset).shuffle(500)
list_train_ds = list_train_normal_dataset.concatenate(list_train_pnem_dataset).shuffle(4600)
list_test_ds = tf.data.Dataset.list_files(str(test_dir + '*/*/*'), shuffle=False)
len(set(list(list_val_ds.as_numpy_iterator())) & set(list(list_train_ds.as_numpy_iterator())) & set(list(list_test_ds.as_numpy_iterator())))
total_train = len(list(list_train_ds.as_numpy_iterator()))
total_val = len(list(list_val_ds.as_numpy_iterator()))
print("Total train after split:",total_train)
print("Total validation after split:", total_val)
print("Total test:", len(list(list_test_ds.as_numpy_iterator())))
for f in list_train_ds.take(5):
  print(f.numpy())
for f in list_val_ds.take(5):
  print(f.numpy())
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

AUTOTUNE = tf.data.experimental.AUTOTUNE
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  #The second to last is the class-directory
  if (parts[-2] == 'PNEUMONIA'):
    class_label = 1
  else:
    class_label = 0
  return class_label

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_train_ds = list_train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_val_ds = list_val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
labeled_test_ds = list_test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
for image, label in labeled_train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())
def prepare_for_training(ds, cache=True, shuffle_buffer_size=4000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)#, reshuffle_each_iteration = True)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds
train_ds = prepare_for_training(labeled_train_ds)
image_batch, label_batch = next(iter(train_ds))
image_batch.numpy().shape
label_batch
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(label_batch[n].numpy())
      plt.axis('off')
show_batch(image_batch, label_batch)
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.0001,
  decay_steps=(total_train // BATCH_SIZE)*15,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.RMSprop(lr_schedule)
base_learning_rate = 0.0001
model.compile(optimizer=get_optimizer(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
initial_epochs = 15
validation_steps=total_val // BATCH_SIZE

loss0,accuracy0 = model.evaluate(labeled_val_ds.batch(32), steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(train_ds,
                    epochs=initial_epochs,
                    steps_per_epoch=total_train // BATCH_SIZE,
                    validation_data=labeled_val_ds.batch(32),
                    validation_steps=total_val // BATCH_SIZE)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
# We should recompile the model for the above changes to take effect
# Also using a much lower learning rate..
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()
len(model.trainable_variables)
fine_tune_epochs = 12
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         steps_per_epoch=total_train // BATCH_SIZE,
                         initial_epoch =  history.epoch[-1],
                         validation_data=labeled_val_ds.batch(32),
                         validation_steps=total_val // BATCH_SIZE
                         )
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
model.evaluate(labeled_test_ds.batch(32))
predictions_logits = model.predict(labeled_test_ds.batch(32))
prediction_labels = np.where(predictions_logits >=0, 1, 0)
orig_test_labels = [image_lable_tuple[1] for image_lable_tuple in list(labeled_test_ds.as_numpy_iterator())]
cf_matrix = tf.math.confusion_matrix(orig_test_labels, prediction_labels.squeeze())
cf_matrix
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True label')
plt.xlabel('Predicted label')
tn, fp, fn, tp = cf_matrix.numpy().ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is {:.4f}".format(recall))
print("Precision of the model is {:.4f}".format(precision))
