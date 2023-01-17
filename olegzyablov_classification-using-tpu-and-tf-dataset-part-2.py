!pip install tfa-nightly -q
import tensorflow as tf
import tensorflow_addons as tfa
tfa.register_all()
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re, math, random, time, gc

try:
    tf.get_logger().propagate = False
    if 'COLAB_TPU_ADDR' in os.environ: #colab
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR'])
    else: #kaggle
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    print('TPU initialized, num of accelerators:', strategy.num_replicas_in_sync)
except (ValueError, KeyError):
    print('TPU not found, using CPU/GPU')
    strategy = tf.distribute.get_strategy()
class_names = [
  'Приора', #0
  'Ford Focus', #1
  'Самара', #2
  'ВАЗ-2110', #3
  'Жигули', #4
  'Нива', #5
  'Калина', #6
  'ВАЗ-2109', #7
  'Volkswagen Passat', #8
  'ВАЗ-21099' #9
]

def visualize_dataset(pairs, grayscale = False): #pair = (image, title)
    images_count = len(pairs)
    fig = plt.figure(figsize = (min(24, images_count * 4), 4))
    for img_index in range(images_count):
        ax = fig.add_subplot(1, images_count, img_index + 1)
        img, title = pairs[img_index]
        if tf.is_tensor(img): img = img.numpy()
        if tf.is_tensor(title): title = title.numpy()
        ax.imshow(img, cmap = 'gray' if grayscale else None)
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def get_dataset(
  size, #например (384, 512)
  normalize = True, #нормализовывать от -1 до 1 (иначе значения от 0 до 255)
  dtype = tf.float32, #или uint8
  validation_ratio = None, #если не None, возвращает 2 датасета: train и val
  random_seed = None, #сид для деления на train и val (если указан validation_ratio) и для перемешивания
  preprocess = None, #функция для всех изображений (до аугментации) img float32 0-255 -> img float32 0-255
  augmentation = None, #функция для тренировочных изображений img float32 0-255 -> img float32 0-255
  augmentation_val = None, #функция для тестовых изображений (обычно не требуется) img float32 0-255 -> img float32 0-255
  postprocess = None, #функция для всех изображений (после аугментации) img float32 0-255 -> img float32 0-255
  shuffle_train = True, #перемещивать тренировочные данные с буфером 30
  shuffle_buffer_size = 300, #размер буфера при перемешивании (меньше буфер - меньше расход памяти)
  visualize = False, #показывать первые 6 изображений
):
  def read_train_tfrecord(serialized_example):
    example = tf.io.parse_single_example(serialized_example, features = {
      'image': tf.io.FixedLenFeature([], tf.string), #tf.string - байтовая строка; [] означает скаляр, т. е. только одна строка
      'class': tf.io.FixedLenFeature([], tf.int64)
    })
    return tf.image.decode_jpeg(example['image'], channels = 3), example['class']
  
  def pad(image, label): #uint8 -> float32
    image = tf.image.resize(image, size, preserve_aspect_ratio = True)
    image = tf.image.resize_with_pad(image, *size)
    return image, label
  
  dataset = tf.data.TFRecordDataset(
    ['gs://oleg-zyablov/car-classification/train_tfrec/%d.tfrec' % i for i in range(16)],
    num_parallel_reads = 16
  ).map(read_train_tfrecord).cache().map(pad)
  
  if preprocess:
    preprocess_tuple_func = lambda img, label: (preprocess(img), label)
    dataset = dataset.map(preprocess_tuple_func)

  make_val_dataset = (validation_ratio is not None)
  
  if not make_val_dataset:
    assert(augmentation_val == None)
    train_dataset = dataset
  else:
    total_len = 15561
    val_len = round(total_len * validation_ratio)
    val_split = np.zeros(total_len)
    np.random.seed(random_seed)
    val_indices = np.random.choice(range(total_len), val_len)
    val_split[val_indices] = 1
    val_split = tf.constant(val_split)

    train_filter_func = lambda i, data: (val_split[i] == 0)
    val_filter_func = lambda i, data: (val_split[i] == 1)
    remove_enumeration = lambda i, data: data
    dataset = dataset.enumerate()
    train_dataset = dataset.filter(train_filter_func).map(remove_enumeration)
    val_dataset = dataset.filter(val_filter_func).map(remove_enumeration)
  
  if shuffle_train:
    train_dataset = train_dataset.shuffle(buffer_size = shuffle_buffer_size, seed = random_seed, reshuffle_each_iteration = True)
  
  if augmentation is not None:
    augmentation_tuple_func = lambda img, label: (augmentation(img), label)
    train_dataset = train_dataset.map(augmentation_tuple_func)
  if augmentation_val is not None:
    val_augmentation_tuple_func = lambda img, label: (augmentation_val(img), label)
    val_dataset = val_dataset.map(val_augmentation_tuple_func)
  
  if postprocess:
    postprocess_tuple_func = lambda img, label: (postprocess(img), label)
    train_dataset = train_dataset.map(postprocess_tuple_func)
    if make_val_dataset:
      val_dataset = val_dataset.map(postprocess_tuple_func)

  if normalize:
    normalization_tuple_func = lambda img, label: (img / 128 - 1, label)
    assert(dtype != tf.uint8)
    train_dataset = train_dataset.map(normalization_tuple_func)
    if make_val_dataset:
      val_dataset = val_dataset.map(normalization_tuple_func)
  
  dtype_cast_tuple_func = lambda img, label: (tf.cast(img, dtype), label)
  train_dataset = train_dataset.map(dtype_cast_tuple_func)
  if make_val_dataset:
    val_dataset = val_dataset.map(dtype_cast_tuple_func)

  if visualize:
    if make_val_dataset:
      print('train examples:')
    if normalize:
      visualize_dataset([(tf.cast((img + 1) * 128, np.uint8), class_names[label]) for img, label in train_dataset.take(6)])
    else:
      visualize_dataset([(tf.cast(img, np.uint8), class_names[label]) for img, label in train_dataset.take(6)])
    if make_val_dataset:
      print('test examples:')
      if normalize:
        visualize_dataset([(tf.cast((img + 1) * 128, np.uint8), class_names[label]) for img, label in val_dataset.take(6)])
      else:
        visualize_dataset([(tf.cast(img, np.uint8), class_names[label]) for img, label in val_dataset.take(6)])
  
  if make_val_dataset:
    return train_dataset, val_dataset
  else:
    return train_dataset

def zoom(x, size): #zooming an image (функция взята из интернета)
    scales = list(np.arange(0.6, 1.2, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=size)
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return random_crop(x)
    #return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def augmentation(image):
  image = tf.cast(image, tf.uint8)

  #image = tfa.image.shear_x(image, 0.5, 0)

  random_val = tf.random.uniform(shape = [], minval = 0.0, maxval = 1.0)
  image = tf.cond(random_val > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
  image = tf.image.random_hue(image, 0.1)
  image = tf.image.random_saturation(image, 0.6, 1.6)
  image = tf.image.random_brightness(image, 0.05)
  image = tfa.image.rotate(image, tf.random.uniform(shape = [], minval = -0.3, maxval = 0.3))

  image = zoom(image, size = tf.shape(image)[:2])
  image = tf.expand_dims(image, 0)
  image = tfa.image.random_cutout(image, (128, 128))
  image = image[0]
  image = tf.cast(image, tf.float32)
  return image

def grayscale(image):
  image = tf.image.rgb_to_grayscale(image)
  image = tf.concat((image, image, image), axis = 2)  
  return image

ds, val_ds = get_dataset(size = (384, 512), validation_ratio = 0.15, augmentation = augmentation, postprocess = None, random_seed = 0, visualize = True)
img = ds.__iter__().__next__()[0]
plt.figure()
plt.imshow(tf.cast((img + 1) * 128, tf.uint8).numpy())
plt.show()
print(img)
size = (384, 512)
val_data_percent = 0.01
batch_size = 64
top_layers = [GlobalMaxPooling2D(), BatchNormalization(), Dropout(0.3), Dense(100, activation = 'relu', kernel_regularizer = 'l2'), BatchNormalization(), Dense(10)]
epochs = 16
verbose = 1
optimizer = Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps = 100, decay_rate = 0.9))

with strategy.scope():
  total_count = 15561
  val_count = int(total_count * val_data_percent)
  train_count = total_count - val_count
  steps_per_epoch = train_count // batch_size
  #steps_per_epoch = total_count // batch_size

  train_dataset, val_dataset = get_dataset(size = size, validation_ratio = val_data_percent, shuffle_buffer_size = 30, augmentation = augmentation)
  #train_dataset = get_dataset(size = size, shuffle_buffer_size = 100, augmentation = augmentation, visualize = True)

  #функция get_real_metrics нужна поскольку keras callback на каждом батче возвращает среднее значение точности по эпохе, нужно найти значение на последнем батче
  total_batches = 0 #счетчик батчей
  prev_sum_metrics = None
  def get_real_metrics(batch, logs, metrics = 'accuracy'):
    global prev_sum_metrics, total_batches
    total_batches += 1
    if batch == 0:
      prev_sum_metrics = logs.get(metrics)
      return prev_sum_metrics
    current_averaged_metrics = logs.get(metrics)
    current_sum_metrics = current_averaged_metrics * (batch + 1)
    real_metrics = current_sum_metrics - prev_sum_metrics
    prev_sum_metrics = current_sum_metrics
    return real_metrics

  #функция on_epoch_end будет схранять значение точности на валидаци после каждой эпохи
  val_acc_batches_history = [] #кол-во обработанных батчей после каждой эпохи
  val_acc_history = [] #точность на валидации после каждой эпохи
  def on_epoch_end(epoch, logs):
    val_acc_batches_history.append(total_batches)
    val_acc_history.append(logs.get('val_accuracy'))

  acc_history = [] #история точности для каждого батча
  print('constructing model')
  gc.collect()

  # conv = tf.keras.applications.xception.Xception(
  #     weights = 'imagenet',
  #     include_top = False,
  #     input_shape = (size[0], size[1], 3)
  # )

  conv = tf.keras.applications.InceptionResNetV2(
    weights = 'imagenet',
    include_top = False,
    input_shape = (size[0], size[1], 3)
  )

  x = conv.output
  for layer in top_layers:
    x = layer(x)
  predictions = x

  model = tf.keras.Model(inputs = conv.input, outputs = predictions)

  model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = optimizer,
    metrics = ['accuracy']
  )

  start_time = time.time()
  model.fit(
      train_dataset.repeat().batch(batch_size),
      steps_per_epoch = steps_per_epoch,
      epochs = epochs,
      validation_data = val_dataset.batch(batch_size),
      callbacks = [
          tf.keras.callbacks.LambdaCallback(on_batch_end = lambda batch, logs: acc_history.append(get_real_metrics(batch, logs))),
          tf.keras.callbacks.LambdaCallback(on_epoch_end = on_epoch_end),
          tf.keras.callbacks.TerminateOnNaN()
      ],
      verbose = verbose
  )
  
elapsed_time = time.time() - start_time
print('validation accuracy = %g, seconds elapsed: %.1f' % (val_acc_history[-1], elapsed_time))
model.save('/inception.h5')
!gcloud auth login
!gsutil cp /inception.h5 gs://oleg-zyablov/car-classification/inception.h5
def get_test_dataset(
  size, #например (384, 512)
  normalize = True, #нормализовывать от -1 до 1 (иначе значения от 0 до 255)
  dtype = tf.float32, #или uint8
  remove_filenames = True, #датасет содержит изображения без имен файлов; автоматически равно True если указан параметр pseudo_labels
  pseudo_labels = None, #словарь имя файла -> класс; датасет будет содержать пары (изображение, класс) и содержать только изображения, присутствующие в словаре
  random_seed = None, #сид для перемешивания
  preprocess = None, #функция для изображений (до аугментации) img float32 0-255 -> img float32 0-255
  augmentation = None, #функция для изображений img float32 0-255 -> img float32 0-255
  postprocess = None, #функция для изображений (после аугментации) img float32 0-255 -> img float32 0-255
  shuffle = False, #перемещивать данные с буфером 30
  shuffle_buffer_size = 300, #размер буфера при перемешивании (меньше буфер - меньше расход памяти)
  visualize = False, #показывать первые 6 изображений
):
  def read_test_tfrecord(serialized_example):
    example = tf.io.parse_single_example(serialized_example, features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'filename': tf.io.FixedLenFeature([], tf.string)
    })
    return tf.image.decode_jpeg(example['image'], channels = 3), example['filename']
  
  def pad(image, filename): #uint8 -> float32
    image = tf.image.resize(image, size, preserve_aspect_ratio = True)
    image = tf.image.resize_with_pad(image, *size)
    return image, filename
  
  dataset = tf.data.TFRecordDataset(
    ['gs://oleg-zyablov/car-classification/test_tfrec/%d.tfrec' % i for i in range(7)],
    num_parallel_reads = 7
  ).map(read_test_tfrecord).cache().map(pad)
  
  if preprocess:
    preprocess_tuple_func = lambda img, filename: (preprocess(img), filename)
    dataset = dataset.map(preprocess_tuple_func)
  
  if shuffle:
    dataset = dataset.shuffle(buffer_size = shuffle_buffer_size, seed = random_seed, reshuffle_each_iteration = True)
  
  if augmentation is not None:
    augmentation_tuple_func = lambda img, filename: (augmentation(img), filename)
    dataset = dataset.map(augmentation_tuple_func)
  
  if postprocess:
    postprocess_tuple_func = lambda img, filename: (postprocess(img), filename)
    dataset = dataset.map(postprocess_tuple_func)

  if normalize:
    normalization_tuple_func = lambda img, filename: (img / 128 - 1, filename)
    assert(dtype != tf.uint8)
    dataset = dataset.map(normalization_tuple_func)
  
  dtype_cast_tuple_func = lambda img, filename: (tf.cast(img, dtype), filename)
  dataset = dataset.map(dtype_cast_tuple_func)

  if pseudo_labels is not None:
    items = list(pseudo_labels.items())
    keys_tensor = tf.constant(value = [item[0] for item in items], dtype = tf.string)
    vals_tensor = tf.constant(value = [item[1] for item in items], dtype = tf.int32)
    dict_intializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    labels_dict = tf.lookup.StaticHashTable(dict_intializer, default_value = -1)
    add_labels_func = lambda img, filename: (img, labels_dict.lookup(filename))
    filter_labels_func = lambda img, label: (label != -1)
    dataset = dataset.map(add_labels_func).filter(filter_labels_func)
  elif remove_filenames:
    remove_filenames_func = lambda img, filename: img
    dataset = dataset.map(remove_filenames_func)
  
  def img_to_uint8(img):
    if normalize:
      img = (img + 1) * 128
    img = tf.cast(img, np.uint8)
    return img
  
  def dataset_pair_to_img_title_pair(pair):
    if pseudo_labels is not None:
      img, label = pair
      img = img_to_uint8(img)
      title = class_names[label]
      return img, title
    elif not remove_filenames:
      img, filename = pair
      img = img_to_uint8(img)
      title = filename.numpy().decode('utf-8')
      return img, title
    else:
      img = pair
      img = img_to_uint8(img)
      return img, ''

  if visualize:
    visualize_dataset([dataset_pair_to_img_title_pair(pair) for pair in dataset.take(6)])
  
  return dataset

#test_ds = get_test_dataset(size = (256, 384), pseudo_labels = {'34725.jpg': 1}, postprocess = grayscale, random_seed = 0, visualize = True)
test_ds = get_test_dataset(size = (384, 512), remove_filenames = False, visualize = True)
filenames = [filename for img, filename in test_ds]
print(len(filenames))
all_predictions = np.zeros((6675, 20))
for i in range(20):
  print(i)
  test_ds_without_filenames = get_test_dataset(size = (384, 512), augmentation = augmentation, remove_filenames = True, visualize = True)
  predictions = model.predict(test_ds_without_filenames.batch(32))
  predictions_labels = predictions.argmax(axis = 1)
  all_predictions[:, i] = predictions_labels
result_predictions = np.zeros(6675, dtype = int)
easy_test_examples_all = [dict() for i in range(21)]
for i in range(6675):
  preds = all_predictions[i]
  counts = np.bincount(preds.astype(int))
  maxresult = np.argmax(counts)
  result_predictions[i] = maxresult
  maxcount = np.max(counts)
  filename = filenames[i].numpy().decode('utf-8')
  for j in range(2, 21):
    if maxcount >= j:
      easy_test_examples_all[j][filename] = maxresult
result_predictions
submission = pd.DataFrame({'Id': [i.numpy().decode('utf-8') for i in filenames], 'Category': result_predictions}, columns=['Id', 'Category'])
submission.head()
submission.to_csv('/submission.csv', index = False)
from google.colab import files
files.download('/submission.csv')
np.save('/all_predictions.npy', all_predictions)
!gsutil cp /all_predictions.npy gs://oleg-zyablov/car-classification/all_predictions.npy
easy_examples = easy_test_examples_all[20]
test_ds_easy = get_test_dataset(size = (384, 512), pseudo_labels = easy_examples, shuffle = True, visualize = True, augmentation = augmentation)
with strategy.scope():

  #train_dataset, val_dataset = get_dataset(size = size, validation_ratio = val_data_percent, shuffle_buffer_size = 30, augmentation = augmentation)

  start_time = time.time()
  model.fit(
      test_ds_easy.repeat().batch(batch_size),
      steps_per_epoch = steps_per_epoch,
      epochs = 3,
      validation_data = val_dataset.batch(batch_size),
      callbacks = [
          tf.keras.callbacks.LambdaCallback(on_batch_end = lambda batch, logs: acc_history.append(get_real_metrics(batch, logs))),
          tf.keras.callbacks.LambdaCallback(on_epoch_end = on_epoch_end),
          tf.keras.callbacks.TerminateOnNaN()
      ],
      verbose = verbose
  )
  
elapsed_time = time.time() - start_time
print('validation accuracy = %g, seconds elapsed: %.1f' % (val_acc_history[-1], elapsed_time))
model.save('/inception_with_pseudo_labels.h5')
!gsutil cp /inception_with_pseudo_labels.h5 gs://oleg-zyablov/car-classification/inception_with_pseudo_labels.h5
gc.collect()
gc.collect()
gc.collect()
predictions_without_augmentations = np.zeros(6675)
test_ds_without_filenames = get_test_dataset(size = (384, 512), remove_filenames = True, visualize = True)
predictions = model.predict(test_ds_without_filenames.batch(32))
predictions_without_augmentations = predictions.argmax(axis = 1)
predictions_without_augmentations
all_predictions2 = np.zeros((6675, 20))
for i in range(20):
  print(i)
  test_ds_without_filenames = get_test_dataset(size = (384, 512), augmentation = augmentation, remove_filenames = True, visualize = True)
  predictions = model.predict(test_ds_without_filenames.batch(32))
  predictions_labels = predictions.argmax(axis = 1)
  all_predictions2[:, i] = predictions_labels
all_predictions2.shape
predictions_without_augmentations
np.save('/all_predictions2.npy', all_predictions2)
!gsutil cp /all_predictions2.npy gs://oleg-zyablov/car-classification/all_predictions2.npy
all_predictions_weighted = np.concatenate(([all_predictions2] + [predictions_without_augmentations[:, np.newaxis]] * 5), axis = 1)
np.save('/all_predictions_weighted.npy', all_predictions_weighted)
!gsutil cp /all_predictions_weighted.npy gs://oleg-zyablov/car-classification/all_predictions_weighted.npy
result_predictions2 = np.zeros(6675, dtype = int)
for i in range(6675):
  preds = all_predictions_weighted[i]
  counts = np.bincount(preds.astype(int))
  maxresult = np.argmax(counts)
  result_predictions2[i] = maxresult
  maxcount = np.max(counts)
result_predictions2
#запускаем ракету
submission = pd.DataFrame({'Id': [i.numpy().decode('utf-8') for i in filenames], 'Category': result_predictions2}, columns=['Id', 'Category'])
submission.to_csv('/submission.csv', index = False)
from google.colab import files
files.download('/submission.csv')
