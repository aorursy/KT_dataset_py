import tensorflow as tf
import os

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
#для Colab: нужен файл kaggle.json, его можно скачать с скайта kaggle.com, раздела My account, кнопка "Create New API Token"
#загружаем этот файл в Colab

from google.colab import files
files.upload();
#для Colab: скачиваем данные https://www.kaggle.com/c/sf-dl-2-car-classification
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle competitions download -c sf-dl-car-classification
!mkdir -p /kaggle/input/sf-dl-car-classification
!mv /content/* /kaggle/input/sf-dl-car-classification
!ls /kaggle/input/sf-dl-car-classification
import zipfile
zip_folder = '/kaggle/input/sf-dl-car-classification/'
out_folder = '/kaggle/working/'
with zipfile.ZipFile(zip_folder + 'train.zip', 'r') as z:
    z.extractall(out_folder)
with zipfile.ZipFile(zip_folder + 'test.zip', 'r') as z:
    z.extractall(out_folder)
!ls /kaggle/working
import tensorflow as tf
import random

with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'): #не используем кластер TPU, даже если он доступен, так как работаем с локальными файлами
    
    #задаем пути к файлам
    paths = tf.io.gfile.glob('/kaggle/working/train/*/*.jpg')
    random.shuffle(paths)

    #определяем классы
    classes = [int(path.split('/')[-2]) for path in paths]
    
    #по сколько изображений будем сохранять в один файл?
    shard_size = 1000

    #создаем датасет, который будет возвращать по 1000 изображений и классов
    dataset = tf.data.Dataset.from_tensor_slices((paths, classes))
    dataset = dataset.map(
        lambda path, class_idx: (tf.io.read_file(path), class_idx)
    ).batch(shard_size)

    #создаем папку для файлов .tfrec
    import shutil, pathlib
    out_folder = '/kaggle/working/train_tfrec'
    shutil.rmtree(out_folder, ignore_errors = True) #удаляем папку со старыми файлами
    pathlib.Path(out_folder).mkdir(parents = True, exist_ok = True)

    images_processed = 0
    for batch_index, (images, classes_idx) in enumerate(dataset):
        filename = out_folder + '/%d.tfrec' % batch_index
        
        images_ndarray = images.numpy() #tf.io.read_file возвращает байты, поэтому images - набор байт, т. е. тензор типа tf.string
        classes_ndarray = classes_idx.numpy()
        
        examples_count = images_ndarray.shape[0] #сколько изображений получено? (в последнем массиве их будет меньше shard_size)
        
        print('Writing file: %s [images %d-%d]' % (filename, images_processed, images_processed + examples_count))
        images_processed += examples_count
        
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(examples_count):
                tfrecord = tf.train.Example(features = tf.train.Features(feature = {
                    'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [images_ndarray[i]])),
                    'class': tf.train.Feature(int64_list = tf.train.Int64List(value = [classes_ndarray[i]]))
                }))
                out_file.write(tfrecord.SerializeToString())
# аналогично сохраним тестовые изображения

with tf.device('/job:localhost/replica:0/task:0/device:CPU:0'):
    
    paths = tf.io.gfile.glob('/kaggle/working/test_upload/*.jpg')

    filenames = [path.split('/')[-1] for path in paths]
    
    shard_size = 1000

    dataset = tf.data.Dataset.from_tensor_slices((paths, filenames))
    dataset = dataset.map(
        lambda path, filename: (tf.io.read_file(path), filename)
    ).batch(shard_size)

    out_folder = '/kaggle/working/test_tfrec'
    shutil.rmtree(out_folder, ignore_errors = True) #удаляем папку со старыми файлами
    pathlib.Path(out_folder).mkdir(parents = True, exist_ok = True)

    images_processed = 0
    for batch_index, (images, img_filenames) in enumerate(dataset):
        filename = out_folder + '/%d.tfrec' % batch_index
        
        images_ndarray = images.numpy()
        img_filenames_ndarray = img_filenames.numpy()
        
        examples_count = images_ndarray.shape[0]
        
        print('Writing file: %s [images %d-%d]' % (filename, images_processed, images_processed + examples_count))
        images_processed += examples_count
        
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(examples_count):
                tfrecord = tf.train.Example(features = tf.train.Features(feature = {
                    'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [images_ndarray[i]])),
                    'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_filenames_ndarray[i]]))
                }))
                out_file.write(tfrecord.SerializeToString())
#для Colab (нужно ввести код акторизации)
!gcloud auth login
!gsutil -m cp -r /kaggle/working/train_tfrec/ gs://oleg-zyablov/car-classification/train_tfrec/
!gsutil -m cp -r /kaggle/working/test_tfrec/ gs://oleg-zyablov/car-classification/test_tfrec/
#для Kaggle (меню -> Add-ons -> Google Cloud SDK, нужно ввести код авторизации)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)

for path in tf.io.gfile.glob('/kaggle/working/train_tfrec/*.tfrec'):
    dst = 'gs://oleg-zyablov/car-classification/train_tfrec/' + path.split('/')[-1]
    tf.io.gfile.copy(path, dst, overwrite = True)
    print('Copied', path, '->', dst)
for path in tf.io.gfile.glob('/kaggle/working/test_tfrec/*.tfrec'):
    dst = 'gs://oleg-zyablov/car-classification/test_tfrec/' + path.split('/')[-1]
    tf.io.gfile.copy(path, dst, overwrite = True)
    print('Copied', path, '->', dst)
import matplotlib.pyplot as plt

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
#тренировочные изображения

with strategy.scope():
    
    def read_train_tfrecord(serialized_example):
        example = tf.io.parse_single_example(serialized_example, features = {
            'image': tf.io.FixedLenFeature([], tf.string), #tf.string - байтовая строка; [] означает скаляр, т. е. только одна строка
            'class': tf.io.FixedLenFeature([], tf.int64)
        })
        return tf.image.decode_jpeg(example['image'], channels = 3), example['class']

    # создаем конвейер
    
    raw_train_dataset = tf.data.TFRecordDataset(
        ['gs://oleg-zyablov/car-classification/train_tfrec/%d.tfrec' % i for i in range(16)],
        num_parallel_reads = 16
        #создаем датасет (конвейер), получающий данные из указанных файлов
    ).map(
       read_train_tfrecord
    ) #.shuffle(300) #конвейер заполняет буфер из 300 элементов, а затем берет случайные элементы из буфера

pairs = list(raw_train_dataset.take(6))
visualize_dataset([(img, class_names[label]) for img, label in pairs])
# тестовые изображения

with strategy.scope():
    
    def read_test_tfrecord(serialized_example):
        example = tf.io.parse_single_example(serialized_example, features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'filename': tf.io.FixedLenFeature([], tf.string)
        })
        return tf.image.decode_jpeg(example['image'], channels = 3), example['filename']
    
    raw_test_dataset = tf.data.TFRecordDataset(
        ['gs://oleg-zyablov/car-classification/test_tfrec/%d.tfrec' % i for i in range(7)],
        num_parallel_reads = 7
    ).map(
       read_test_tfrecord
    )
    
pairs = list(raw_test_dataset.take(6))
visualize_dataset([(img, filename.numpy().decode('utf-8')) for img, filename in pairs])
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import time, gc

def test_xception(optimizer, lr, resizing_func, size, preprocess_func_train, preprocess_func_val, global_pooling, dense_layer,
                  val_data_percent = 0.5, epochs = 1, batch_size = 64, message = '', verbose = 0, optimizer_ready = None, callbacks = [],
                  reuse_model = None, return_model = None):
  with strategy.scope():
    total_count = 15561
    val_count = int(total_count * val_data_percent)
    train_count = total_count - val_count
    steps_per_epoch = train_count // batch_size

    raw_train_dataset_shuffled = raw_train_dataset

    resizing_func_tuple = lambda img, label: (resizing_func(img), label)
    preprocess_func_tuple_train = lambda img, label: (preprocess_func_train(img), label)
    preprocess_func_tuple_val = lambda img, label: (preprocess_func_val(img), label)

    val_dataset = raw_train_dataset_shuffled.take(val_count).map(resizing_func_tuple).map(preprocess_func_tuple_val)

    train_dataset = raw_train_dataset_shuffled.skip(val_count).shuffle(300).repeat().map(resizing_func_tuple).map(preprocess_func_tuple_train)

    #функция get_real_metrics нужна поскольку keras callback на каждом батче возвращает среднее значение точности по эпохе, нужно найти значение на последнем батче
    total_batches = 0 #счетчик батчей
    prev_sum_metrics = None
    def get_real_metrics(batch, logs, metrics = 'accuracy'):
      nonlocal prev_sum_metrics, total_batches
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
    #print('constructing model')
    gc.collect()

    if not reuse_model:
      xception = tf.keras.applications.xception.Xception(
          weights = 'imagenet',
          include_top = False,
          input_shape = (size[0], size[1], 3)
      )

      x = xception.output

      if global_pooling == 'average':
        x = GlobalAveragePooling2D()(x)
      elif global_pooling == 'max':
        x = GlobalMaxPooling2D()(x)
      else:
        assert(0)
      x = BatchNormalization()(x)
      x = Dropout(0.3)(x)

      if dense_layer == '128':
        x = Dense(128, activation = 'relu')
        x = BatchNormalization()(x)
      elif type(dense_layer) == int:
        x = Dense(dense_layer, activation = 'relu')(x)
        x = BatchNormalization()(x)
      elif dense_layer != None:
        assert(0)

      predictions = Dense(10)(x)

      model = tf.keras.Model(inputs = xception.input, outputs = predictions)
    else:
      model = reuse_model

    model.compile(
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
      optimizer = optimizer(learning_rate = lr) if not optimizer_ready else optimizer_ready,
      metrics = ['accuracy']
    )

    start_time = time.time()
    model.fit(
        train_dataset.batch(batch_size),
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = val_dataset.batch(batch_size),
        callbacks = [
            tf.keras.callbacks.LambdaCallback(on_batch_end = lambda batch, logs: acc_history.append(get_real_metrics(batch, logs))),
            tf.keras.callbacks.LambdaCallback(on_epoch_end = on_epoch_end)
        ] + callbacks,
        verbose = verbose
    )
    
  elapsed_time = time.time() - start_time
  print(message, 'validation accuracy = %g, seconds elapsed: %.1f' % (val_acc_history[-1], elapsed_time))

  if return_model:
    return model, acc_history, val_acc_batches_history, val_acc_history
  del model, x, xception
  return acc_history, val_acc_batches_history, val_acc_history
def stretch(image, size): #uint8 0-255 -> float32 0-255
  return tf.image.resize(image, size, preserve_aspect_ratio = False)

def crop(image, size):
  # if image.shape[0] == None: #Tensor("args_0:0", shape=(None, None, 3), dtype=uint8)
  #   return tf.image.resize(image, size)
  # print(image)
  ratio = tf.shape(image)[1] / tf.shape(image)[0]
  target_ratio = size[1] / size[0]
  if target_ratio > ratio:
    h = size[1]
    w = int(size[1] / ratio)
  else:
    w = size[0]
    h = int(size[0] * ratio)
  image = tf.image.resize(image, (w, h))
  return tf.image.resize_with_crop_or_pad(image, size[0], size[1])

def pad(image, size):
  image = tf.image.resize(image, size, preserve_aspect_ratio = True)
  return tf.image.resize_with_pad(image, *size)

def multipad(image, side_size, ratio = 0.64):
  h1 = int(side_size * ratio)
  h2 = side_size - h1
  w1 = int(h2 / ratio)
  w2 = side_size - w1
  h3 = int(w2 * ratio)
  h4 = h2 - h3
  result = tf.constant(0.0, shape = (h4, w2, 3))
  result = tf.concat((pad(image, (h3, w2)), result), axis = 0)
  result = tf.concat((pad(image, (h2, w1)), result), axis = 1)
  result = tf.concat((pad(image, (h1, side_size)), result), axis = 0)
  return result

pairs_train = list(raw_train_dataset.skip(18).take(6))
print('Исходные изображения')
visualize_dataset([(img, '') for img, label in pairs_train])
print('stretch')
visualize_dataset([(tf.cast(stretch(img, (384, 512)), tf.uint8), '') for img, label in pairs_train])
print('crop')
visualize_dataset([(tf.cast(crop(img, (384, 512)), tf.uint8), '') for img, label in pairs_train])
print('pad')
visualize_dataset([(tf.cast(pad(img, (384, 512)), tf.uint8), '') for img, label in pairs_train])
print('multipad')
visualize_dataset([(tf.cast(multipad(img, 512), tf.uint8), '') for img, label in pairs_train])
normalize = lambda img: img / 128 - 1.0

test_epochs = 4
test_examples = 4

stretch_acc = [test_xception(optimizer = Adam, lr = 0.001, resizing_func = lambda img: stretch(img, (384, 512)), size = (384, 512),
    preprocess_func_train = normalize, preprocess_func_val = normalize, global_pooling = 'average',
    dense_layer = 128, message = 'stretch:', epochs = test_epochs) for _ in range(test_examples)]

crop_acc = [test_xception(optimizer = Adam, lr = 0.001, resizing_func = lambda img: crop(img, (384, 512)), size = (384, 512),
    preprocess_func_train = normalize, preprocess_func_val = normalize, global_pooling = 'average',
    dense_layer = 128, message = 'crop:', epochs = test_epochs) for _ in range(test_examples)]

pad_acc = [test_xception(optimizer = Adam, lr = 0.001, resizing_func = lambda img: pad(img, (384, 512)), size = (384, 512),
    preprocess_func_train = normalize, preprocess_func_val = normalize, global_pooling = 'average',
    dense_layer = 128, message = 'pad:', epochs = test_epochs) for _ in range(test_examples)]

multipad_acc = [test_xception(optimizer = Adam, lr = 0.001, resizing_func = lambda img: multipad(img, 444), size = (444, 444), #444*444 ~ 384*512
    preprocess_func_train = normalize, preprocess_func_val = normalize, global_pooling = 'average',
    dense_layer = 128, message = 'multipad:', epochs = test_epochs) for _ in range(test_examples)]
import numpy as np

steps_per_epoch = len(stretch_acc[0][0]) / test_epochs

fig = plt.figure(figsize = (17, 8))

def ax_settings(ax):
  ax.set_yscale('logit')
  ax.set_xscale('logit')
  ax.set_xlim(0.8, 0.97)
  ax.set_ylim(0.2, 0.95)
  ax.set_xlabel('train accuracy')
  ax.set_ylabel('test accuracy')

ax = fig.add_subplot(1, 2, 1)
ax_settings(ax)
ax.set_title('all results')
for data_idx, (data, data_label) in enumerate(zip([stretch_acc, crop_acc, pad_acc, multipad_acc], ['stretch', 'crop', 'pad', 'multipad'])):
  color = 'C%d' % data_idx
  for data_tuple_idx, data_tuple in enumerate(data):
    acc_history = data_tuple[0]
    val_acc_batches_history = data_tuple[1]
    val_acc_history = data_tuple[2]
    train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, test_epochs)]
    ax.plot(train_acc_history_smoothed, val_acc_history[:-1], color = color, label = data_label if data_tuple_idx == 0 else None, linewidth = 1.5)
    ax.scatter(train_acc_history_smoothed, val_acc_history[:-1], color = color, s = 20)
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax_settings(ax)
ax.set_title('averaged results')
for data, data_label in zip([stretch_acc, crop_acc, pad_acc, multipad_acc], ['stretch', 'crop', 'pad', 'multipad']):
  acc_history = np.mean(np.array([i[0] for i in data]), axis = 0)
  val_acc_batches_history = np.mean(np.array([i[1] for i in data]), axis = 0)
  val_acc_history = np.mean(np.array([i[2] for i in data]), axis = 0)
  train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, test_epochs)]
  ax.plot(train_acc_history_smoothed, val_acc_history[:-1], label = data_label, linewidth = 3)
  ax.scatter(train_acc_history_smoothed, val_acc_history[:-1], s = 50)
ax.legend()

plt.show()
normalize = tf.image.per_image_standardization

test_epochs = 4

test_xception(optimizer = Adam, lr = 0.001, resizing_func = lambda img: stretch(img, (384, 512)), size = (384, 512),
    preprocess_func_train = normalize, preprocess_func_test = normalize, global_pooling = 'average',
    dense_layer = 128, message = 'stretch:', epochs = test_epochs, verbose = 1)
import random, numpy as np

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

#применяется к тренировочному набору
def preprocess_train(image):
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    #image = tf.image.random_contrast(image, 0.7, 1.3)
    #image = tf.keras.preprocessing.image.random_rotation(image, 15, row_axis = 0, col_axis = 1, channel_axis = 2)
    image = zoom(image, size = tf.shape(image)[:2])
    return image / 128 - 1

#применяется к тестовому набору
def preprocess_val(image):
    image = tf.cast(image, tf.float32)
    return image / 128 - 1

#применяется для отрисовки
def back_to_uint8(image):
    return tf.cast((image + 1) * 128, tf.uint8)

pairs_train = list(raw_train_dataset.skip(18).take(6))
print('Исходные изображения')
visualize_dataset([(img, '') for img, label in pairs_train])
print('pad + augmentation')
visualize_dataset([(back_to_uint8(preprocess_train(pad(img, (384, 512)))), '') for img, label in pairs_train])
test_epochs = 10
lr = lambda: tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps = 100, decay_rate = 0.9)

size = (384, 512)
acc_512 = test_xception(optimizer = Adam, lr = lr(), resizing_func = lambda img: pad(img, size), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = '512:', epochs = test_epochs, verbose = 1)

size = (256, 384)
acc_384 = test_xception(optimizer = Adam, lr = lr(), resizing_func = lambda img: pad(img, size), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = '384:', epochs = test_epochs, verbose = 1)

size = (192, 256)
acc_256 = test_xception(optimizer = Adam, lr = lr(), resizing_func = lambda img: pad(img, size), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = '256:', epochs = test_epochs, verbose = 1)

size = (128, 192)
acc_192 = test_xception(optimizer = Adam, lr = lr(), resizing_func = lambda img: pad(img, size), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = '192:', epochs = test_epochs, verbose = 1)
import numpy as np

steps_per_epoch = len(acc_512[0]) / test_epochs

fig = plt.figure(figsize = (17, 8))

def ax_settings(ax):
  ax.set_yscale('logit')
  ax.set_xscale('logit')
  ax.set_xlim(0.5, 0.992)
  ax.set_ylim(0.4, 0.975)
  ax.set_xlabel('train accuracy')
  ax.set_ylabel('test accuracy')

ax = fig.add_subplot(1, 2, 2)
ax_settings(ax)
ax.set_title('all results')
for data, data_label in zip([acc_512, acc_384, acc_256, acc_192], ['512x384', '384x256', '256x192', '192x128']):
  acc_history = np.array(data[0])
  val_acc_batches_history = np.array(data[1])
  val_acc_history = np.array(data[2])
  train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, test_epochs)]
  ax.plot(train_acc_history_smoothed, val_acc_history[:-1], label = data_label, linewidth = 3)
  ax.scatter(train_acc_history_smoothed, val_acc_history[:-1], s = 50)
ax.legend()

plt.show()
test_epochs = 15
size = (256, 384)

optimizer = Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps = 100, decay_rate = 0.9))
callbacks = []
test1 = test_xception(None, None, resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = 'Adam decay:', epochs = test_epochs, verbose = 1, optimizer_ready = optimizer, callbacks = callbacks)

optimizer = Nadam(learning_rate = 0.0005)
callbacks = []
test2 = test_xception(None, None, resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = 'Nadam:', epochs = test_epochs, verbose = 1, optimizer_ready = optimizer, callbacks = callbacks)

optimizer = Adam(learning_rate = 0.0005)
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.7, patience = 1, verbose = 1, cooldown = 3, min_lr = 0.00001)]
test3 = test_xception(None, None, resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = 'Adam plateau:', epochs = test_epochs, verbose = 1, optimizer_ready = optimizer, callbacks = callbacks)

optimizer = SGD(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.05, decay_steps = 100, decay_rate = 0.8), momentum = 0.9)
callbacks = []
test4 = test_xception(None, None, resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = 'SGD momentum decay:', epochs = test_epochs, verbose = 1, optimizer_ready = optimizer, callbacks = callbacks)
import numpy as np

steps_per_epoch = len(test1[0]) / test_epochs

fig = plt.figure(figsize = (17, 8))

def ax_settings(ax):
  ax.set_yscale('logit')
  ax.set_xscale('logit')
  ax.set_xlim(0.7, 0.995)
  ax.set_ylim(0.3, 0.975)
  ax.set_xlabel('train accuracy')
  ax.set_ylabel('test accuracy')

ax = fig.add_subplot(1, 2, 2)
ax_settings(ax)
ax.set_title('all results')
for data, data_label in zip([test1, test2, test3, test4], ['Adam decay', 'Nadam', 'Adam plateau reduce', 'SGD momentum decay']):
  acc_history = np.array(data[0])
  val_acc_batches_history = np.array(data[1])
  val_acc_history = np.array(data[2])
  train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, test_epochs)]
  ax.plot(train_acc_history_smoothed, val_acc_history[:-1], label = data_label, linewidth = 3)
  ax.scatter(train_acc_history_smoothed, val_acc_history[:-1], s = 50)
ax.legend()

plt.show()
global_pooling_index = 132 #индекс слоя global average pooling в Xception

layer_groups = [
  (
    max(0, global_pooling_index-5*(i+1)),
    max(0, global_pooling_index-1-5*i)
  )
  for i in range(27)
] + [None] * 23
print(len(layer_groups))
print(layer_groups)

def set_untrainable_all(model):
  for layer in model.layers[:global_pooling_index]:
    layer.trainable = False

def set_trainable_group(model, group):
  if group:
    print('set_trainable_group(): unlocking layers %d-%d' % (group[0], group[1]))
    for layer in model.layers[group[0]:group[1] + 1]:
      layer.trainable = True
  else:
    print('set_trainable_group(): do nothing')
test_params = {
    'SGD nesterov decay': {
        'optimizer': SGD(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.06, decay_steps = 100, decay_rate = 0.85), momentum = 0.97, nesterov = True),
        'on_start': None,
        'on_epoch_end': None
    },
    'adam decay': {
        'optimizer': Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)),
        'on_start': None,
        'on_epoch_end': None
    },
    'adam decay TEST': {
        'optimizer': Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)),
        'on_start': set_untrainable_all,
        'on_epoch_end': (lambda model, epoch: set_trainable_group(model, layer_groups[epoch]))
    },
    'SGD decay TEST': {
        'optimizer': SGD(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.06, decay_steps = 100, decay_rate = 0.85), momentum = 0.9),
        'on_start': set_untrainable_all,
        'on_epoch_end': (lambda model, epoch: set_trainable_group(model, layer_groups[epoch]))
    },
}

def test_with_params(params_name, epochs):
  params = test_params[params_name]
  size = (192, 256)
  with strategy.scope():
    batch_size = 64
    total_count = 15561
    val_count = int(total_count * 0.5)
    train_count = total_count - val_count
    steps_per_epoch = train_count // batch_size

    raw_train_dataset_shuffled = raw_train_dataset.shuffle(300)

    resizing_func_tuple = lambda img, label: (pad(img, (size)), label)
    preprocess_func_tuple_train = lambda img, label: (preprocess_train(img), label)
    preprocess_func_tuple_val = lambda img, label: (preprocess_val(img), label)

    val_dataset = raw_train_dataset_shuffled.take(val_count).map(resizing_func_tuple).map(preprocess_func_tuple_val)

    train_dataset = raw_train_dataset_shuffled.skip(val_count).repeat().map(resizing_func_tuple).map(preprocess_func_tuple_train)

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

    #функция on_epoch_end будет схранять значение точности на валидации после каждой эпохи
    val_acc_batches_history = [] #кол-во обработанных батчей после каждой эпохи
    val_acc_history = [] #точность на валидации после каждой эпохи
    def on_epoch_end(epoch, logs):
      val_acc_batches_history.append(total_batches)
      val_acc_history.append(logs.get('val_accuracy'))

    acc_history = [] #история точности для каждого батча
    print(params_name, '- constructing model')
    gc.collect()

    xception = tf.keras.applications.xception.Xception(
        weights = 'imagenet',
        include_top = False,
        input_shape = (size[0], size[1], 3)
    )

    x = xception.output

    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation = 'relu')(x)
    x = BatchNormalization()(x)

    predictions = Dense(10)(x)

    model = tf.keras.Model(inputs = xception.input, outputs = predictions)

    model.compile(
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
      optimizer = params['optimizer'],
      metrics = ['accuracy']
    )

    start_time = time.time()

    if params['on_start']:
      params['on_start'](model)

    for epoch in range(epochs):
      model.fit(
          train_dataset.batch(batch_size),
          steps_per_epoch = steps_per_epoch,
          epochs = 1,
          validation_data = val_dataset.batch(batch_size),
          callbacks = [
              tf.keras.callbacks.LambdaCallback(on_batch_end = lambda batch, logs: acc_history.append(get_real_metrics(batch, logs))),
              tf.keras.callbacks.LambdaCallback(on_epoch_end = on_epoch_end)
          ],
          verbose = 1
      )
      if params['on_epoch_end']:
        params['on_epoch_end'](model, epoch)
    
    elapsed_time = time.time() - start_time
    print(params_name, '- validation accuracy = %g, seconds elapsed: %.1f' % (val_acc_history[-1], elapsed_time))

    del model, x, xception

    return {
        'epochs': epochs,
        'acc_history': acc_history,
        'val_acc_batches_history': val_acc_batches_history,
        'val_acc_history': val_acc_history
    }
results = {}
epochs = 32
results['SGD nesterov decay'] = test_with_params('SGD nesterov decay', epochs)
results['adam decay'] = test_with_params('adam decay', epochs)
results['adam decay TEST'] = test_with_params('adam decay TEST', epochs)
results['SGD decay TEST'] = test_with_params('SGD decay TEST', epochs)
# #@title Code
import numpy as np

fig = plt.figure(figsize = (29, 8))

def ax_settings(ax):
  ax.set_yscale('logit')
  ax.set_xscale('logit')
  ax.set_xlim(0.5, 0.999)
  ax.set_ylim(0.5, 0.975)
  ax.set_xlabel('train accuracy')
  ax.set_ylabel('test accuracy')

ax = fig.add_subplot(1, 2, 2)
ax_settings(ax)
ax.set_title('all results')
for title, result in results.items():
  epochs = result['epochs']
  acc_history = result['acc_history']
  val_acc_history = result['val_acc_history']
  steps_per_epoch = len(acc_history) / epochs
  train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
  ax.plot(train_acc_history_smoothed, val_acc_history[:-1], linewidth = 2, label = title)
  ax.scatter(train_acc_history_smoothed[-1:], val_acc_history[-2:-1], s = 50)

plt.legend()
plt.show()
optimizer = Adam()
size = (192, 256)
callbacks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 / (epoch % 10 + 1))]
test1 = test_xception(None, None, resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = 'test 1:', epochs = 30, verbose = 1, optimizer_ready = optimizer, callbacks = callbacks)
optimizer = Adam()
size = (192, 256)
callbacks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0005 / (epoch % 10 + 1))]
test2 = test_xception(None, None, resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_train, preprocess_func_val = preprocess_val, global_pooling = 'average',
    dense_layer = 128, message = 'test 1:', epochs = 30, verbose = 1, optimizer_ready = optimizer, callbacks = callbacks)
import numpy as np

fig = plt.figure(figsize = (17, 8))

def ax_settings(ax):
  ax.set_yscale('logit')
  ax.set_xscale('logit')
  ax.set_xlim(0.6, 0.995)
  ax.set_ylim(0.1, 0.98)
  ax.set_xlabel('train accuracy')
  ax.set_ylabel('test accuracy')

ax = fig.add_subplot(1, 2, 2)
ax_settings(ax)
ax.set_title('all results')
for test, title in zip([test1, test2], ['Adam lr = 0.001 / (epoch % 10 + 1)', 'Adam lr = 0.0005 / (epoch % 10 + 1)']):
  epochs = 30
  acc_history, val_acc_batches_history, val_acc_history = test
  steps_per_epoch = len(acc_history) / epochs
  train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
  ax.plot(train_acc_history_smoothed, val_acc_history[:-1], linewidth = 2, label = title)
  ax.scatter(train_acc_history_smoothed[-1:], val_acc_history[-2:-1], s = 50)

plt.legend()
plt.show()
def visualize_channels(img, title = None): #float, 0-255
  total_channels = img.shape[2]
  fig = plt.figure(figsize = (6*total_channels, 4))
  for channel in range(total_channels):
    ax = fig.add_subplot(1, total_channels, 1 + channel)
    ax.imshow(img[..., channel], cmap = 'gray')
    ax.set_title('%s channel %d' % (title or '', channel))
    ax.axis('off')
  plt.show()

#оператор собеля

def sobel(img_uint8, one_output_channel = True, downscaling = 1): #float, 0-255
    img = tf.cast(img_uint8, tf.float32)
    size = tf.shape(img)[:2]
    if downscaling != 1:
      new_size = tf.math.floordiv(size, downscaling)
      img = tf.image.resize(img, new_size)
    img = tf.expand_dims(img, 0)
    img = tf.image.sobel_edges(img)
    img = img[0]
    img = tf.math.reduce_max(img, axis = 2)
    if one_output_channel:
      img = tf.math.sqrt(img[..., 0]*img[..., 0] / 9 + img[..., 1]*img[..., 1] / 9)
      img = tf.expand_dims(img, 2)
    img = tf.math.minimum(img, 255)
    img = tf.math.maximum(img, 0)
    if downscaling != 1:
      img = tf.image.resize(img, size)
    return img

img = list(raw_train_dataset.take(1))[0][0]
visualize_channels(img, 'RGB')
visualize_channels(sobel(img, one_output_channel = True), 'Sobel (by both axes)')
visualize_channels(sobel(img, one_output_channel = False), 'Sobel (by X and by Y)')
visualize_channels(sobel(img, one_output_channel = True, downscaling = 3), 'Sobel (by both axes, scaled)')
def RGB(img_uint8): #float, 0-255
  return tf.cast(img_uint8, tf.float32)

def Gray(img_uint8): #float, 0-255
  channel = tf.cast(tf.image.rgb_to_grayscale(img_uint8), tf.float32) #float, 0-255
  return tf.concat((channel, channel, channel), axis = 2)

def Gray_Sobel_Sobel3(img_uint8): #float, 0-255
  channel1 = tf.cast(tf.image.rgb_to_grayscale(img_uint8), tf.float32) #float, 0-255
  channel2 = sobel(img_uint8, one_output_channel = True) #float, 0-255
  channel3 = sobel(img_uint8, one_output_channel = True, downscaling = 3) #float, 0-255
  return tf.concat((channel1, channel2, channel3), axis = 2)

def Gray_SobelX_SobelY(img_uint8): #float, 0-255
  channel1 = tf.cast(tf.image.rgb_to_grayscale(img_uint8), tf.float32) #float, 0-255
  channels2_3 = sobel(img_uint8, one_output_channel = False) #float, 0-255
  return tf.concat((channel1, channels2_3), axis = 2)

visualize_channels(Gray(img))
visualize_channels(Gray_Sobel_Sobel3(img))
visualize_channels(Gray_SobelX_SobelY(img))
get_lr = lambda: tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)
size = (192, 256)

def preprocess_train_norescale(image):
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    #image = tf.image.random_contrast(image, 0.7, 1.3)
    #image = tf.keras.preprocessing.image.random_rotation(image, 15, row_axis = 0, col_axis = 1, channel_axis = 2)
    image = zoom(image, size = tf.shape(image)[:2])
    return image

#применяется к тестовому набору
def preprocess_val_norescale(image):
    image = tf.cast(image, tf.float32)
    return image
# это нагромождение кода конечно выглядит не очень хорошо, но для пары экспериментов подойдет
# сложность еще и в том, что разные функции tensorflow принимают и возвращают изображения в разных форматах
# форматы бывают следующие: uint8 от 0 до 255, float от 0 до 255, float от 0 до 1, float от -1 до 1

channels_func = Gray
preprocess_func_train = lambda img: channels_func(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: channels_func(preprocess_val_norescale(img)) / 128 - 1
test_Gray = test_xception(Adam, get_lr(), resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_func_train,
    preprocess_func_val = preprocess_func_val,
    global_pooling = 'average', dense_layer = 128, message = 'Gray:', epochs = 30, verbose = 1, return_model = True)

channels_func = RGB
preprocess_func_train = lambda img: channels_func(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: channels_func(preprocess_val_norescale(img)) / 128 - 1
test_RGB = test_xception(Adam, get_lr(), resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_func_train,
    preprocess_func_val = preprocess_func_val,
    global_pooling = 'average', dense_layer = 128, message = 'RGB:', epochs = 30, verbose = 1, return_model = True)

channels_func = Gray_Sobel_Sobel3
preprocess_func_train = lambda img: channels_func(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: channels_func(preprocess_val_norescale(img)) / 128 - 1
test_Gray_Sobel_Sobel3 = test_xception(Adam, get_lr(), resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_func_train,
    preprocess_func_val = preprocess_func_val,
    global_pooling = 'average', dense_layer = 128, message = 'Gray_Sobel_Sobel3:', epochs = 30, verbose = 1, return_model = True)

channels_func = Gray_SobelX_SobelY
preprocess_func_train = lambda img: channels_func(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: channels_func(preprocess_val_norescale(img)) / 128 - 1
test_Gray_SobelX_SobelY = test_xception(Adam, get_lr(), resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_func_train,
    preprocess_func_val = preprocess_func_val,
    global_pooling = 'average', dense_layer = 128, message = 'Gray_SobelX_SobelY:', epochs = 30, verbose = 1, return_model = True)
fig = plt.figure(figsize = (17, 8))

def ax_settings(ax):
  ax.set_yscale('logit')
  ax.set_xscale('logit')
  ax.set_xlim(0.6, 0.998)
  ax.set_ylim(0.3, 0.98)
  ax.set_xlabel('train accuracy')
  ax.set_ylabel('test accuracy')

ax = fig.add_subplot(1, 2, 2)
ax_settings(ax)
ax.set_title('all results')
for test, title, epochs in zip([test_Gray, test_RGB, test_Gray_Sobel_Sobel3, test_Gray_SobelX_SobelY], ['Gray', 'RGB', 'Gray_Sobel_Sobel3', 'Gray_SobelX_SobelY'], [30, 30, 30, 30]):
  model, acc_history, val_acc_batches_history, val_acc_history = test
  steps_per_epoch = len(acc_history) / epochs
  train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
  ax.plot(train_acc_history_smoothed, val_acc_history[:-1], linewidth = 2, label = title)
  ax.scatter(train_acc_history_smoothed[-1:], val_acc_history[-2:-1], s = 50)

plt.legend()
plt.show()
channels_func = Gray
preprocess_func_train = lambda img: channels_func(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: channels_func(preprocess_val_norescale(img)) / 128 - 1
test_Gray = test_xception(Adam, get_lr(), resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_func_train,
    preprocess_func_val = preprocess_func_val,
    global_pooling = 'average', dense_layer = 128, message = 'Gray:', epochs = 20, verbose = 1, return_model = True)
test_Gray[0].save('/gray.h5')
!gcloud auth login
!gsutil cp /gray.h5 gs://oleg-zyablov/car-classification/models/gray.h5
channels_func = RGB
preprocess_func_train = lambda img: channels_func(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: channels_func(preprocess_val_norescale(img)) / 128 - 1
test_RGB = test_xception(Adam, get_lr(), resizing_func = lambda img: pad(img, (size)), size = size,
    preprocess_func_train = preprocess_func_train,
    preprocess_func_val = preprocess_func_val,
    global_pooling = 'average', dense_layer = 128, message = 'RGB:', epochs = 20, verbose = 1, return_model = True)
test_RGB[0].save('/rgb.h5')
!gsutil cp /rgb.h5 gs://oleg-zyablov/car-classification/models/rgb.h5
!gsutil cp gs://oleg-zyablov/car-classification/models/gray.h5 /gray.h5
!gsutil cp gs://oleg-zyablov/car-classification/models/rgb.h5 /rgb.h5
val_data_percent = 0.5
batch_size = 64
total_count = 15561
val_count = int(total_count * val_data_percent)
train_count = total_count - val_count
steps_per_epoch = train_count // batch_size

raw_train_dataset_shuffled = raw_train_dataset #.shuffle(300)

channels_func = Gray
#preprocess_func_train = lambda img: channels_func(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: channels_func(preprocess_val_norescale(img)) / 128 - 1

resizing_func_tuple = lambda img, label: (pad(img, size = (192, 256)), label)
#preprocess_func_tuple_train = lambda img, label: (preprocess_func_train(img), label)
preprocess_func_tuple_val = lambda img, label: (preprocess_func_val(img), label)

val_dataset = raw_train_dataset_shuffled.take(val_count).map(resizing_func_tuple).map(preprocess_func_tuple_val)
#train_dataset = raw_train_dataset_shuffled.skip(val_count).repeat().map(resizing_func_tuple).map(preprocess_func_tuple_train)

logits_gray = tf.keras.models.load_model('/gray.h5').predict(val_dataset.batch(64))
channels_func = RGB
#preprocess_func_train = lambda img: channels_func(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: channels_func(preprocess_val_norescale(img)) / 128 - 1

resizing_func_tuple = lambda img, label: (pad(img, size = (192, 256)), label)
#preprocess_func_tuple_train = lambda img, label: (preprocess_func_train(img), label)
preprocess_func_tuple_val = lambda img, label: (preprocess_func_val(img), label)

val_dataset = raw_train_dataset_shuffled.take(val_count).map(resizing_func_tuple).map(preprocess_func_tuple_val)
#train_dataset = raw_train_dataset_shuffled.skip(val_count).repeat().map(resizing_func_tuple).map(preprocess_func_tuple_train)

logits_rgb = tf.keras.models.load_model('/rgb.h5').predict(val_dataset.batch(64))
correct_answers = [x[1] for x in val_dataset]
correct_answers_count = np.bincount(correct_answers)
print(correct_answers_count)
labels_gray = np.argmax(logits_gray, axis = 1)
labels_rgb = np.argmax(logits_rgb, axis = 1)
confusion_matrix_gray = np.zeros((10, 10))
confusion_matrix_rgb = np.zeros((10, 10))

for i, correct_answer in enumerate(correct_answers):
  confusion_matrix_gray[labels_gray[i], correct_answer] += 1
  confusion_matrix_rgb[labels_rgb[i], correct_answer] += 1

print('confusion matrix gray')
print(confusion_matrix_gray)
print('confusion matrix rgb')
print(confusion_matrix_rgb)
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (16, 8))
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Grayscale recognition')
ax.imshow(confusion_matrix_gray, cmap = 'gray', vmin = 0, vmax = 100)
for i in range(10):
  for j in range(10):
    ax.text(j, i, str(int(confusion_matrix_gray[i][j])), horizontalalignment = 'center',
            verticalalignment = 'center', color = 'black' if confusion_matrix_gray[i][j] > 50 else 'white')
ax.set_ylabel('prediction')
ax.set_xlabel('correct answer')
ax = fig.add_subplot(1, 2, 2)
ax.set_title('RGB recognition')
ax.imshow(confusion_matrix_rgb, cmap = 'gray', vmin = 0, vmax = 100)
for i in range(10):
  for j in range(10):
    ax.text(j, i, str(int(confusion_matrix_rgb[i][j])), horizontalalignment = 'center',
            verticalalignment = 'center', color = 'black' if confusion_matrix_rgb[i][j] > 50 else 'white')
ax.set_ylabel('prediction')
ax.set_xlabel('correct answer')
plt.show()
pairs = []
for i, pair in enumerate(val_dataset):
  if labels_rgb[i] != correct_answers[i]:
    title = 'Индекс: %d\nРаспознано: %s\nПравильно:%s' % (i, class_names[labels_rgb[i]], class_names[correct_answers[i]])
    pairs.append((tf.cast((pair[0] + 1) * 128, tf.uint8), title))
  if len(pairs) == 6:
    visualize_dataset(pairs)
    pairs = []
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import time, gc

def test_xception_v2(optimizer, resizing_func, size, preprocess_func_train, preprocess_func_val, top_layers, on_compile = None,
                  val_data_percent = 0.5, epochs = 1, batch_size = 64, message = '', verbose = 0, callbacks = [],
                  return_model = False, reuse_model = None):
  with strategy.scope():
    total_count = 15561
    val_count = int(total_count * val_data_percent)
    train_count = total_count - val_count
    steps_per_epoch = train_count // batch_size

    raw_train_dataset_shuffled = raw_train_dataset

    resizing_func_tuple = lambda img, label: (resizing_func(img), label)
    preprocess_func_tuple_train = lambda img, label: (preprocess_func_train(img), label)
    preprocess_func_tuple_val = lambda img, label: (preprocess_func_val(img), label)

    val_dataset = raw_train_dataset_shuffled.take(val_count).map(resizing_func_tuple).map(preprocess_func_tuple_val)

    train_dataset = raw_train_dataset_shuffled.skip(val_count).shuffle(300).repeat().map(resizing_func_tuple).map(preprocess_func_tuple_train)

    #функция get_real_metrics нужна поскольку keras callback на каждом батче возвращает среднее значение точности по эпохе, нужно найти значение на последнем батче
    total_batches = 0 #счетчик батчей
    prev_sum_metrics = None
    def get_real_metrics(batch, logs, metrics = 'accuracy'):
      nonlocal prev_sum_metrics, total_batches
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
    #print('constructing model')
    gc.collect()

    if not reuse_model:
      xception = tf.keras.applications.xception.Xception(
          weights = 'imagenet',
          include_top = False,
          input_shape = (size[0], size[1], 3)
      )
      x = xception.output
      for layer in top_layers:
        x = layer(x)
      predictions = x

      model = tf.keras.Model(inputs = xception.input, outputs = predictions)
    else:
      model = reuse_model

    model.compile(
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
      optimizer = optimizer,
      metrics = ['accuracy']
    )

    if on_compile:
      on_compile(model)

    start_time = time.time()
    model.fit(
        train_dataset.batch(batch_size),
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        validation_data = val_dataset.batch(batch_size),
        callbacks = [
            tf.keras.callbacks.LambdaCallback(on_batch_end = lambda batch, logs: acc_history.append(get_real_metrics(batch, logs))),
            tf.keras.callbacks.LambdaCallback(on_epoch_end = on_epoch_end)
        ] + callbacks,
        verbose = verbose
    )
    
  elapsed_time = time.time() - start_time
  print(message, 'validation accuracy = %g, seconds elapsed: %.1f' % (val_acc_history[-1], elapsed_time))

  if return_model:
    return model, acc_history, val_acc_batches_history, val_acc_history
  del model, x, xception
  return acc_history, val_acc_batches_history, val_acc_history
size = (192, 256)
preprocess_func_train = lambda img: RGB(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: RGB(preprocess_val_norescale(img)) / 128 - 1

def freeze_output_layer(model): #получаем ответы напрямую из слоя global average pooling
  dense = model.layers[-1]
  #dense.trainable = False
  dense.set_weights((
      np.concatenate(
          (np.identity(10), np.zeros((2038, 10))),
          axis = 0)
      ,
      np.zeros(10)
  ))

test1 = test_xception_v2(optimizer = Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)),
                 resizing_func = lambda img: pad(img, size),
                 size = size,
                 preprocess_func_train = preprocess_func_train,
                 preprocess_func_val = preprocess_func_val,
                 top_layers = [GlobalAveragePooling2D(), Dense(10, trainable = False)],
                 on_compile = freeze_output_layer,
                val_data_percent = 0.5, epochs = 25, batch_size = 64, message = '', verbose = 1, callbacks = [], return_model = True)
#проверим, что выходной слой не обучался
test1[0].layers[-1].get_weights()[0]
#удалим модель
model = test1[0]
test1 = test1[1:]
del model
acc_history, val_acc_batches_history, val_acc_history = test1
epochs = 25
steps_per_epoch = len(acc_history) / epochs
train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
test1 = (train_acc_history_smoothed, val_acc_history[:-1])
test2 = test_xception_v2(optimizer = Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)),
                 resizing_func = lambda img: pad(img, size),
                 size = size,
                 preprocess_func_train = preprocess_func_train,
                 preprocess_func_val = preprocess_func_val,
                 top_layers = [GlobalAveragePooling2D(), BatchNormalization(), Dense(10)],
                 val_data_percent = 0.5, epochs = 25, batch_size = 64, verbose = 1)
acc_history, val_acc_batches_history, val_acc_history = test2
epochs = 25
steps_per_epoch = len(acc_history) / epochs
train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
test2 = (train_acc_history_smoothed, val_acc_history[:-1])
#batchNorm, Dropout(0.3), Dense(100, relu, l2 regularizer), batchNorm, Dense(10)
test3 = test_xception_v2(optimizer = Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)),
                 resizing_func = lambda img: pad(img, size),
                 size = size,
                 preprocess_func_train = preprocess_func_train,
                 preprocess_func_val = preprocess_func_val,
                 top_layers = [GlobalAveragePooling2D(), BatchNormalization(), Dropout(0.3), Dense(100, activation = 'relu', kernel_regularizer = 'l2'), BatchNormalization(), Dense(10)],
                 val_data_percent = 0.5, epochs = 25, batch_size = 64, verbose = 1)
acc_history, val_acc_batches_history, val_acc_history = test3
epochs = 25
steps_per_epoch = len(acc_history) / epochs
train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
test3 = (train_acc_history_smoothed, val_acc_history[:-1])
#batchNorm, Dropout(0.3), Dense(1000, relu, l2 regularizer), batchNorm, Dropout(0.5), Dense(10)
test4 = test_xception_v2(optimizer = Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)),
                 resizing_func = lambda img: pad(img, size),
                 size = size,
                 preprocess_func_train = preprocess_func_train,
                 preprocess_func_val = preprocess_func_val,
                 top_layers = [GlobalAveragePooling2D(), BatchNormalization(), Dropout(0.3), Dense(1000, activation = 'relu', kernel_regularizer = 'l2'), BatchNormalization(), Dropout(0.5), Dense(10)],
                 val_data_percent = 0.5, epochs = 25, batch_size = 64, verbose = 1)
acc_history, val_acc_batches_history, val_acc_history = test4
epochs = 25
steps_per_epoch = len(acc_history) / epochs
train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
test4 = (train_acc_history_smoothed, val_acc_history[:-1])
fig = plt.figure(figsize = (18, 12))

def ax_settings(ax):
  ax.set_yscale('logit')
  ax.set_xscale('logit')
  ax.set_xlim(0.6, 0.999)
  ax.set_ylim(0.5, 0.96)
  ax.set_xlabel('train accuracy')
  ax.set_ylabel('test accuracy')

ax = fig.add_subplot(1, 2, 2)
ax_settings(ax)
ax.set_title('all results')
for (train_acc_history_smoothed, val_acc_history), title in zip([test1, test2, test3, test4], ['test 1', 'test 2', 'test 3', 'test 4']):
  ax.plot(train_acc_history_smoothed, val_acc_history, linewidth = 2, label = title)
  ax.scatter(train_acc_history_smoothed[-1:], val_acc_history[-1:], s = 50)

plt.legend()
plt.show()
size = (384, 512)
preprocess_func_train = lambda img: RGB(preprocess_train_norescale(img)) / 128 - 1
preprocess_func_val = lambda img: RGB(preprocess_val_norescale(img)) / 128 - 1

test_GlobalAveragePooling2D = test_xception_v2(optimizer = Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)),
                 resizing_func = lambda img: pad(img, size),
                 size = size,
                 preprocess_func_train = preprocess_func_train,
                 preprocess_func_val = preprocess_func_val,
                 top_layers = [GlobalAveragePooling2D(), BatchNormalization(), Dropout(0.3), Dense(100, activation = 'relu', kernel_regularizer = 'l2'), BatchNormalization(), Dense(10)],
                 val_data_percent = 0.5, epochs = 15, batch_size = 64, verbose = 1)
acc_history, val_acc_batches_history, val_acc_history = test_GlobalAveragePooling2D
epochs = 15
steps_per_epoch = len(acc_history) / epochs
train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
test_GlobalAveragePooling2D = (train_acc_history_smoothed, val_acc_history[:-1])
test_GlobalMaxPooling2D = test_xception_v2(optimizer = Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.0008, decay_steps = 100, decay_rate = 0.9)),
                 resizing_func = lambda img: pad(img, size),
                 size = size,
                 preprocess_func_train = preprocess_func_train,
                 preprocess_func_val = preprocess_func_val,
                 top_layers = [GlobalMaxPooling2D(), BatchNormalization(), Dropout(0.3), Dense(100, activation = 'relu', kernel_regularizer = 'l2'), BatchNormalization(), Dense(10)],
                 val_data_percent = 0.5, epochs = 15, batch_size = 64, verbose = 1)
acc_history, val_acc_batches_history, val_acc_history = test_GlobalMaxPooling2D
epochs = 15
steps_per_epoch = len(acc_history) / epochs
train_acc_history_smoothed = [np.mean(np.array(acc_history[int(steps_per_epoch * (epoch - 0.5)):int(steps_per_epoch * (epoch + 0.5))])) for epoch in range(1, epochs)]
test_GlobalMaxPooling2D = (train_acc_history_smoothed, val_acc_history[:-1])
fig = plt.figure(figsize = (18, 12))

def ax_settings(ax):
  ax.set_yscale('logit')
  ax.set_xscale('logit')
  ax.set_xlim(0.7, 0.996)
  ax.set_ylim(0.2, 0.98)
  ax.set_xlabel('train accuracy')
  ax.set_ylabel('test accuracy')

ax = fig.add_subplot(1, 2, 2)
ax_settings(ax)
ax.set_title('all results')
for (train_acc_history_smoothed, val_acc_history), title in zip([test_GlobalAveragePooling2D, test_GlobalMaxPooling2D], ['GlobalAveragePooling', 'GlobalMaxPooling']):
  ax.plot(train_acc_history_smoothed, val_acc_history, linewidth = 2, label = title)
  ax.scatter(train_acc_history_smoothed[-1:], val_acc_history[-1:], s = 50)

plt.legend()
plt.show()
 