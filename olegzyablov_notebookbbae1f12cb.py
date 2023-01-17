# import keras

# x = keras.layers.Input(shape = (10, 10, 3))

# x2 = keras.layers.Input(shape = (10, 10, 3))

# y = keras.layers.Conv2D(16, (3, 3))(x)



# model = keras.Model(inputs = x, outputs = y)
# model = keras.applications.InceptionResNetV2(input_shape = (100, 100, 3), weights = 'imagenet', include_top = False)
# layer_to_layer_dict = {}

# input_layer = model.layers[0]

# for layer in model.layers[1:]:

#     cfg = layer.get_config()

#     cls = layer.__class__

#     del cfg['name']

#     layer_copy = cls.from_config(cfg)

#     layer_to_layer_dict[layer] = (layer_copy, None)

# for source_layer, (copied_layer, variable) in layer_to_layer_dict.items():

#     assert(len(source_layer._inbound_nodes) == 1)

#     inbound_layer_or_list = source_layer._inbound_nodes[0].inbound_layers

#     if inbound_layer_or_list == input_layer:

#         variable = copied_layer(input_layer.output)

#     elif type(inbound_layer_or_list) == list:

#         inbound_layers_variables = []

#         for inbound_layer in inbound_layer_or_list:

#             inbound_layer_copied, inbound_layer_variable = layer_to_layer_dict[inbound_layer]

#             inbound_layers_variables.append(inbound_layer_variable)

#         variable = copied_layer(inbound_layers_variables)

#     else:

#         inbound_layer = inbound_layer_or_list

#         inbound_layer_copied, inbound_layer_variable = layer_to_layer_dict[inbound_layer]

#         if source_layer.__class__ == keras.layers.Conv2D:

#             #layer_input = keras.layers.Concatenate(axis = 3)([inbound_layer_variable, inbound_layer.output])

#             layer_input = inbound_layer_variable + inbound_layer.output

#             variable = copied_layer(layer_input)

#         else:

#             variable = copied_layer(inbound_layer_variable)

#     layer_to_layer_dict[source_layer] = (copy_layer, variable)

# final_layer = model.layers[-1]

# final_layer_copied, final_variable = layer_to_layer_dict[final_layer]

# print(final_variable)
# new_model = keras.Model(inputs = model.input, outputs = final_variable)

# new_model.compile(loss = 'mse', optimizer = 'adam')
# test_image = np.ones((1, 100, 100, 3))

# new_model(test_image)
# test_image = np.ones((1, 100, 100, 3))

# model(test_image)
# w = model.layers[4].get_weights()[0]

# model.layers[4].set_weights([np.zeros(w.shape)])
# len(new_model.layers)
# model.layers[:5]
# tf.keras.utils.plot_model(new_model, to_file="/kaggle/working/model.png");
# !pip install gluoncv --upgrade
# !pip install -U --pre mxnet -f https://dist.mxnet.io/python/mkl
# import mxnet as mx

# from mxnet import image

# from mxnet.gluon.data.vision import transforms

# import gluoncv

# # using cpu

# ctx = mx.cpu(0)
# url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/1.jpg'

# filename = 'example.jpg'

# gluoncv.utils.download(url, filename)
# img = image.imread(filename)



# from matplotlib import pyplot as plt

# plt.imshow(img.asnumpy())

# plt.show()
# from gluoncv.data.transforms.presets.segmentation import test_transform

# img = test_transform(img, ctx)
# model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
# output = model.predict(img)

# predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
# from gluoncv.utils.viz import get_color_pallete

# import matplotlib.image as mpimg

# mask = get_color_pallete(predict, 'pascal_voc')

# mask.save('output.png')
# mmask = mpimg.imread('output.png')

# plt.imshow(mmask)

# plt.show()
# import tensorflow as tf

# def read_train_tfrecord(serialized_example):

#     example = tf.io.parse_single_example(serialized_example, features = {

#       'image': tf.io.FixedLenFeature([], tf.string), #tf.string - байтовая строка; [] означает скаляр, т. е. только одна строка

#       'class': tf.io.FixedLenFeature([], tf.int64)

#     })

#     return tf.image.decode_jpeg(example['image'], channels = 3), example['class']



# dataset = tf.data.TFRecordDataset(

#     ['gs://oleg-zyablov/car-classification/train_tfrec/%d.tfrec' % i for i in range(16)],

#     num_parallel_reads = 16

#   ).map(read_train_tfrecord).cache()
# it = dataset.shuffle(300).__iter__()

# models = [

#     gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True), #автомобиль=7.0

#     gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True), #автомобиль=20.0

#     gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True) #автомобиль=20.0

# ]

# auto_idx = [7, 20, 20]

# for i in range(100):

#     import mxnet as mx

#     import numpy as np

#     img_orig = it.__next__()[0].numpy()

#     img = mx.nd.array(img_orig)

#     #plt.imshow(img.asnumpy().astype(np.uint8))

#     #plt.show()

#     img = test_transform(img, ctx)

#     outputs = [model.predict(img) for model in models]

#     predicts = [mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy() for output in outputs]

#     predicts = [(predict == idx).astype(int) for predict, idx in zip(predicts, auto_idx)]

#     mask = np.mean(np.array(predicts), axis = 0)

#     #plt.imshow(mask)

#     #plt.show()

#     r, g, b = img_orig[:, :, 0], img_orig[:, :, 1], img_orig[:, :, 2]

#     r, g, b = r * mask, g * mask, b * mask

#     masked_image = np.moveaxis(np.array([r, g, b]), 0, 2).astype(np.uint8)

#     plt.imshow(masked_image)

#     plt.show()

# #     print(predict)

# #     mask = get_color_pallete(predict, 'pascal_voc')

# #     mask.save('output.png')

# #     mmask = mpimg.imread('output.png')

# #     plt.imshow(mmask)

# #     plt.show()
from google.cloud import storage

storage_client = storage.Client(project = 'oleg-zyablov')



def upload_blob(bucket_name, source_file_name, destination_blob_name, silent = False):

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    if not silent: print('File {} uploaded to {}.'.format(source_file_name, destination_blob_name))



def download_blob(bucket_name, source_blob_name, destination_file_name, silent = False):

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    if not silent: print("File {} downloaded to {}.".format(source_blob_name, destination_file_name))
download_blob('oleg-zyablov', 'car-classification/2109.zip', '2109.zip')
!unzip 2109.zip
from matplotlib import pyplot as plt

import numpy as np



import mxnet as mx

from mxnet import image

from mxnet.gluon.data.vision import transforms

from gluoncv.data.transforms.presets.segmentation import test_transform

import gluoncv

# using cpu

ctx = mx.gpu(0)



models = [

    gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True, ctx=ctx), #автомобиль=7.0

    gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True, ctx=ctx), #автомобиль=20.0

    gluoncv.model_zoo.get_model('deeplab_resnet101_ade', pretrained=True, ctx=ctx) #автомобиль=20.0

]

auto_idx = [7, 20, 20]



def get_mask(img_orig):

    img = mx.nd.array(img_orig)

    img = test_transform(img, ctx)

    outputs = [model.predict(img) for model in models]

    predicts = [mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy() for output in outputs]

    predicts = [(predict == idx).astype(int) for predict, idx in zip(predicts, auto_idx)]

    mask = np.mean(np.array(predicts), axis = 0)

    #plt.imshow(mask)

    #plt.show()

    return mask

    #r, g, b = img_orig[:, :, 0], img_orig[:, :, 1], img_orig[:, :, 2]

    #r, g, b = r * mask, g * mask, b * mask

    #masked_image = np.moveaxis(np.array([r, g, b]), 0, 2).astype(np.uint8)

    #plt.imshow(masked_image)

    #plt.show()



import tensorflow as tf

def read_train_tfrecord(serialized_example):

    example = tf.io.parse_single_example(serialized_example, features = {

      'image': tf.io.FixedLenFeature([], tf.string), #tf.string - байтовая строка; [] означает скаляр, т. е. только одна строка

      'class': tf.io.FixedLenFeature([], tf.int64)

    })

    return tf.image.decode_jpeg(example['image'], channels = 3), example['class']



# dataset = tf.data.TFRecordDataset(

#     ['gs://oleg-zyablov/car-classification/train_tfrec/%d.tfrec' % i for i in range(16)],

#     num_parallel_reads = 16

#   ).map(read_train_tfrecord).cache()



# img = dataset.__iter__().__next__()[0].numpy()

# mask = get_mask(img)

# plt.imshow(mask)

# plt.show()
import shutil, pathlib

import tensorflow as tf

import random



paths = tf.io.gfile.glob('2109/*.jpg')



filenames = [path.split('/')[-1] for path in paths]



shard_size = 1000



dataset = tf.data.Dataset.from_tensor_slices((paths, filenames))

dataset = dataset.map(

    lambda path, filename: (tf.io.read_file(path), filename)

).batch(shard_size)



out_folder = '2109_tfrec'

shutil.rmtree(out_folder, ignore_errors = True) #удаляем папку со старыми файлами

pathlib.Path(out_folder).mkdir(parents = True, exist_ok = True)



images_processed = 0

for batch_index, (images, img_filenames) in enumerate(dataset):

    filename = out_folder + '/%d.tfrec' % batch_index



    images_ndarray = images.numpy()

    masks = []

    for i, img_bytes in enumerate(images_ndarray):

        print(i)

        img_array = tf.image.decode_jpeg(img_bytes, channels = 3)

        img_array = tf.image.resize(img_array, (384, 512), preserve_aspect_ratio = True)

        mask = (get_mask(img_array)[:, :, np.newaxis] * 255).astype(np.uint8)

        #img_array = np.concatenate((img_array, mask), axis = 2)

        mask_bytes = tf.image.encode_jpeg(mask).numpy()

        masks.append(mask_bytes)

    masks = np.array(masks)

    

    img_filenames_ndarray = img_filenames.numpy()



    examples_count = images_ndarray.shape[0]



    print('Writing file: %s [images %d-%d]' % (filename, images_processed, images_processed + examples_count))

    images_processed += examples_count



    with tf.io.TFRecordWriter(filename) as out_file:

        for i in range(examples_count):

            tfrecord = tf.train.Example(features = tf.train.Features(feature = {

                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [images_ndarray[i]])),

                'filename': tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_filenames_ndarray[i]])),

                'mask': tf.train.Feature(bytes_list = tf.train.BytesList(value = [masks[i]])),

            }))

            out_file.write(tfrecord.SerializeToString())

    upload_blob('oleg-zyablov', filename, 'car-classification/2109/%d.tfrec' % batch_index)