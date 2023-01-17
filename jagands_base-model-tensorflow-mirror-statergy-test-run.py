import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom as dcm
import cv2
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

def get_available_gpus():
    local_device_protos = tf.python.client.device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

get_available_gpus()
strategy = tf.distribute.MirroredStrategy()
from vtk import vtkDICOMImageReader
from vtk import vtkImageShiftScale
import vtk
from vtk.util import numpy_support
from vtk import vtkPNGWriter


reader = vtk.vtkDICOMImageReader()
def get_img(path):
    reader.SetFileName(path)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    ConstPixelSpacing = reader.GetPixelSpacing()
    imageData = reader.GetOutput()
    pointData = imageData.GetPointData()
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    ArrayDicom = cv2.resize(ArrayDicom,(512,512))
    return ArrayDicom

def window_image(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dcm.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def decode_dcm(file, dim = 1,  rescale = True):
    data = dcm.dcmread(file)
    try:
        image = data.pixel_array
    except:
        image = get_img(file)
    window_center , window_width, intercept, slope = get_windowing(data)
    output = window_image(image, window_center, window_width, intercept, slope, rescale = rescale)
    if dim != 1:
        output = np.repeat(output[..., np.newaxis], dim, -1)
    return output
def tags(data):
    tags = tio.image.dicom_tags()
    window_center = tio.image.decode_dicom_data(data, tags = tags.WindowCenter)
    WindowWidth = tio.image.decode_dicom_data(data, tags = tags.WindowWidth)
    RescaleIntercept = tio.image.decode_dicom_data(data, tags= tags.RescaleIntercept)
    RescaleSlope = tio.image.decode_dicom_data(data, tags = tags.RescaleSlope)
    return tf.strings.to_number(window_center), tf.strings.to_number(WindowWidth), tf.strings.to_number(RescaleIntercept), tf.strings.to_number(RescaleSlope)

def dcm_to_image(filename):
    return tio.image.decode_dicom_image(filename, on_error = 'lossy', dtype = tf.float32, )

def window_image_tf(img, window_center,window_width, intercept, slope, rescale=True):
    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img = tf.clip_by_value(img, img_min, img_max)
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img

def string_to_image(filename):
    encodedata = tf.io.read_file(filename)
    image = dcm_to_image(encodedata)
    window_center,window_width, intercept, slope = tags(encodedata)
    return window_image_tf(image, window_center,window_width, intercept, slope)
def preprocess_image(x, y):
    x = x.numpy()
    if isinstance(x, np.ndarray):
        x = list(map(lambda a : decode_dcm(a.decode('utf-8'), dim = 3), x))
    else:
        x = decode_dcm(x.decode('utf-8'), dim = 3)
    return (np.array(x), y)
def process_string(x, y):
    return tf.strings.unicode_decode(x,input_encoding='UTF-8'), y
train_df = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
x_columns = train_df.columns[:3]
output = ['pe_present_on_image','negative_exam_for_pe','indeterminate', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
            'chronic_pe','acute_and_chronic_pe', 'indeterminate',
            'leftsided_pe', 'rightsided_pe', 'central_pe']
root_path = "../input/rsna-str-pulmonary-embolism-detection/train/"
train_df_x = list(map(lambda x : root_path + "/".join(x) +".dcm", train_df[x_columns].values.tolist()))
train_df_y = train_df[output].values.astype('float32')
train_df_x = tf.data.Dataset.from_tensor_slices(train_df_x)
# train_df_x = train_df_x.map(lambda x: tf.py_function(preprocess_image, [x], [tf.float32]))
train_df_y = tf.data.Dataset.from_tensor_slices(train_df_y)
ds = tf.data.Dataset.zip((train_df_x, train_df_y))
BUFFER_SIZE=1300
PER_REPLICA_BATCH_SIZE=64
try:
    NUM_REPLICAS=strategy.num_replicas_in_sync
except:
    NUM_REPLICAS = 1
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * NUM_REPLICAS
train_set = int((np.round(train_df.shape[0] * 0.70)) // GLOBAL_BATCH_SIZE)

ds = ds.shuffle(BUFFER_SIZE)
ds = ds.batch(GLOBAL_BATCH_SIZE)
train_ds = ds.take(train_set)
test_ds = ds.skip(train_set)
#train_ds = train_ds.interleave(train_ds, num_parallel_calls = tf.data.experimental.AUTOTUNE)
train_ds = train_ds.map(lambda x, y : tf.py_function(preprocess_image, [x, y], [tf.float32, tf.float32]))
test_ds = test_ds.map(lambda x, y : tf.py_function(preprocess_image, [x, y], [tf.float32, tf.float32]))
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

train_example_ds = train_ds.take(100)
test_example_ds = test_ds.take(100)
train_dist_dataset = strategy.experimental_distribute_dataset(train_example_ds)
test_dist_dataset = strategy.experimental_distribute_dataset(test_example_ds)
it = iter(train_example_ds)
next(it)
next(it)
del train_df_x, train_df_y

import gc 
gc.collect()
x, y = next(iter(train_dist_dataset))
print(x.numpy().shape)
feature_extraction = tf.keras.applications.ResNet50(include_top = False, input_shape= (512, 512, 3))
for layer in feature_extraction.layers:
    layer.trainable = False
kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.2)
def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    if filters_1x1 == 0:
        conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    if filters_1x1 == 0:
        output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    else:
        output = tf.keras.layers.concatenate([conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    return output
def create_model():
    inputs = tf.keras.Input(shape=(512, 512, 3))
    features = feature_extraction(inputs)
    inception_downsample = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='max_pool_1')(features)
    inception_downsample = inception_module(inception_downsample,filters_1x1=32,filters_3x3_reduce=32,filters_3x3=32,filters_5x5_reduce=32,
                     filters_5x5=32,filters_pool_proj=10,name='inception_1')
    inception_downsample = tf.keras.layers.BatchNormalization(name = 'BatchNormalization_1')(inception_downsample)
    inception_downsample = tf.keras.layers.MaxPool2D((2, 2), name='max_pool_2')(inception_downsample)
    # Classification Layer
    x = tf.keras.layers.Flatten()(inception_downsample)
    x = tf.keras.layers.Dense(230)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    y1 = tf.keras.layers.Dense(1,'sigmoid',name = 'pe_present_on_image')(x)
    y2 = tf.keras.layers.Dense(1,'sigmoid',name = 'negative_exam_for_pe')(x)
    y3 = tf.keras.layers.Dense(1,'sigmoid',name = 'indeterminate')(x)

    model = tf.keras.Model(inputs = inputs, outputs = [y1, y2,y3])
    print(model.summary())
    return model
with strategy.scope():
    loss_object = tf.keras.losses.BinaryCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
        labels1 = labels[:, 0:1]
        labels2 = labels[:, 1:2]
        labels3 = labels[:, 2:3]
        predictions1 = predictions[0]
        predictions2 = predictions[1]
        predictions3 = predictions[2]
        per_example_loss1 = loss_object(labels1, predictions1)
        per_example_loss2 = loss_object(labels2, predictions2)
        per_example_loss3 = loss_object(labels3, predictions3)
        return tf.nn.compute_average_loss(per_example_loss1, global_batch_size=GLOBAL_BATCH_SIZE), tf.nn.compute_average_loss(per_example_loss2, global_batch_size=GLOBAL_BATCH_SIZE), tf.nn.compute_average_loss(per_example_loss3, global_batch_size=GLOBAL_BATCH_SIZE)
with strategy.scope():
    test_loss1 = tf.keras.metrics.Mean(name='test_loss1')
    test_loss2 = tf.keras.metrics.Mean(name='test_loss2')
    test_loss3 = tf.keras.metrics.Mean(name='test_loss3')
    train_accuracy1 = tf.keras.metrics.BinaryAccuracy(name='pe_present_on_image_accuracy')
    train_accuracy2 = tf.keras.metrics.BinaryAccuracy(name='negative_exam_for_pe')
    train_accuracy3 = tf.keras.metrics.BinaryAccuracy(name='indeterminate')
    test_accuracy1 = tf.keras.metrics.BinaryAccuracy(name='test_accuracy1')
    test_accuracy2 = tf.keras.metrics.BinaryAccuracy(name='test_accuracy2')
    test_accuracy3 = tf.keras.metrics.BinaryAccuracy(name='test_accuracy3')

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# model, optimizer, and checkpoint must be created under `strategy.scope`.
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(images, training=True)
        loss1, loss2, loss3 = compute_loss(labels, predictions)
    gradients1 = tape.gradient(loss1, model.trainable_variables)
    gradients2 = tape.gradient(loss2, model.trainable_variables)
    gradients3 = tape.gradient(loss3, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients1, model.trainable_variables))
    optimizer.apply_gradients(zip(gradients2, model.trainable_variables))
    optimizer.apply_gradients(zip(gradients3, model.trainable_variables))
    labels1 = labels[:, 0:1]
    labels2 = labels[:, 1:2]
    labels3 = labels[:, 2:3]
    predictions1 = predictions[0]
    predictions2 = predictions[1]
    predictions3 = predictions[2]
    train_accuracy1.update_state(labels1, predictions1)
    train_accuracy2.update_state(labels2, predictions2)
    train_accuracy3.update_state(labels3, predictions3)
    return loss1, loss2, loss3

def test_step(inputs):
    images, labels = inputs
    predictions = model(images, training=False)
    t_loss1, t_loss2, t_loss3 = compute_loss(labels, predictions)
    labels1 = labels[:, 0:1]
    labels2 = labels[:, 1:2]
    labels3 = labels[:, 2:3]
    predictions1 = predictions[0]
    predictions2 = predictions[1]
    predictions3 = predictions[2]
    test_loss1.update_state(t_loss1)
    test_loss2.update_state(t_loss2)
    test_loss3.update_state(t_loss3)
    test_accuracy1.update_state(labels1, predictions1)
    test_accuracy2.update_state(labels2, predictions2)
    test_accuracy3.update_state(labels3, predictions3)
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses1, per_replica_losses2, per_replica_losses3 = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses1, axis=None),strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses2,
                         axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses3,
                         axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))
EPOCHS = 5
for epoch in range(EPOCHS):
    # TRAIN LOOP
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    num_batches = 0
    for x in train_dist_dataset:
        loss1, loss2, loss3 = distributed_train_step(x)
        total_loss1 += loss1
        total_loss1 += loss2
        total_loss1 += loss3
        num_batches += 1
        train_loss1 = total_loss1 / num_batches
        train_loss2 = total_loss2 / num_batches
        train_loss3 = total_loss3 / num_batches

    # TEST LOOP
    for x in test_dist_dataset:
        distributed_test_step(x)

    if epoch % 10 == 0:
        checkpoint.save(checkpoint_prefix)

    template = ("Epoch {},\n\t Loss: \n\t\t pe_present_on_image {} \n\t\t negative_exam_for_pe {} \n\t\t indeterminate {}, \n\tAccuracy:  \n\t\t pe_present_on_image {} \n\t\t negative_exam_for_p {} \n\t\t indeterminate {}, \n\t Test Loss: \n\t\t pe_present_on_image {} \n\t\t negative_exam_for_pe {} \n\t\t indeterminate {}, \n\t Test Accuracy: \n\t\t pe_present_on_image {} \n\t\t negative_exam_for_pe {} \n\t\t indeterminate {}")
    print (template.format(epoch+1, train_loss1, train_loss2, train_loss3,
                         train_accuracy1.result()*100, train_accuracy2.result()*100, train_accuracy3.result()*100
                        ,test_loss1.result(),test_loss2.result(),test_loss3.result(),
                         test_accuracy1.result()*100, test_accuracy2.result()*100, test_accuracy3.result()*100))

    test_loss1.reset_states()
    train_accuracy1.reset_states()
    test_accuracy1.reset_states()
    test_loss2.reset_states()
    train_accuracy2.reset_states()
    test_accuracy2.reset_states()
    test_loss3.reset_states()
    train_accuracy3.reset_states()
    test_accuracy3.reset_states()
x, y = next(iter(test_example_ds))
x = x[0]
y = y[0][:3]
tf.round(model(x[np.newaxis, ...]))
y
model.save("model.h5")