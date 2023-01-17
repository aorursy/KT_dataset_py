# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, UpSampling2D, BatchNormalization
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend as K
from keras.applications.resnet50 import ResNet50,preprocess_input
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

import numpy as np
import pandas as pd
import os
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label
from skimage.util import crop
from skimage.transform import resize
from skimage.segmentation import find_boundaries, mark_boundaries
from scipy.ndimage.morphology import binary_erosion as erosion
from scipy.ndimage.morphology import binary_dilation as dilation
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
%matplotlib inline
train_path = '../input/train_color'
label_path = '../input/train_label'
def get_data():
    label = os.listdir(label_path)
    label.sort()
    train = os.listdir(train_path)
    train.sort()
    df_id = pd.DataFrame()
    df_id['label'] = label
    df_id['train'] = train
    df_id['label_path'] = df_id['label'].apply(lambda x: os.path.join(label_path, x))
    df_id['train_path'] = df_id['train'].apply(lambda x: os.path.join(train_path, x))
    return df_id

df_id = get_data()
ix = np.random.randint(0, len(df_id.index))
train_ex = imread(df_id.loc[ix, 'train_path'])
label_ex = imread(df_id.loc[ix, 'label_path'])
f, ax = plt.subplots(1,2, figsize = (10,10))
ax[0].imshow(train_ex)
ax[1].imshow(np.squeeze(label_ex))
ax[0].axis('off')
ax[1].axis('off')
ax[0].set_title('Photo')
ax[1].set_title('Mask')
plt.show()
print(df_id.info())
df_id.head()
test_path = '../input/test'
def get_test_data():
    test = os.listdir(test_path)
    test.sort()
    df_id = pd.DataFrame()
    df_id['test'] = test
    df_id['test_path'] = df_id['test'].apply(lambda x: os.path.join(test_path, x))
    return df_id

df_test_id = get_test_data()
plt.imshow(imread(df_test_id.sample(n = 1)['test_path'].values[0]))
plt.axis('off')
print(df_test_id.info())
df_test_id.head()
labelmap_target = {33:'car', 34:'motorbicycle', 35:'bicycle', 36:'person', 38:'truck', 39:'bus', 40:'tricycle'}
datagen_arg = dict(horizontal_flip = True)

maskgen_arg = dict(horizontal_flip = True)

x_gen = ImageDataGenerator(**datagen_arg)
mask_gen = ImageDataGenerator(**maskgen_arg)

#a custom generator that outputs x and y
#x is the raw input unit 8 image
# y is the 7 dims array of masks
def label_to_mask(y, classes):
    mask_core = np.zeros((y.shape[0], y.shape[1], len(classes)))
    mask_edge = np.zeros((y.shape[0], y.shape[1], 1))
    foreground = (y*((y >= classes[0]*1000) & (y < (classes[-1]+1)*1000))).astype(np.uint16)
    unique_objects = np.delete(np.unique(foreground), 0)
    # mask core
    for i, class_ in enumerate(classes):
        mask_core[:,:,i] = np.squeeze(((foreground/1000).astype(np.int32) == class_).astype(np.bool))
    # This operation makes the edge ~2 pixels thick,
    # remember to fill the pixels back in post processing
    mask_edge = find_boundaries(foreground, mode = 'outer').astype(np.bool)
    return mask_core, mask_edge

def resize_random_crop(images, random_state, resize_w = 1280, resize_h = 720, crop_w = 224, crop_h = 224):
    np.random.seed(random_state)
    # function to shrink and random crop input to desired size for training
    # resize is done so random crop won't output images with no targets too often
    images = np.array([resize(image, (resize_h, resize_w), mode = 'constant', preserve_range = True) for image in images])
    height = images.shape[1]
    width = images.shape[2]
    # The randn function is to prevent data augmentation from producing too many crop with sky only
    rand_h_start = (np.clip(0.9*np.random.randn()+0.7, 0, 1)*(height - crop_h)).astype(np.uint16)
    rand_w_start = np.random.randint(0, (width - crop_w))
    return crop(images, ((0, 0), (rand_h_start, height - (rand_h_start + crop_h)),
                        (rand_w_start, width - (rand_w_start + crop_w)), (0, 0)))

def augmentation_checker(mask, classes, percentage = 0.005):
    # function to ensure that the cropped input has at least some mask in it
    mask_core, mask_edge = label_to_mask(mask, classes)
    while np.mean(mask_core) < percentage:
        mask_core, mask_edge = label_to_mask(mask, classes)
    return mask_core, mask_edge

def data_generator(x_gen_, mask_gen_, df_data, mini_bat_size, classes = sorted(labelmap_target.keys())):
    while True:
        seed = np.random.randint(0,1000)
        sampled_set = df_data.sample(n = mini_bat_size)
        sampled_train = sampled_set['train_path']
        sampled_label = sampled_set['label_path']
        X = preprocess_input(np.array([imread(train_path) for train_path in sampled_train]).astype(np.float32))
        y = np.expand_dims(np.array([imread(label_path) for label_path in sampled_label]), axis = -1)

        x_generator = x_gen_.flow(X, batch_size = mini_bat_size, seed = seed)
        mask_generator = mask_gen_.flow(y, batch_size = mini_bat_size, seed = seed)
        X = resize_random_crop(x_generator.next(), seed)
        y = resize_random_crop(mask_generator.next(), seed)
        y_mask = []
        y_edge = []
        for mask in y:
            mask_core, mask_edge = label_to_mask(mask, classes)
            y_mask.append(mask_core)
            y_edge.append(mask_edge)
        yield (X, {'masks': np.array(y_mask), 'edges': np.array(y_edge)})
mini_bat_size = 2
X, y= next(data_generator(x_gen, mask_gen, df_id, mini_bat_size))
ix = 1
X_cropped = X[ix]
y_edge_cropped = np.squeeze(y['edges'][ix])
y_mask_cropped = np.squeeze(y['masks'][ix,:,:,0])
y_mask_seg = y_mask_cropped - y_edge_cropped
f, ax = plt.subplots(1,4, figsize = (20,20))
ax[0].imshow(X_cropped/255 + 150) # This is my lazy work to roughly undo preprocessing for this visualization
ax[0].axis('off')
ax[0].set_title('X cropped')
ax[1].imshow(y_edge_cropped)
ax[1].axis('off')
ax[1].set_title('edge: all')
ax[2].imshow(y_mask_cropped)
ax[2].axis('off')
ax[2].set_title('mask: car')
ax[3].imshow(y_mask_seg > 0)
ax[3].axis('off')
ax[3].set_title('post: segmented by instance')
try:
    R50 = ResNet50(include_top = False, input_shape = (224, 224, 3))
except:
    # I can't get the pretrained weight on Kaggle kernel, so we will set weights = None for now.
    R50 = ResNet50(include_top = False, weights = None, input_shape = (224, 224, 3))
# pop off the last pooling layer
R50.layers.pop()
for layer in R50.layers:
    layer.trainable = False
    
# Layers from ResNet50 to make skip connections
skip_ix = [172, 140, 78, 36, 3]
# Layers in decoder to connect to encoder
skip_end = []
for i in skip_ix:
    skip_end.append(R50.layers[i])
R50.summary()
# Use billinear additive upsampling (BAU) and separable convolution to reduce the total amount of hyperparameters
def BAU_layer(last_layer, channel_num):
    additive = []
    depth = int(last_layer.get_shape()[-1])
    step = int(depth / channel_num)
    last_layer[:, :, :, 1*step:(1*step+step)]
    for i in range(channel_num):
        layersum = K.mean(last_layer[:, :, :, i*step:(i*step+step)], axis = -1)
        additive.append(layersum)
    additive = K.stack(additive, axis = -1)
    return additive

def upsampling_step(skipped_conv, num_output_filters, prev_conv = None):
    num_filters = skipped_conv.output_shape[-1]
    if prev_conv != None:
        concat_layer = concatenate([skipped_conv.output, prev_conv])
    else:
        concat_layer = skipped_conv.output
    conv1 = SeparableConv2D(num_filters, 3, padding = 'same', activation = 'relu',
                           depthwise_initializer = 'he_normal', pointwise_initializer = 'he_normal')(concat_layer)
    conv2 = SeparableConv2D(num_filters, 3, padding = 'same', activation = 'relu',
                           depthwise_initializer = 'he_normal', pointwise_initializer = 'he_normal')(conv1)
    up = UpSampling2D()(conv2)
    BAU = Lambda(BAU_layer, arguments = {'channel_num': num_output_filters})(up)
    conv3 = SeparableConv2D(num_output_filters, 2, padding = 'same', activation = 'relu',
                           depthwise_initializer = 'he_normal', pointwise_initializer = 'he_normal')(BAU)
    return conv3

def output(feature_map, mask = True):
    if mask:
        conv3 = Conv2D(7, 1, padding = 'same', activation = 'sigmoid', kernel_initializer = 'he_normal', name = 'masks')(feature_map)
    else:
        conv3 = Conv2D(1, 1, padding = 'same', activation = 'sigmoid', kernel_initializer = 'he_normal', name = 'edges')(feature_map)
    return conv3
up_num_filters = [1024, 512, 256, 64, 64]
for n, i in enumerate(up_num_filters):
    if n == 0:
        conv_layer = upsampling_step(skip_end[n], i)
    else:
        conv_layer = upsampling_step(skip_end[n], i, conv_layer)
conv1 = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_layer)
conv2 = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv1)

masks = output(conv2)
edges = output(conv2, False)
model = Model(inputs = R50.inputs, outputs = [masks, edges])
model.compile(optimizer = 'adam', loss = {'masks': binary_crossentropy, 'edges': binary_crossentropy})

# Optional callbacks
# checkpointer = ModelCheckpoint('./models/ResNet50_U-Net_chkpt.h5', verbose = 1, save_best_only = True)
# tbCallback = TensorBoard(log_dir = './Graph', histogram_freq = 0, write_graph = True, write_images = True)
# earlystopper = EarlyStopping(patience = 5, verbose = 1)
bat_size = 2
result = model.fit_generator(data_generator(x_gen, mask_gen, df_id, bat_size), steps_per_epoch = bat_size, epochs = 50,
                            verbose = 1)