# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
image_path = '/kaggle/input/vocdataset/VOCdevkit/VOC2012/JPEGImages/'

os.mkdir('/kaggle/working/sketch-img')

os.mkdir('/kaggle/working/real-img-test')

os.mkdir('/kaggle/working/sketch-img-test')

os.mkdir('/kaggle/working/PaintsTensorFlowDraftModel/')

os.mkdir('/kaggle/working/real-img')

os.mkdir('/kaggle/working/model-save/')
import cv2

import matplotlib.pyplot as plt

%matplotlib inline



def sketch(image):

    #converting_image_to_grayscale

    img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #blurring_image_to_remove_noise

    img_blur=cv2.GaussianBlur(img_gray,(3,3),0)

    #extracting_edges

    edges=cv2.Canny(img_blur,10,80)

    #applying_threshold_inverse

    ret,mask=cv2.threshold(edges,50,255,cv2.THRESH_BINARY_INV)

    return mask

import imageio



count = 0

for images in os.listdir(image_path):

    img = str(image_path + '/'+images)

    s = imageio.imread(img)

    save_image = '/kaggle/working/sketch-img/img_'+ str(count)+'.jpeg'

    plt.imsave(save_image, sketch(s), cmap='gray', vmin=0, vmax=255)

    count += 1 

    if count == 3000:

        break
count = 0

for images in os.listdir(image_path):

    if count > 3000 and count < 4000:

        img = str(image_path + '/'+images)

        s = imageio.imread(img)

        save_image = '/kaggle/working/sketch-img-test/img_'+ str(count)+'.jpeg'

        plt.imsave(save_image, sketch(s), cmap='gray', vmin=0, vmax=255)

    count += 1 

    if count == 4000:

        break
from PIL import Image

count = 0

for files in os.listdir(image_path):

    if count > 3000 and count < 4000:

        img = Image.open(image_path +'/'+files)

        path = '/kaggle/working/real-img-test/imgreal_'+str(count)+'.jpeg'

        img.save(path)

    count += 1

    if count == 4000:

        break
count = 0

for files in os.listdir(image_path):

    img = Image.open(image_path +'/'+files)

    path = '/kaggle/working/real-img/imgreal_'+str(count)+'.jpeg'

    img.save(path)

    count += 1

    if count == 3000:

        break
loadEpochs = 10

import tensorflow as tf

import cv2

from glob import glob



config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

tf.compat.v1.enable_eager_execution(config=config)



from tqdm import tqdm



# edit by your path

__SAVED_MODEL_PATH__ = "/kaggle/working/PaintsTensorFlowDraftModel/"

# hyperparameters.py

batch_steps = 0



gf_dim = 64

df_dim = 64

c_dim = 3



lr = 1e-5

beta1 = 0.9

beta2 = 0.99



l1_scaling = 100

l2_scaling = 10



epoch = 2

batch_size = 4



log_interval = 10

sampling_interval = 200

save_interval = 4000



train_image_datasets_path = "/kaggle/working/real-img/*"

train_line_datasets_path = "/kaggle/working/sketch-img/*"

test_image_datasets_path = "/kaggle/working/real-img-test/*"

test_line_datasets_path = "/kaggle/working/sketch-img-test/*"
# utils.py

def get_line(imgs):

    def img_liner(img):

        k = 3

        kernal = np.ones((k, k), dtype=np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        dilated = cv2.dilate(gray, kernal, iterations=1)

        diff = cv2.absdiff(dilated, gray)

        img = 255 - diff

        return img



    lines = np.array([img_liner(l) for l in imgs])

    return np.expand_dims(lines, 3)





def convert2f32(img):

    img = img.astype(np.float32)

    return (img / 127.5) - 1.0





def convert2uint8(img):

    img = (img + 1) * 127.5

    return img.astype(np.uint8)





def convertRGB(imgs):

    imgs = np.asarray(imgs, np.uint8)

    return np.array([cv2.cvtColor(img, cv2.COLOR_YUV2RGB) for img in imgs])





def mkdir(path):

    try:

        os.mkdir(path)

    except FileExistsError:

        pass





def initdir(model_name):

    base = os.path.join("/kaggle/working/", model_name)

    mkdir(base)

    mkdir(os.path.join(base, "board"))

    mkdir(os.path.join(base, "image"))
#SubNet.py

__INITIALIZER__ = tf.random_normal_initializer(0., 0.02)

__MOMENTUM__ = 0.9

__EPSILON__ = 1e-5





def res_net_block_v2(inputs, filters):

    with tf.name_scope("ResNetBlock"):

        shortcut = inputs

        tensor = tf.keras.layers.BatchNormalization()(inputs)

        tensor = tf.keras.layers.ReLU()(tensor)

        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="SAME")(tensor)



        tensor = tf.keras.layers.BatchNormalization()(tensor)

        tensor = tf.keras.layers.ReLU()(tensor)

        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="SAME")(tensor)

        tensor = tf.keras.layers.add([shortcut, tensor])

    return tensor





def GenConvBlock(inputs, filters, k, s, res_net_block=True, name="GenConvBlock"):

    filters = int(filters)

    with tf.name_scope(name):

        tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=k, strides=s, use_bias=False,

                                        padding="SAME", kernel_initializer=__INITIALIZER__)(inputs)



        if res_net_block:

            tensor = res_net_block_v2(tensor, filters)

        else:

            tensor = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)(tensor)

            tensor = tf.keras.layers.LeakyReLU()(tensor)



        return tensor





def GenUpConvBlock(inputs_a, inputs_b, filters, k, s, res_net_block=True, name="GenUpConvBlock"):

    filters = int(filters)

    with tf.name_scope(name):

        tensor = tf.keras.layers.Concatenate(3)([inputs_a, inputs_b])

        tensor = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=k, strides=s, use_bias=False,

                                                 padding="SAME", kernel_initializer=__INITIALIZER__)(tensor)



        if res_net_block:

            tensor = res_net_block_v2(tensor, filters)

        else:

            tensor = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)(tensor)

            tensor = tf.keras.layers.ReLU()(tensor)



        return tensor





class DisConvBlock(tf.keras.Model):

    def __init__(self, filters, k, s, apply_bat_norm=True, name=None):

        super(DisConvBlock, self).__init__(name=name)

        initializer = tf.random_normal_initializer(0., 0.02)

        filters = int(filters)

        self.apply_bat_norm = apply_bat_norm

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=k, strides=s,

                                           padding="SAME", kernel_initializer=initializer)

        if self.apply_bat_norm:

            self.bn = tf.keras.layers.BatchNormalization(momentum=__MOMENTUM__, epsilon=__EPSILON__)



        self.act = tf.keras.layers.LeakyReLU(alpha=0.2)



    def call(self, inputs, training):

        tensor = self.conv(inputs)



        if self.apply_bat_norm:

            tensor = self.bn(tensor, training=training)



        tensor = self.act(tensor)

        return tensor





def tf_int_round(num):

    return tf.cast(tf.round(num), dtype=tf.int32)





class resize_layer(tf.keras.layers.Layer):

    def __init__(self, size=(512, 512), **kwargs, ):

        super(resize_layer, self).__init__(**kwargs)

        (self.height, self.width) = size



    def build(self, input_shape):

        super(resize_layer, self).build(input_shape)



    def call(self, x, method="nearest"):

        height = 512

        width = 512



        if method == "nearest":

            return tf.image.resize_nearest_neighbor(x, size=(height, width))

        elif method == "bicubic":

            return tf.image.resize_bicubic(x, size=(height, width))

        elif method == "bilinear":

            return tf.image.resize_bilinear(x, size=(height, width))



    def get_output_shape_for(self, input_shape):

        return (self.input_shape[0], 512, 512, 3)
#PaintsTensorflow

def Generator(inputs_size=None, res_net_block=True, name="PaintsTensorFlow"):

    inputs_line = tf.keras.Input(shape=[inputs_size, inputs_size, 1], dtype=tf.float32, name="inputs_line")

    inputs_hint = tf.keras.Input(shape=[inputs_size, inputs_size, 3], dtype=tf.float32, name="inputs_hint")

    tensor = tf.keras.layers.Concatenate(3)([inputs_line, inputs_hint])



    e0 = GenConvBlock(tensor,gf_dim / 2, 3, 1, res_net_block=res_net_block, name="E0")  # 64

    e1 = GenConvBlock(e0, gf_dim * 1, 4, 2, res_net_block=res_net_block, name="E1")

    e2 = GenConvBlock(e1, gf_dim * 1, 3, 1, res_net_block=res_net_block, name="E2")

    e3 = GenConvBlock(e2, gf_dim * 2, 4, 2, res_net_block=res_net_block, name="E3")

    e4 = GenConvBlock(e3, gf_dim * 2, 3, 1, res_net_block=res_net_block, name="E4")

    e5 = GenConvBlock(e4, gf_dim * 4, 4, 2, res_net_block=res_net_block, name="E5")

    e6 = GenConvBlock(e5, gf_dim * 4, 3, 1, res_net_block=res_net_block, name="E6")

    e7 = GenConvBlock(e6, gf_dim * 8, 4, 2, res_net_block=res_net_block, name="E7")

    e8 = GenConvBlock(e7, gf_dim * 8, 3, 1, res_net_block=res_net_block, name="E8")



    d8 = GenUpConvBlock(e7, e8, gf_dim * 8, 4, 2, res_net_block=res_net_block, name="D8")

    d7 = GenConvBlock(d8, gf_dim * 4, 3, 1, res_net_block=res_net_block, name="D7")

    d6 = GenUpConvBlock(e6, d7, gf_dim * 4, 4, 2, res_net_block=res_net_block, name="D6")

    d5 = GenConvBlock(d6, gf_dim * 2, 3, 1, res_net_block=res_net_block, name="D5")

    d4 = GenUpConvBlock(e4, d5, gf_dim * 2, 4, 2, res_net_block=res_net_block, name="D4")

    d3 = GenConvBlock(d4, gf_dim * 1, 3, 1, res_net_block=res_net_block, name="D3")

    d2 = GenUpConvBlock(e2, d3, gf_dim * 1, 4, 2, res_net_block=res_net_block, name="D2")

    d1 = GenConvBlock(d2, gf_dim / 2, 3, 1, res_net_block=res_net_block, name="D1")



    tensor = tf.keras.layers.Concatenate(3)([e0, d1])

    outputs = tf.keras.layers.Conv2D(c_dim, kernel_size=3, strides=1, padding="SAME",

                                     use_bias=True, name="output", activation=tf.nn.tanh,

                                     kernel_initializer=tf.random_normal_initializer(0., 0.02))(tensor)



    inputs = [inputs_line, inputs_hint]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model



class Discriminator(tf.keras.Model):

    def __init__(self):

        super(Discriminator, self).__init__()



        self.h0 = DisConvBlock(df_dim / 2, 4, 2)

        self.h1 = DisConvBlock(df_dim / 2, 3, 1)

        self.h2 = DisConvBlock(df_dim * 1, 4, 2)

        self.h3 = DisConvBlock(df_dim * 1, 3, 1)

        self.h4 = DisConvBlock(df_dim * 2, 4, 2)

        self.h5 = DisConvBlock(df_dim * 2, 3, 1)

        self.h6 = DisConvBlock(df_dim * 4, 4, 2)

        self.flatten = tf.keras.layers.Flatten()

        self.last = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=tf.initializers.he_normal())



#     @tf.contrib.eager.defun

    def call(self, inputs, training):

        tensor = self.h0(inputs, training)

        tensor = self.h1(tensor, training)

        tensor = self.h2(tensor, training)

        tensor = self.h3(tensor, training)

        tensor = self.h4(tensor, training)

        tensor = self.h5(tensor, training)

        tensor = self.h6(tensor, training)

        tensor = self.flatten(tensor)  # (?,16384)

        tensor = self.last(tensor)

        return tensor
#Datasets.py

import cv2

class Datasets:

    def __init__(self, prefetch=-1, batch_size=1, shuffle=False):

        self.prefetch = prefetch

        self.batch_size = batch_size

        self.shuffle = shuffle



    def _preprocess(self, image, line, training = 'True'):

        if training == 'True':

            if np.random.rand() < 0.5:

                image = cv2.flip(np.float32(image), 0)

                line = cv2.flip(np.float32(line), 0)

#                 line = np.expand_dims(line, 3)



            if np.random.rand() < 0.5:

                image = cv2.flip(np.float32(image), 1)

                line = cv2.flip(np.float32(line), 1)

#                 line = np.expand_dims(line, 3)



        return image, line, self._buildHint_resize(image)



    def _buildHint_resize(self, image):

        random = np.random.rand

        hint = np.ones_like(image)

        hint += 1

        leak_count = np.random.randint(16, 120)



        if random() < 0.4:

            leak_count = 0

        elif random() < 0.7:

            leak_count = np.random.randint(2, 16)



        # leak position

        x = np.random.randint(1, image.shape[0] - 1, leak_count)

        y = np.random.randint(1, image.shape[1] - 1, leak_count)



        def paintCel(i):

            color = image[x[i]][y[i]]

            hint[x[i]][y[i]] = color



            if random() > 0.5:

                hint[x[i]][y[i] + 1] = color

                hint[x[i]][y[i] - 1] = color



            if random() > 0.5:

                hint[x[i] + 1][y[i]] = color

                hint[x[i] - 1][y[i]] = color



        for i in range(leak_count):

            paintCel(i)



        return hint



    def convert2float(self, image):

        image = tf.cast(image, tf.float32)

        image = (image / 127.5) - 1

        return image



    def __line_threshold(self, line):

        if np.random.rand() < 0.3:

            line = np.reshape(line, newshape=(512, 512))

            _, line = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            line = np.reshape(line, newshape=(512, 512, 1))

        return line



    def loadImage(self, imagePath, linePath, isTrain='True'):

        image = tf.io.read_file(imagePath)

        image = tf.image.decode_jpeg(image, channels=3)



        line = tf.io.read_file(linePath)

        line = tf.image.decode_jpeg(line, channels=1)



        image = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        line = tf.image.resize(line, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)



        image = self.convert2float(image)

        line = self.convert2float(line)



        image, line, hint = tf.py_function(self._preprocess,

                                       [np.float32(image), np.float32(line), str(isTrain)],

                                       [tf.float32, tf.float32, tf.float32])



        return image, line, hint



    def buildDataSets(self):

        def build_dataSets(image, line, shuffle=False, isTrain=False):

            image = glob(image)

            image.sort()

            line = glob(line)

            line.sort()



            if shuffle is False and isTrain is False:

                image.reverse()

                line.reverse()



            batch_steps = int(len(line) / self.batch_size)

            datasets = tf.data.Dataset.from_tensor_slices((image, line))

#             datasets = datasets.map(lambda x, y: self.loadImage(x, y, isTrain))

            image_data = []

            line_data = []

            hint_data = []

            for ele in datasets:

                x, y ,z = self.loadImage(ele[0], ele[1], str(isTrain))

                image_data.append(x)

                line_data.append(y)

                hint_data.append(z)

#             datasets = datasets.batch(self.batch_size)

            image_data_batch = []

            line_data_batch = []

            hint_data_batch = []

            for i in range(0, len(image_data), self.batch_size):

                image_data_batch.append(image_data[i:i+self.batch_size])

                line_data_batch.append(line_data[i:i+self.batch_size])

                hint_data_batch.append(hint_data[i:i+self.batch_size])



            return image_data_batch, line_data_batch, hint_data_batch



        testDatasets = build_dataSets(test_image_datasets_path,

                                      test_line_datasets_path,

                                      shuffle=False, isTrain=False)



        trainDatasets = build_dataSets(train_image_datasets_path,

                                       train_line_datasets_path,

                                       shuffle=False, isTrain=True)



        return trainDatasets, testDatasets



class Datasets_512(Datasets):

    def __init__(self ,batch_size):

        self.batch_size = batch_size

        super().__init__(self, batch_size = self.batch_size)

    def _flip(self, image, line, training):

        if training:

            if np.random.rand() < 0.5:

                image = cv2.flip(image, 0)

                line = cv2.flip(line, 0)

                line = np.expand_dims(line, 3)



            if np.random.rand() < 0.5:

                image = cv2.flip(image, 1)

                line = cv2.flip(line, 1)

                line = np.expand_dims(line, 3)



        return image, line



    def _buildHint(self, image):

        random = np.random.rand

        hint = np.ones_like(image)

        hint += 1

        leak_count = np.random.randint(16, 128)



        # leak position

        x = np.random.randint(1, image.shape[0] - 1, leak_count)

        y = np.random.randint(1, image.shape[1] - 1, leak_count)



        def paintCel(i):

            color = image[x[i]][y[i]]

            hint[x[i]][y[i]] = color



            if random() > 0.5:

                hint[x[i]][y[i] + 1] = color

                hint[x[i]][y[i] - 1] = color



            if random() > 0.5:

                hint[x[i] + 1][y[i]] = color

                hint[x[i] - 1][y[i]] = color



        for i in range(leak_count):

            paintCel(i)

        return hint



    def loadImage(self, imagePath, linePath, isTrain='True'):

        image = tf.io.read_file(imagePath)

        image = tf.image.decode_jpeg(image, channels=3)

        line = tf.io.read_file(linePath)

        line = tf.image.decode_jpeg(line, channels=1)



        image_128 = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        line_128 = tf.image.resize(line, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)



        image = self.convert2float(image)

        line = self.convert2float(line)

        image_128 = self.convert2float(image_128)

        line_128 = self.convert2float(line_128)



        hint_128 = tf.py_function(self._buildHint,

                                  [image_128],

                                  tf.float32)



        hint_128.set_shape(shape=image_128.shape)

        hint = tf.image.resize(hint_128, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)



        return line_128, hint_128, image, line, hint

#NEW BUILD

    def buildDataSets(self):

        def build_dataSets(image, line, shuffle=False, isTrain=False):

            image = glob(image)

            image.sort()

            line = glob(line)

            line.sort()



            if shuffle is False and isTrain is False:

                image.reverse()

                line.reverse()



            batch_steps = int(len(line) / self.batch_size)

            datasets = tf.data.Dataset.from_tensor_slices((image, line))

#             datasets = datasets.map(lambda x, y: self.loadImage(x, y, isTrain))

            line_128 = []

            hint_128 = []

            image_data = []

            line_data = []

            hint_data = []

            for ele in datasets:

                a, b, x, y ,z = self.loadImage(ele[0], ele[1], str(isTrain))

                line_128.append(a)

                hint_128.append(b)

                image_data.append(x)

                line_data.append(y)

                hint_data.append(z)

#             datasets = datasets.batch(self.batch_size)

            line_128_data_batch = []

            hint_128_data_batch = []

            image_data_batch = []

            line_data_batch = []

            hint_data_batch = []

            for i in range(0, len(image_data), self.batch_size):

                line_128_data_batch.append(line_128[i:i+self.batch_size])

                hint_128_data_batch.append(hint_128[i:i+self.batch_size])

                image_data_batch.append(image_data[i:i+self.batch_size])

                line_data_batch.append(line_data[i:i+self.batch_size])

                hint_data_batch.append(hint_data[i:i+self.batch_size])



            return line_128_data_batch ,hint_128_data_batch ,image_data_batch, line_data_batch, hint_data_batch



        testDatasets = build_dataSets(test_image_datasets_path,

                                      test_line_datasets_path,

                                      shuffle=False, isTrain=False)



        trainDatasets = build_dataSets(train_image_datasets_path,

                                       train_line_datasets_path,

                                       shuffle=False, isTrain=True)



        return trainDatasets, testDatasets
def loadImage(self, imagePath, linePath, isTrain='True'):

    image = tf.io.read_file(imagePath)

    image = tf.image.decode_jpeg(image, channels=3)

    line = tf.io.read_file(linePath)

    line = tf.image.decode_jpeg(line, channels=1)

    image_128 = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    line_128 = tf.image.resize(line, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = self.convert2float(image)

    line = self.convert2float(line)

    image_128 = self.convert2float(image_128)

    line_128 = self.convert2float(line_128)

    hint_128 = tf.py_function(self._buildHint,[image_128],tf.float32)

    hint_128.set_shape(shape=image_128.shape)

    hint = tf.image.resize(hint_128, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return line_128, hint_128, image, line, hint
def buildDataSets():

    def build_dataSets(image, line, shuffle=False, isTrain=False):

        image = glob(image)

        image.sort()

        line = glob(line)

        line.sort()

        if shuffle is False and isTrain is False:

            image.reverse()

            line.reverse()



        batch_steps = int(len(line) / 4)

        datasets = tf.data.Dataset.from_tensor_slices((image, line))

#             datasets = datasets.map(lambda x, y: self.loadImage(x, y, isTrain))

        line_128 = []

        hint_128 = []

        image_data = []

        line_data = []

        hint_data = []

        for ele in datasets:

            a, b, x, y ,z = loadImage(ele[0], ele[1], str(isTrain))

            line_128.append(a)

            hint_128.append(b)

            image_data.append(x)

            line_data.append(y)

            hint_data.append(z)

#             datasets = datasets.batch(self.batch_size)

        line_128_data_batch = []

        hint_128_data_batch = []

        image_data_batch = []

        line_data_batch = []

        hint_data_batch = []

        for i in range(0, len(image_data), self.batch_size):

            line_128_data_batch.append(line_128[i:i+self.batch_size])

            hint_128_data_batch.append(hint_128[i:i+self.batch_size])

            image_data_batch.append(image_data[i:i+self.batch_size])

            line_data_batch.append(line_data[i:i+self.batch_size])

            hint_data_batch.append(hint_data[i:i+self.batch_size])



        return line_128_data_batch ,hint_128_data_batch ,image_data_batch, line_data_batch, hint_data_batch



    testDatasets = build_dataSets(test_image_datasets_path,test_line_datasets_path,shuffle=False, isTrain=False)



    trainDatasets = build_dataSets(train_image_datasets_path,train_line_datasets_path,shuffle=False, isTrain=True)



    return trainDatasets, testDatasets
# train_sets, test_sets = buildDataSets()

# print (train_sets, test_sets)
#PaintsTensorflowTraining

class PaintsTensorFlowTrain:

    def __init__(self, model_name="PaintsTensorFlow"):

        self.data_sets = Datasets_512(batch_size=batch_size)

        self.model_name = "{}".format(model_name)

        initdir(self.model_name)



        self.global_steps = tf.compat.v1.train.get_or_create_global_step()

        self.epochs = tf.Variable(0, trainable=False, dtype=tf.int32)

        self.ckpt_path = "./ckpt/{}/".format(self.model_name) + "ckpt_E:{}"

        self.ckpt_prefix = os.path.join(self.ckpt_path, "model_GS:{}")



#         self.generator_128 =  tf.keras.models.load_model(__SAVED_MODEL_PATH__)

        self.generator_512 = Generator(res_net_block=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)



        self.check_point = tf.train.Checkpoint(generator_512=self.generator_512,

                                               optimizer=self.optimizer,

                                               globalSteps=self.global_steps,

                                               epochs=self.epochs)



#     def __loging(self, name, scalar):

#         with tf.contrib.summary.always_record_summaries():

#             tf.contrib.summary.scalar(name, scalar)



    def __loss(self, output, target):

        loss = tf.reduce_mean(tf.abs(target - output))

        return loss



    def __pred_image(self, model, image, line, hint, draft, epoch=None):

        gs = self.global_steps.numpy()

        predImage = model.predict([line, draft])

        file_name = "./ckpt/{}/image/{}.jpg".format(self.model_name, gs)



        if epoch is not None:

            loss = self.__loss(predImage, image)

#             self.__loging("Sample_LOSS", loss)

            loss = "{:0.05f}".format(loss).zfill(7)

            print("Epoch:{} GS:{} LOSS:{}".format(epoch, self.global_steps.numpy(), loss))

            file_name = "./ckpt/{}/image/{}_loss:{}.jpg".format(self.model_name, gs, loss)



        hint = np.array(hint)

        hint[hint > 1] = 1



        lineImage = np.concatenate([line, line, line], -1)

        save_img = np.concatenate([lineImage, hint, draft, predImage, image], 1)

        save_img = utils.convert2uint8(save_img)

        tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)



    def __draft_image(self, line_128, hint_128):

        draft = self.generator_128.predict([line_128, hint_128])

        draft = tf.image.resize(draft, size=(512, 512),

                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return draft



    def training(self, loadEpochs=0):

        train_sets, test_sets = self.data_sets.buildDataSets()

        print (train_sets, test_sets)

#         log = self.__loging



        self.check_point.restore(tf.train.latest_checkpoint(self.ckpt_path.format(loadEpochs)))



        if self.global_steps.numpy() == 0:

            self.check_point.save(file_prefix=self.ckpt_prefix.format(0, 0))

            print("------------------------------SAVE_INIT-------------------------------------")



        for epochss in range(epoch):

            print("GS: ", self.global_steps.numpy())



            for line_128, hint_128, image, line, hint in tqdm(train_sets, total=batch_steps):

                draft = self.__draft_image(line_128, hint_128)



                with tf.GradientTape() as tape:

                    genOut = self.generator_512(inputs=[line, draft], training=True)

                    loss = self.__loss(genOut, image)



                # Training

                gradients = tape.gradient(loss, self.generator_512.variables)

                self.optimizer.apply_gradients(zip(gradients, self.generator_512.variables),

                                               global_step=self.global_steps)

                # Loging

                gs = self.global_steps.numpy()

                if gs % log_interval == 0:

                    log("LOSS", loss)

                    if gs % sampling_interval == 0:

                        # test image Save

                        for line_128, hint_128, image, line, hint in test_sets.take(1):

                            draft = self.__draft_image(line_128, hint_128)

                            self.__pred_image(self.generator_512, image, line, hint, draft, self.epochs.numpy())



                    if gs % save_interval == 0:

                        self.check_point.save(file_prefix=self.ckpt_prefix.format(self.epochs.numpy(), gs))

                        print("------------------------------SAVE_E:{}_G:{}-------------------------------------"

                              .format(self.epochs.numpy(), gs))



            self.check_point.save(file_prefix=self.ckpt_prefix.format(self.epochs.numpy(), self.global_steps.numpy()))

            self.epochs = self.epochs + 1



        for line_128, hint_128, image, line, hint in test_sets.take(1):

            hint = hint.numpy()

            draft = self.__draft_image(line_128, hint_128)

            self.__pred_image(self.generator_512, image, line, hint, draft, self.epochs.numpy())



        self.generator_512.summary()

        print(self.global_steps)



        save_path = "/kaggle/working/model-save" + self.model_name + "/{}.h5".format(self.generator_512.name)

        self.generator_512.save(save_path, include_optimizer=False)  # for keras Model

        save_path = tf.keras.models.save_model(self.generator_512, save_path)  # saved_model

        print("saved_model path = {}".format(save_path))

        print("------------------------------Training Done-------------------------------------")
model = PaintsTensorFlowTrain()
# model.training(loadEpochs=1)