# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir('/kaggle/input/colorization-to-caffee/')
# import cv2
# imagePath = '/kaggle/input/colorization-to-sketch-ru/real-img'
# for img in os.listdir(imagePath):
#     image = cv2.imread(imagePath+'/'+img)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     (thresh, blackAndWhiteImage) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
#     path = '/kaggle/working/train_sketch/'+img
#     cv2.imwrite(path, blackAndWhiteImage)
loadEpochs = 10
import tensorflow as tf
import cv2
from glob import glob

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)

from tqdm import tqdm

# edit by your path
__SAVED_MODEL_PATH__ = '/kaggle/input/colorization-to-sketch-ru/model-save'

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

train_image_datasets_path = "/kaggle/input/colorization-to-sketch-ru/real-img/*"
train_line_datasets_path = "/kaggle/input/colorization-to-sketch-ru/sketch-img/*"
test_image_datasets_path = "/kaggle/input/colorization-to-sketch-ru/real-img-test/*"
test_line_datasets_path = "/kaggle/input/colorization-to-sketch-ru/sketch-img-test/*"
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
        
    def get_image_and_line(self):
        image = glob(train_image_datasets_path)
        image.sort()
        line = glob(train_line_datasets_path)
        line.sort()
        return image, line
        
    def _next(self, image, line):   
        return self.buildDataSets(image, line)
        
  

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
        image = (image / 127.5) 
        return image

    def __line_threshold(self, line):
        if np.random.rand() < 0.3:
            line = np.reshape(line, newshape=(512, 512))
            _, line = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            line = np.reshape(line, newshape=(512, 512, 1))
        return line

    def loadImage(self, imagePath, linePath, isTrain='True'):
#         print (imagePath)
        image = tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)

        line = tf.io.read_file(linePath)
        line = tf.image.decode_jpeg(line, channels=1)

        image = tf.image.resize(image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
#         plt.imshow(image)
        line = tf.image.resize(line, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image = self.convert2float(image)
        line = self.convert2float(line)

        image, line, hint = tf.py_function(self._preprocess,
                                       [np.float32(image), np.float32(line), str(isTrain)],
                                       [tf.float32, tf.float32, tf.float32])
    

        return image, line, hint

    def buildDataSets(self, image, line):
        def build_dataSets(image, line, shuffle=False, isTrain=False):
#             image = glob(image)
#             image.sort()
#             line = glob(line)
#             line.sort()
            if shuffle is False and isTrain is False:
                image.reverse()
                line.reverse()

            batch_steps = int(3000 / self.batch_size)
            datasets = tf.data.Dataset.from_tensor_slices((image, line))
#             datasets = datasets.map(lambda x, y: self.loadImage(x, y, isTrain))
            image_data = []
            line_data = []
            hint_data = []
            line128_data = []
            hint128_data = []
            for ele in datasets:
#                 p, q, x, y ,z  = self.loadImage(ele[0], ele[1], str(isTrain))
                x, y ,z  = self.loadImage(ele[0], ele[1], str(isTrain))
                #line_128, hint_128, image, line, hint
#                 line128_data.append(p)
#                 hint128_data.append(q)
                image_data.append(x)
                
                line_data.append(y)
                hint_data.append(z)
#                 del (p)
#                 del (q)
                del (x)
                del (y)
                del (z)

            del(datasets)
            del (line)
            del(image)
#             datasets = datasets.batch(self.batch_size)
#             line128_data_batch = []
#             hint128_data_batch = []
#             image_data_batch = []
#             line_data_batch = []
#             hint_data_batch = []
#             for i in range(0,self.batch_size):
#                 line128_data_batch.append(line128_data[i:i+self.batch_size])
#                 hint128_data_batch.append(hint128_data[i:i+self.batch_size])
#                 image_data_batch.append(image_data[i:i+self.batch_size])
#                 line_data_batch.append(line_data[i:i+self.batch_size])
#                 hint_data_batch.append(hint_data[i:i+self.batch_size])
                
#             del(line128_data)
#             del(hint128_data)
#             del(image_data)
#             del(line_data)
#             del(hint_data)

#             return line128_data, hint128_data, image_data, line_data, hint_data
            return image_data, line_data, hint_data

#         testDatasets = build_dataSets(image, line, shuffle=False, isTrain=False)

        trainDatasets = build_dataSets(image, line, shuffle=False, isTrain=True)
        return trainDatasets

#         return trainDatasets, testDatasets

class Datasets_512(Datasets):
    def __init__(self ,batch_size):
        self.batch_size = batch_size
        super().__init__(self, batch_size = self.batch_size)
        
    def get_image_and_line(self):
        image = glob(train_image_datasets_path)
        image.sort()
        line = glob(train_line_datasets_path)
        line.sort()
        return image, line
        
    def _next(self, image, line):   
        return self.buildDataSets(image, line)
        
    def _flip(self, image, line, training):
        if training:
            if np.random.rand() < 0.5:
                image = cv2.flip(image, 0)
                line = cv2.flip(line, 0)
#                 line = np.expand_dims(line, 3)

            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                line = cv2.flip(line, 1)
#                 line = np.expand_dims(line, 3)

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

    def loadImage(self, imagePath, linePath, train):
#         print (imagePath)
        image = tf.io.read_file(imagePath)
        image = tf.image.decode_jpeg(image, channels=3)
        line = tf.io.read_file(linePath)
        line = tf.image.decode_jpeg(line, channels=1)
        image = tf.image.resize(image, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        line = tf.image.resize(line, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
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
# os.listdir('/kaggle/input/colorization-to-caffee')
import matplotlib.pyplot as plt
%matplotlib inline
__SAVED_MODEL_PATH__ = '/kaggle/input/colorization-to-caffee/PaintsTensorFlowDraftNet_35.h5'
# saved_refined = '/kaggle/working/PaintsTensorFlow_refined_1.h5'
# os.listdir("/kaggle/input/colorization-to-caffee/PaintsTensorFlow_refined_3.h5")
#PaintsTensorflowTraining
import time
class PaintsTensorFlowTrain:
    def __init__(self, model_name="PaintsTensorFlow"):
        self.data_sets = Datasets_512(batch_size=batch_size)
        self.model_name = "{}".format(model_name)
        initdir(self.model_name)

        self.global_steps = tf.compat.v1.train.get_or_create_global_step()
        self.epochs = tf.Variable(0, trainable=False, dtype=tf.int32)
#         self.ckpt_path = "./ckpt/{}/".format(self.model_name) + "ckpt_E:{}"
#         self.ckpt_prefix = os.path.join(self.ckpt_path, "model_GS:{}")
        self.saved_refined = '/kaggle/input/colorization-to-caffee/PaintsTensorFlow_refined_7.h5'
        self.generator_128 =  tf.keras.models.load_model(__SAVED_MODEL_PATH__)
        self.generator_512 = tf.keras.models.load_model(self.saved_refined)
#         self.generator_512 = Generator(res_net_block=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)

#         self.check_point = tf.train.Checkpoint(generator_512=self.generator_512,
#                                                optimizer=self.optimizer,
#                                                globalSteps=self.global_steps,
#                                                epochs=self.epochs)

#     def __loging(self, name, scalar):
#         with tf.contrib.summary.always_record_summaries():
#             tf.contrib.summary.scalar(name, scalar)

    def __loss(self, output, target):
        loss = tf.reduce_mean(tf.abs(target - output))
        return loss

    def __pred_image(self, model, image, line, hint, draft, epoch=None):
        gs = self.global_steps.numpy()
        predImage = model.predict([line, draft])
#         file_name = "./ckpt/{}/image/{}.jpg".format(self.model_name, gs)

        if epoch is not None:
            loss = self.__loss(predImage, image)
#             self.__loging("Sample_LOSS", loss)
            loss = "{:0.05f}".format(loss).zfill(7)
            print("Epoch:{} GS:{} LOSS:{}".format(epoch, self.global_steps.numpy(), loss))
#             file_name = "./ckpt/{}/image/{}_loss:{}.jpg".format(self.model_name, gs, loss)

        hint = np.array(hint)
        hint[hint > 1] = 1

        lineImage = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([lineImage, hint, draft, predImage, image], 1)
        save_img = utils.convert2uint8(save_img)
        tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)

    def load_caffee(self):
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe("/kaggle/input/reuirements/colorization_deploy_v2.prototxt", "/kaggle/input/reuirements/colorization_release_v2.caffemodel")
        self.pts = np.load("/kaggle/input/reuirements/pts_in_hull.npy")
        
    def pred_caffee(self, line_128):
        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        self.pts = self.pts.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(class8).blobs = [self.pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        
        pred_img = []
        for i in range(0,4):
            print (line_128[i])
            image = cv2.imread(line_128[i])
            plt.imshow(image)
            scaled = image.astype("float32") / 255.0
#             lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            resized = cv2.resize(np.float32(scaled), (224, 224))
            L = cv2.split(resized)[0]
            L -= 50
            self.net.setInput(cv2.dnn.blobFromImage(L))
            ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
            L = cv2.split(np.float32(scaled))[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            colorized = tf.image.resize(colorized, size=(512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            pred_img.append(colorized)
        return pred_img
        
        
        
        
        
    def __draft_image(self, line_128, hint_128):
        draft = self.generator_128.predict([line_128, hint_128])
#         print (draft.shape
        draft = tf.image.resize(draft, size=(512, 512),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return draft
    
    
    
    def fast_fetch(self,size, Traindata):        
        if (size[3]==3):
            x=np.zeros(size)
            for i in range(0,4):
                t = Traindata[i]
                x[i] = np.asarray(t, np.float32)        
            return x
        elif (size[3] ==1):
            x=np.zeros(size)
            for i in range(4):
                t = Traindata[i]
                x[i,:,:,0] = np.asarray(t,np.float32).reshape(size[1],size[2])
            return x
          
    
    

    def training(self, loadEpochs=0):
        
#         #predict image
#         line_flower = tf.io.read_file('/kaggle/input/predimg/out.jpeg')
#         line_flower = tf.image.decode_jpeg(line_flower, channels=1)
#         line_flower = tf.image.resize(line_flower, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# # #         %matplotlib inline
# #         plt.imshow(line_flower[:,:,0])
# #         print (line_flower.shape)
#         hint_pred = line = tf.io.read_file('/kaggle/input/predimg/hint.jpg')
#         hint_pred = tf.image.decode_jpeg(hint_pred, channels=3)
#         hint_pred = tf.image.resize(hint_pred, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#         draft = self.__draft_image(line_128, hint_128)


        images, lines = self.data_sets.get_image_and_line()
#         self.load_caffee()
#         print (len(images))
        for epoch in range(loadEpochs):
#             t = time.time()
#             print ('starting time of{} epoch is {}'.format(epoch, t))
            print ('epoch {}'.format(epoch))
            print("GS: ", self.global_steps.numpy())
            for batchs in range(750):
                
                print (batchs , end = '\r')
#                 print (batchs)
                trainData = self.data_sets._next(images[(batchs*4):(batchs*4)+4], lines[(batchs*4):(batchs*4)+4])
    
#                 prediction = self.pred_caffee(images[(batchs*4):(batchs*4)+4])
#                 draft = tf.convert_to_tensor(fast_fetch((4,512,512,3), prediction), dtype=tf.float32, dtype_hint=None, name=None)
#                 return prediction
                
#                 trainData[0][0] = line_flower
#                 trainData[1][0] = hint_pred
                image = tf.convert_to_tensor(self.fast_fetch((4,512,512,3), trainData[2]), dtype=tf.float32, dtype_hint=None, name=None)        
                line = tf.convert_to_tensor(self.fast_fetch((4,512,512,1), trainData[3]), dtype=tf.float32, dtype_hint=None, name=None)
                hint = tf.convert_to_tensor(self.fast_fetch((4,512,512,3), trainData[4]), dtype=tf.float32, dtype_hint=None, name=None)
                line_128 = tf.convert_to_tensor(self.fast_fetch((4,128,128,1), trainData[0]), dtype=tf.float32, dtype_hint=None, name=None)
                hint_128 = tf.convert_to_tensor(self.fast_fetch((4,128,128,3), trainData[1]), dtype=tf.float32, dtype_hint=None, name=None)
                
                return image,line,hint,line_128,hint_128
                
#                 print (line_128.shape)
#                 print ('fetch')
                draft = self.__draft_image(line_128, hint_128)
#                 return draft, image
#                 print ('pred')
                with tf.GradientTape() as tape:
                    genOut = self.generator_512(inputs=[line, draft], training=True)
                    loss = self.__loss(genOut, image)
#                 print ('loss calc')
                # Training
                gradients = tape.gradient(loss, self.generator_512.variables)
                self.optimizer.apply_gradients(zip(gradients, self.generator_512.variables))
#                 print ('opt')
                
#                 # Loging
                gs = self.global_steps.numpy()
#                 if (batchs%100 == 0):
#                     print (batchs)
            print ("gen loss {}".format(loss))
            print("------------------------------SAVE_E:{}_G:{}-------------------------------------"
                              .format(self.epochs.numpy(), gs))
                        
            self.generator_512.summary()
            print(self.global_steps)
    
    def save_refined(self):
#         os.mkdir('/kaggle/working/model-save')
        save_path = "/kaggle/working/{}_refined_7.h5".format(self.generator_512.name)
        self.generator_512.save(save_path, include_optimizer=False)  # for keras Model
#         save_path = tf.keras.models.save_model(self.generator_512, save_path)  # saved_model

    def make_prediction(self, line128_img, hint128_img, line512_img):
        pred512 = self.__draft_image(line128_img, hint128_img)
        plt.imshow(pred512[0])
        cv2.imwrite('/kaggle/working/demo.jpg',np.float32(pred512[0]))
        final_pred = self.generator_512.predict([line512_img, pred512])
        return final_pred
    
    def convert_data(self, image_path, line_path):
        
        def fast_fetch(size, data):        
            if (size[2]==3):
                x=np.zeros((4,size[0],size[1],size[2]))
                x[0] = np.asarray(data, np.float32)        
                return x
            elif (size[2] ==1):
                x=np.zeros((4,size[0],size[1],size[2]))
                x[0,:,:,0] = np.asarray(data,np.float32).reshape(size[0],size[1])
                return x
          
        
        loadData = line_to_data(image_path, line_path)
        line128, hint128, line512 = loadData.convert_to_128()
        line128 = tf.convert_to_tensor(fast_fetch((128,128,1), line128), dtype=tf.float32, dtype_hint=None, name=None)
        hint128 = tf.convert_to_tensor(fast_fetch((128,128,3), hint128), dtype=tf.float32, dtype_hint=None, name=None)
        line512 = tf.convert_to_tensor(fast_fetch((512,512,1), line512), dtype=tf.float32, dtype_hint=None, name=None)
        prediction = self.make_prediction(line128, hint128, line512)
        return prediction
        
        
        
# 
class line_to_data:
    def __init__(self, image_path, line_path):
        self.line_real_path = line_path
        self.image_path = image_path
        
    def convert2float(self, img):
        img = tf.cast(img, tf.float32)
        img = (img / 127.5) - 1
        return img
        
    def built_hint(self, image):
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
        
    def convert_to_128(self):
        self.image = tf.io.read_file(self.image_path)
        self.image = tf.image.decode_jpeg(self.image, channels=3)
        self.line_real = tf.io.read_file(self.line_real_path)
        self.line_real = tf.image.decode_jpeg(self.line_real, channels=1)
        self.image_128 = tf.image.resize(self.image, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.line_128 = tf.image.resize(self.line_real, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.hint_128 = self.built_hint(self.image_128)
        self.line_512 = tf.image.resize(self.line_real, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.image = self.convert2float(self.image)
        self.line_real = self.convert2float(self.line_real)
        self.line_128 = self.convert2float(self.line_128)
        self.line_512 = self.convert2float(self.line_512)

        return self.line_128, self.hint_128, self.line_512
        

        


        
        
model = PaintsTensorFlowTrain()
image,line,hint,line_128,hint_128 = model.training(loadEpochs=4)
image_path = '../input/realdemo/JPMorgan.jpg'
line_path = '../input/demo12/output.jpg'



prediction = model.convert_data(image_path, line_path)
model.save_refined()

#PaintsTensorflowDraftModel
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
import time
class PaintsTensorFlowDraftModelTrain:
    def __init__(self, model_name="PaintsTensorFlowDraftModel"):
        self.data_sets = Datasets(batch_size=4)
        self.model_name = model_name
        # utils.initdir(self.model_name)

        self.global_steps = tf.compat.v1.train.get_or_create_global_step()
        self.epochs = tf.Variable(0, trainable=False, dtype=tf.int32)

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
#         self.ckptPath = "./ckpt/{}/".format(self.model_name) + "ckpt_E:{}"
#         self.ckptPrefix = os.path.join(self.ckptPath, "model_GS:{}")

#         self.generator = Generator(name="PaintsTensorFlowDraftNet")
        self.generator = tf.keras.models.load_model(__SAVED_MODEL_PATH__)
        
        self.discriminator = Discriminator()

#         self.logWriter = tf.contrib.summary.create_file_writer("./ckpt/{}/board/log".format(self.model_name))
# #         self.logWriter.set_as_default()

#         self.check_point = tf.train.Checkpoint(generator=self.generator,
#                                                genOptimizer=self.generator_optimizer,
#                                                disOptimizer=self.discriminator_optimizer,
#                                                discriminator=self.discriminator,
#                                                globalSteps=self.global_steps,
#                                                epochs=self.epochs)

    def __discriminator_loss(self, real, fake):
        SCE = tf.nn.sigmoid_cross_entropy_with_logits
        self.real_loss = SCE(tf.ones_like(real), logits=real)
        self.fake_loss = SCE(tf.zeros_like(fake), logits=fake)
        loss = self.real_loss + self.fake_loss
        return loss

    def __generator_loss(self, disOutput, output, target):
        SCE = tf.nn.sigmoid_cross_entropy_with_logits
        self.gan_loss = SCE(tf.ones_like(disOutput), logits=disOutput)
        self.image_loss = tf.reduce_mean(tf.abs(target - output)) * l1_scaling
        loss = self.image_loss + self.gan_loss
        return loss

    def __pred_image(self, model, image, line, hint, epoch=None):
        global_steps = self.global_steps.numpy()
        pred_image = model.predict([line, hint])

        zero_hint = tf.ones_like(hint)
        zero_hint += 1
        pred_image_zero = model.predict([line, zero_hint])

        dis_fake = self.discriminator(pred_image, training=False)
        loss = self.__generator_loss(dis_fake, pred_image, image)

#         self.__loging("Sample_LOSS", loss)
        loss = "{:0.05f}".format(loss).zfill(7)
        print("Epoch:{} GS:{} LOSS:{}".format(epoch, global_steps, loss))
#         file_name = "./ckpt/{}/image/{}_loss:{}.jpg".format(self.model_name, global_steps, loss)

        hint = np.array(hint)
        hint[hint > 1] = 1

        line_image = np.concatenate([line, line, line], -1)
        save_img = np.concatenate([line_image, hint, pred_image_zero, pred_image, image], 1)
        save_img = utils.convert2uint8(save_img)
        #use plt.save for viasualization
        # tl.visualize.save_images(save_img, [1, save_img.shape[0]], file_name)

#     def __loging(self, name, scalar):
#         with tf.contrib.summary.always_record_summaries():
#             tf.contrib.summary.scalar(name, scalar)

#     def __check_point_save(self):
#         file_prefix = self.ckptPrefix.format(self.epochs.numpy(), self.global_steps.numpy())
#         self.check_point.save(file_prefix=file_prefix)

    def training(self, loadEpochs=0):
        images, lines = self.data_sets.get_image_and_line()
#         return (images, lines)
        t = time.time()
        batch_steps = 750
        def fast_fetch(size, Traindata):        
            if (size==(4,128,128,3)):
                x=np.zeros(size)
                for i in range(0,4):
                    t = Traindata[i]
                    x[i] = np.asarray(t, np.float32)        
                return x
            else:
                x=np.zeros(size)
                for i in range(4):
                    t = Traindata[i]
                    x[i,:,:,0] = np.asarray(t,np.float32).reshape(128,128)
#                     plt.imshow(x[i,:,:,0])
                return x
            
        for epoch in range(loadEpochs):

            print("GS: ", self.global_steps.numpy(), "Epochs:  ", self.epochs.numpy())
            for batch in range(batch_steps):
                print (batch, end='\r')
#                 t1 = time.time()
#                 print("Time of for loop {}".format(t1-t))
                trainData = self.data_sets._next(images[(batch*4):(batch*4)+4], lines[(batch*4):(batch*4)+4])
#                 return trainData
#                 trainData[0][0] = line_flower
                image = tf.convert_to_tensor(fast_fetch((4,128,128,3), trainData[0]), dtype=tf.float32, dtype_hint=None, name=None)        
                line = tf.convert_to_tensor(fast_fetch((4,128,128,1), trainData[1]), dtype=tf.float32, dtype_hint=None, name=None)
                hint = tf.convert_to_tensor(fast_fetch((4,128,128,3), trainData[2]), dtype=tf.float32, dtype_hint=None, name=None)
                del (trainData)
#                 return image, line, hint
                
                
                with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
                    pred_image = self.generator(inputs=[line, hint], training=True)
                    dis_real = self.discriminator(inputs=image, training=True)
                    dis_fake = self.discriminator(inputs=pred_image, training=True)
                    generator_loss = self.__generator_loss(dis_fake, pred_image, image)
                    discriminator_loss = self.__discriminator_loss(dis_real, dis_fake)
                discriminator_gradients = discTape.gradient(discriminator_loss, self.discriminator.variables)
                generator_gradients = genTape.gradient(generator_loss, self.generator.variables)
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.variables))
                self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.variables))
                gs = self.global_steps.numpy()
                del(image)
                del(line)
                del(hint)

                
#                     log("LOSS_G", generator_loss)
#                     log("LOSS_G_Image", self.image_loss)
#                     log("LOSS_G_GAN", self.gan_loss)
#                     log("LOSS_D", discriminator_loss)
#                     log("LOSS_D_Real", self.real_loss)
#                     log("LOSS_D_Fake", self.fake_loss)
                    
#                     if gs % sampling_interval == 0:
#                         for image, line, hint in test_sets[:]:
#                             self.__pred_image(self.generator, image, line, hint, self.epochs.numpy())

#                 if gs % save_interval == 0:
#                     print("------------------------------SAVE_E:{}_G:{}-------------------------------------".format(self.epochs.numpy(), gs))
            self.epochs = self.epochs + 1
            print ('LOSS_G {}\nLOSS_G_Image {}\nLOSS_G_GAN {} \nLOSS_D {}\nLOSS_D_Real {}\nLOSS_D_Fake {}'.format(generator_loss,self.image_loss,self.gan_loss,discriminator_loss,self.real_loss,self.fake_loss))
        self.generator.summary()
    def save_model(self):
        save_path = "/kaggle/working/{}_40.h5".format(self.generator.name)
        self.generator.save(save_path, include_optimizer=False)  # for keras Model
#         save_path = tf.contrib.saved_model.save_keras_model(self.generator, "./saved_model")
        

                

#         log = self.__loging

#         self.check_point.restore(tf.train.latest_checkpoint(self.ckptPath.format(loadEpochs)))

#         for epoch in range(10):
#             print("GS: ", self.global_steps.numpy(), "Epochs:  ", self.epochs.numpy())

#             for image, line, hint in tqdm(train_sets, total=batch_steps):
#                 # get loss
#                 with tf.GradientTape() as genTape, tf.GradientTape() as discTape:

#                     pred_image = self.generator(inputs=[line, hint], training=True)

#                     dis_real = self.discriminator(inputs=image, training=True)
#                     dis_fake = self.discriminator(inputs=pred_image, training=True)

#                     generator_loss = self.__generator_loss(dis_fake, pred_image, image)
#                     discriminator_loss = self.__discriminator_loss(dis_real, dis_fake)

#                 # Gradients
#                 discriminator_gradients = discTape.gradient(discriminator_loss, self.discriminator.variables)
#                 generator_gradients = genTape.gradient(generator_loss, self.generator.variables)

#                 self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.variables))
#                 self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.variables),
#                                                          global_step=self.global_steps)
#                 gs = self.global_steps.numpy()

#                 if gs % log_interval == 0:
#                     log("LOSS_G", generator_loss)
#                     log("LOSS_G_Image", self.image_loss)
#                     log("LOSS_G_GAN", self.gan_loss)
#                     log("LOSS_D", discriminator_loss)
#                     log("LOSS_D_Real", self.real_loss)
#                     log("LOSS_D_Fake", self.fake_loss)

#                     if gs % sampling_interval == 0:
#                         for image, line, hint in test_sets.take(1):
#                             self.__pred_image(self.generator, image, line, hint, self.epochs.numpy())

#                     if gs % save_interval == 0:
#                         self.__check_point_save()
#                         print("------------------------------SAVE_E:{}_G:{}-------------------------------------"
#                               .format(self.epochs.numpy(), gs))
#             self.epochs = self.epochs + 1

#         self.generator.summary()
#         save_path = "'/kaggle/working/model-save/'" + self.model_name + "/{}.h5".format(self.generator.name)
#         self.generator.save(save_path, include_optimizer=False)  # for keras Model
#         save_path = tf.contrib.saved_model.save_keras_model(self.generator, "./saved_model")  # saved_model
#         print("saved_model path = {}".format(save_path))

#         print("------------------------------Training Done-------------------------------------")
model = PaintsTensorFlowDraftModelTrain()
model.training(loadEpochs=5)

model.save_model()
