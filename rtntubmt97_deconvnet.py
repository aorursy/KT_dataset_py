import datetime

print(datetime.datetime.now())

import tensorflow as tf

import SimpleITK as sitk



from tensorflow.keras import Model, Input

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation

from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model, save_model

from tensorflow.keras.applications import VGG16

from tensorflow.keras.optimizers import SGD, Adam



import os

import random

import time

# import wget

import tarfile

import numpy as np

import cv2



from os import listdir, mkdir, makedirs

from tensorflow.keras import losses

from tensorflow.keras.callbacks import Callback

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from random import randint



from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer



data_path_format = "../input/data24/scan/scan/liver-orig{:0>3}.mhd"

label_path_format = "../input/data24/label/label/liver-seg{:0>3}.mhd"

model_path = "./models/deconv_sigmoid/"

# makedirs(model_path, exist_ok=True)

            

base_num_filter = 64



def my_gen(batch_size = 32, start_idx=1, end_idx=19):

#     means = dict()

#     stds = dict()

#     for i in range(start_idx, end_idx + 1, 1):

#         images = sitk.ReadImage(data_path_format.format(i))

#         images = sitk.GetArrayFromImage(images)

#         means[i] = np.mean(images)

#         stds[i] = np.std(images)

    data_generator = ImageDataGenerator(zoom_range=0.5, rotation_range=5, fill_mode='constant')

    while True:

        slice_num = randint(start_idx,end_idx)

        images = sitk.ReadImage(data_path_format.format(slice_num))

        images = sitk.GetArrayFromImage(images)

        labels = sitk.ReadImage(label_path_format.format(slice_num))

        labels = sitk.GetArrayFromImage(labels)

        depth = labels.shape[0]

        chosen_images_labels = []

        for i in range(batch_size):

#             x = randint(0, 256)

#             y = randint(0, 256)

            z = randint(0, depth - 1)

#             image = images[z, x:x+256, y:y+256][...,None]

#             label = labels[z, x:x+256, y:y+256][...,None]

            image = images[z][...,None]

            label = labels[z][...,None]

            image_label = np.concatenate([image, label], axis=-1)

            tran_para = data_generator.get_random_transform(image_label.shape)

            image_label = data_generator.apply_transform(image_label, tran_para)

            chosen_images_labels.append(image_label)

        

        chosen_images_labels = np.array(chosen_images_labels)



        yield chosen_images_labels[...,0, None], chosen_images_labels[...,1, None]





def weigth_ce(true,pred): 

#     true = true[..., 1]

#   pred = pred + 0.0000001

  return - tf.reduce_sum(true * tf.log(pred + 0.0000001) + (1 - true)*tf.log(1 - pred + 0.0000001), -1)

        

        

def iou_score(true,pred):  #this can be used as a loss if you make it negative

#   true = true[..., 1]

#   pred = pred[..., 1]

    pred = tf.round(pred)

    intersection = true * pred

    notTrue = 1 - true

    union = true + (notTrue * pred)



    return (tf.reduce_sum(intersection) + tf.constant(0.0000001)) / (tf.reduce_sum(union) + tf.constant(0.0000001))



def recall(true, pred):

    pred = tf.round(pred)

    return (tf.reduce_sum(true*pred) + tf.constant(0.0000001))/(tf.reduce_sum(true) + tf.constant(0.0000001))



def precision(true, pred):

    pred = tf.round(pred)

    return (tf.reduce_sum(true*pred) + tf.constant(0.0000001))/(tf.reduce_sum(pred) + tf.constant(0.0000001))



class Valid_own(Callback):

    def __init__(self, img, label, focus=False, freq_save=2):

        super(Callback, self).__init__()

        self.img = img

        self.label = label

        self.focus = focus

        self.freq_save = freq_save

        self.freq_count = freq_save

        try:

            self.best_iou_score = np.load("best_valid_iou.npy")

        except:

            self.best_iou_score = np.float32(0.0)





    def on_epoch_end(self, batch, logs={}):

        self.freq_count -= 1

        if self.freq_count == 0:

            self.model.save_weights('model.h5')

            self.freq_count = self.freq_save

        eva_ret = self.model.evaluate(self.img, self.label, batch_size=16)

        eva_ret = dict(zip(self.model.metrics_names, eva_ret))

        if self.best_iou_score < eva_ret["iou_score"]:

            self.best_iou_score = eva_ret["iou_score"]

            np.save("best_valid_iou.npy", self.best_iou_score)

            self.model.save_weights('best_model.h5')

        print("train:       ", logs)

        print("valid:       ", eva_ret)





class MaxUnpoolWithArgmax(Layer):



    def __init__(self, pooling_argmax, stride = [1, 2, 2, 1], **kwargs):

        self.pooling_argmax = pooling_argmax    

        self.stride = stride

        super(MaxUnpoolWithArgmax, self).__init__(**kwargs)

        

    def build(self, input_shape):

        super(MaxUnpoolWithArgmax, self).build(input_shape)



    def call(self, inputs):

        input_shape = K.cast(K.shape(inputs), dtype='int64')



        output_shape = (input_shape[0],

                        input_shape[1] * self.stride[1],

                        input_shape[2] * self.stride[2],

                        input_shape[3])



        #output_list = []

        #output_list.append(self.pooling_argmax // (output_shape[2] * output_shape[3]))

        #output_list.append(self.pooling_argmax % (output_shape[2] * output_shape[3]) // output_shape[3])

        argmax = self.pooling_argmax #K.stack(output_list)



        one_like_mask = K.ones_like(argmax)

        batch_range = K.reshape(K.arange(start=0, stop=input_shape[0], dtype='int64'), 

                                 shape=[input_shape[0], 1, 1, 1])



        b = one_like_mask * batch_range

        y = argmax // (output_shape[2] * output_shape[3])

        x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]

        feature_range = K.arange(start=0, stop=output_shape[3], dtype='int64')

        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension

        updates_size = tf.size(inputs)

        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))

        values = K.reshape(inputs, [updates_size])

        return tf.scatter_nd(indices, values, output_shape)



    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])



    def get_config(self):

        base_config = super(MaxUnpoolWithArgmax, self).get_config()

        base_config['pooling_argmax'] = self.pooling_argmax

        base_config['stride'] = self.stride

        return base_config



    @classmethod

    def from_config(cls, config):

        return cls(**config)





class DeconvNet:

    def __init__(self, use_cpu=False, print_summary=False):

#         self.maybe_download_and_extract()

        self.build(use_cpu=use_cpu, print_summary=print_summary)



        

    def maybe_download_and_extract(self):

        """Download and unpack VOC data if data folder only contains the .gitignore file"""

        if os.listdir('data') == ['.gitignore']:

            filenames = ['VOC_OBJECT.tar.gz', 'VOC2012_SEG_AUG.tar.gz', 'stage_1_train_imgset.tar.gz', 'stage_2_train_imgset.tar.gz']

            url = 'http://cvlab.postech.ac.kr/research/deconvnet/data/'



            for filename in filenames:

                wget.download(url + filename, out=os.path.join('data', filename))



                tar = tarfile.open(os.path.join('data', filename))

                tar.extractall(path='data')

                tar.close()



                os.remove(os.path.join('data', filename))



                

    def predict(self, image):

        return self.model.predict(np.array([image]))

    

    def save(self, file_path = model_path + 'model.h5'):

        print(self.model.to_json())

        self.model.save_weights(file_path)

        

    def load(self, file_path = model_path + 'model.h5'):

        self.model.load_weights(file_path)

    

    def random_crop_or_pad(self, image, truth, size=(224, 224)):

        assert image.shape[:2] == truth.shape[:2]



        if image.shape[0] > size[0]:

            crop_random_y = random.randint(0, image.shape[0] - size[0])

            image = image[crop_random_y:crop_random_y + size[0],:,:]

            truth = truth[crop_random_y:crop_random_y + size[0],:]

        else:

            zeros = np.zeros((size[0], image.shape[1], image.shape[2]), dtype=np.float32)

            zeros[:image.shape[0], :image.shape[1], :] = image                                          

            image = np.copy(zeros)

            zeros = np.zeros((size[0], truth.shape[1]), dtype=np.float32)

            zeros[:truth.shape[0], :truth.shape[1]] = truth

            truth = np.copy(zeros)



        if image.shape[1] > size[1]:

            crop_random_x = random.randint(0, image.shape[1] - size[1])

            image = image[:,crop_random_x:crop_random_x + 224,:]

            truth = truth[:,crop_random_x:crop_random_x + 224]

        else:

            zeros = np.zeros((image.shape[0], size[1], image.shape[2]))

            zeros[:image.shape[0], :image.shape[1], :] = image

            image = np.copy(zeros)

            zeros = np.zeros((truth.shape[0], size[1]))

            zeros[:truth.shape[0], :truth.shape[1]] = truth

            truth = np.copy(zeros)            



        return image, truth



    #(0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 

    # 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 

    # 17=sheep, 18=sofa, 19=train, 20=tv/monitor, 255=no_label)



    def max_pool_with_argmax(self, x):

        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    

    def BatchGenerator(self, train_stage=1, batch_size=8, image_size=(256, 256, 3), labels=21):

        if train_stage == 1:

            trainset = open('data/stage_1_train_imgset/train.txt').readlines()

        else:

            trainset = open('data/stage_2_train_imgset/train.txt').readlines()



        while True:

            images = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))

            truths = np.zeros((batch_size, image_size[0], image_size[1], labels))



            for i in range(batch_size):

                random_line = random.choice(trainset)

                image_file = random_line.split(' ')[0]

                truth_file = random_line.split(' ')[1]

                image = np.float32(cv2.imread('data' + image_file)/255.0)



                truth_mask = cv2.imread('data' + truth_file[:-1], cv2.IMREAD_GRAYSCALE)

                truth_mask[truth_mask == 255] = 0 # replace no_label with background  

                images[i], truth = self.random_crop_or_pad(image, truth_mask, image_size)

                truths[i] = (np.arange(labels) == truth[...,None]-1).astype(int) # encode to one-hot-vector

            yield images, truths

            



    def get_model(self):

        return self.model

    

    def train(self, steps_per_epoch=1000, epochs=10, batch_size=32):

        batch_generator = self.BatchGenerator(batch_size=batch_size)

        self.model.fit_generator(batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)



    def buildConv2DBlock(self, block_input, filters, block, depth):

        for i in range(1, depth + 1):

            if i == 1:

                conv2d = Conv2D(filters, 3, padding='same', name='conv{}-{}'.format(block, i), use_bias=False)(block_input)

            else:

                conv2d = Conv2D(filters, 3, padding='same', name='conv{}-{}'.format(block, i), use_bias=False)(conv2d)

            

            conv2d = BatchNormalization(name='batchnorm{}-{}'.format(block, i))(conv2d)

            conv2d = Activation('relu', name='relu{}-{}'.format(block, i))(conv2d)

            

        return conv2d

 



    def build(self, use_cpu=False, print_summary=False):

#         vgg16 = VGG16(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))

        

        if use_cpu:

            device = '/cpu:0'

        else:

            device = '/gpu:0'



        with tf.device(device):

            inputs = Input(shape=(512, 512, 1))



            conv_block_0 = self.buildConv2DBlock(inputs, base_num_filter, 0, 2)

            pool0, pool0_argmax = Lambda(self.max_pool_with_argmax, name='pool0')(conv_block_0) 



            conv_block_1 = self.buildConv2DBlock(pool0, base_num_filter, 1, 2)

            pool1, pool1_argmax = Lambda(self.max_pool_with_argmax, name='pool1')(conv_block_1) 



            conv_block_2 = self.buildConv2DBlock(pool1, base_num_filter*2, 2, 2)

            pool2, pool2_argmax = Lambda(self.max_pool_with_argmax, name='pool2')(conv_block_2) 



            conv_block_3 = self.buildConv2DBlock(pool2, base_num_filter*4, 3, 3)

            pool3, pool3_argmax = Lambda(self.max_pool_with_argmax, name='pool3')(conv_block_3) 



            conv_block_4 = self.buildConv2DBlock(pool3, base_num_filter*8, 4, 3)

            pool4, pool4_argmax = Lambda(self.max_pool_with_argmax, name='pool4')(conv_block_4) 



            conv_block_5 = self.buildConv2DBlock(pool4, base_num_filter*8, 5, 3)

            pool5, pool5_argmax = Lambda(self.max_pool_with_argmax, name='pool5')(conv_block_5)



            fc6 = Conv2D(base_num_filter*8, 7, use_bias=False, padding='valid', name='fc6')(pool5) #4096

            fc6 = BatchNormalization(name='batchnorm_fc6')(fc6)

            fc6 = Activation('relu', name='relu_fc6')(fc6)

            

            fc7 = Conv2D(base_num_filter*8, 1, use_bias=False, padding='valid', name='fc7')(fc6)   #4096

            fc7 = BatchNormalization(name='batchnorm_fc7')(fc7)

            fc7 = Activation('relu', name='relu_fc7')(fc7)

            

            x = Conv2DTranspose(base_num_filter*8, 7, use_bias=False, padding='valid', name='deconv-fc6')(fc7)

            x = BatchNormalization(name='batchnorm_deconv-fc6')(x)

            x = Activation('relu', name='relu_deconv-fc6')(x)            

            x = MaxUnpoolWithArgmax(pool5_argmax, name='unpool5')(x)

            x.set_shape(conv_block_5.get_shape())



            x = Conv2DTranspose(base_num_filter*8, 3, use_bias=False, padding='same', name='deconv5-1')(x)

            x = BatchNormalization(name='batchnorm_deconv5-1')(x)

            x = Activation('relu', name='relu_deconv5-1')(x)  

            

            x = Conv2DTranspose(base_num_filter*8, 3, use_bias=False, padding='same', name='deconv5-2')(x)

            x = BatchNormalization(name='batchnorm_deconv5-2')(x)

            x = Activation('relu', name='relu_deconv5-2')(x)  

            

            x = Conv2DTranspose(base_num_filter*8, 3, use_bias=False, padding='same', name='deconv5-3')(x)

            x = BatchNormalization(name='batchnorm_deconv5-3')(x)

            x = Activation('relu', name='relu_deconv5-3')(x)  

            

            x = MaxUnpoolWithArgmax(pool4_argmax, name='unpool4')(x)

            x.set_shape(conv_block_4.get_shape())



            x = Conv2DTranspose(base_num_filter*8, 3, use_bias=False, padding='same', name='deconv4-1')(x)

            x = BatchNormalization(name='batchnorm_deconv4-1')(x)

            x = Activation('relu', name='relu_deconv4-1')(x)  

            

            x = Conv2DTranspose(base_num_filter*8, 3, use_bias=False, padding='same', name='deconv4-2')(x)

            x = BatchNormalization(name='batchnorm_deconv4-2')(x)

            x = Activation('relu', name='relu_deconv4-2')(x) 

            

            x = Conv2DTranspose(base_num_filter*4, 3, use_bias=False, padding='same', name='deconv4-3')(x)

            x = BatchNormalization(name='batchnorm_deconv4-3')(x)

            x = Activation('relu', name='relu_deconv4-3')(x) 

            

            x = MaxUnpoolWithArgmax(pool3_argmax, name='unpool3')(x)

            x.set_shape(conv_block_3.get_shape())



            x = Conv2DTranspose(base_num_filter*4, 3, use_bias=False, padding='same', name='deconv3-1')(x)

            x = BatchNormalization(name='batchnorm_deconv3-1')(x)

            x = Activation('relu', name='relu_deconv3-1')(x)  

            

            x = Conv2DTranspose(base_num_filter*4, 3, use_bias=False, padding='same', name='deconv3-2')(x)

            x = BatchNormalization(name='batchnorm_deconv3-2')(x)

            x = Activation('relu', name='relu_deconv3-2')(x)  

            

            x = Conv2DTranspose(base_num_filter*2, 3, use_bias=False, padding='same', name='deconv3-3')(x)

            x = BatchNormalization(name='batchnorm_deconv3-3')(x)

            x = Activation('relu', name='relu_deconv3-3')(x)  

            

            x = MaxUnpoolWithArgmax(pool2_argmax, name='unpool2')(x)

            x.set_shape(conv_block_2.get_shape())



            x = Conv2DTranspose(base_num_filter*2, 3, use_bias=False, padding='same', name='deconv2-1')(x)

            x = BatchNormalization(name='batchnorm_deconv2-1')(x)

            x = Activation('relu', name='relu_deconv2-1')(x)  

            

            x = Conv2DTranspose(base_num_filter, 3, use_bias=False, padding='same', name='deconv2-2')(x)

            x = BatchNormalization(name='batchnorm_deconv2-2')(x)

            x = Activation('relu', name='relu_deconv2-2')(x)  

            

            x = MaxUnpoolWithArgmax(pool1_argmax, name='unpool1')(x)

            x.set_shape(conv_block_1.get_shape())



            x = Conv2DTranspose(base_num_filter, 3, use_bias=False, padding='same', name='deconv1-1')(x)

            x = BatchNormalization(name='batchnorm_deconv1-1')(x)

            x = Activation('relu', name='relu_deconv1-1')(x)  

            

            x = Conv2DTranspose(base_num_filter, 3, use_bias=False, padding='same', name='deconv1-2')(x)

            x = BatchNormalization(name='batchnorm_deconv1-2')(x)

            x = Activation('relu', name='relu_deconv1-2')(x)     

            

            x = MaxUnpoolWithArgmax(pool0_argmax, name='unpool0')(x)

            x.set_shape(conv_block_0.get_shape())



            x = Conv2DTranspose(base_num_filter, 3, use_bias=False, padding='same', name='deconv0-1')(x)

            x = BatchNormalization(name='batchnorm_deconv0-1')(x)

            x = Activation('relu', name='relu_deconv0-1')(x)    



            x = Conv2DTranspose(base_num_filter, 3, use_bias=False, padding='same', name='deconv0-2')(x)

            x = BatchNormalization(name='batchnorm_deconv0-2')(x)

            x = Activation('relu', name='relu_deconv0-2')(x)            

            

            output = Conv2DTranspose(1, 1, activation='sigmoid', padding='same', name='output')(x)



            self.model = Model(inputs=inputs, outputs=output)

#             vgg16 = VGG16(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))

            

            if print_summary:

                print(self.model.summary())

            

#             for layer in self.model.layers:

#                 if layer.name.startswith('conv'):

#                     block = layer.name[4:].split('-')[0]

#                     depth = layer.name[4:].split('-')[1]

#                     # apply vgg16 weights without bias

#                     layer.set_weights([vgg16.get_layer('block{}_conv{}'.format(block, depth)).get_weights()[0]])

            optimizer = Adam(lr=0.001)

            self.model.compile(optimizer=optimizer,

                          loss=weigth_ce,

                          metrics=['accuracy', iou_score, recall, precision])

 



model = DeconvNet(print_summary=False).get_model()
# model.load_weights("../input/512-model/image_512_32f_bb.15-0.04-0.04-0.93-0.97-0.97.h5")
import datetime

print(datetime.datetime.now())

import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint



gen_train = my_gen(batch_size=8, start_idx=2, end_idx=24)

gen_valid = my_gen(batch_size=8, start_idx=1, end_idx=1)

mcp = ModelCheckpoint(filepath="image_512_64f_6b.{epoch:02d}-{loss:.2f}-{val_loss:.2f}-{val_iou_score:.2f}-{val_recall:.2f}-{val_precision:.2f}.h5", monitor='val_loss',save_weights_only=True)

model.fit_generator(gen_train, steps_per_epoch=2000, epochs=40, callbacks=[mcp], validation_data=gen_valid, validation_steps=200)
# model.load_weights('../input/deconv-model/best_model.h5')
import numpy as np



import random

def get_smooth_predict(model, image, loop, threshold=0.5):

    poss = [[0, 0], [0, 256], [256, 0], [256, 256]]

    ret = np.zeros_like(image, dtype=np.float32)

    for i in range(loop - 4):

        x = random.randint(50, 236)

        y = random.randint(0, 200)

        poss.append([x, y])

    poss = np.array(poss)

    image_part = np.array([image[pos[0]: pos[0] + 256, pos[1]: pos[1] + 256] for pos in poss])

    image_part = model.predict(image_part) 

    start_count = 4

    for i in range(len(poss)):

        if i < 4:

            ret[poss[i, 0]: poss[i, 0] + 256, poss[i, 1]: poss[i, 1] + 256] = image_part[i]

        else:

            f = 0.65

            ret[poss[i, 0]: poss[i, 0] + 256, poss[i, 1]: poss[i, 1] + 256] = (image_part[i]*f + ret[poss[i, 0]: poss[i, 0] + 256, poss[i, 1]: poss[i, 1] + 256]*(1-f))

            start_count = start_count + 1

    

    return np.where(ret > threshold, 1, 0)

    



    

def predict_a_sample(model, sample, loop, threshold=0.5):

    return np.array([get_smooth_predict(model, image, loop, threshold=threshold) for image in sample])

import numpy as np

import SimpleITK as sitk

from scipy import ndimage, misc



def predict_from_file_to_file(model, src_path, des_path, threshold=0.5,loop=64):

    image_src = sitk.ReadImage(src_path)

    spacing = image_src.GetSpacing()

    origin = image_src.GetOrigin()

    direction = image_src.GetDirection()

    image_arr = sitk.GetArrayFromImage(image_src)[..., None]

#     pre_arr = predict_a_sample(model, image_arr, loop, threshold=threshold)

    pre_arr = predict_a_sample_v2(model, image_arr)

    pre_arr = ndimage.median_filter(np.where(pre_arr > 0.5, 1, 0)[..., 0], size=[3, 3, 3], mode='constant')

    pre_arr = np.int8(pre_arr)

    image_pre = sitk.GetImageFromArray(pre_arr)

    image_pre.SetSpacing(spacing)

#     image_pre.SetOrigin(origin)

    image_pre.SetDirection(direction)

    sitk.WriteImage(image_pre, des_path, False)

    return pre_arr



def predict_a_sample_v2(model, image_arr):

    return model.predict(image_arr)



data_path_format = "../input/test-scans/test-scans/scan/liver-orig{:0>3}.mhd"

out_path_format = "liver-out{:0>3}.mhd"

for i in range(1, 11, 1):

    sample_pre = predict_from_file_to_file(model, data_path_format.format(i), out_path_format.format(i), threshold=0.35)

# sample_pre = predict_from_file_to_file(model, 

#                                        "../input/test-scans/test-scans/scan/liver-orig001.mhd", 

#                                        "liver-out001.mhd", 

#                                        threshold=0.35,loop=64)
# import datetime

# print(datetime.datetime.now())



# import cv2

# import numpy as np

# from os import listdir, mkdir, makedirs

# from os.path import isfile, join, isdir



# valid_data_file = "../input/sliver07-train/training-scans/scan/liver-orig020.mhd"

# valid_label_file = "../input/sliver07-train/training-labels/label/liver-seg020.mhd"

# images = sitk.ReadImage(valid_data_file)

# images = sitk.GetArrayFromImage(images)[..., None]

# labels = sitk.ReadImage(valid_label_file)

# labels = sitk.GetArrayFromImage(labels)[..., None]

    

# v_images = []

# v_labels = []    



# for i in range(images.shape[0]):

#     v_images.append(images[i, 0:256, 0:256])

#     v_images.append(images[i, 0:256, 256:512])

#     v_images.append(images[i, 256:512, 0:256])

#     v_images.append(images[i, 256:512, 256:512])

#     v_labels.append(labels[i, 0:256, 0:256])

#     v_labels.append(labels[i, 0:256, 256:512])

#     v_labels.append(labels[i, 256:512, 0:256])

#     v_labels.append(labels[i, 256:512, 256:512])

  



# v_labels = np.array(v_labels)

# v_images = np.float32(v_images)







# valid_data_file = "../input/sliver07-train/training-scans/scan/liver-orig019.mhd"

# valid_label_file = "../input/sliver07-train/training-labels/label/liver-seg019.mhd"

# images = sitk.ReadImage(valid_data_file)

# images = sitk.GetArrayFromImage(images)[..., None]

# labels = sitk.ReadImage(valid_label_file)

# labels = sitk.GetArrayFromImage(labels)[..., None]

    

# v_images2 = []

# v_labels2 = []    



# for i in range(images.shape[0]):

#     v_images2.append(images[i, 0:256, 0:256])

#     v_images2.append(images[i, 0:256, 256:512])

#     v_images2.append(images[i, 256:512, 0:256])

#     v_images2.append(images[i, 256:512, 256:512])

#     v_labels2.append(labels[i, 0:256, 0:256])

#     v_labels2.append(labels[i, 0:256, 256:512])

#     v_labels2.append(labels[i, 256:512, 0:256])

#     v_labels2.append(labels[i, 256:512, 256:512])

  



# v_labels2 = np.array(v_labels2)

# v_images2 = np.float32(v_images2)



# next(my_gen(images, labels))

# val_cb = Valid_own(v_images, v_labels)

# model.load_weights(model_path + 'best_model.h5')

# model.fit_generator(my_gen(batch_size=16), steps_per_epoch=750, epochs=80, callbacks=[val_cb])
# model.evaluate(v_images2, v_labels2)

# v_pred2 = model.predict(v_images2)
# valid_data_file = "../input/sliver07-train/training-scans/scan/liver-orig019.mhd"

# valid_label_file = "../input/sliver07-train/training-labels/label/liver-seg019.mhd"

# images = sitk.ReadImage(valid_data_file)

# images = sitk.GetArrayFromImage(images)[..., None]

# labels = sitk.ReadImage(valid_label_file)

# labels = sitk.GetArrayFromImage(labels)[..., None]



# valid_data_file = "../input/sliver07-train/training-scans/scan/liver-orig020.mhd"

# valid_label_file = "../input/sliver07-train/training-labels/label/liver-seg020.mhd"

# images2 = sitk.ReadImage(valid_data_file)

# images2 = sitk.GetArrayFromImage(images2)[..., None]

# labels2 = sitk.ReadImage(valid_label_file)

# labels2 = sitk.GetArrayFromImage(labels2)[..., None]
# pred = predict_a_sample(model, images, 64)

# pred2 = predict_a_sample(model, images2, 64)
# from sklearn.metrics import accuracy_score

# from sklearn.metrics import jaccard_similarity_score

# from sklearn.metrics import recall_score

# from sklearn.metrics import precision_score

# acc = accuracy_score(labels.flatten(), np.where(pred.flatten() > 0.35, 1, 0))

# acc2 = accuracy_score(labels2.flatten(), np.where(pred2.flatten() > 0.35, 1, 0))

# iou = jaccard_similarity_score(labels.flatten(), np.where(pred.flatten() > 0.35, 1, 0))

# iou2 = jaccard_similarity_score(labels2.flatten(), np.where(pred2.flatten() > 0.35, 1, 0))

# prec = precision_score(labels.flatten(), np.where(pred.flatten() > 0.35, 1, 0))

# prec2 = precision_score(labels2.flatten(), np.where(pred2.flatten() > 0.35, 1, 0))

# rec = recall_score(labels.flatten(), np.where(pred.flatten() > 0.35, 1, 0))

# rec2 = recall_score(labels2.flatten(), np.where(pred2.flatten() > 0.35, 1, 0))

# print(acc)

# print(acc2)

# print(iou)

# print(iou2)

# print(prec)

# print(prec2)

# print(rec)

# print(rec2)
# model.evaluate(v_images, v_labels)

# model.evaluate(v_images2, v_labels2)

# v_predict = model.predict(v_images)

# v_predict2 = model.predict(v_images2)
# def quarter_to_one(ori_data):

#     depth = ori_data.shape[0]//4

#     height = ori_data.shape[1]*2

#     width = ori_data.shape[2]*2

#     channel = ori_data.shape[3]

#     ret_data = np.zeros([depth, height, width, channel],dtype=ori_data.dtype)

#     for i in range(depth):

#         ret_data[i, 0: 256, 0:256, :] = ori_data[4*i]

#         ret_data[i, 0: 256, 256:512, :] = ori_data[4*i + 1]

#         ret_data[i, 256: 512, 0:256, :] = ori_data[4*i + 2]

#         ret_data[i, 256: 512, 256:512, :] = ori_data[4*i + 3]

#     return ret_data

               

# image_complete = quarter_to_one(v_images)        

# label_complete = quarter_to_one(v_labels)
# take = range(145, 175, 1)

# chosen_images = images2[take]

# chosen_labels = labels2[take]

# chosen_predicts = pred2[take]

# pred2 = predict_a_sample(model, images2[160:170], 128)

# chosen_predicts = pred2



import numpy as np

import matplotlib.pyplot as plt 

import cv2

from scipy import ndimage, misc



def display_sample(sample, label=None, predict=None):

    shape = sample.shape

    if label is not None:

        if np.any(label.shape[:3] != sample.shape[:3]):

            return

    depth = shape[0]

    height = shape[1]

    width = shape[2]

#   channel_num = shape[3]

  

    if label is None:

        fig=plt.figure(figsize=(5, 5*depth))

        for i in range(depth):

            fig.add_subplot(depth, 1, i+1)

            plt.imshow(sample[i])

    elif predict is None:

        print("hi")

        fig=plt.figure(figsize=(10, 5*depth))

        for i in range(depth):

            fig.add_subplot(depth, 2, 2*i + 1)

            plt.imshow(sample[i]/255.0)

            fig.add_subplot(depth, 2, 2*i + 2)

            plt.imshow(label[i]/255.0)

    else:

        fig=plt.figure(figsize=(15, 5*depth))

        for i in range(depth):

            fig.add_subplot(depth, 3, 3*i + 1)

            plt.imshow(sample[i]/255.0)

            fig.add_subplot(depth, 3, 3*i + 2)

            plt.imshow(label[i]/255.0)

            fig.add_subplot(depth, 3, 3*i + 3)

            plt.imshow(predict[i]/255.0)





    plt.show()

  

# print(labels.shape)  

# display_sample(chosen_images[...,0], chosen_labels[..., 0], ndimage.median_filter(np.where(chosen_predicts > 0.35, 1, 0)[..., 0], size=[3, 3, 3], mode='constant'))

# display_sample(v_images2[1000:1010,...,0], v_labels2[1000:1010,...,0], v_pred2[1000:1010,...,0])

# np.where(chosen_predicts[160] > 0.35, 1, 0)[..., 0]

# display_sample(focus_pos_images[150:160, ..., 0], np.round(focus_pos_labels[150:160, ...,0]), focus_pos_predict[150:160, ...,0])
import numpy as np

from skimage.measure import label   

import matplotlib.pyplot as plt 





def getLargestCC(segmentation):

    labels = label(segmentation)

    unique, counts = np.unique(labels, return_counts=True)

    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest

    largest=max(list_seg, key=lambda x:x[1])[0]

    labels_max=(labels == largest).astype(int)

    return labels_max




