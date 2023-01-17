#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

Descripstion(어떤 방법을 어떤 이유로 선택했는지)



I used the local computer without using the kernel of the Kaggle.

Since I thought the kernel lacked resources such as RAM and disks.



Currently, this code has only been enabled to execute. 



Remove the Args object and Modify main function to run it in a local computer.



Model:

    I implemented basic CNN model introduced Quick, Draw! Doodle Recognition paper from scratch.

    And I used Transfer Learning Models like Densenet, Mobilnet, Xception.



    These models such as ResNet are "large" deep learning model. 

    These models require a big image size. 

    So, I can't use these models since I don't have large resources.

    

    Mobilenet is shallow deep learning model.

    It didn't come out as much as I expected.



    DenseNet201 doesn't requires a big image size.



    I used a range of models and hyperparameters.



    Best performance is the result of DenseNet201 + FCN(512, 512).



Multi GPUs:

    This code was implemented for using multi gpus.

    If you use multi gpus model, you must use multi gpus in test.



Data:

    I used only simplified train data because my server computer's hard disk capacity was insufficient.

    Anyway, I employed Beluga's pre-processing method.



"""



###########

# imports #

###########



import os

import json

import datetime as dt

import cv2

import numpy as np

import pandas as pd

import tensorflow as tf

from argparse import ArgumentParser, Namespace

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import MobileNet, Xception, DenseNet201

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint





DATA_DIR = '../input/shuffle-csvs/'

INPUT_DIR = '../input/quickdraw-doodle-recognition'



BASE_SIZE = 256

num_csvs = 100

num_classes = 340



lw = 1

channel = 3

border = 2  





#############

# Converter #

#############



def f2cat(filename):

    return filename.split('.')[0]



def list_all_categories():

    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))

    return sorted([f2cat(f) for f in files], key=str.lower)



def preds2catids(predictions):

	return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])





###########

# Metrics #

###########



def apk(actual, predicted, k=3):

    """ref: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    """

    print('actual:', actual, 'predicted:', predicted)

    if len(predicted) > k:

        predicted = predicted[:k]

    score = 0.0

    num_hits = 0.0

    for i, p in enumerate(predicted):

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i + 1.0)

    if not actual:

        return 0.0

    return score / min(len(actual), k)



def mapk(actual, predicted, k=3):

    """ref: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    """

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)





#############

# Generator #

#############





shift_colors = (

        (255, 0, 0),

        (255, 128, 0),

        (255, 255, 0),

        (128, 255, 0),

        (0, 255, 0),

        (0, 255, 128),

        (0, 255, 255),

        (0, 128, 255),

        (0, 0, 255),

        (128, 0, 255),

        (255, 0, 255),

        (255, 0, 128)

)



def list2drawing(raw_strokes, img_size):

    img = np.zeros((img_size, img_size, channel), np.uint8)

    coef = (img_size - 2 * lw - 2 * border) / (BASE_SIZE - 1)

    nb_stokes = len(raw_strokes)

    for t, stroke in enumerate(raw_strokes[::-1]):

        rgb = shift_colors[(nb_stokes-t-1)%12]



        for i in range(len(stroke[0]) - 1):

            p1 = (int(coef * stroke[0][i] + lw + border), int(coef * stroke[1][i] + lw+ border))

            p2 = (int(coef * stroke[0][i + 1] + lw + border), int(coef * stroke[1][i + 1] + lw + border))

            _ = cv2.line(img, p1, p2, rgb, lw, cv2.LINE_AA)

    if img_size != BASE_SIZE:

        return cv2.resize(img, (img_size, img_size))

    else:

        return img



def draw_cv2(raw_strokes, img_size=256, lw=lw, time_color=True):

    """Ref: https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892

    """

    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)

    for t, stroke in enumerate(raw_strokes):

        for i in range(len(stroke[0]) - 1):

            color = 255 - min(t, 10) * 13 if time_color else 255

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),

                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

    if img_size != BASE_SIZE:

        return cv2.resize(img, (img_size, img_size))

    else:

        return img



def image_generator_test(img_size, batch_size, model_name, lw=lw, time_color=True):

    while True:

        filename = os.path.join(INPUT_DIR, 'test_simplified.csv') 

        for df in pd.read_csv(filename, chunksize=batch_size):

            df['drawing'] = df['drawing'].apply(json.loads)

            x = np.zeros((len(df), img_size, img_size, channel))

            for i, raw_strokes in enumerate(df.drawing.values):

                x[i, :, :, 0] = draw_cv2(raw_strokes, img_size=img_size, lw=lw,

                                        time_color=time_color)

#                     x[i, :, :, :] = list2drawing(raw_strokes, img_size=img_size)

            if model_name == 'mobilenet':

                x = tf.keras.applications.mobilenet.preprocess_input(x).astype(np.float32)

            elif model_name =='xception':

                x = tf.keras.applications.xception.preprocess_input(x).astype(np.float32)

            elif model_name == 'densenet201':

                x = tf.keras.applications.densenet.preprocess_input(x).astype(np.float32)

            else:

                x /= 255.

                x.astype(np.float32)

            yield x



def image_generator_xd(img_size, batch_size, ks, model_name, lw=lw, time_color=True):

    while True:

        for k in np.random.permutation(ks):

            filename = os.path.join(DATA_DIR, 'train_k{}.csv.gz'.format(k))

            for df in pd.read_csv(filename, chunksize=batch_size):

                df['drawing'] = df['drawing'].apply(json.loads)

                x = np.zeros((len(df), img_size, img_size, channel))

                for i, raw_strokes in enumerate(df.drawing.values):

                    x[i, :, :, 0] = draw_cv2(raw_strokes, img_size=img_size, lw=lw,

                                            time_color=time_color)

#                     x[i, :, :, :] = list2drawing(raw_strokes, img_size=img_size)

                if model_name == 'mobilenet':

                    x = tf.keras.applications.mobilenet.preprocess_input(x).astype(np.float32)

                elif model_name =='xception':

                    x = tf.keras.applications.xception.preprocess_input(x).astype(np.float32)

                elif model_name == 'densenet201':

                    x = tf.keras.applications.densenet.preprocess_input(x).astype(np.float32)

                else:

                    x /= 255.

                    x.astype(np.float32)

                y = tf.keras.utils.to_categorical(df.y, num_classes=num_classes)

                yield x, y

                

                

def df_to_image_array_xd(df, img_size, model_name, lw=lw, time_color=True):

    df['drawing'] = df['drawing'].apply(json.loads)

    x = np.zeros((len(df), img_size, img_size, channel))

    for i, raw_strokes in enumerate(df.drawing.values):

        x[i, :, :, 0] = draw_cv2(raw_strokes, img_size=img_size, lw=lw, time_color=time_color)

#         x[i, :, :, :] = list2drawing(raw_strokes, img_size=img_size)

    if model_name == 'mobilenet':

        x = tf.keras.applications.mobilenet.preprocess_input(x).astype(np.float32)

    elif model_name =='xception':

        x = tf.keras.applications.xception.preprocess_input(x).astype(np.float32)

    elif model_name == 'densenet201':

        x = tf.keras.applications.densenet.preprocess_input(x).astype(np.float32)

    else:

        x /= 255.

        x.astype(np.float32)

    return x





#########

# Model #

#########



class StanfordCNN(tf.keras.layers.Layer):

    """ref: http://cs229.stanford.edu/proj2018/report/98.pdf

    It is an CNN architectural model introduced in the present paper.

    """

    def __init__(self, filters, kernel_size, pool_size, num_classes):

        super(StanfordCNN, self).__init__()

        

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')

        self.conv3 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')

        self.maxpool = tf.keras.layers.MaxPool2D(pool_size)

        

        self.fcn = FCN([700,500,400], num_classes)



    def call(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.maxpool(x)

        

        x = self.fcn(x)



        return x



class FCN(tf.keras.layers.Layer):

    def __init__(self, num_unit, num_classes):

        super(FCN, self).__init__()

        self.layers = []



        num_layer = len(num_unit)



        self.flatten = tf.keras.layers.Flatten()



        for i in range(num_layer):

            W = tf.keras.layers.Dense(num_unit[i], activation='relu')

            self.layers.append(W)



        self.W_o = tf.keras.layers.Dense(num_classes, activation='softmax')



    def call(self, x):

        x = self.flatten(x)

        for layer in self.layers:

            x = layer(x)

        return self.W_o(x)





def create_model(model_name, input_shape, num_gpu, learning_rate):



    if model_name.lower()=='mobilenet':

        model = MobileNet(input_shape=input_shape, weights=None, classes=num_classes)



    elif model_name.lower() == 'xception':

        xcep = Xception(input_shape=input_shape, weights=None, include_top=False)

        fcn = FCN([512, 512], num_classes)



        output = fcn(xcep.output)

        model = tf.keras.Model(inputs=xcep.input, outputs=output)



    elif model_name.lower() =='densenet201':

        dense = DenseNet201(input_shape=input_shape, weights='imagenet', include_top=False) 

        x = dense.output

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        fcn = FCN([1024, 512], num_classes)



        output = fcn(dense.output)

        model = tf.keras.Model(inputs=dense.input, outputs=output)



    elif model_name.lower() == 'resnet50':

        resnet = tf.keras.applications.ResNet50(input_shape=input_shape, weights=None, include_top=False, pooling='avg')

        fcn = FCN([512, 512], num_classes)



        output = fcn(resnet.output)

        model = tf.keras.Model(inputs=resnet.input, outputs=output)



    elif model_name.lower() == 'base':

        stanCNN = StanfordCNN(5, (3,3), (2,2), num_classes)

        input_img = tf.keras.layers.Input(shape=input_shape)

        output = stanCNN(input_img)

        model = tf.keras.Model(inputs=input_img, outputs=output)



    elif model_name.lower() == 'inception':

        incep = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=input_shape, weights=None, include_top=False, pooling='avg')

        fcn = FCN([512], num_classes)



        output = fcn(incep.output) 

        model = tf.keras.Model(inputs=incep.input, outputs=output)







    if num_gpu != 1:

        model = tf.keras.utils.multi_gpu_model(model, gpus=num_gpu)



    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',

                  metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])



    print(model.summary())



    return model





############

# Executor #

############



def test(args: Namespace, model=None):



    img_size = args.img_size 

    input_shape = (img_size, img_size, channel)

    model_name = args.model_name



    test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))

#     x_test = df_to_image_array_xd(test, img_size, model_name)

#     print(test.shape, x_test.shape)

#     print('Test array memory {:.2f} GB'.format(x_test.nbytes / 2.**30))



    test_datagen = image_generator_test(img_size=img_size, batch_size=128, model_name=model_name)





    if model == None:

        print('Loading saved model')

        model = create_model(model_name, input_shape, args.num_gpu, args.learning_rate)

        model.load_weights('model.h5')



#         valid_df = pd.read_csv(os.path.join(DATA_DIR, 'train_k{}.csv.gz'.format(num_csvs - 1)), nrows=3000)

#         x_valid = df_to_image_array_xd(valid_df, img_size, model_name)

#         y_valid = tf.keras.utils.to_categorical(valid_df.y, num_classes=num_classes)

#         print(x_valid.shape, y_valid.shape)

#         print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 2.**30 ))

#         valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)

#         map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)

#         print('Map3: {:.3f}'.format(map3))



#     test_predictions = model.predict(x_test, batch_size=128, verbose=1)

    test_predictions = model.predict_generator(test_datagen, verbose=1, steps=len(test)//128)



    top3 = preds2catids(test_predictions)



    cats = list_all_categories()

    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}

    top3cats = top3.replace(id2cat)



    test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']

    submission = test[['key_id', 'word']]

    now_time = dt.datetime.now().strftime("%m%d%H%M")

    submission.to_csv('my_submission_{}.csv'.format(now_time), index=False)

    print('submission shape:', submission.shape)



def train(args: Namespace):

    start = dt.datetime.now()



    num_gpu = args.num_gpu

    epochs = args.epoch 

    batch_size = args.batch_size * num_gpu

    if args.steps == -1:

        steps_per_epoch = (49209839) // batch_size 

    else: 

        steps_per_epoch = args.steps 

    learning_rate = args.learning_rate

    img_size = args.img_size

    input_shape = (img_size, img_size, channel)

    model_name = args.model_name



    model = create_model(model_name, input_shape, num_gpu, learning_rate) 



    valid_df = pd.read_csv(os.path.join(DATA_DIR, 'train_k{}.csv.gz'.format(num_csvs - 1)), nrows=3000)

    x_valid = df_to_image_array_xd(valid_df, img_size, model_name)

    y_valid = tf.keras.utils.to_categorical(valid_df.y, num_classes=num_classes)

    print(x_valid.shape, y_valid.shape)

    print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 2.**30 ))



    train_datagen = image_generator_xd(img_size=img_size, batch_size=batch_size, ks=range(num_csvs - 1), model_name=model_name)



    callbacks = [

        ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.75, patience=3, min_delta=0.001,

                              mode='max', min_lr=1e-5, verbose=1),

        ModelCheckpoint('model.h5', monitor='val_top_3_accuracy', mode='max', save_best_only=True,

                        save_weights_only=True),

    ]



    model.fit_generator(

        train_datagen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,

        validation_data=(x_valid, y_valid),

        callbacks = callbacks

    )



    valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)

    map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)

    print('Map3: {:.3f}'.format(map3))



    end = dt.datetime.now()

    print('Total time {}s'.format((end - start).seconds))



    return model



class Args():

    """This Object is for using Kaggle Kernel.

    """

    def __init__(self):

        self.mode = 'train'

        self.model_name = 'densenet201'

        self.img_size = 75

        self.batch_size = 256

        self.epoch = 15

        self.steps = 500

        self.learning_rate = 0.002

        self.num_gpu = 1

        

def main():

    

    """Remove the following annotations if you want to use local resources."""



    # parser = ArgumentParser(description='train model from data')



    # parser.add_argument('--mode', help='train or test <default: train>', metavar='STRING', default='train')

    # parser.add_argument('--model-name', help='select a model <default: mobilenet>', metavar='STRING', default='mobilenet')

    # parser.add_argument('--img-size', help='image size <default: 70>', metavar='INT', 

    #                     type=int, default=75)



    # parser.add_argument('--batch-size', help='batch size <default: 256>', metavar='INT', 

    #                     type=int, default=256)

    # parser.add_argument('--epoch', help='epoch number <default: 3>', metavar='INT', 

    #                     type=int, default=3)

    # parser.add_argument('--steps', help='steps <default: -1>', metavar='INT', 

    #                     type=int, default=-1)

    # parser.add_argument('--learning-rate', help='learning_rate <default: 1>', 

    #                     metavar='REAL', type=float, default=0.002)



    # parser.add_argument('--num-gpu', help='the number of GPUs to use <default: 1>', 

    #                     metavar='INT', type=int, default=1)



    # args = parser.parse_args()

    

    """Remove this object if you want to use local resources."""

    args = Args() 

    



#     if args.mode=='train':

#         train(args)

#     elif args.mode=='test':

#         test(args)



    model = train(args)

#     model = None

    

    test(args, model)

    



if __name__=='__main__':

    main()
