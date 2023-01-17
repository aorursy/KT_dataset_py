from random import randint

import tensorflow as tf

import json

import os

from PIL import Image

from glob import glob

from zipfile import ZipFile

import pandas as pd

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.models import Model, Input, load_model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.applications.nasnet import NASNetLarge

from keras import backend as K

from matplotlib import pyplot as plt

from matplotlib.image import imread

import os

import sys

import cv2

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score

import scipy

from tqdm import tqdm

from skimage.io import imsave
def process_csv(dataframe: pd.DataFrame, image_column_name: str,

                label_column_name: str,

                folder_with_images: str) -> pd.DataFrame:

    """This function process Pandas DataFrame, which contains image filenames

    and their corresponding labels.



    Args:

        dataframe: Pandas DataFrame object. It should consist of 2 columns

        image_column_name: The name of the column containing the image

            filenames

        label_column_name: The name of the column containing the image

            labels

        folder_with_images: Folder with images



    Returns:

        dataframe: processed DataFrame with full paths to images

    """

    dataframe[image_column_name] = dataframe[image_column_name].apply(

        lambda x: f"{folder_with_images}{x}.png")

    dataframe[label_column_name] = dataframe[label_column_name].astype('str')

    return dataframe
train_datagen = ImageDataGenerator(rescale=1. / 255,

                                   rotation_range=15,

                                   width_shift_range=0.1,

                                   height_shift_range=0.1,

                                   shear_range=0.01,

                                   zoom_range=[0.9, 1.25],

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                   fill_mode='reflect',

                                   data_format='channels_last',

                                   brightness_range=[0.5, 1.5],

                                   validation_split=0.3)
!mkdir /kaggle/dataset_with_ben
def crop_image1(img,tol=7):

    # img is image data

    # tol  is tolerance

        

    mask = img>tol

    return img[np.ix_(mask.any(1),mask.any(0))]



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
def ben_color(path, sigmaX=20):

    image = Image.open(path)

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (299, 299))



    height, width, depth = image.shape    



    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))



    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    image = cv2.bitwise_and(image, image, mask=circle_img)



    image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX=20) ,-4 ,128)

    

    return image
for path in glob(f"/kaggle/input/aptos2019-blindness-detection/train_images/*"):

    img = ben_color(path, sigmaX=30)

    name = path.split('/')[-1]

    name = name.split('.')[0]

    imsave('/kaggle/dataset_with_ben/'+str(name)+'.png', img)
train_csv = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")

train_csv = process_csv(

    dataframe=train_csv,

    image_column_name="id_code",

    label_column_name="diagnosis",

    folder_with_images="/kaggle/dataset_with_ben/")
train_generator = train_datagen.flow_from_dataframe(

    dataframe=train_csv, x_col="id_code", y_col="diagnosis", subset="training",

    batch_size=32, target_size=(299, 299))

val_generator = train_datagen.flow_from_dataframe(

    dataframe=train_csv, x_col="id_code", y_col="diagnosis",

    subset="validation", batch_size=32, target_size=(299, 299))
from keras.optimizers import Optimizer

from keras import backend as K





class RAdam(Optimizer):



    def __init__(self, lr, beta1=0.9, beta2=0.99, decay=0, **kwargs):

        super(RAdam, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.lr = K.variable(lr)

            self._beta1 = K.variable(beta1, dtype="float32")

            self._beta2 = K.variable(beta2, dtype="float32")

            self._max_sma_length = 2 / (1 - self._beta2)

            self._iterations = K.variable(0)

            self._decay = K.variable(decay)



    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self._iterations, 1)]

        first_moments = [K.zeros(K.int_shape(p), dtype=K.dtype(p))

                         for (i, p) in enumerate(params)]

        second_moments = [K.zeros(K.int_shape(p), dtype=K.dtype(p))

                          for (i, p) in enumerate(params)]



        self.weights = [self._iterations] + first_moments + second_moments

        bias_corrected_beta1 = K.pow(self._beta1, self._iterations)

        bias_corrected_beta2 = K.pow(self._beta2, self._iterations)

        for i, (curr_params, curr_grads) in enumerate(zip(params, grads)):

            # Updating moving moments



            new_first_moment = self._beta1 * first_moments[i] + (

                    1 - self._beta1) * curr_grads

            new_second_moment = self._beta2 * second_moments[i] + (

                    1 - self._beta2) * K.square(curr_grads)

            self.updates.append(K.update(first_moments[i],

                                         new_first_moment))

            self.updates.append(K.update(second_moments[i],

                                         new_second_moment))



            # Computing length of approximated SMA



            bias_corrected_moving_average = new_first_moment / (

                    1 - bias_corrected_beta1)

            sma_length = self._max_sma_length - 2 * (

                    self._iterations * bias_corrected_beta2) / (

                                 1 - bias_corrected_beta2)



            # Bias correction



            variance_rectification_term = K.sqrt(

                self._max_sma_length * (sma_length - 4) * (sma_length - 2) / (

                        sma_length * (self._max_sma_length - 4) *

                        (self._max_sma_length - 2) + K.epsilon()))

            resulting_parameters = K.switch(

                sma_length > 5, variance_rectification_term *

                bias_corrected_moving_average / K.sqrt(

                    K.epsilon() + new_second_moment / (1 -

                                                       bias_corrected_beta2)),

                bias_corrected_moving_average)

            resulting_parameters = curr_params - self.lr * resulting_parameters

            self.updates.append(K.update(curr_params, resulting_parameters))

        if self._decay != 0:

            new_lr = self.lr * (1. / (1. + self._decay * K.cast(

                self._iterations, K.dtype(self._decay))))

            self.updates.append(K.update(self.lr, new_lr))

        return self.updates



    def get_config(self):

        config = {

            "lr": float(K.get_value(self.lr)),

            "beta1": float(K.get_value(self._beta1)),

            "beta2": float(K.get_value(self._beta2)),

        }

        base_config = super(RAdam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
from keras.models import Model, Input

from keras.layers import Dense, GlobalAveragePooling2D

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.xception import Xception

from keras.applications.inception_v3 import InceptionV3

from keras.layers import Average, Input



sys.path.append(os.path.abspath('../input/kerasefficientnetsmaster/keras-efficientnets-master/keras-efficientnets-master/'))

from keras_efficientnets import EfficientNetB7







def create_model():

    input_tensor = Input((299, 299, 3))

    outputs = []

    

    effnet = EfficientNetB7(input_shape=(299,299,3),

                        weights=sys.path.append(os.path.abspath('/kaggle/input/efficientnetb0b7-keras-weights/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')),

                        include_top=False)

    

    #InceptionResNetV2_model = InceptionResNetV2(weights=None, input_shape=(299, 299, 3),include_top=False)                          

    #InceptionResNetV2_model.load_weights("/kaggle/input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5")

    

    xception_model = Xception(weights=None,include_top=False,input_shape=(299,299,3))

    xception_model.load_weights("/kaggle/input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5")

    

    #InceptionV3_model = InceptionV3(weights=None,include_top=False,input_shape=(299, 299, 3))

    #InceptionV3_model.load_weights("/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")

    

    pretrained_models = [

       effnet,xception_model #xception_model #InceptionResNetV2_model #xception_model #InceptionV3_model, InceptionResNetV2_model,  #xception_model,InceptionV3_model

    ]

    for i, model in enumerate(pretrained_models):

        curr_output = model(input_tensor)

        curr_output = GlobalAveragePooling2D()(curr_output)

        curr_output = Dense(1024, activation="relu")(curr_output)

        outputs.append(curr_output)

    output_tensor = Average()(outputs)

    output_tensor = Dense(5, activation="softmax")(output_tensor)



    model = Model(input_tensor, output_tensor)

    return model 
start_lr = 1e-10

end_lr = 1
def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=256, name='kappa'):

    """A continuous differentiable approximation of discrete kappa loss.

        Args:

            y_pred: 2D tensor or array, [batch_size, num_classes]

            y_true: 2D tensor or array,[batch_size, num_classes]

            y_pow: int,  e.g. y_pow=2

            N: typically num_classes of the model

            bsize: batch_size of the training or validation ops

            eps: a float, prevents divide by zero

            name: Optional scope/name for op_scope.

        Returns:

            A tensor with the kappa loss."""



    with tf.name_scope(name):

        y_true = tf.to_float(y_true)

        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))

        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))

        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

    

        pred_ = y_pred ** y_pow

        try:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))

        except Exception:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

    

        hist_rater_a = tf.reduce_sum(pred_norm, 0)

        hist_rater_b = tf.reduce_sum(y_true, 0)

    

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

    

        nom = tf.reduce_sum(weights * conf_mat)

        denom = tf.reduce_sum(weights * tf.matmul(

            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /

                              tf.to_float(bsize))

    

        return nom / (denom + eps)
from keras.callbacks import ReduceLROnPlateau



reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,

                              patience=5, min_lr=1e-5)
callbacks = [

    ModelCheckpoint(

        "best_weights.hdf5",

        monitor='val_acc',

        verbose=1, save_best_only=True,

        save_weights_only=True),

    EarlyStopping(monitor='val_acc', patience=5),

    reduce_lr

]
model = create_model()

model.compile(optimizer=Adam(1e-4),

              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(generator=train_generator,

                    steps_per_epoch=len(train_generator),

                    validation_data=val_generator,

                    validation_steps=len(val_generator),

                    epochs=10,

                    callbacks=callbacks)
def test_time_augmentation(image, network_model):

    datagen = ImageDataGenerator()



    all_images = np.expand_dims(image, axis=0)

    

    flip_horizontal_image = np.expand_dims(datagen.apply_transform(

        x=image, transform_parameters={"flip_horizontal": True}), axis=0)

    all_images = np.append(all_images, flip_horizontal_image, axis=0)

    

    flip_vertical_image = np.expand_dims(datagen.apply_transform(

        x=image, transform_parameters={"flip_vertical": True}), axis=0)

    all_images = np.append(all_images, flip_vertical_image, axis=0)

    

    rotated_image = np.expand_dims(datagen.apply_transform(

        x=image, transform_parameters={"theta": randint(0, 15)}), axis=0)

    all_images = np.append(all_images, rotated_image, axis=0)

    

    prediction = int(np.argmax(np.mean(network_model.predict(all_images), axis=0)))

    return prediction
model.load_weights("best_weights.hdf5")
test_csv = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")

predicted_csv = pd.DataFrame(columns=["id_code", "diagnosis"])



for id_code in test_csv["id_code"]:

    filename = f"/kaggle/input/aptos2019-blindness-detection/test_images/{id_code}.png"

    img = imread(filename)

    img = cv2.resize(img, dsize=(299, 299) , interpolation=cv2.INTER_CUBIC)

    prediction = test_time_augmentation(img, model)

    predicted_csv = predicted_csv.append(

        {'id_code':id_code ,"diagnosis": prediction}, ignore_index=True)



with open("submission.csv", "w") as f:

    f.write(predicted_csv.to_csv(index=False))
!rm -r /kaggle/dataset_with_ben/