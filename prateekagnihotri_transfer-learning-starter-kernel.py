#### Set number of epochs and learning rate ####



set_epochs = 2

set_lr = 1e-4
import numpy as np

import pandas as pd

import os

import skimage.io

from glob import glob

from random import shuffle

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator



import keras.backend as K

from keras import layers,Input



from keras.models import Model

from keras.applications.nasnet import  preprocess_input

from keras.optimizers import Adam, RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from imgaug import augmenters as iaa

import imgaug as ia

print(os.listdir("../input/"))

def quadratic_kappa_coefficient(y_true, y_pred):

    y_true = K.cast(y_true, "float32")

    n_classes = K.cast(y_pred.shape[-1], "float32")

    weights = K.arange(0, n_classes, dtype="float32") / (n_classes - 1)

    weights = (weights - K.expand_dims(weights, -1)) ** 2



    hist_true = K.sum(y_true, axis=0)

    hist_pred = K.sum(y_pred, axis=0)



    E = K.expand_dims(hist_true, axis=-1) * hist_pred

    E = E / K.sum(E, keepdims=False)



    O = K.transpose(K.transpose(y_true) @ y_pred)  # confusion matrix

    O = O / K.sum(O)



    num = weights * O

    den = weights * E



    QWK = (1 - K.sum(num) / K.sum(den))

    return QWK
def get_model_classif_nasnet_1():  

    

    inputs = Input((256, 256, 3))



    x1 = layers.Conv2D(32,3,padding='same')(inputs)

    x1 = layers.BatchNormalization()(x1)

    x1 = layers.Activation('relu')(x1)

    x1 = layers.Conv2D(32,3,padding='same')(x1)

    x1 = layers.BatchNormalization()(x1)

    x1 = layers.Activation('relu')(x1)

    x1 = layers.Conv2D(32,3,padding='same')(x1)

    x1 = layers.BatchNormalization()(x1)

    x1 = layers.Activation('relu')(x1)



    

    x1_s = layers.SeparableConv2D(32,3,padding='same')(inputs)

    x1_s = layers.BatchNormalization()(x1_s)

    x1_s = layers.Activation('relu')(x1_s)

    x1_s = layers.SeparableConv2D(32,3,padding='same')(x1_s)

    x1_s = layers.BatchNormalization()(x1_s)

    x1_s = layers.Activation('relu')(x1_s)    

    x1_s = layers.SeparableConv2D(32,3,padding='same')(x1_s)

    x1_s = layers.BatchNormalization()(x1_s)

    x1_s = layers.Activation('relu')(x1_s)

    concetenated_0 = layers.concatenate([x1,x1_s])



    x2 = layers.Conv2D(64,3,padding='same')(concetenated_0)

    x2 = layers.BatchNormalization()(x2)

    x2 = layers.Activation('relu')(x2)

    x2 = layers.Conv2D(64,3,padding='same')(x2)

    x2 = layers.BatchNormalization()(x2)

    x2 = layers.Activation('relu')(x2)

    x2 = layers.Conv2D(64,3,padding='same')(x2)

    x2 = layers.BatchNormalization()(x2)

    x2 = layers.Activation('relu')(x2)

    residual_concetenated_0 = layers.Conv2D(64,1,strides=1,padding='same')(concetenated_0)

    x2 = layers.add([x2,residual_concetenated_0])

    concetenates_x2_x1_s = layers.concatenate([x2,x1_s])

    x2 = layers.MaxPool2D(2,2)(concetenates_x2_x1_s)

    

    



    x2_s = layers.SeparableConv2D(64,3,padding='same')(x2)

    x2_s = layers.BatchNormalization()(x2_s)

    x2_s = layers.Activation('relu')(x2_s)

    x2_s = layers.SeparableConv2D(64,3,padding='same')(x2_s)

    x2_s= layers.BatchNormalization()(x2_s)

    x2_s= layers.Activation('relu')(x2_s)

    x2_s = layers.SeparableConv2D(64,3,padding='same')(x2_s)

    x2_s= layers.BatchNormalization()(x2_s)

    x2_s= layers.Activation('relu')(x2_s)

    x2_s = layers.Conv2D(96,1,strides=1,padding='same')(x2_s)

    x2_s = layers.add([x2_s,x2]) 

    x2_s = layers.MaxPool2D(2,2)(x2_s)

    

    x3 = layers.Conv2D(128,3,padding='same')(x2)

    x3 = layers.BatchNormalization()(x3)

    x3 = layers.Activation('relu')(x3)

    x3 = layers.Conv2D(128,3,padding='same')(x3)

    x3 = layers.BatchNormalization()(x3)

    x3 = layers.Activation('relu')(x3)

    x3 = layers.Conv2D(128,3,padding='same')(x3)

    x3 = layers.BatchNormalization()(x3)

    x3 = layers.Activation('relu')(x3)

    residual_x2 = layers.Conv2D(128,1,strides=1,padding='same')(x2)

    x3 = layers.add([residual_x2,x3]) 

    

    x3_x3 = layers.Conv2D(128,3,padding='same')(x3)

    x3_x3 = layers.BatchNormalization()(x3_x3)

    x3_x3 = layers.Activation('relu')(x3_x3)

    x3_x3 = layers.Conv2D(128,3,padding='same')(x3_x3)

    x3_x3 = layers.BatchNormalization()(x3_x3)

    x3_x3 = layers.Activation('relu')(x3_x3)

    x3_x3 = layers.Conv2D(128,3,padding='same')(x3_x3)

    x3_x3 = layers.BatchNormalization()(x3_x3)

    x3_x3 = layers.Activation('relu')(x3_x3)

    x3_x3 = layers.add([x3,x3_x3]) 

    x3_x3 = layers.MaxPool2D(2,2)(x3_x3)

    

    

    concetenated_1 = layers.concatenate([x3_x3,x2_s])

    x3_s = layers.SeparableConv2D(128,3,padding='same')(concetenated_1)

    x3_s = layers.BatchNormalization()(x3_s)

    x3_s = layers.Activation('relu')(x3_s)

    x3_s = layers.SeparableConv2D(128,3,padding='same')(x3_s)

    x3_s= layers.BatchNormalization()(x3_s)

    x3_s= layers.Activation('relu')(x3_s)

    x3_s = layers.SeparableConv2D(128,3,padding='same')(x3_s)

    x3_s= layers.BatchNormalization()(x3_s)

    x3_s= layers.Activation('relu')(x3_s)

    x3_s = layers.add([x3_s,x3_x3]) 

    x3_s = layers.MaxPool2D(2,2)(x3_s)

    

    x4 = layers.Conv2D(256,3,padding='same')(x3_x3)

    x4 = layers.BatchNormalization()(x4)

    x4 = layers.Activation('relu')(x4)

    x4 = layers.Conv2D(256,3,padding='same')(x4)

    x4 = layers.BatchNormalization()(x4)

    x4 = layers.Activation('relu')(x4)

    x4 = layers.Conv2D(256,3,padding='same')(x4)

    x4 = layers.BatchNormalization()(x4)

    x4 = layers.Activation('relu')(x4)

    residual_x3 = layers.Conv2D(256,1,strides=1,padding='same')(x3_x3)

    x4 = layers.add([residual_x3,x4]) 

    

    x4_x4 = layers.Conv2D(256,3,padding='same')(x4)

    x4_x4 = layers.BatchNormalization()(x4_x4)

    x4_x4 = layers.Activation('relu')(x4_x4)

    x4_x4 = layers.Conv2D(256,3,padding='same')(x4_x4)

    x4_x4 = layers.BatchNormalization()(x4_x4)

    x4_x4 = layers.Activation('relu')(x4_x4)

    x4_x4 = layers.Conv2D(256,3,padding='same')(x4_x4)

    x4_x4 = layers.BatchNormalization()(x4_x4)

    x4_x4 = layers.Activation('relu')(x4_x4)

    x4_x4 = layers.add([x4,x4_x4]) 

    x4_x4 = layers.MaxPool2D(2,2)(x4_x4)

    



    concetenated_2 = layers.concatenate([x4_x4,x3_s])

    x4_s = layers.SeparableConv2D(256,3,padding='same')(concetenated_2)

    x4_s = layers.BatchNormalization()(x4_s)

    x4_s = layers.Activation('relu')(x4_s)

    x4_s = layers.SeparableConv2D(256,3,padding='same')(x4_s)

    x4_s= layers.BatchNormalization()(x4_s)

    x4_s= layers.Activation('relu')(x4_s)

    x4_s = layers.SeparableConv2D(256,3,padding='same')(x4_s)

    x4_s= layers.BatchNormalization()(x4_s)

    x4_s= layers.Activation('relu')(x4_s)

    x4_s = layers.add([x4_s,x4_x4]) 

    x4_s = layers.MaxPool2D(2,2)(x4_s)

    

    x5 = layers.Conv2D(512,3,padding='same')(x4_x4)

    x5 = layers.BatchNormalization()(x5)

    x5 = layers.Activation('relu')(x5)

    x5 = layers.Conv2D(512,3,padding='same')(x5)

    x5 = layers.BatchNormalization()(x5)

    x5 = layers.Activation('relu')(x5)

    x5 = layers.Conv2D(512,3,padding='same')(x5)

    x5 = layers.BatchNormalization()(x5)

    x5 = layers.Activation('relu')(x5)

    residual_x4 = layers.Conv2D(512,1,strides=1,padding='same')(x4_x4)

    x5 = layers.add([residual_x4,x5])



    x5_x5 = layers.Conv2D(512,3,padding='same')(x5)

    x5_x5 = layers.BatchNormalization()(x5_x5)

    x5_x5 = layers.Activation('relu')(x5_x5)

    x5_x5 = layers.Conv2D(512,3,padding='same')(x5_x5)

    x5_x5 = layers.BatchNormalization()(x5_x5)

    x5_x5 = layers.Activation('relu')(x5_x5)

    x5_x5 = layers.Conv2D(512,3,padding='same')(x5_x5)

    x5_x5 = layers.BatchNormalization()(x5_x5)

    x5_x5 = layers.Activation('relu')(x5_x5)

    x5_x5 = layers.add([x5,x5_x5])

    x5_x5 = layers.MaxPool2D(2,2)(x5_x5)

    

    concetenated_3 = layers.concatenate([x5_x5,x4_s])

    x5_s = layers.SeparableConv2D(512,3,padding='same')(concetenated_3)

    x5_s = layers.BatchNormalization()(x5_s)

    x5_s = layers.Activation('relu')(x5_s)

    x5_s = layers.SeparableConv2D(512,3,padding='same')(x5_s)

    x5_s= layers.BatchNormalization()(x5_s)

    x5_s= layers.Activation('relu')(x5_s)

    x5_s = layers.SeparableConv2D(512,3,padding='same')(x5_s)

    x5_s= layers.BatchNormalization()(x5_s)

    x5_s= layers.Activation('relu')(x5_s)

    x5_s = layers.add([x5_s,x5_x5]) 



    x = layers.GlobalAveragePooling2D()(x5_s)



    x = layers.Dense(64)(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.3)(x)

    

    output_tensor_1 = layers.Dense(1,activation='sigmoid')(x)



    ### Extra layers for model 2 ####    

    

    x = layers.Dense(32)(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation('relu')(x)

    x = layers.Dropout(0.3)(x)

    output_tensor_2 = layers.Dense(6,activation='softmax')(x)

    

    model_2 = Model(inputs,output_tensor_2)

    

    model_1 = Model(inputs,output_tensor_1)

    

    return model_1, model_2
model_1,model_2 = get_model_classif_nasnet_1()

model_1.load_weights('../input/pretrained-model/model_4.h5')
### Freeze weights

for layer in model_1.layers:

    if 'conv' in layer.name:

        layer.trainable = False

        

### Compile model 2

model_2.compile(optimizer=Adam(set_lr), loss='categorical_crossentropy', metrics=['acc',quadratic_kappa_coefficient])
def get_seq():

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.5), # horizontally flip 50% of all images

            iaa.Flipud(0.2), # vertically flip 20% of all images

            sometimes(iaa.Affine(

                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)

                rotate=(-10, 10), # rotate by -45 to +45 degrees

                shear=(-5, 5), # shear by -16 to +16 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 5),

                [

                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                    iaa.OneOf([

                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0

                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7

                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7

                    ]),

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    # search either for all edges or for directed edges,

                    # blend the result with the original image using a blobby mask

                    iaa.SimplexNoiseAlpha(iaa.OneOf([

                        iaa.EdgeDetect(alpha=(0.5, 1.0)),

                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                    ])),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images

                    iaa.OneOf([

                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels

                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),

                    ]),

                    iaa.Invert(0.01, per_channel=True), # invert color channels

                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation

                    # either change the brightness of the whole image (sometimes

                    # per channel) or change the brightness of subareas

                    iaa.OneOf([

                        iaa.Multiply((0.9, 1.1), per_channel=0.5),

                        iaa.FrequencyNoiseAlpha(

                            exponent=(-1, 0),

                            first=iaa.Multiply((0.9, 1.1), per_channel=True),

                            second=iaa.ContrastNormalization((0.9, 1.1))

                        )

                    ]),

                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around

                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

                ],

                random_order=True

            )

        ],

        random_order=True

    )

    return seq
def chunker(seq, size):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



def get_image(img_name, data_dir='../input/prostate-cancer-grade-assessment/train_images'):

    

    img_path = os.path.join(data_dir, f'{img_name}.tiff')

    img = skimage.io.MultiImage(img_path)

    img = cv2.resize(img[-1], (256,256))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



def data_gen(list_files, id_label_map, batch_size, augment=False):

    seq = get_seq()

    while True:

        shuffle(list_files)

        for batch in chunker(list_files, batch_size):

            X = [get_image(x) for x in batch]

            Y = np.zeros((len(batch),6))

            for i in range(len(batch)):

                Y[i,id_label_map[get_id_from_file_path(batch[i])]] = 1.0

            if augment:

                X = seq.augment_images(X)

            X = [preprocess_input(x) for x in X]



            yield np.array(X), np.array(Y)

def get_id_from_file_path(file_path):

    return file_path.split(os.path.sep)[-1].replace('.tiff', '')
batch_size=16

df_train = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")

id_label_map = {k:v for k,v in zip(df_train.image_id.values, df_train.isup_grade.values)}
labeled_files = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv").image_id.values

test_files = pd.read_csv("../input/prostate-cancer-grade-assessment/test.csv").image_id.values



train, val = train_test_split(labeled_files, test_size=0.1, random_state=101010)
check_point = ModelCheckpoint('./model.h5',monitor='val_loss',verbose=True, save_best_only=True, save_weights_only=True)

early_stop = EarlyStopping(monitor='val_loss',patience=5,verbose=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1)
history = model_2.fit_generator(

    data_gen(train, id_label_map, batch_size, augment=True),

    validation_data=data_gen(val, id_label_map, batch_size),

    epochs=set_epochs, verbose=1,

    callbacks=[check_point,early_stop,reduce_lr],

    steps_per_epoch=len(train) // batch_size,

    validation_steps=len(val) // batch_size)
def TTA(img):

    img1 = img

    img2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    img3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img4 = cv2.rotate(img, cv2.ROTATE_180)

    images = [img1, img2, img3, img4]

    

    return model_2.predict(np.array(images), batch_size=4)
def post_process(preds):

    avg = np.sum(preds,axis = 0)

    label = np.argmax(avg)

    return label
data_dir = '../input/prostate-cancer-grade-assessment/test_images'

sample_submission = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')

# data_dir = '../input/prostate-cancer-grade-assessment/train_images'

# sample_submission = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').head(50)



test_images = sample_submission.image_id.values

labels = []



try:    

    for image in tqdm(test_images):

        img = get_image(image, data_dir)

        preds = TTA(img)

        label = post_process(preds)

        labels.append(label)

    sample_submission['isup_grade'] = labels

except:

    print('Test dir not found')

    

sample_submission['isup_grade'] = sample_submission['isup_grade'].astype(int)

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()