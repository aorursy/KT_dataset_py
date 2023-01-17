import os

import sys

import string

import random

import warnings



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from itertools import chain

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.measure import label, regionprops

from PIL import Image, ImageDraw

from ast import literal_eval

from tqdm.notebook import tqdm



import tensorflow as tf



from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Dropout, Lambda,Conv2D, Conv2DTranspose, MaxPooling2D, MaxPooling2D, concatenate, BatchNormalization



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())



tf.debugging.set_log_device_placement(True)



# Create some tensors

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)



print(c)
# # Detect hardware, return appropriate distribution strategy

# try:

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

#     print('Running on TPU ', tpu.master())

# except ValueError:

#     tpu = None



# if tpu:

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# else:

#     strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



# print("REPLICAS: ", strategy.num_replicas_in_sync)
# Set some parameters

IMG_WIDTH = 256

IMG_HEIGHT = 256



#customization for running on laptop

# PATH =  os.getcwd() + "\\"

# TRAIN_PATH = PATH + "train\\"

# TEST_PATH = PATH + "test\\"



PATH = "../input/global-wheat-detection/"

TRAIN_PATH = '/kaggle/input/global-wheat-detection/train/'

TEST_PATH = '/kaggle/input/global-wheat-detection/test/'

SC_FACTOR = int(1024 / IMG_WIDTH)



warnings.filterwarnings('ignore')

SEED = 42

random.seed(SEED)

np.random.seed(SEED)
# PATH = "../input/global-wheat-detection/"

train_folder = os.path.join(PATH, "train")

test_folder = os.path.join(PATH, "test")



train_csv_path = os.path.join(PATH, "train.csv")
df = pd.read_csv(train_csv_path)

sample_sub = pd.read_csv(PATH + "sample_submission.csv")



df.head()
# Get train and test IDs and paths

train_ids = os.listdir(TRAIN_PATH)

test_ids = os.listdir(TEST_PATH)



# train_ids = train_ids[0:100]
def make_polygon(coords):

    xm, ym, w, h = coords

    xm, ym, w, h = xm / SC_FACTOR, ym / SC_FACTOR, w / SC_FACTOR, h / SC_FACTOR   # scale values if image was downsized

    return [(xm, ym), (xm, ym + h), (xm + w, ym + h), (xm + w, ym)]



masks = dict() # dictionnary containing all masks



for img_id, gp in tqdm(df.groupby("image_id")):

    gp['polygons'] = gp['bbox'].apply(eval).apply(lambda x: make_polygon(x))



    img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 0)

    for pol in gp['polygons'].values:

        ImageDraw.Draw(img).polygon(pol, outline=1, fill=1)



    mask = np.array(img, dtype=np.uint8)

    masks[img_id] = mask
im = Image.fromarray(masks[list(masks.keys())[7]])

plt.imshow(im)
# Get and resize train images and masks

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks... ')

sys.stdout.flush()



for n, id_ in tqdm(enumerate(train_ids[:]), total=len(train_ids)):

    path = TRAIN_PATH + id_

    img = imread(path)

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    

    id_clean = id_.split('.')[0]

    if id_clean in masks.keys():

        Y_train[n] = masks[id_clean][:, :, np.newaxis]



# Get and resize test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

sizes_test = list()

print('Getting and resizing test images...')

sys.stdout.flush()



for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    img = imread(path)

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img



print('Done!')
X_train.shape, Y_train.shape
def show_images(images, num=2):

    

    images_to_show = np.random.choice(images, num)



    for image_id in images_to_show:



        image_path = os.path.join(train_folder, image_id + ".jpg")

        image = Image.open(image_path)



        # get all bboxes for given image in [xmin, ymin, width, height]

        bboxes = [literal_eval(box) for box in df[df['image_id'] == image_id]['bbox']]



        # visualize them

        draw = ImageDraw.Draw(image)

        for bbox in bboxes:    

            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)



        plt.figure(figsize = (15,15))

        plt.imshow(image)

        plt.show()





unique_images = df['image_id'].unique()

show_images(unique_images)
#Credits to : https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63044



def castF(x):

    return K.cast(x, K.floatx())



def castB(x):

    return K.cast(x, bool)



def iou_loss_core(true,pred):  #this can be used as a loss if you make it negative

    intersection = true * pred

    notTrue = 1 - true

    union = true + (notTrue * pred)



    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())



def competitionMetric2(true, pred):



    tresholds = [0.5 + (i * 0.05)  for i in range(5)]



    #flattened images (batch, pixels)

    true = K.batch_flatten(true)

    pred = K.batch_flatten(pred)

    pred = castF(K.greater(pred, 0.5))



    #total white pixels - (batch,)

    trueSum = K.sum(true, axis=-1)

    predSum = K.sum(pred, axis=-1)



    #has mask or not per image - (batch,)

    true1 = castF(K.greater(trueSum, 1))    

    pred1 = castF(K.greater(predSum, 1))



    #to get images that have mask in both true and pred

    truePositiveMask = castB(true1 * pred1)



    #separating only the possible true positives to check iou

    testTrue = tf.boolean_mask(true, truePositiveMask)

    testPred = tf.boolean_mask(pred, truePositiveMask)



    #getting iou and threshold comparisons

    iou = iou_loss_core(testTrue,testPred) 

    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]



    #mean of thressholds for true positives and total sum

    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)

    truePositives = K.sum(truePositives)



    #to get images that don't have mask in both true and pred

    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1

    trueNegatives = K.sum(trueNegatives) 



    return (truePositives + trueNegatives) / castF(K.shape(true)[0])
#### Build U-Net model

    

inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

s = Lambda(lambda x: x / 255) (inputs)  # rescale inputs



c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)

c1 = BatchNormalization()(c1)

c1 = Dropout(0.1) (c1)

c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)

c1 = BatchNormalization()(c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)

c2 = BatchNormalization()(c2)

c2 = Dropout(0.1) (c2)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)

c2 = BatchNormalization()(c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)

c3 = BatchNormalization()(c3)

c3 = Dropout(0.2) (c3)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)

c3 = BatchNormalization()(c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)

c4 = BatchNormalization()(c4)

c4 = Dropout(0.2) (c4)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)

c4 = BatchNormalization()(c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)

c5 = Dropout(0.3) (c5)

c5 = BatchNormalization()(c5)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)

c5 = BatchNormalization()(c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)

c6 = BatchNormalization()(c6)

c6 = Dropout(0.2) (c6)

c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)

c6 = BatchNormalization()(c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)

c7 = BatchNormalization()(c7)

c7 = Dropout(0.2) (c7)

c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)

c7 = BatchNormalization()(c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)

c8 = BatchNormalization()(c8)

c8 = Dropout(0.1) (c8)

c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

c8 = BatchNormalization()(c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1])

c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)

c9 = BatchNormalization()(c9)

c9 = Dropout(0.1) (c9)

c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

c9 = BatchNormalization()(c9)



# p5 = Conv2DTranspose(16, (8, 8), strides=(16,16), padding = 'same') (c5)

# u10 = concatenate([p5, c9], axis=3)

# c10 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u10)

# c10 = Dropout(0.1) (c10)

# c10 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c10)



outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)





# instantiating the model in the strategy scope creates the model on the TPU

print("build model")

model = Model(inputs=[inputs], outputs=[outputs])



print("Compiling model")

model.compile(optimizer='adam', 

              loss='binary_crossentropy',

              metrics=[competitionMetric2])



model.summary()
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True)
#data augmentation/ generation

data_gen_args = dict(#featurewise_center=True,

                 #featurewise_std_normalization=True,

                      #samplewise_center=True,

                    #samplewise_std_normalization=False,

                    # rotation_range=10,

                    #shear_range = 10,

                 #zca_whitening=True,

                 #brightness_range=(0.4, 0.6),

                 channel_shift_range = 15,

                 width_shift_range=0.2,

                 height_shift_range=0.2,

                 zoom_range=0.2,

                 horizontal_flip=True,

                 vertical_flip=True,

                validation_split=0.2

                )



#val_idx = np.random.randint(low=0,high=len(train_ids),size=(1,int(0.25*len(train_ids))))



image_datagen = ImageDataGenerator(**data_gen_args,)

mask_datagen = ImageDataGenerator(**data_gen_args)



# val_image_datagen = ImageDataGenerator()

# val_mask_datagen = ImageDataGenerator()

# Provide the same seed and keyword arguments to the fit and flow methods

#seed = 10

#image_datagen.fit(X_train, augment=True#, seed=seed)

#mask_datagen.fit(Y_train, augment=True#, seed=seed)

train_image_generator = image_datagen.flow(

    X_train,

    batch_size=8,

    seed=42,

    subset='training'

)

train_mask_generator = mask_datagen.flow(

    Y_train,

    batch_size=8,

    seed=42,

    subset='training'

)



val_image_generator = image_datagen.flow(

    X_train,

    batch_size=8,

    seed=42,

    subset='validation'

)

val_mask_generator = mask_datagen.flow(

    Y_train,

    batch_size=8,

    seed=42,

    subset='validation'

)

# combine generators into one which yields image and masks

train_generator = (pair for pair in zip(train_image_generator, train_mask_generator))

val_generator = (pair for pair in zip(val_image_generator, val_mask_generator))
# # Fit model

# earlystop = EarlyStopping(patience=10, verbose=1, restore_best_weights=True)



# model.fit(x=X_train,y=Y_train,

#             validation_split=0.20,

#             batch_size=8, 

#             #steps_per_epoch=6*len(X_train)/8,

#             epochs=20, 

#             callbacks=[earlystop]

#          )
earlystop = EarlyStopping(patience=10, verbose=1, restore_best_weights=True)



model.fit_generator(train_generator,

                    validation_data=val_generator,

                    #validation_split=0.10,

                    #batch_size=4, 

                    steps_per_epoch= train_image_generator.x.shape[0]/train_image_generator.batch_size,

                    epochs=100,

                    validation_steps= val_image_generator.x.shape[0]/val_image_generator.batch_size,

#                     callbacks=[earlystop]

                   )
THRESH = 0.6



preds = model.predict(X_test)[:, :, :, 0]

masked_preds = preds > THRESH

preds.shape
n_rows = 3

f, ax = plt.subplots(n_rows, 3, figsize=(14, 10))



for j, idx in enumerate([4,5,6]):

    for k, kind in enumerate(['original', 'pred', 'masked_pred']):

        if kind == 'original':

            img = X_test[idx]

        elif kind == 'pred':

            img = preds[idx]

        elif kind == 'masked_pred':

            masked_pred = preds[idx] > THRESH

            img = masked_pred

        ax[j, k].imshow(img)



plt.tight_layout()
def get_params_from_bbox(coords, scaling_factor=SC_FACTOR):

    xmin, ymin = coords[1] * scaling_factor, coords[0] * scaling_factor

    w = (coords[3] - coords[1]) * scaling_factor

    h = (coords[2] - coords[0]) * scaling_factor

    

    return xmin, ymin, w, h
# Allows to extract bounding boxes from binary masks

bboxes = list()



for j in range(masked_preds.shape[0]):

    label_j = label(masked_preds[j, :, :]) 

    props = regionprops(label_j,intensity_image=preds[j,:,:])   # that's were the job is done

    bboxes.append(props)
bboxes[0][0].bbox
np.max(bboxes[0][0].intensity_image)
# Here we format the bboxes into the required format

output = dict()

significant_scores = list()

significant_bboxes = list()



for i in range(masked_preds.shape[0]):

    

    bboxes_processed = [get_params_from_bbox(bb.bbox, scaling_factor=SC_FACTOR) for bb in bboxes[i]]

    scores = [np.max(bb.intensity_image) for bb in bboxes[i]]

    

    bbareas = [bb.bbox_area*SC_FACTOR*SC_FACTOR for bb in bboxes[i]]

    

#     df_test = pd.DataFrame({'bboxe_processed':bboxes_processed,

#                             'scores':scores,

#                             'bbares':bbareas})

#     print(df_test.shape)

    

    significant_scores.append([score for score,area in zip(scores,bbareas) if ((score > 0.5) & (area > 100))])

    significant_bboxes.append([' '.join(map(str, bb_m)) for bb_m,score,area in zip(bboxes_processed,scores,bbareas) if ((score > 0.5) & (area > 100))])

    

    print("There are " + str(len(significant_scores[i])) + " Scores in "+ test_ids[i])

    print("There are " + str(len(significant_bboxes[i])) + " BBoxes in "+ test_ids[i])

    

    assert(len(significant_scores[i]) == len(significant_bboxes[i]))

    

    

    

    

    formated_boxes = [str(score) + ' ' + bb_m for score,bb_m in zip(significant_scores[i],significant_bboxes[i])]

    

#     formated_boxes = formated_boxes[i > 100 for i in bbareas]

    

    output[str.split(test_ids[i],'.')[0]] = " ".join(formated_boxes)
sub  = pd.DataFrame()

sub['image_id'] = output.keys()

sub["PredictionString"] = output.values()

sub
for image_id in sub['image_id'][0:10]:



        image_path = os.path.join(test_folder, image_id + ".jpg")

        image = Image.open(image_path)



        # get all bboxes for given image in [xmin, ymin, width, height]

        bboxes = significant_bboxes[pd.Index(sub['image_id']).get_loc(image_id)]



        # visualize them

        draw = ImageDraw.Draw(image)

        for bbox in bboxes:  

            bbox = tuple(map(int ,bbox.split(' ')))

            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=5)



        plt.figure(figsize = (5,5))

        plt.imshow(image)

        plt.show()
sub.to_csv('submission.csv', index=False)