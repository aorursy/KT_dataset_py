# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# !pip uninstall imgaug

!pip install --upgrade albumentations

!pip install git+https://github.com/aleju/imgaug.git

# !pip install --upgrade imgaug
import os

import glob 



import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as img



import imageio

import imgaug as ia

from imgaug import augmenters as iaa

import cv2



from keras.preprocessing import image

from keras.models import Model, load_model

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, LeakyReLU

from keras.layers import MaxPooling2D, Dropout, UpSampling2D

from keras import regularizers

import keras.backend as K



%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 5.0) # set default size of plots
import zipfile



# Will unzip the files so that you can see them..

zipfiles = ['train','test','train_cleaned', 'sampleSubmission.csv']



for each_zip in zipfiles:

    with zipfile.ZipFile("../input/denoising-dirty-documents/"+each_zip+".zip","r") as z:

        z.extractall(".")

bsb = img.imread('/kaggle/working/train/216.png')

# test = img.imread('../kaggle/working/test/1.png')

plt.imshow(bsb, cmap=plt.cm.gray)
target_width = 540

target_height = 420
def load_image_from_dir(img_path):

    file_list = glob.glob(img_path+'/*.png')

    file_list.sort()

    img_list = np.empty((len(file_list), target_height, target_width, 1))

    for i, fig in enumerate(file_list):

        img = image.load_img(fig, color_mode='grayscale', target_size=(target_height, target_width))

        img_array = image.img_to_array(img).astype('float32')

        img_array = img_array / 255.0

        img_list[i] = img_array

    

    return img_list



def train_test_split(data,random_seed=55,split=0.75):

    set_rdm = np.random.RandomState(seed=random_seed)

    dsize = len(data)

    ind = set_rdm.choice(dsize,dsize,replace=False)

    train_ind = ind[:int(0.75*dsize)]

    val_ind = ind[int(0.75*dsize):]

    return data[train_ind],data[val_ind]



def augment_pipeline(pipeline, images, seed=5):

    ia.seed(seed)

    processed_images = images.copy()

    for step in pipeline:

        temp = np.array(step.augment_images(images))

        processed_images = np.append(processed_images, temp, axis=0)

    return(processed_images)
full_train = load_image_from_dir('/kaggle/working/train')

full_target = load_image_from_dir('/kaggle/working/train_cleaned')

# test = load_image_from_dir('/kaggle/working/test')
rotate90 = iaa.Rot90(1, name="Rotate90") # rotate image 90 degrees

rotate180 = iaa.Rot90(2, name="Rotate180") # rotate image 180 degrees

rotate270 = iaa.Rot90(3, name="Rotate270") # rotate image 270 degrees

random_rotate = iaa.Rot90((1,3), name="RandomRotate") # randomly rotate image from 90,180,270 degrees

perc_transform = iaa.PerspectiveTransform(scale=(0.02, 0.1), name="Perc_transform") # Skews and transform images without black bg

rotate10 = iaa.Affine(rotate=(10), name="Rotate10") # rotate image 10 degrees

rotate10r = iaa.Affine(rotate=(-10), name="Rotate10r") # rotate image 30 degrees in reverse

crop = iaa.Crop(px=(5, 32), name="RandomCrop") # Crop between 5 to 32 pixels

hflip = iaa.Fliplr(1, name="Flip_horizontal") # horizontal flips for 100% of images

vflip = iaa.Flipud(1, name="Filp_vertical") # vertical flips for 100% of images

gblur = iaa.GaussianBlur(sigma=(1, 1.5), name="Gaussian_blur") # gaussian blur images with a sigma of 1.0 to 1.5

motionblur = iaa.MotionBlur(8, name="Motion_blur") # motion blur images with a kernel size 8



seq_rp = iaa.Sequential([

    iaa.Rot90((1,3)), # randomly rotate image from 90,180,270 degrees

    iaa.PerspectiveTransform(scale=(0.02, 0.1)) # Skews and transform images without black bg

], name="Combination1")



seq_cfg = iaa.Sequential([

    iaa.Crop(px=(5, 32)), # crop images from each side by 5 to 32px (randomly chosen)

    iaa.Fliplr(0.5), # horizontally flip 50% of the images

    iaa.GaussianBlur(sigma=(0, 1.5)) # blur images with a sigma of 0 to 1.5

], name="Combination2")



seq_fm = iaa.Sequential([

    iaa.Flipud(1), # vertical flips all the images

    iaa.MotionBlur(k=6) # motion blur images with a kernel size 6

], name="Combination3")
def build_autoencoder(optimizer):

    K.clear_session()

    ### Multi layer auto encoder with LeakyRelu and Normalization

    input_layer = Input(shape=(None,None,1))



    # encoder

    e = Conv2D(32, (3, 3), padding='same')(input_layer)

    e = LeakyReLU(alpha=0.3)(e)

    e = BatchNormalization()(e)

    e = Conv2D(64, (3, 3), padding='same')(e)

    e = LeakyReLU(alpha=0.3)(e)

    e = BatchNormalization()(e)

    e = Conv2D(64, (3, 3), padding='same')(e)

    e = LeakyReLU(alpha=0.3)(e)

    e = MaxPooling2D((2, 2), padding='same')(e)



    # decoder

    d = Conv2D(64, (3, 3), padding='same')(e)

    d = LeakyReLU(alpha=0.3)(d)

    d = BatchNormalization()(d)



    d = Conv2D(64, (3, 3), padding='same')(d)

    d = LeakyReLU(alpha=0.3)(d)

    # e = BatchNormalization()(e)

    d = UpSampling2D((2, 2))(d)

    d = Conv2D(32, (3, 3), padding='same')(d)

    d = LeakyReLU(alpha=0.2)(d)

    # d = Conv2D(128, (3, 3), padding='same')(d)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)



    model = Model(input_layer,output_layer)

    model.compile(loss='mse', optimizer=optimizer)

    

    return model
def augment_testing(pipeline, train_images, target_images, seed=6):

    results = []

    ia.seed(seed)

    train, val = train_test_split(train_images, random_seed=seed, split=0.8)

    target_train, target_val = train_test_split(target_images, random_seed=seed, split=0.8)





    optimizer = Adam(lr=1e-4)

    AEmodel = build_autoencoder(optimizer)

    AEmodel.fit(train, target_train, batch_size=8, epochs=20, verbose=0)

    val_loss = AEmodel.evaluate(val, target_val, verbose=0)

    train_loss = AEmodel.evaluate(train, target_train, verbose=0)

    results.append({'Augmentation':'Original','Val_mse':val_loss,'Train_mse':train_loss})

    

    processed_train = train.copy()

    processed_target = target_train.copy()

    for step in pipeline:



        temp1 = np.array(step.augment_images(train))

        processed_train = np.append(processed_train, temp1, axis=0)

        temp2 = np.array(step.augment_images(target_train))

        processed_target = np.append(processed_target, temp2, axis=0)

        

#         print(processed_train.shape)



        AEmodel = build_autoencoder(optimizer)

        AEmodel.fit(processed_train, processed_target,

                    validation_data = (val, target_val), batch_size=8, epochs=20, verbose=0)



        val_loss = AEmodel.evaluate(val, target_val, verbose=0)

        train_loss = AEmodel.evaluate(processed_train, processed_target, verbose=0)

        results.append({'Augmentation':step.name,'Val_mse':val_loss,'Train_mse':train_loss})



    return results
# pipeline = []

# pipeline.append(rotate90)

# pipeline.append(rotate180)

# pipeline.append(rotate270)

# # pipeline.append(random_rotate)

# pipeline.append(perc_transform)

# # pipeline.append(rotate10)

# # pipeline.append(rotate10r)

# pipeline.append(crop)

# pipeline.append(hflip)

# pipeline.append(vflip)

# pipeline.append(gblur)

# pipeline.append(motionblur)

# # pipeline.append(seq_rp)

# pipeline.append(seq_cfg)

# pipeline.append(seq_fm)
pipeline = []

pipeline.append(perc_transform)

pipeline.append(motionblur)

pipeline.append(seq_rp)

pipeline.append(seq_cfg)

pipeline.append(rotate180)

pipeline.append(random_rotate)

# pipeline.append(rotate10)

# pipeline.append(rotate10r)

# pipeline.append(crop)

pipeline.append(hflip)

pipeline.append(vflip)

pipeline.append(gblur)

pipeline.append(seq_fm)

pipeline.append(rotate90)

pipeline.append(rotate270)
%%time

results = augment_testing(pipeline, full_train, full_target)
resultsdf = pd.DataFrame(results)

resultsdf.to_csv("Progressive Augment results.csv")

resultsdf.head(30)
plt.plot(range(resultsdf.shape[0]), resultsdf['Val_mse'])

plt.plot(range(resultsdf.shape[0]), resultsdf['Train_mse'])

plt.title('Progressive Augmentation vs Loss')

plt.ylabel('MSE')

plt.xlabel('Number of Augmentations')

plt.xticks(range(resultsdf.shape[0]), range(resultsdf.shape[0]))

plt.legend(['Train','Val'], loc='upper left')

plt.show()
# %%time

# processed_train = augment_pipeline(pipeline, full_train.reshape(-1,target_height,target_width))

# processed_target = augment_pipeline(pipeline, full_target.reshape(-1,target_height,target_width))



# processed_train = processed_train.reshape(-1,target_height,target_width,1)

# processed_target = processed_target.reshape(-1,target_height,target_width,1)



# processed_train.shape
# train, val = train_test_split(processed_train, random_seed=9, split=0.8)

# target_train, target_val = train_test_split(processed_target, random_seed=9, split=0.8)
# train, val = train_test_split(full_train, random_seed=9, split=0.8)

# target_train, target_val = train_test_split(full_target, random_seed=9, split=0.8)
# %%time

# pre_train, pre_val = train_test_split(full_train, random_seed=9, split=0.7)

# pre_target_train, pre_target_val = train_test_split(full_target, random_seed=9, split=0.7)



# print(pre_train.shape,pre_val.shape)



# train = augment_pipeline(pipeline, pre_train.reshape(-1,target_height,target_width), seed=10)

# target_train = augment_pipeline(pipeline, pre_target_train.reshape(-1,target_height,target_width), seed=10)



# train = train.reshape(-1,target_height,target_width,1)

# target_train = target_train.reshape(-1,target_height,target_width,1)



# val_pipeline = pipeline + [seq_fm]



# val = augment_pipeline(val_pipeline, pre_val.reshape(-1,target_height,target_width))

# target_val = augment_pipeline(val_pipeline, pre_target_val.reshape(-1,target_height,target_width))



# val = val.reshape(-1,target_height,target_width,1)

# target_val = target_val.reshape(-1,target_height,target_width,1)



# print("Shape of Train set:",train.shape)

# print("Shape of Validation set:",val.shape)
# optimizer = Adam(lr=9e-4, decay=1e-5)

# # optimizer = Adam(lr=1e-4, decay=7e-6)

# # AEmodel = Model(input_layer,output_layer)

# AEmodel = build_autoencoder(optimizer)

# # AEmodel.compile(loss='mse', optimizer=optimizer)

# AEmodel.summary()
# early_stopping = EarlyStopping(monitor='val_loss',

#                                min_delta=0,

#                                patience=30,

#                                verbose=1, 

#                                mode='auto')



# checkpoint1 = ModelCheckpoint('best_val_loss.h5',

#                              monitor='val_loss',

#                              save_best_only=True)



# checkpoint2 = ModelCheckpoint('best_loss.h5',

#                              monitor='loss',

#                              save_best_only=True)
# history = AEmodel.fit(processed_train, processed_target,

#                       batch_size=16,

#                       epochs=300,

# #                       validation_split=0.2,

#                       callbacks=[checkpoint2])

# #                                      validation_data=(val, target_val))
# plt.plot(history.history['loss'])

# plt.plot(history.history['val_loss'])

# plt.title('Model loss')

# plt.ylabel('Loss')

# plt.xlabel('Epoch')

# plt.legend(['Train','Val'], loc='upper left')

# plt.show()
# AEmodel.save('AutoEncoderModelFull.h5')
# # full_model_preds = AEmodel.predict(test)

# full_train_preds = AEmodel.predict(full_train)



# AEmodel.load_weights('best_loss.h5')

# AEmodel.compile(loss='mse', optimizer=optimizer)

# # preds = AEmodel.predict(test)

# train_preds = AEmodel.predict(full_train)
# AEmodel.evaluate(full_train, full_target)
# AEmodel.save('AutoEncoderModelBestLoss.h5')
# bsb = img.imread('https://github.com/sampath9dasari/GSU/raw/master/denoise_test.png')

# # test = img.imread('../kaggle/working/test/1.png')

# plt.imshow(bsb, cmap=plt.cm.gray)
# # ii = cv2.imread("https://github.com/sampath9dasari/GSU/raw/master/denoise_test.png")

# gray_image = cv2.cvtColor(bsb, cv2.COLOR_BGR2GRAY)

# # print(gray_image)

# plt.imshow(gray_image,cmap=plt.cm.gray)

# plt.show()
# gpred = AEmodel.predict(gray_image.reshape(1,1599,1200,1))
# fig, ax = plt.subplots(1,2,figsize=(22,12))

# ax[0].imshow(gray_image, cmap=plt.cm.gray)

# ax[1].imshow(gpred.reshape(1600,1200), cmap=plt.cm.gray)
# fig, ax = plt.subplots(3,2,figsize=(22,16))

# ax[0][0].imshow(full_train[42].reshape(target_height,target_width), cmap=plt.cm.gray)

# ax[0][1].imshow(full_target[42].reshape(target_height,target_width), cmap=plt.cm.gray)

# ax[1][0].imshow(full_train_preds[42].reshape(target_height,target_width), cmap=plt.cm.gray)

# ax[1][1].imshow(train_preds[42].reshape(target_height,target_width), cmap=plt.cm.gray)

# reshape = cv2.resize(full_train_preds[42],(target_width,258))

# ax[2][0].imshow(reshape.reshape(258,target_width), cmap=plt.cm.gray)

# reshape = cv2.resize(train_preds[42],(target_width,258))

# ax[2][1].imshow(reshape.reshape(258,target_width), cmap=plt.cm.gray)
# %%time

# ids = []

# vals = []

# file_list = glob.glob('/kaggle/working/test/*.png')

# file_list.sort()

# for i, f in enumerate(file_list):

#     file = os.path.basename(f)

#     imgid = int(file[:-4])

#     test_img = cv2.imread(f, 0)

#     img_shape = test_img.shape

# #     print('processing: {}'.format(imgid))

# #     print(img_shape)

#     preds_reshaped = cv2.resize(preds[i], (img_shape[1], img_shape[0]))

#     for r in range(img_shape[0]):

#         for c in range(img_shape[1]):

#             ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))

#             vals.append(preds_reshaped[r, c])



# print('Writing to csv file')

# pd.DataFrame({'id': ids, 'value': vals}).to_csv('submission.csv', index=False)
# #Load and Scale test images into one big list.

# file_list = glob.glob('/kaggle/working/test/*.png')

# file_list.sort()

# test_size = len(file_list)



# #initailize data arrays.

# img_ids = []

# test = []



# #read data

# for i, img_dir in enumerate(file_list):

#     file = os.path.basename(img_dir)

#     imgid = int(file[:-4])

#     img_ids.append(imgid)

#     img_pixels = image.load_img(img_dir, color_mode='grayscale')

#     w, h = img_pixels.size

#     test.append(np.array(img_pixels).reshape(1, h, w, 1) / 255.)

    

# print('Test sample shape: ', test[0].shape)

# print('Test sample dtype: ', test[0].dtype)
# #Predict test images one by one and store them into a list.

# test_preds = []

# for img in test:

#     test_preds.append(AEmodel.predict(img)[0, :, :, 0])
# fig, ax = plt.subplots(1,2,figsize=(22,12))

# ax[0].imshow(test[45].reshape(test[45].shape[1],test[45].shape[2]), cmap=plt.cm.gray)

# ax[1].imshow(test_preds[45].reshape(test[45].shape[1],test[45].shape[2]), cmap=plt.cm.gray)
# fig, ax = plt.subplots(1,2,figsize=(16,8))

# ax[0].imshow(test[42].reshape(test[42].shape[1],test[42].shape[2]), cmap=plt.cm.gray)

# ax[1].imshow(test_preds[42].reshape(test[42].shape[1],test[42].shape[2]), cmap=plt.cm.gray)
# # First column will be raw data, second column will be the corresponding cleaned images.

# f, ax = plt.subplots(2,3, figsize=(20,10))

# f.subplots_adjust(hspace = .1, wspace=.05)

# for i, (img, lbl) in enumerate(zip(test[:3], test_preds[:3])):

#     ax[0, i].imshow(img[0,:,:,0], cmap='gray')

#     ax[0, i].title.set_text('Original Image')

#     ax[0, i].axis('off')



#     ax[1, i].imshow(lbl, cmap='gray')

#     ax[1, i].title.set_text('Cleaned Image')

#     ax[1, i].axis('off')

# plt.show()
# #Flatten the 'test_preds' list into 1-d list for submission.

# submit_vector = []

# submit_ids = []

# for imgid, img in zip(img_ids,test_preds):

#     h, w = img.shape

#     for c in range(w):

#         for r in range(h):

#             submit_ids.append(str(imgid)+'_'+str(r + 1)+'_'+str(c + 1))

#             submit_vector.append(img[r,c])

# print(len(submit_vector))
# len(submit_vector)
# sample_csv = pd.read_csv('/kaggle/working/sampleSubmission.csv')

# sample_csv.head(10)
# id_col = sample_csv['id']

# value_col = pd.Series(submit_vector, name='value')

# submission = pd.concat([id_col, value_col], axis=1)

# submission.head(10)
# submission.to_csv('submission.csv',index = False)
# import shutil



# shutil.rmtree('/kaggle/working/train')

# shutil.rmtree('/kaggle/working/test')

# shutil.rmtree('/kaggle/working/train_cleaned')