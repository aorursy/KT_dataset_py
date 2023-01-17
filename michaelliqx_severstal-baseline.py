# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # tensorflow

from tensorflow import reduce_sum

from tensorflow.keras.backend import pow

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten

from tensorflow.keras.losses import binary_crossentropy

from sklearn.model_selection import train_test_split

from tensorflow.keras.activations import relu

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.initializers import he_normal

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.applications.resnet50 import ResNet50

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt # plt显示图片

import matplotlib.image as mpimg # mpimg 用于读取图片

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/input'):

    if 'test_images' in dirname:

        for filename in filenames:

            file = os.path.join(dirname,filename)

            print(file)

            img = mpimg.imread(file)

            print(img.shape)

            plt.imshow(img)

            break

    continue
# Kernel Configurations

make_submission = False # used to turn off lengthy model analysis so a submission version doesn't run into memory error

load_pretrained_model = True # load a pre-trained model

save_model = True # save the model after training

train_dir = '../input/severstal-steel-defect-detection/' # directory of training images

# pretrained_model_path = '../input/severstal-pretrained-model/ResUNetSteel_z.h5' # path of pretrained model

model_save_path = './ResUNetSteel_w800e50_z.h5' # path of model to save

train_image_dir = os.path.join(train_dir, 'train_images') # 
# network configuration parameters

# original image is 1600x256, so we will resize it

img_w = 512 # resized weidth

img_h = 256 # resized height

batch_size = 16

epochs = 10

# batch size for training unet

k_size = 3 # kernel size 3x3

val_size = 0.2 # split of training set between train and validation set

# we will repeat the images with lower samples to make the training process more fair

repeat = False

# only valid if repeat is True

class_1_repeat = 1 # repeat class 1 examples x times

class_2_repeat = 1

class_3_repeat = 1

class_4_repeat = 1
# load full data and label no mask as -1

train_df = pd.read_csv(os.path.join(train_dir, 'train.csv')).fillna(-1)
print(train_df)
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

train_df['ClassId_EncodedPixels'] = train_df.apply(lambda row: (row['ClassId'], row['EncodedPixels']), axis = 1)

grouped_EncodedPixels = train_df.groupby('ImageId')['ClassId_EncodedPixels'].apply(list)
print('ClassId_EncodedPixels:',train_df.ClassId_EncodedPixels)

print('grouped_EncodedPixels:',grouped_EncodedPixels)

print('\n')

print(train_df['ImageId'].size)

train_img_list = np.unique(train_df['ImageId'])

print(train_img_list)
# #  use imggenerator to process the images

# dg_args = dict(featurewise_center = False, 

#                   samplewise_center = False,

#                   rotation_range = 45, 

#                   width_shift_range = 0.1, 

#                   height_shift_range = 0.1, 

#                   shear_range = 0.01,

#                   zoom_range = [0.9, 1.25],  

#                   horizontal_flip = True, 

#                   vertical_flip = True,

#                   fill_mode = 'reflect',

#                    data_format = 'channels_last')

# valid_args = dict(

#                     fill_mode = 'reflect',

#                    data_format = 'channels_last')



# core_idg = ImageDataGenerator(**dg_args)

# valid_idg = ImageDataGenerator(**valid_args)





# # def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):

# #     base_dir = os.path.dirname(in_df[path_col].values[0])

# #     print('## Ignore next message from keras, values are replaced anyways')

# #     df_gen = img_data_gen.flow_from_directory(base_dir, 

# #                                      class_mode = 'sparse',

# #                                     **dflow_args)

# #     df_gen.filenames = in_df[path_col].values

# #     df_gen.classes = np.stack(in_df[y_col].values)

# #     df_gen.samples = in_df.shape[0]

# #     df_gen.n = in_df.shape[0]

# #     df_gen._set_index_array()

# #     df_gen.directory = '' # since we have the full path

# #     print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

# #     return df_gen



# # train_gen = flow_from_dataframe(core_idg, train_df, 

# #                              path_col = 'path',

# #                             y_col = 'has_ship_vec', 

# #                             target_size = (256,800),

# #                              color_mode = 'rgb',

# #                             batch_size = 30)



# # # used a fixed dataset for evaluating the algorithm

# # valid_x, valid_y = next(flow_from_dataframe(valid_idg, 

# #                                valid_df, 

# #                              path_col = 'path',

# #                             y_col = 'has_ship_vec', 

# #                             target_size = (256,800),

# #                              color_mode = 'rgb',

# #                             batch_size = 1000)) # one big batch

# # print(valid_x.shape, valid_y.shape)



class DataGen(tf.keras.utils.Sequence):

    def __init__(self, list_ids, labels, image_dir, batch_size=32,

                 img_h=256, img_w=512, shuffle=True):

        self.list_ids = list_ids

        self.labels = labels

        self.image_dir = image_dir

        self.batch_size = batch_size

        self.img_h = img_h

        self.img_w = img_w

        self.shuffle = shuffle

        self.on_epoch_end()

    

    def __len__(self):

        # 代表每个epoch有多少个batch

        return int(np.floor(len(self.list_ids)) / self.batch_size)

    

    def on_epoch_end(self):

        'update ended after each epoch'

        # 每个epoch之后重新打乱index

        self.indexes = np.arange(len(self.list_ids))

        if self.shuffle:

            np.random.shuffle(self.indexes)

        

    

    def __data_generation(self, list_ids_temp):

        'generate data containing batch_size samples'

        X = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        y = np.empty((self.batch_size, self.img_h, self.img_w, 4))

        

        for idx, id in enumerate(list_ids_temp):

            # 对每一张图片

            file_path =  os.path.join(self.image_dir, id)

            image = cv2.imread(file_path, 0)

            image_resized = cv2.resize(image, (self.img_w, self.img_h))

            image_resized = cv2.GaussianBlur(image_resized,(5,5),0)

            image_resized = np.array(image_resized, dtype=np.float32)

            '进行标准化（可能会过拟合，之后考虑一下用整体的mean和方差来进行标准化'

            image_resized /= 255

            image_resized -= image_resized.mean()

            image_resized /= image_resized.std()

            

            

            mask = np.empty((img_h, img_w, 4))

            

            for idm, image_class in enumerate(['1','2','3','4']):

                rle = self.labels.get(id + '_' + image_class)

                # if there is no mask create empty mask

                if rle is None:

                    class_mask = np.zeros((1600, 256))

                else:

                    class_mask = rle_to_mask(rle, width=1600, height=256)

             

                class_mask_resized = cv2.resize(class_mask, (self.img_w, self.img_h))

                mask[...,idm] = class_mask_resized

            

            X[idx,] = np.expand_dims(image_resized, axis=2)

            y[idx,] = mask

        

        # normalize Y

        y = (y > 0).astype(int)

        

        return X, y



    def __getitem__(self, index):

        # 生成一个batch的数据

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # get list of IDs

        list_ids_temp = [self.list_ids[k] for k in indexes]

        # generate data

        X, y = self.__data_generation(list_ids_temp)

        # return data 

        return X, y

    

    

def rle_to_mask(rle_string,height,width):

    '''

    convert RLE(run length encoding) string to numpy array



    Parameters: 

    rleString (str): Description of arg1 

    height (int): height of the mask

    width (int): width of the mask 



    Returns: 

    numpy.array: numpy array of the mask

    '''

    rows, cols = height, width

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]

        rlePairs = np.array(rleNumbers).reshape(-1,2)

        img = np.zeros(rows*cols,dtype=np.uint8)

        for index,length in rlePairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img

    

def mask_to_rle(mask):

    '''

    Convert a mask into RLE

    

    Parameters: 

    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background



    Returns: 

    sring: run length encoding 

    '''

    pixels= mask.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
'生成一个包含所有mask的字典'

masks = {}

for index, row in train_df[train_df['EncodedPixels']!=-1].iterrows():

    masks[row['ImageId_ClassId']] = row['EncodedPixels']
# repeat low represented samples more frequently to balance our dataset



class_1_img_id = train_df[(train_df['EncodedPixels']!=-1) & (train_df['ClassId']=='1')]['ImageId'].values

class_1_img_id = np.repeat(class_1_img_id, class_1_repeat)

class_2_img_id = train_df[(train_df['EncodedPixels']!=-1) & (train_df['ClassId']=='2')]['ImageId'].values

class_2_img_id = np.repeat(class_2_img_id, class_2_repeat)

class_3_img_id = train_df[(train_df['EncodedPixels']!=-1) & (train_df['ClassId']=='3')]['ImageId'].values

class_3_img_id = np.repeat(class_3_img_id, class_3_repeat)

class_4_img_id = train_df[(train_df['EncodedPixels']!=-1) & (train_df['ClassId']=='4')]['ImageId'].values

class_4_img_id = np.repeat(class_4_img_id, class_4_repeat)

train_image_ids = np.concatenate([class_1_img_id, class_2_img_id, class_3_img_id, class_4_img_id])

'划分训练集和验证集'

X_train, X_val = train_test_split(train_image_ids, test_size=val_size, random_state=2)
params = {'img_h': img_h,

          'img_w': img_w,

          'image_dir': train_image_dir,

          'batch_size': batch_size,

          'shuffle': True }



# Get Generators

training_generator = DataGen(X_train, masks, **params)

validation_generator = DataGen(X_val, masks, **params)
x, y = training_generator.__getitem__(0)

print(x.shape, y.shape)
# visualize steel image with four classes of faults in seperate columns

def viz_steel_img_mask(img, masks):

    img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(20,10))

    cmaps = ["Reds", "Blues", "Greens", "Purples"]

    for idx, mask in enumerate(masks):

        ax[idx].imshow(img)

        ax[idx].imshow(mask, alpha=0.3, cmap=cmaps[idx])
for idx in range(0,batch_size):

    if y[idx].sum() > 0:

        img = x[idx]

        masks_temp = [y[idx][...,i] for i in range(0,4)]

        viz_steel_img_mask(img, masks_temp)
def bn_act(x, act=True):

    'batch normalization layer with an optinal activation layer'

    x = tf.keras.layers.BatchNormalization()(x)

    if act == True:

        x = tf.keras.layers.Activation('relu')(x)

    return x
def conv_block(x, filters, kernel_size=3, padding='same', strides=1):

    'convolutional layer which always uses the batch normalization layer'

    conv = bn_act(x)

    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer="he_normal")(conv)

    return conv
def stem(x, filters, kernel_size=3, padding='same', strides=1):

    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer="he_normal")(x)

    conv = conv_block(conv, filters, kernel_size, padding, strides)

    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides, kernel_initializer="he_normal")(x)

    shortcut = bn_act(shortcut, act=False)

    output = Add()([conv, shortcut])

    return output
def residual_block(x, filters, kernel_size=3, padding='same', strides=1):

    res = conv_block(x, filters, k_size, padding, strides)

    res = conv_block(res, filters, k_size, padding, 1)

    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer="he_normal")(x)

    shortcut = bn_act(shortcut, act=False)

    output = Add()([shortcut, res])

    return output
def upsample_concat_block(x, xskip):

    u = UpSampling2D((2,2))(x)

    c = Concatenate()([u, xskip])

    return c
def ResUNet(img_h, img_w):

    f = [16, 32, 64, 128, 256, 512]

    inputs = Input((img_h, img_w, 1))

    ## Encoder

    e0 = inputs

    e1 = stem(e0, f[0])

    e2 = residual_block(e1, f[1], strides=2)

    e3 = residual_block(e2, f[2], strides=2)

    e4 = residual_block(e3, f[3], strides=2)

    e5 = residual_block(e4, f[4], strides=2)

    e6 = residual_block(e5, f[5], strides=2)

    

    ## Bridge

    b0 = conv_block(e6, f[5], strides=1)

    b1 = conv_block(b0, f[5], strides=1)

    

    ## Decoder

    u1 = upsample_concat_block(b1, e5)

    d1 = residual_block(u1, f[5])

    

    u2 = upsample_concat_block(d1, e4)

    d2 = residual_block(u2, f[4])

    

    u3 = upsample_concat_block(d2, e3)

    d3 = residual_block(u3, f[3])

    

    u4 = upsample_concat_block(d3, e2)

    d4 = residual_block(u4, f[2])

    

    u5 = upsample_concat_block(d4, e1)

    d5 = residual_block(u5, f[1])

    

    outputs = tf.keras.layers.Conv2D(4, (1, 1), padding="same", activation="sigmoid", kernel_initializer="he_normal")(d5)

    model = Model(inputs, outputs)

    print(model.summary())

    return model
model = ResUNet(img_h,img_w)
# Dice similarity coefficient loss, brought to you by: https://github.com/nabsabraham/focal-tversky-unet

def dsc(y_true, y_pred):

    smooth = 1.

    y_true_f = Flatten()(y_true)

    y_pred_f = Flatten()(y_pred)

    intersection = reduce_sum(y_true_f * y_pred_f)

    score = (2. * intersection + smooth) / (reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth)

    return score



def dice_loss(y_true, y_pred):

    loss = 1 - dsc(y_true, y_pred)

    return loss



def bce_dice_loss(y_true, y_pred):

    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss
# Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet

def tversky(y_true, y_pred, smooth=1e-6):

    y_true_pos = tf.keras.layers.Flatten()(y_true)

    y_pred_pos = tf.keras.layers.Flatten()(y_pred)

    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)

    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))

    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)

    alpha = 0.7

    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)



def tversky_loss(y_true, y_pred):

    return 1 - tversky(y_true,y_pred)



def focal_tversky_loss(y_true,y_pred):

    pt_1 = tversky(y_true, y_pred)

    gamma = 0.75

    return tf.keras.backend.pow((1-pt_1), gamma)
adam = tf.keras.optimizers.Adam(lr = 0.05, epsilon = 0.1)

model.compile(optimizer=adam, loss=focal_tversky_loss, metrics=[tversky])
# checkpoint = ModelCheckpoint("mymodel_1.h5", monitor='tversky', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# early = EarlyStopping(monitor='tversky', min_delta=0, patience=10, verbose=1, mode='auto')



# aug = ImageDataGenerator(rotation_range=20, zoom_range=0.10,rescale=1./255,

#                          width_shift_range=0.1, height_shift_range=0.1, shear_range=0.10,

#                          horizontal_flip=False, fill_mode="nearest")



model.fit_generator(training_generator, validation_data=validation_generator, epochs=20

#                     ,callbacks = [checkpoint],

#                     verbose = 2

                   )

model.save("Severstal_baseline2.h5")
# return tensor in the right shape for prediction 

def get_test_tensor(img_dir, img_h, img_w, channels=1):



    X = np.empty((1, img_h, img_w, channels))

    # Store sample

    image = cv2.imread(img_dir, 0)

    image_resized = cv2.resize(image, (img_w, img_h))

    image_resized = np.array(image_resized, dtype=np.float64)

    # normalize image

    image_resized -= image_resized.mean()

    image_resized /= image_resized.std()

    

    X[0,] = np.expand_dims(image_resized, axis=2)



    return X
# this is an awesome little function to remove small spots in our predictions



from skimage import morphology



def remove_small_regions(img, size):

    """Morphologically removes small (less than size) connected regions of 0s or 1s."""

    img = morphology.remove_small_objects(img, size)

    img = morphology.remove_small_holes(img, size)

    return img
import glob

# get all files using glob

test_files = [f for f in glob.glob('../input/severstal-steel-defect-detection/test_images/' + "*.jpg", recursive=True)]
submission = []



# a function to apply all the processing steps necessery to each of the individual masks

def process_pred_mask(pred_mask):

    

    pred_mask = cv2.resize(pred_mask.astype('float32'),(1600, 256))

    pred_mask = (pred_mask > .5).astype(int)

    pred_mask = remove_small_regions(pred_mask, 0.02 * np.prod(512)) * 255

    pred_mask = mask_to_rle(pred_mask)

    

    return pred_mask



# loop over all the test images

for f in test_files:

    # get test tensor, output is in shape: (1, 256, 512, 3)

    test = get_test_tensor(f, img_h, img_w) 

    # get prediction, output is in shape: (1, 256, 512, 4)

    pred_masks = model.predict(test) 

    # get a list of masks with shape: 256, 512

    pred_masks = [pred_masks[0][...,i] for i in range(0,4)]

    # apply all the processing steps to each of the mask

    pred_masks = [process_pred_mask(pred_mask) for pred_mask in pred_masks]

    # get our image id

    id = f.split('/')[-1]

    # create ImageId_ClassId and get the EncodedPixels for the class ID, and append to our submissions list

    [submission.append((id+'_%s' % (k+1), pred_mask)) for k, pred_mask in enumerate(pred_masks)]
# convert to a csv

submission_df = pd.DataFrame(submission, columns=['ImageId_ClassId', 'EncodedPixels'])

# check out some predictions and see if RLE looks ok

submission_df[ submission_df['EncodedPixels'] != ''].head()
# take a look at our submission 

submission_df.head()
# write it out

submission_df.to_csv('./submission.csv', index=False)
# model.fit_generator(training_generator, validation_data=validation_generator, epochs=50

# #                     ,callbacks = [checkpoint],

# #                     verbose = 2

#                    )

# model.save("./Severstal_baseline1.h5")
# model.save(model_save_path)
# ResUnet = load_model('./MyModel.h5', custom_objects={'focal_tversky_loss':focal_tversky_loss,'tversky':tversky})
# ResUnet_history = ResUnet.fit_generator(

#                             generator=training_generator, 

#                             validation_data=validation_generator,

#                             epochs=15,

#                             verbose=1)

# model.save('./MyModel2.h5')
# ResUnet_history = ResUnet.fit_generator(

#                             generator=training_generator, 

#                             validation_data=validation_generator,

#                             epochs=30,

#                             verbose=1)

# model.save('./MyModel3.h5')