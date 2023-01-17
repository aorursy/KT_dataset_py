import pandas as pd

import numpy as np

from PIL import Image

import gc

import os

from pathlib import Path

import matplotlib.pyplot as plt

import keras

import cv2

from collections import defaultdict

from skimage.data import imread

from sklearn.model_selection import train_test_split

from keras import backend as K





path=Path('../input/understanding_cloud_organization')

os.listdir(path)
### Reading files

train=pd.read_csv(path/'train.csv')

train.shape
train['ImageId']=train['Image_Label'].apply(lambda x : x.split('_')[0])

train['cat']=train['Image_Label'].apply(lambda x : x.split('_')[1])

train[train['EncodedPixels'].notnull()].head()
cat=train[train['EncodedPixels'].notnull()]['cat'].value_counts()

plt.bar(cat.index,cat)

plt.xlabel('category of cloud')

plt.ylabel('number of masked samples')

plt.show()
x1=train[train['EncodedPixels'].notnull()].shape[0]

x2=train[train['EncodedPixels'].isnull()].shape[0]

plt.bar(['has Mask','not Masked'],[x1,x2])
train['has_mask']= ~pd.isna(train['EncodedPixels'])

train['missing']= pd.isna(train['EncodedPixels'])
train_nan=train.groupby('ImageId').agg('sum')

train_nan.columns=['No: of Masks','Missing masks']

train_nan['Missing masks'].hist()



mask_count_df=pd.DataFrame(train_nan)



mask_count_df = train.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('has_mask', ascending=False, inplace=True)

print(mask_count_df.shape)

mask_count_df.head()
train_nan['No: of Masks'].hist()
image_size=defaultdict(int)

image_file=path/'train_images'

for img in image_file.iterdir():

    img=Image.open(img)

    image_size[img.size]+=1



    
image_size
image_size=defaultdict(int)

image_file=path/'test_images'

for img in image_file.iterdir():

    img=Image.open(img)

    image_size[img.size]+=1

    
image_size
no_patterns=0

patterns=0



for i in range(0,len(train),4):

    samples=[x.split('_')[0] for x in train.iloc[i:i+4,0].values]

    if(samples[0]!=samples[1]!=samples[2]!=samples[3]):

        raise ValueError

    labels=train.iloc[i:i+4]['EncodedPixels']

    if labels.isna().all():

        no_patterns+=1

    else:

        patterns+=1

        

    
print('Number of images with patters {} '.format(patterns))

print("Number of images without patters {} ".format(no_patterns))

labels = sorted(list(set(train['Image_Label'].apply(lambda x: x.split('_')[1]))))

print(labels)
def rle_decode(mask,shape=(1400,2100)):

    

    s=mask.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts-=1

    end=starts+lengths

    img=np.zeros(shape[0]*shape[1],dtype=np.uint8)

    for l,m in zip(starts,end):

        img[l:m]=1

    return img.reshape(shape[0],shape[1],order='F')





    
train_nan[train_nan['No: of Masks']==4].iloc[0]
image_name = '00dec6a.jpg'

img = imread(str(path)+'/train_images/' + image_name)



fig, ax = plt.subplots(2, 2, figsize=(15, 10))



for e, label in enumerate(labels):

    axarr = ax.flat[e]

    image_label = image_name + '_' + label

    mask_rle = train.loc[train['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    try: # label might not be there!

        mask = rle_decode(mask_rle)

    except:

        mask = np.zeros((1400, 2100))

    axarr.axis('off')

    axarr.imshow(img)

    axarr.imshow(mask, alpha=0.5, cmap='gray')

    axarr.set_title(label, fontsize=24)

plt.tight_layout(h_pad=0.1, w_pad=0.1)

plt.show()




def np_resize(img, input_shape):

    """

    Reshape a numpy array, which is input_shape=(height, width), 

    as opposed to input_shape=(width, height) for cv2

    """

    height, width = input_shape

    return cv2.resize(img, (width, height))

    

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):

    width, height = input_shape[:2]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return mask.reshape(height, width).T



def build_masks(rles, input_shape, reshape=None):

    depth = len(rles)

    if reshape is None:

        masks = np.zeros((*input_shape, depth))

    else:

        masks = np.zeros((*reshape, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            if reshape is None:

                masks[:, :, i] = rle2mask(rle, input_shape)

            else:

                mask = rle2mask(rle, input_shape)

                reshaped_mask = np_resize(mask, reshape)

                masks[:, :, i] = reshaped_mask

    

    return masks



def build_rles(masks, reshape=None):

    width, height, depth = masks.shape

    

    rles = []

    

    for i in range(depth):

        mask = masks[:, :, i]

        

        if reshape:

            mask = mask.astype(np.float32)

            mask = np_resize(mask, reshape).astype(np.int64)

        

        rle = mask2rle(mask)

        rles.append(rle)

        

    return rles
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path='../input/understanding_cloud_organization/train_images',

                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None,

                 n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.reshape = reshape

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.random_state = random_state

        

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        

        X = self.__generate_X(list_IDs_batch)

        

        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            return X, y

        

        elif self.mode == 'predict':

            return X



        else:

            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.seed(self.random_state)

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        if self.reshape is None:

            X = np.empty((self.batch_size, *self.dim, self.n_channels))

        else:

            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            img_path = f"{self.base_path}/{im_name}"

            

            if self.n_channels == 3:

                img = self.__load_rgb(img_path)

            else:

                img = self.__load_grayscale(img_path)

            

            if self.reshape is not None:

                img = np_resize(img, self.reshape)

            

            if len(img.shape) == 2:

                img = np.expand_dims(img, axis=-1)

            

            # Store samples

            X[i,] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        if self.reshape is None:

            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        else:

            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            

            rles = image_df['EncodedPixels'].values

            

            if self.reshape is not None:

                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)

            else:

                masks = build_masks(rles, input_shape=self.dim)

            

            y[i, ] = masks



        return y

    

    def __load_grayscale(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.

        img = np.expand_dims(img, axis=-1)



        return img

    

    def __load_rgb(self, img_path):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.



        return img

def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

BATCH_SIZE = 32



train_idx, val_idx = train_test_split(

    mask_count_df.index, random_state=2019, test_size=0.15

)



train_generator = DataGenerator(

    train_idx, 

    df=mask_count_df,

    target_df=train,

    batch_size=BATCH_SIZE,

    reshape=(256, 384),

    n_channels=3,

    n_classes=4

)



val_generator = DataGenerator(

    val_idx, 

    df=mask_count_df,

    target_df=train,

    batch_size=BATCH_SIZE, 

    reshape=(256, 384),

    n_channels=3,

    n_classes=4

)
! pip install segmentation-models
from segmentation_models import Unet

from segmentation_models.backbones import get_preprocessing



# LOAD UNET WITH PRETRAINING FROM IMAGENET

preprocess = get_preprocessing('resnet34') # for resnet, img = (img-110.0)/1.0

model = Unet('resnet34', input_shape=(256, 384, 3), classes=4, activation='sigmoid')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef])

model.summary()
history=model.fit_generator(train_generator,validation_data=val_generator,epochs=4,verbose=3)
# PLOT TRAINING

plt.figure(figsize=(15,5))

plt.plot(range(history.epoch[-1]+1),history.history['val_dice_coef'],label='val_dice_coef')

plt.plot(range(history.epoch[-1]+1),history.history['dice_coef'],label='trn_dice_coef')

plt.title('Training Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Dice_coef');plt.legend(); 

plt.show()