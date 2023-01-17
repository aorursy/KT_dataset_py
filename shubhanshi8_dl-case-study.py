# https://www.kaggle.com/titericz/building-and-visualizing-masks

def rle2maskResize(rle):

    # CONVERT RLE TO MASK 

    if (pd.isnull(rle))|(rle==''): 

        return np.zeros((128,800) ,dtype=np.uint8)

    

    height= 256

    width = 1600

    mask= np.zeros( width*height ,dtype=np.uint8)



    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]-1

    lengths = array[1::2]    

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

    

    return mask.reshape( (height,width), order='F' )[::2,::2]



def mask2contour(mask, width=3):

    # CONVERT MASK TO ITS CONTOUR

    w = mask.shape[1]

    h = mask.shape[0]

    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)

    mask2 = np.logical_xor(mask,mask2)

    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)

    mask3 = np.logical_xor(mask,mask3)

    return np.logical_or(mask2,mask3) 



def mask2pad(mask, pad=2):

    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT

    w = mask.shape[1]

    h = mask.shape[0]

    

    # MASK UP

    for k in range(1,pad,2):

        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)

        mask = np.logical_or(mask,temp)

    # MASK DOWN

    for k in range(1,pad,2):

        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)

        mask = np.logical_or(mask,temp)

    # MASK LEFT

    for k in range(1,pad,2):

        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)

        mask = np.logical_or(mask,temp)

    # MASK RIGHT

    for k in range(1,pad,2):

        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)

        mask = np.logical_or(mask,temp)

    

    return mask 
# import basics

import numpy as np, pandas as pd, os, gc

import warnings

warnings.filterwarnings("ignore")

from glob import glob



# import plotting

from matplotlib import pyplot as plt

import matplotlib.patches as patches

import matplotlib

import seaborn as sns



# import image manipulation

from PIL import Image 

import cv2



import json

import keras

from keras import backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers import Dropout

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.losses import binary_crossentropy

from keras.callbacks import Callback, ModelCheckpoint

from tqdm import tqdm

from sklearn.model_selection import train_test_split

path = '../input/severstal-steel-defect-detection/'
# set paths to train and test image datasets

TRAIN_PATH =  path+'train_images/'

TEST_PATH =  path+'test_images/'



# load dataframe with train labels

train_df = pd.read_csv( path+'train.csv')

train_fns = sorted(glob(TRAIN_PATH + '*.jpg'))

test_fns = sorted(glob(TEST_PATH + '*.jpg'))



print('There are {} images in the train set.'.format(len(train_fns)))

print('There are {} images in the test set.'.format(len(test_fns)))
# plotting a pie chart which demonstrates train and test sets

labels = 'Train', 'Test'

sizes = [len(train_fns), len(test_fns)]

explode = (0, 0.1)



fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal')

ax.set_title('Train and Test Sets')



plt.show()
print('There are {} rows with empty segmentation maps.'.format(len(train_df) - train_df.EncodedPixels.count()))
# plotting a pie chart

labels = 'Non-empty', 'Empty'

sizes = [train_df.EncodedPixels.count(), len(train_df) - train_df.EncodedPixels.count()]

explode = (0, 0.1)



fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal')

ax.set_title('Non-empty and Empty Masks')
# split column

split_df = train_df["ImageId_ClassId"].str.split("_", n = 1, expand = True)



# add new columns to train_df

train_df['Image'] = split_df[0]

train_df['Label'] = split_df[1]



# check the result

train_df.head()
#Analyse the number of labels for each defect type

defect1 = train_df[train_df['Label'] == '1'].EncodedPixels.count()

defect2 = train_df[train_df['Label'] == '2'].EncodedPixels.count()

defect3 = train_df[train_df['Label'] == '3'].EncodedPixels.count()

defect4 = train_df[train_df['Label'] == '4'].EncodedPixels.count()



labels_per_image = train_df.groupby('Image')['EncodedPixels'].count()



no_defects = labels_per_image[labels_per_image == 0].count()



print('There are {} defect1 images'.format(defect1))

print('There are {} defect2 images'.format(defect2))

print('There are {} defect3 images'.format(defect3))

print('There are {} defect4 images'.format(defect4))

print('There are {} images with no defects'.format(no_defects))
# plotting a pie chart

labels = 'Defect 1', 'Defect 2', 'Defect 3', 'Defect 4', 'No defects'

sizes = [defect1, defect2, defect3, defect4, no_defects]



fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal')

ax.set_title('Defect Types')



plt.show()
# Number of Labels per Image

print('There are {} images with no labels'.format(labels_per_image[labels_per_image == 0].count()))

print('There are {} images with 1 label'.format(labels_per_image[labels_per_image == 1].count()))

print('There are {} images with 2 labels'.format(labels_per_image[labels_per_image == 2].count()))

print('There are {} images with 3 labels'.format(labels_per_image[labels_per_image == 3].count()))
def plot_mask(image_filename):

    '''

    Function to plot an image and segmentation masks.

    INPUT:

        image_filename - filename of the image (with full path)

    '''

    img_id = image_filename.split('/')[-1]

    image = Image.open(image_filename)

    train = train_df.fillna('-1')

    rle_masks = train[(train['Image'] == img_id) & (train['EncodedPixels'] != '-1')]['EncodedPixels'].values

    

    defect_types = train[(train['Image'] == img_id) & (train['EncodedPixels'] != '-1')]['Label'].values

    

    if (len(rle_masks) > 0):

        fig, axs = plt.subplots(1, 1 + len(rle_masks), figsize=(20, 3))



        axs[0].imshow(image)

        axs[0].axis('off')

        axs[0].set_title('Original Image')



        for i in range(0, len(rle_masks)):

            mask = rle2maskResize(rle_masks[i])

            axs[i + 1].imshow(image)

            axs[i + 1].imshow(mask, alpha = 0.5, cmap = "Reds")

            axs[i + 1].axis('off')

            axs[i + 1].set_title('Mask with defect #{}'.format(defect_types[i]))



        plt.suptitle('Image with defect masks')

    else:

        fig, axs = plt.subplots(figsize=(20, 3))

        axs.imshow(image)

        axs.axis('off')

        axs.set_title('Original Image without Defects')
# plot image example with one defects

for image_code in train_df.Image.unique():

    if (train_df.groupby(['Image'])['EncodedPixels'].count().loc[image_code] == 1):

        plot_mask(TRAIN_PATH + image_code)

        break;
# plot image example with more than one defects

for image_code in train_df.Image.unique():

    if (train_df.groupby(['Image'])['EncodedPixels'].count().loc[image_code] > 1):

        plot_mask(TRAIN_PATH + image_code)

        break;
train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()



print(train_df.shape)

train_df.head()
mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

print(mask_count_df.shape)

mask_count_df.head()
sub_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')

sub_df['ImageId'] = sub_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
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



def rle2mask(mask_rle, shape=(256,1600)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
def build_masks(rles, input_shape):

    depth = len(rles)

    height, width = input_shape

    masks = np.zeros((height, width, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            masks[:, :, i] = rle2mask(rle, (width, height))

    

    return masks



def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
sample_filename = 'db4867ee8.jpg'

sample_image_df = train_df[train_df['ImageId'] == sample_filename]

sample_path = f"../input/severstal-steel-defect-detection/train_images/{sample_image_df['ImageId'].iloc[0]}"

sample_img = cv2.imread(sample_path)

sample_rles = sample_image_df['EncodedPixels'].values

sample_masks = build_masks(sample_rles, input_shape=(256, 1600))



fig, axs = plt.subplots(5, figsize=(12, 12))

axs[0].imshow(sample_img)

axs[0].axis('off')



for i in range(4):

    axs[i+1].imshow(sample_masks[:, :, i])

    axs[i+1].axis('off')
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 base_path='../input/severstal-steel-defect-detection/train_images',

                 batch_size=32, dim=(256, 1600), n_channels=1,

                 n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

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

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            img_path = f"{self.base_path}/{im_name}"

            img = self.__load_grayscale(img_path)

            

            # Store samples

            X[i,] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            

            rles = image_df['EncodedPixels'].values

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
BATCH_SIZE = 16



train_idx, val_idx = train_test_split(

    mask_count_df.index, random_state=2019, test_size=0.15

)



train_generator = DataGenerator(

    train_idx, 

    df=mask_count_df,

    target_df=train_df,

    batch_size=BATCH_SIZE, 

    n_classes=4

)



val_generator = DataGenerator(

    val_idx, 

    df=mask_count_df,

    target_df=train_df,

    batch_size=BATCH_SIZE, 

    n_classes=4

)
def build_model(input_shape):

    inputs = Input(input_shape)



    c1 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)

    c1 = Dropout(0.1) (c1)

    c1 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)

    p1 = MaxPooling2D((2, 2)) (c1)

    

    c2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)

    c2 = Dropout(0.1) (c2)

    c2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)

    c3 = Dropout(0.2) (c3)

    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)

    p3 = MaxPooling2D((2, 2)) (c3)

    

    c4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)

    c4 = Dropout(0.2) (c4)

    c4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    

    c5 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)

    c5 = Dropout(0.3) (c5)

    c5 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    

    u6 = Conv2DTranspose(64,(2, 2), strides=(2, 2),padding='same') (c5)

    u6 = concatenate([u6, c4])

    c6 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)

    c6 = Dropout(0.2) (c6)

    c6 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    

    u7 = Conv2DTranspose(32,(2, 2), strides=(2, 2),padding='same') (c6)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)

    c7 = Dropout(0.2) (c7)

    c7 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    

    u8 = Conv2DTranspose(16,(2, 2), strides=(2, 2),padding='same') (c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)

    c8 = Dropout(0.1) (c8)

    c8 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    

    u9 = Conv2DTranspose(8,(2, 2), strides=(2, 2),padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)

    c9 = Dropout(0.1) (c9)

    c9 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    

    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)



    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coef])

    

    return model
model = build_model((256, 1600, 1))

model.summary()
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_dice_coef', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit_generator(

    train_generator,

    validation_data=val_generator,

    callbacks=[checkpoint],

    use_multiprocessing=False,

    workers=1,

    epochs=9

)
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['dice_coef', 'val_dice_coef']].plot()
model.load_weights('model.h5')

test_df = []



for i in range(0, test_imgs.shape[0], 500):

    batch_idx = list(

        range(i, min(test_imgs.shape[0], i + 500))

    )

    

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/severstal-steel-defect-detection/test_images',

        target_df=sub_df,

        batch_size=1,

        n_classes=4

    )

    

    batch_pred_masks = model.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )

    

    for j, b in tqdm(enumerate(batch_idx)):

        filename = test_imgs['ImageId'].iloc[b]

        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        

        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks)

        

        image_df['EncodedPixels'] = pred_rles

        test_df.append(image_df)
test_df = pd.concat(test_df)

test_df.drop(columns='ImageId', inplace=True)

test_df.to_csv('submission.csv', index=False)
test_df.head(40)