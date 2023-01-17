import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt 

import random

import cv2



from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint

from keras.applications import ResNet50, InceptionResNetV2

from keras.layers import Dense, Input, Lambda

from tensorflow import keras

from keras.models import Model 

import tensorflow as tf

from math import ceil

from keras import Model

from keras.optimizers import Adam



import random

import os

import gc

from keras.callbacks import ReduceLROnPlateau

!pip install efficientnet

import albumentations

from albumentations import RandomCrop, Compose, HorizontalFlip, VerticalFlip, OneOf

from albumentations.core.transforms_interface import DualTransform

from albumentations.augmentations import functional as F

import efficientnet.keras as efn
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



seed_everything(42)
seed = 69

img_dir = "../input/plant-pathology-2020-fgvc7//images/"

path = "../input/plant-pathology-2020-fgvc7/"

input_shape = (512,512,3)

#Version - 1.0 4D_augmentation  LB: 0.941

#Version - 2.0 4D_augmentation + TTA LB: 0.963

#Version -3.0 4D_augmentation + TTA + GridMask 0.967

#Version -4.0 4D_augmentation + TTA + reduce_lr + GridMask + Effnet 0.966

#Version -5.0 4D_augmentation + TTA + reduce_lr + GridMask + Effnet 0.966
class GridMask(DualTransform):

    

    """GridMask augmentation for image classification and object detection.

    

    Author: Qishen Ha

    Email: haqishen@gmail.com

    2020/01/29



    Args:

        num_grid (int): number of grid in a row or column.

        fill_value (int, float, lisf of int, list of float): value for dropped pixels.

        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int

            an angle is picked from (-rotate, rotate). Default: (-90, 90)

        mode (int):

            0 - cropout a quarter of the square of each grid (left top)

            1 - reserve a quarter of the square of each grid (left top)

            2 - cropout 2 quarter of the square of each grid (left top & right bottom)



    Targets:

        image, mask



    Image types:

        uint8, float32



    Reference:

    |  https://arxiv.org/abs/2001.04086

    |  https://github.com/akuxcw/GridMask

    """



    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):

        super(GridMask, self).__init__(always_apply, p)

        if isinstance(num_grid, int):

            num_grid = (num_grid, num_grid)

        if isinstance(rotate, int):

            rotate = (-rotate, rotate)

        self.num_grid = num_grid

        self.fill_value = fill_value

        self.rotate = rotate

        self.mode = mode

        self.masks = None

        self.rand_h_max = []

        self.rand_w_max = []



    def init_masks(self, height, width):

        if self.masks is None:

            self.masks = []

            n_masks = self.num_grid[1] - self.num_grid[0] + 1

            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):

                grid_h = height / n_g

                grid_w = width / n_g

                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)

                for i in range(n_g + 1):

                    for j in range(n_g + 1):

                        this_mask[

                             int(i * grid_h) : int(i * grid_h + grid_h / 2),

                             int(j * grid_w) : int(j * grid_w + grid_w / 2)

                        ] = self.fill_value

                        if self.mode == 2:

                            this_mask[

                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),

                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)

                            ] = self.fill_value

                

                if self.mode == 1:

                    this_mask = 1 - this_mask



                self.masks.append(this_mask)

                self.rand_h_max.append(grid_h)

                self.rand_w_max.append(grid_w)



    def apply(self, image, mask, rand_h, rand_w, angle, **params):

        h, w = image.shape[:2]

        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask

        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask

        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)

        return image



    def get_params_dependent_on_targets(self, params):

        img = params['image']

        height, width = img.shape[:2]

        self.init_masks(height, width)



        mid = np.random.randint(len(self.masks))

        mask = self.masks[mid]

        rand_h = np.random.randint(self.rand_h_max[mid])

        rand_w = np.random.randint(self.rand_w_max[mid])

        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0



        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}



    @property

    def targets_as_params(self):

        return ['image']



    def get_transform_init_args_names(self):

        return ('num_grid', 'fill_value', 'rotate', 'mode')
def plot_validate(model, loss_acc):

    '''

    Plot model accuracy or loss for both train and test validation per epoch

    model : fitted model

    loss_acc : input 'loss' or 'acc' to plot respective graph

    '''

    history = model.history.history



    if loss_acc == 'loss':

        axis_title = 'loss'

        title = 'Loss'

        epoch = len(history['loss'])

    elif loss_acc == 'acc':

        axis_title = 'categorical_accuracy'

        title = 'Accuracy'

        epoch = len(history['loss'])



    plt.figure(figsize=(15,4))

    plt.plot(history[axis_title])

    plt.plot(history['val_' + axis_title])

    plt.title('Model ' + title)

    plt.ylabel(title)

    plt.xlabel('Epoch')



    plt.grid(b=True, which='major')

    plt.minorticks_on()

    plt.grid(b=True, which='minor', alpha=0.2)



    plt.legend(['Train', 'Test'])

    plt.show()
def augment(aug, image):

    '''

    image augmentation

    aug : augmentation from albumentations



    '''

    aug_img = aug(image=image)['image']

    return aug_img



def VH_augment(image):

    

    '''

    Vertical and horizontal flip image

    '''

    image = HorizontalFlip(p=1)(image=image)['image']

    image = VerticalFlip(p=1)(image=image)['image']

    return image



def strong_aug(p=1.0):

    

    '''

    4D - augmentations

    '''

    return  OneOf([

            HorizontalFlip(p=0.33),

            VerticalFlip(p=0.33),

           Compose([HorizontalFlip(p=1),

                    VerticalFlip(p=1)], p=0.33)

        ], p=1)



def four_D_augment(name):

    name = str(name)

    image = cv2.imread(img_dir + name)

    aug_img_1 = augment(HorizontalFlip(p=1), image)

    aug_img_2 = augment(VerticalFlip(p=1), image)

    aug_img_3 = VH_augment(image)

    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (25,25))

    ax[0].imshow(image[...,[2,1,0]])

    ax[0].set_title('Original image', fontsize=14)

    ax[1].imshow(aug_img_1[...,[2,1,0]])

    ax[1].set_title('Horizontal flip image', fontsize=14)

    ax[2].imshow(aug_img_2[...,[2,1,0]])

    ax[2].set_title('Vertical flip image', fontsize=14)

    ax[3].imshow(aug_img_3[...,[2,1,0]])

    ax[3].set_title('Vertical and horizontal flip image', fontsize=14)

    plt.show()
#4D-augment

four_D_augment('Train_1400.jpg')

four_D_augment('Train_1420.jpg')

four_D_augment('Train_1430.jpg')
# GridMask

transforms_train = albumentations.Compose([

    GridMask(num_grid=4, rotate=15, p=0.6),])



fig, ax = plt.subplots(nrows=1 , ncols=2, figsize=(15,15))

image = cv2.imread(img_dir+'Train_100.jpg')

ax[0].imshow(image[...,[2,1,0]])

aug = augment(transforms_train, image)

ax[1].imshow(aug[...,[2,1,0]])
train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')

sample_submission = pd.read_csv(path + 'sample_submission.csv')

train.head()
gm_exp = tf.Variable(3.0, dtype = tf.float32)

def generalized_mean_pool_2d(X):

    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)), 

                        axis = [1, 2], 

                        keepdims = False) + 1.e-7)**(1./gm_exp)

    return pool





def create_model(input_shape):

    input = Input(shape = input_shape)



    #Create and complite model and show summary

    

    x_model = ResNet50(weights = 'imagenet', include_top = False, input_tensor = input, pooling = None,

                        classes = None)

    for layer in x_model.layers:

        layer.trainable = True

    

    # Gem 

    lambda_layer = Lambda(generalized_mean_pool_2d)

    lambda_layer.trainable_weights.extend([gm_exp])

    x = lambda_layer(x_model.output)

    

    #output 

    healthy = Dense(4, activation = 'softmax', name = 'plan_diseases')(x)

   

    

    #model 

    model = Model(inputs = x_model.input, outputs = healthy )

    

    return model

def _read(path):

    img = cv2.imread(path)    

    return img



class TrainDataGenerator(tf.keras.utils.Sequence):

    

    def __init__(self , X_set, Y_set, img_dir, ids, batch_size = 2, img_size = (512,512,3), augmentation = False, GridMask = False):

        self.X = X_set

        self.Y = Y_set

        self.batch_size = batch_size

        self.ids = ids

        self.img_size  = img_size  

        self.img_dir = img_dir

        self.augmentation = augmentation

        self.GridMask = GridMask

        self.on_epoch_end()

        

        #Split Data

        self.x_indexed = X_set[self.ids]

        

    def __len__(self):

        return int(ceil(len(self.ids)/self.batch_size))



    def __getitem__(self, index):

        indices = self.ids[index*self.batch_size:(index+1)*self.batch_size]

        X, Y = self.__generator__(indices)

        return X, Y

    

    def on_epoch_end(self):

        self.indices = np.arange(len(self.ids))

        

        

    def __generator__(self, indices):

        X = np.empty((self.batch_size, *self.img_size))

        Y = np.empty((self.batch_size, 4))

        for i, index in enumerate(indices):

            ID = self.X[index]

            image = _read(self.img_dir+ID+".jpg")

            #image = image[200:1100, 200:1700]

            image = cv2.resize(image, (512,512))

            if self.augmentation == True:

                aug = strong_aug(p=1.0)

                image = augment(aug, image)

            elif self.GridMask == True:

                image = augment(transforms_train, image)

            X[i,] = image/255.

            Y[i,] = self.Y.loc[index].values

        return X, Y    

    

class TestDataGenerator(tf.keras.utils.Sequence):

    

    def __init__(self, Test_set, img_dir, ids, batch_size = 3, img_size = (512,512,3), augmentation = None):

        self.X = Test_set

        self.img_dir = img_dir

        self.ids = ids

        self.batch_size = batch_size

        self.img_size = img_size

        self.augmentation = augmentation

                 

    def __len__(self):

        return int(ceil(len(self.ids)/self.batch_size))

    

    def __getitem__(self, index):

        indices = self.ids[index*self.batch_size:(index+1)*self.batch_size]

        X = self.__generator__(indices)

        return X

    

    def __generator__(self, indices):

        X = np.empty((self.batch_size, *self.img_size))

        for i, index in enumerate(indices):

            ID = self.X[index]

            image = _read(self.img_dir+ID+".jpg")

           #image = image[200:1100, 200:1700]

            image = cv2.resize(image, (512,512))

                 # TTA

            if self.augmentation is not None:

                if self.augmentation == "HorizontalFlip":

                    image = augment(HorizontalFlip(p=1), image)

                elif self.augmentation == "VerticalFlip":

                    image = augment(VerticalFlip(p=1), image)

                elif self.augmentation == 'VH':

                    image = VH_augment(image)

                 

            X[i,] = image/255.

        return X

#Prepary X and Y

tgt_cols = ['healthy', 'multiple_diseases', 'rust' , 'scab']

train_df = train[tgt_cols]

Y = pd.get_dummies(train_df)

X = train['image_id']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = seed)

ids_train = np.array(X_train.index)

ids_test = X_test.index

del train

del X

del Y
model = create_model(input_shape)

model.compile(optimizer = Adam(lr = 0.00016),

              loss = 'categorical_crossentropy',

              metrics = ['categorical_accuracy'])
best_w = ModelCheckpoint('plant_best.h5',

                                monitor='val_loss',

                                verbose=0,

                                save_best_only=True,

                                save_weights_only=True,

                                mode='auto',

                                period=1)



last_w = ModelCheckpoint('plant_last.h5',

                               monitor='val_loss',

                                verbose=0,

                                save_best_only=False,

                              mode='auto',

                                period=1)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=2, min_lr=0.000001, verbose=1,cooldown=1)

callbacks = [best_w, last_w, reduce_lr]

batch_size = 5

model.fit_generator(TrainDataGenerator(X_train,Y_train , img_dir, ids_train, 

                                              batch_size, augmentation = True, GridMask = True),

               epochs=15, 

               verbose=1,

               callbacks=callbacks,

               validation_data=TrainDataGenerator(X_test, Y_test , img_dir, ids_test,

                                                  batch_size, augmentation = False, GridMask = False),

               #max_queue_size=100,

               workers=1,

               use_multiprocessing=False,

               shuffle=True)
gc.collect()
plot_validate(model, 'loss')
plot_validate(model, 'acc')
model.load_weights('plant_best.h5')


ids = test.index

Test_set = test['image_id']

data_generator_test = TestDataGenerator(Test_set, img_dir, ids, batch_size, input_shape, augmentation = None ) 

data_generator_test_Horizontal = TestDataGenerator(Test_set, img_dir, ids, batch_size, input_shape, augmentation = "HorizontalFlip" ) 

data_generator_test_Vertical =  TestDataGenerator(Test_set, img_dir, ids, batch_size, input_shape, augmentation = "VerticalFlip" ) 

data_generator_test_VH = TestDataGenerator(Test_set, img_dir, ids, batch_size, input_shape, augmentation = "VH" ) 

preds_1 = model.predict_generator(data_generator_test, verbose = 1)

preds_2 = model.predict_generator(data_generator_test_Horizontal, verbose = 1)

preds_3 = model.predict_generator(data_generator_test_Vertical, verbose = 1)

preds_4 = model.predict_generator(data_generator_test_VH, verbose = 1)



for index, pred in enumerate(zip(preds_1,preds_2, preds_3, preds_4)):

    sample_submission["healthy"][index] = (pred[0][0] + pred[1][0] + pred[2][0]  + pred[3][0])/4

    sample_submission["multiple_diseases"][index] = (pred[0][1] + pred[1][1] + pred[2][1]  + pred[3][1])/4

    sample_submission["rust"][index] = (pred[0][2] + pred[1][2] + pred[2][2]  + pred[3][2])/4

    sample_submission["scab"][index] = (pred[0][3] + pred[1][3] + pred[2][3]  + pred[3][3])/4
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)