# Let's install tensorflow 2.2 first

!pip install tensorflow==2.2rc2
import os

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras import layers, models

from tensorflow.keras import optimizers, callbacks

from tensorflow.keras.utils import to_categorical, Sequence

from tensorflow.keras.applications import vgg19



from albumentations import (

                        PadIfNeeded,

                        HorizontalFlip,

                        VerticalFlip,    

                        CenterCrop,    

                        Crop,

                        Compose,

                        Transpose,

                        RandomRotate90,

                        Rotate,

                        RandomSizedCrop,

                        OneOf,

                        CLAHE,

                        RandomBrightnessContrast,    

                        RandomGamma    

                    )



# always set the seed

seed=1234

np.random.seed(seed)

tf.random.set_seed(seed)

sns.set()

%matplotlib inline
# path to the original files

files_path = Path("../input/digitally-reconstructed-radiographs-drr-bones")



# get all the file names as a list of strings

files = list(map(str, list(files_path.glob("**/*.png"))))

print("Total number of files found: ", len(files))





# store the above info in a pandas dataframe

bone_drr = pd.DataFrame([(x, x.replace('.png','_mask.png')) for x in files if not x.endswith('_mask.png')])

bone_drr.columns = ['image','bones']

print(f'Total instances: {bone_drr.shape[0]}')

bone_drr.head()
def plot_random_images(nb_images, df, idx=None, figsize=(15,8)):

    """Plots random images from the data

    Args:

        nb_images: Number of images to plot

        df: dataframe object

        idx: list of the indices to plot

        figsize: size of the plot

    """

    

    if idx is not None:

        idx = idx

        if nb_images != len(idx):

            raise ValueError("""Number of indices and the 

            number of images to plot should be same""")

    else:

        idx = np.random.choice(len(df), size=nb_images)

        

    ncols = 2

    nrows = nb_images

        

    f, ax = plt.subplots(nrows, ncols, figsize=figsize)

    

    for i, index in enumerate(idx):

        img = cv2.imread(df['image'][index], 0)

        bone = cv2.imread(df['bones'][index], 0)

        ax[i, 0].imshow(img, cmap='gray')

        ax[i, 1].imshow(bone, cmap='gray')

        ax[i, 0].axis("off")

        ax[i, 1].axis("off")

    plt.show()
plot_random_images(nb_images=4, df=bone_drr, figsize=(15, 20))
IMG_SHAPE = (512, 512, 3)

SPLIT_IDX = 160



# Shuffle rows in dataframe

bone_drr = bone_drr.sample(frac=1, random_state=seed)

df_train = bone_drr[:SPLIT_IDX].reset_index(drop=True)

df_val = bone_drr[SPLIT_IDX:].reset_index(drop=True)
def load_data_as_numpy_array(df, im_shape):

    X, y = [], []

    for i in range(len(df)):

        img = cv2.imread(df['image'][i], 0)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, im_shape[:2])

        

        mask = cv2.imread(df['bones'][i])

        mask = cv2.resize(mask, im_shape[:2])

        

        X.append(img)

        y.append(mask)

    

    X = np.array(X)

    y = np.array(y)

    return X, y
# Load training and validation data

X_train, y_train = load_data_as_numpy_array(df_train, IMG_SHAPE)

X_val, y_val = load_data_as_numpy_array(df_val, IMG_SHAPE)
class DataGenerator(Sequence):

    """Performs augmentation using albumentations"""

    def __init__(self, 

                 data, 

                 labels,

                 img_dim=IMG_SHAPE, 

                 batch_size=32,  

                 shuffle=True,

                 augment=True,

                ):

        """

        Args:

            data: numpy array containing images

            labels: numpy array containing corresponding masks

            img_dim: fixed image shape that is to be used

            batch_size: batch size for one step

            shuffle: (bool) whether to shuffle the data or not

            augment: (bool) whether to augment the data or not

        

        Returns:

            A batch of images and corresponding masks

        """

        

        self.data = data

        self.labels = labels

        self.img_dim = img_dim

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.augment = augment

        self.indices = np.arange(len(self.data))

        self.augmentations()

        self.on_epoch_end()

        

        

    def augmentations(self):

        self.aug = OneOf([VerticalFlip(p=0.2),

                        HorizontalFlip(p=1.0),

                        RandomBrightnessContrast(p=0.5),

                        Rotate(p=1.0, limit=20, border_mode=0)])

        

    

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(self.indices)

    

    def augment_data(self, img, label):

        augmenetd = self.aug(image=img, mask=label)

        return augmenetd['image'], augmenetd['mask']





    def __len__(self):

        return int(np.ceil(len(self.data) / self.batch_size))

    

    

    def __getitem__(self, idx):

        curr_batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        # print(curr_batch)

        batch_len = len(curr_batch)  

        X = np.zeros((batch_len, *self.img_dim), dtype=np.float32)

        y = np.zeros((batch_len, *self.img_dim), dtype=np.float32)

        

        for i, index in enumerate(curr_batch):

            img = self.data[index]

            label = self.labels[index]

            if self.augment:

                img, label = self.augment_data(img, label)

            img = img.astype(np.float32)

            img -= img.mean()

            img /= img.std()

            label = label.astype(np.float32) / 127.5 -1.

            

            X[i], y[i] = img, label

        return X, y
def conv_block(inputs,

               filters, 

               kernel_size, 

               dilation_rate=1, 

               padding="same", 

               activation="relu",

               kernel_initializer="he_normal"):

    

    x = layers.Conv2D(filters,

                      kernel_size=kernel_size,

                      dilation_rate=dilation_rate,

                      kernel_initializer=kernel_initializer,

                      padding=padding,

                      activation=activation

                     )(inputs)

    return x



def pool_block(inputs, pool="max", pool_size=((2,2)), strides=(2,2)):

    return layers.MaxPooling2D(strides=strides, pool_size=pool_size)(inputs)
def dilated_unet(im_shape, addition=1, dilate=1, dilate_rate=1):

    x = inputs = layers.Input(im_shape)

    

    down1 = conv_block(x, 44, 3)

    down1 = conv_block(x, 44, 3, dilation_rate=dilate_rate)

    down1pool = pool_block(down1)

    

    down2 = conv_block(down1pool, 88, 3)

    down2 = conv_block(down1pool, 88, 3, dilation_rate=dilate_rate)

    down2pool = pool_block(down2)

    

    down3 = conv_block(down2pool, 176, 3)

    down3 = conv_block(down3, 176, 3, dilation_rate=dilate_rate)

    down3pool = pool_block(down3)

    

    if dilate == 1:

        dilate1 = conv_block(down3pool, 176, 3, dilation_rate=1)

        dilate2 = conv_block(dilate1, 176, 3, dilation_rate=2)

        

        if addition == 1:

            dilate_all_added = layers.add([dilate1, dilate2])

            up3 = layers.UpSampling2D((2, 2))(dilate_all_added)

        else:

            up3 = layers.UpSampling2D((2, 2))(dilate2)

            

    up3 = conv_block(up3, 88, 3)

    up3 = layers.concatenate([down3, up3])

    up3 = conv_block(up3, 88, 3)

    up3 = conv_block(up3, 88, 3)

    

    up2 = layers.UpSampling2D((2, 2))(up3)

    up2 = conv_block(up2, 44, 3)

    up2 = layers.concatenate([down2, up2])

    up2 = conv_block(up2, 44, 3)

    up2 = conv_block(up2, 44, 3)

    

    up1 = layers.UpSampling2D((2, 2))(up2)

    up1 = conv_block(up1, 22, 3)

    up1 = layers.concatenate([down1, up1])

    up1 = conv_block(up1, 22, 3)

    up1 = conv_block(up1, 22, 3)

    

    out = layers.Conv2D(1, 1, activation="tanh")(up1)

    model = models.Model(inputs=inputs, outputs=out)

    return model
dunet = dilated_unet(IMG_SHAPE)

dunet.summary()
vgg = vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(512, 512, 3))

loss_model = models.Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output)

loss_model.trainable = False

loss_model.summary()
def perceptual_loss_vgg19(y_true, y_pred):

    y_pred = tf.image.grayscale_to_rgb(y_pred, name=None)

    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))
class BoneUNet(tf.keras.Model):

    def __init__(self, base_model):

        super().__init__()

        self.base_model = base_model

        

    def compile(self, base_model_opt, loss_fn):

        super().compile()

        self.base_model_optimizer = base_model_opt

        self.loss_fn = loss_fn

        

    def train_step(self, inputs):

        images, labels =  inputs

        

        with tf.GradientTape() as tape:

            preds = self.base_model(images, training=True)

            loss = self.loss_fn(labels, preds)

        

        grads = tape.gradient(loss, self.base_model.trainable_weights)

        self.base_model_optimizer.apply_gradients(

                    zip(grads, self.base_model.trainable_weights)) 

        

        return {'loss':loss}

    

    def call(self, images):

        preds = self.base_model(images, training=False)    

        return preds

        

    def test_step(self, inputs):

        images, labels = inputs

        preds = self.call(images)

        loss = self.loss_fn(labels, preds)

        return {'loss': loss}
bone_model = BoneUNet(base_model=dunet)

bone_model.compile(optimizers.Adam(), loss_fn=perceptual_loss_vgg19)



batch_size = 8

epochs = 100

es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)



nb_train_steps = int(np.ceil(len(X_train) / batch_size))

nb_valid_steps = int(np.ceil(len(X_val)) / batch_size)



train_data_gen = DataGenerator(data=X_train,

                               labels=y_train,

                               batch_size=batch_size,

                               shuffle=True,

                               augment=False)



valid_data_gen = DataGenerator(data=X_val,

                              labels=y_val,

                              batch_size=batch_size,

                              shuffle=False,

                              augment=False)
bone_model.fit(train_data_gen,

                validation_data=valid_data_gen,

                epochs=epochs,

                steps_per_epoch=nb_train_steps,

                validation_steps=nb_valid_steps,

                callbacks=[es])
def plot_prediction(filepath, figsize=(16, 8)):

    orig_img = cv2.imread(filepath, 0)

    img = orig_img.copy()

    

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (512, 512))

    

    img = img.astype(np.float32)

    img -= img.mean()

    img /= img.std()

    

    pred = bone_model.predict(np.expand_dims(np.array(img), 0))[0, :, :, 0]

    pred = ((pred + 1)*127.5).astype(np.uint8)

    pred = cv2.resize(pred, (img.shape[1], img.shape[0]))

    

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    axes[0].imshow(orig_img, 'gray')

    axes[1].imshow(pred, 'gray')

    plt.show()
sample_img_path = '../input/padchest-chest-xrays-sample/sample/216840111366964012819207061112010316094555679_04-017-068.png'

plot_prediction(sample_img_path)