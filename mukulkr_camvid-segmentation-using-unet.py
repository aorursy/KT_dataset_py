# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import cv2

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.utils import to_categorical ,Sequence

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout

from tensorflow.keras.optimizers import Adadelta, Nadam ,Adam

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from glob import glob

from pathlib import Path

import shutil

from tqdm import tqdm_notebook

from random import sample, choice



# Any results you write to the current directory are saved as output.
dataset_path = Path("../input/camvid/CamVid/")

list(dataset_path.iterdir())
def tree(directory):

    print(f'+ {directory}')

    for path in sorted(directory.rglob('*')):

        depth = len(path.relative_to(directory).parts)

        spacer = '    ' * depth

        print(f'{spacer}+ {path.name}')       

#tree(dataset_path)
train_imgs = list((dataset_path / "train").glob("*.png"))

train_labels = list((dataset_path / "train_labels").glob("*.png"))

val_imgs = list((dataset_path / "val").glob("*.png"))

val_labels = list((dataset_path / "val_labels").glob("*.png"))

test_imgs = list((dataset_path / "test").glob("*.png"))

test_labels = list((dataset_path / "test_labels").glob("*.png"))



(len(train_imgs),len(train_labels)), (len(val_imgs),len(val_labels)) , (len(test_imgs),len(test_labels))



img_size = 512
assert len(train_imgs) == len(train_labels), "No of Train images and label mismatch"

assert len(val_imgs) == len(val_labels), "No of Train images and label mismatch"

assert len(test_imgs) == len(test_labels), "No of Train images and label mismatch"



sorted(train_imgs), sorted(train_labels), sorted(val_imgs), sorted(val_labels), sorted(test_imgs), sorted(test_labels);
for im in train_imgs:

    assert dataset_path / "train_labels" / (im.stem +"_L.png") in train_labels , "{im} not there in label folder"

for im in val_imgs:

    assert dataset_path / "val_labels" / (im.stem +"_L.png") in val_labels , "{im} not there in label folder"

for im in test_imgs:

    assert dataset_path / "test_labels" / (im.stem +"_L.png") in test_labels , "{im} not there in label folder"
def make_pair(img,label,dataset):

    pairs = []

    for im in img:

        pairs.append((im , dataset / label / (im.stem +"_L.png")))

    

    return pairs
train_pair = make_pair(train_imgs, "train_labels", dataset_path)

val_pair = make_pair(val_imgs, "val_labels", dataset_path)

test_pair = make_pair(test_imgs, "test_labels", dataset_path)
temp = choice(train_pair)

img = img_to_array(load_img(temp[0], target_size=(img_size,img_size)))

mask = img_to_array(load_img(temp[1], target_size = (img_size,img_size)))

plt.figure(figsize=(10,10))

plt.subplot(121)

plt.imshow(img/255)

plt.subplot(122)

plt.imshow(mask/255)
class_map_df = pd.read_csv(dataset_path / "class_dict.csv")
class_map = []

for index,item in class_map_df.iterrows():

    class_map.append(np.array([item['r'], item['g'], item['b']]))

    

len(class_map)
def assert_map_range(mask,class_map):

    mask = mask.astype("uint8")

    for j in range(img_size):

        for k in range(img_size):

            assert mask[j][k] in class_map , tuple(mask[j][k])
def form_2D_label(mask,class_map):

    mask = mask.astype("uint8")

    label = np.zeros(mask.shape[:2],dtype= np.uint8)

    

    for i, rgb in enumerate(class_map):

        label[(mask == rgb).all(axis=2)] = i

    

    return label
lab = form_2D_label(mask,class_map)

np.unique(lab,return_counts=True)
class DataGenerator(Sequence):

    'Generates data for Keras'

    

    def __init__(self, pair, class_map, batch_size=16, dim=(224,224,3), shuffle=True):

        'Initialization'

        self.dim = dim

        self.pair = pair

        self.class_map = class_map

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.pair) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [k for k in indexes]



        # Generate data

        X, y = self.__data_generation(list_IDs_temp)



        return X, y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.pair))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization

        batch_imgs = list()

        batch_labels = list()



        # Generate data

        for i in list_IDs_temp:

            # Store sample

            img = load_img(self.pair[i][0] ,target_size=self.dim)

            img = img_to_array(img)/255.

            batch_imgs.append(img)



            label = load_img(self.pair[i][1],target_size=self.dim)

            label = img_to_array(label)

            label = form_2D_label(label,self.class_map)

            label = to_categorical(label , num_classes = 32)

            batch_labels.append(label)

            

        return np.array(batch_imgs) ,np.array(batch_labels)
train_generator = DataGenerator(train_pair+test_pair,class_map,batch_size=4, dim=(img_size,img_size,3) ,shuffle=True)

train_steps = train_generator.__len__()

train_steps
X,y = train_generator.__getitem__(1)

y.shape
val_generator = DataGenerator(val_pair, class_map, batch_size=4, dim=(img_size,img_size,3) ,shuffle=True)

val_steps = val_generator.__len__()

val_steps
def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):

    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    return x





def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):

    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)

    y = concatenate([y, residual], axis=3)

    y = conv_block(y, nfilters)

    return y





def Unet(h, w, filters):

# down

    input_layer = Input(shape=(h, w, 3), name='image_input')

    conv1 = conv_block(input_layer, nfilters=filters)

    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(conv1_out, nfilters=filters*2)

    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(conv2_out, nfilters=filters*4)

    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(conv3_out, nfilters=filters*8)

    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv4_out = Dropout(0.5)(conv4_out)

    conv5 = conv_block(conv4_out, nfilters=filters*16)

    conv5 = Dropout(0.5)(conv5)

# up

    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)

    deconv6 = Dropout(0.5)(deconv6)

    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)

    deconv7 = Dropout(0.5)(deconv7) 

    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)

    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)

    output_layer = Conv2D(filters=32, kernel_size=(1, 1), activation='softmax')(deconv9)



    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')

    return model
model = Unet(img_size , img_size , 64)

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=['accuracy'])
mc = ModelCheckpoint(mode='max', filepath='top-weights.h5', monitor='val_acc',save_best_only='True', save_weights_only='True', verbose=1)

es = EarlyStopping(mode='max', monitor='val_acc', patience=10, verbose=0)

tb = TensorBoard(log_dir="logs/", histogram_freq=0, write_graph=True, write_images=False)

rl = ReduceLROnPlateau(monitor='val_acc',factor=0.1,patience=5,verbose=1,mode="max",min_lr=0.0001)

cv = CSVLogger("logs/log.csv" , append=True , separator=',')
results = model.fit_generator(train_generator , steps_per_epoch=train_steps ,epochs=30,

                              validation_data=val_generator,validation_steps=val_steps,callbacks=[mc,es,tb,rl,cv])
img_mask = choice(val_pair)

img= img_to_array(load_img(img_mask[0] , target_size= (img_size,img_size)))

gt_img = img_to_array(load_img(img_mask[1] , target_size= (img_size,img_size)))
def make_prediction(model,img_path,shape):

    img= img_to_array(load_img(img_path , target_size= shape))/255.

    img = np.expand_dims(img,axis=0)

    labels = model.predict(img)

    labels = np.argmax(labels[0],axis=2)

    return labels
pred_label = make_prediction(model, img_mask[0], (img_size,img_size,3))

pred_label.shape
def form_colormap(prediction,mapping):

    h,w = prediction.shape

    color_label = np.zeros((h,w,3),dtype=np.uint8)    

    color_label = mapping[prediction]

    color_label = color_label.astype(np.uint8)

    return color_label
pred_colored = form_colormap(pred_label,np.array(class_map))
plt.figure(figsize=(15,15))

plt.subplot(131);plt.title('Original Image')

plt.imshow(img/255.)

plt.subplot(132);plt.title('True labels')

plt.imshow(gt_img/255.)

plt.subplot(133)

plt.imshow(pred_colored/255.);plt.title('predicted labels')