!pip install keras==2.4.3
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.utils import shuffle
from getpass import getpass
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from skimage import img_as_float
from scipy.fftpack import fft2, ifft2, fftshift
from collections import OrderedDict, defaultdict
from scipy.ndimage.filters import gaussian_filter, median_filter, maximum_filter, minimum_filter
import random
import os


from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from glob import glob
from tqdm import tqdm
from PIL import Image


path = '/kaggle/input/data/'
bbox_csv = 'BBox_List_2017.csv'
data_csv = 'Data_Entry_2017.csv'
imgs_folders = ['images_001', 'images_002', 'images_003',
       'images_004', 'images_005', 'images_006', 'images_007',
       'images_008', 'images_009', 'images_010', 'images_011',
       'images_012']
Categories = ['No Finding','Atelectasis', 'Cardiomegaly', 'Consolidation',
        'Edema', 'Effusion', 'Fibrosis', 'Infiltration', 'Mass',
        'Pneumothorax', 'Emphysema', 'Pneumonia', 'Pleural_Thickening',
        'Nodule', 'Hernia']


def Add_path(df, path = '/kaggle/input/data/images*/images/*.png'):
    my_glob = glob(path)
    full_img_paths = {os.path.basename(x): x for x in my_glob}
    dataset_path = df['Image Index'].map(full_img_paths.get)
    df['full_path'] = dataset_path.astype(str)
    return df


def Adjust_data(data):
    data = data[['Image Index', 'Finding Labels']]
    #data = data[data['Finding Labels'] != 'No Finding']
    new_data = data.rename(columns={'Finding Labels': 'All Labels'})
    return new_data


def dfcat2dfid(df, Categories): # change dataframe of category names into category numbers
    cat2id = {i:j for j,i in enumerate(Categories)}
    id2cat = {i:j for i,j in enumerate(Categories)}

    All_Cat = df['All Labels'].values.astype(str)
    All_cat_list = [i.split('|') for i in All_Cat]

    mcs_All_Labels = np.array([np.array([cat2id[p] for p in o]) for o in All_cat_list])

    df['Class_All'] = mcs_All_Labels
    
    return df

df = pd.read_csv(path + data_csv)

df = Adjust_data(df)
df = dfcat2dfid(df, Categories)
df = Add_path(df)

df.head()
Negative = df[df['All Labels'] == 'No Finding'][:1000]
Cardiomegaly = df[df['All Labels']=='Cardiomegaly'][:1000]
Emphysema = df[df['All Labels']=='Emphysema']
Hernia = df[df['All Labels']=='Hernia']
Effusion = df[df['All Labels']=='Effusion'][:1000]
Atelectasis = df[df['All Labels']=='Atelectasis'][:1000]
Consolidation = df[df['All Labels']=='Consolidation'][:1000]
Pleural_Thickening = df[df['All Labels']=='Pleural_Thickening'][:1000]
Nodule = df[df['All Labels']=='Nodule'][:1000]
Fibrosis = df[df['All Labels']=='Fibrosis']
Infiltration = df[df['All Labels']=='Infiltration'][:1000]
Mass = df[df['All Labels']=='Mass'][:1000]
Pneumothorax = df[df['All Labels']=='Pneumothorax'][:1000]
Pneumonia = df[df['All Labels']=='Pneumonia']
Edema = df[df['All Labels']=='Edema']
print('length of No Finding = {}\n'.format(len(Negative)))
print('length of Cardiomegaly = {}\n'.format(len(Cardiomegaly)))
print('length of Emphysema = {}\n'.format(len(Emphysema)))
print('length of Hernia = {}\n'.format(len(Hernia)))
print('length of Effusion = {}\n'.format(len(Effusion)))
print('length of Atelectasis = {}\n'.format(len(Atelectasis)))
print('length of Consolidation = {}\n'.format(len(Consolidation)))
print('length of Pleural_Thickening = {}\n'.format(len(Pleural_Thickening)))
print('length of Mass = {}\n'.format(len(Mass)))
print('length of Infiltration = {}\n'.format(len(Infiltration)))
print('length of Pneumonia = {}\n'.format(len(Pneumonia)))
print('length of Pneumothorax = {}\n'.format(len(Pneumothorax)))
print('length of Nodule = {}\n'.format(len(Nodule)))
print('length of Fibrosis = {}\n'.format(len(Fibrosis)))
print('length of Edema = {}\n'.format(len(Edema)))
mixed = df[df['Class_All'].map(len) > 1]

New_df = pd.concat([Negative, Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
                    Fibrosis, Infiltration, Mass, Pneumothorax, Emphysema,
                    Pneumonia, Pleural_Thickening, Nodule, Hernia], ignore_index = True )
print(len(New_df))
New_df.head()
total_df = pd.concat([New_df, mixed], ignore_index = True)

print(len(total_df))
total_df.head()
mc = []
for i in New_df['Class_All'].values:
    c = np.zeros(len(Categories) ,dtype=int)
    c[i] = 1 
    mc.append(c)

for i in mixed['Class_All'].values:
    c = np.zeros(len(Categories) ,dtype=int)
    for p in i:
        c[p]=1
    mc.append(c)
mc = np.asarray(mc)

mc_df =pd.DataFrame(mc,columns=Categories)

print(len(mc_df))
mc_df.head()
data_df = shuffle(pd.concat([total_df, mc_df], axis=1).drop(columns = ['All Labels', 'Class_All'])).reset_index(drop=True)
print(len(data_df))
data_df.head()
def Split_Train_Valid(df,Split_train_val=0.7):
    # step 1: shuffle the data
    df = df.reindex(np.random.permutation(df.index))
    df=df.set_index(np.arange(len(df)))
    
    # step 2: split in training and testing
    df_train = df[:int(len(df)*Split_train_val)]
    df_valid = df[int(len(df)*Split_train_val):]
    df_train=df_train.set_index(np.arange(len(df_train)))
    df_valid=df_valid.set_index(np.arange(len(df_valid)))
    
    return df_train,df_valid

df_train, df_valid = Split_Train_Valid(data_df,0.7)
!wget https://www.kaggleusercontent.com/kf/6702147/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kZj25u0GwqbLC5aHU41g1g.V-9itkt_Aez3KlCyqN1LuLxYe6dCeWYjKuqvhxmI_vy3jeunKtRjwowu1Zd5AhzKFOSUJ2JsUkPQiK2pK_8tGacLrmPI9oNzmhHhV3g7AopvY6KcQ20fXJqQ0ET5QuIsqnR_FijrYLCoL8sifr7k4QyS_3NpnR0Rhadlr3T59MKRUWGsAVvtek0Fr4o9S-OleCjkAxBkFqiObFeYScvYGrnVoVFu6eoy7YVEa0_Q8FW7L-9J4mh7Fn2zFSv5J0D2lzyKS_J2C8uaeRpiPSZkxxHDWWzPeA4kTyE7mxqltSOs7uj9X6EfO5QLjx2FBWD3zIX3HDTHA5uqucxnausr8NDJVYeicQ1pFNzhMwWIVkcYMmrg9AePKmfFTP1hX9Vc_8As11yLd48PFIel-PqIjR1IrWklp7ldKeB-YbGiUgKGJvW6u0JXVDIbPjpI3U6TQ6QGem0w8VkEwUYtiRDyg-fRJocFFwVJlXaTihpuK2iIyk5OMj7i3rfK83n6M_csZdj35lJ8y6FAJbPr4TMuD_6Q5HyXdnzKo9WZyWQ0iKuKU_gAOT62lOPfyCmGr3DT1uHjjT2Ee1p_k9f2YvFCVOpqj6MYR6RS3KkDiEr4OGpeVORXqOpnLYkTfxkfr1ab2ACOLvwgedddJSgJLNSfOS9Vsr7q_GSyMX6ly44jI6s.Wrf_NCxlzPT3k-uMStwWyw/unet_lung_seg.hdf5
# From: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
# From: https://github.com/zhixuhao/unet/blob/master/data.py
def test_load_image(img, target_size=(256,256)):
    s = max(img.shape[0:2])
    f = np.zeros((s,s),np.uint8)
    ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
    img = f
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

seg_model = unet(input_size=(256,256,1))
seg_model.load_weights('unet_lung_seg.hdf5')
from scipy.ndimage.filters import median_filter

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_and_rescale(model, img, path='', target_size=(256,256)):
  image = test_load_image(img, target_size)
  mask = model.predict(image)[0]
  mask[mask < .5] = .0
  mask[mask >= .5] = 1.0
  image = image[0]
  try:
    rmin, rmax, cmin, cmax = bbox(mask)
  except IndexError:
    return cv2.resize(img, target_size)
  cropped_img = image[rmin:rmax, cmin:cmax]
  s = max(cropped_img.shape[0:2])
  f = np.zeros((s,s,1),np.float64)
  ax,ay = (s - cropped_img.shape[1])//2,(s - cropped_img.shape[0])//2
  f[ay:cropped_img.shape[0]+ay,ax:ax+cropped_img.shape[1]] = cropped_img
  cropped_img = cv2.resize(cropped_img, target_size)
  cropped_img = median_filter(cropped_img, 3)
  cropped_img = (cropped_img - np.min(cropped_img)) / (np.max(cropped_img) - np.min(cropped_img))
  # plt.imshow(cropped_img)
  return cropped_img

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
import imgaug as ia
from imgaug import augmenters as iaa


class Data_gen(keras.utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self, df, categories, seg_model, batch_size=32, dim=(256,256), shuffle=True, augment=True):
        'Initialization'
        self.categories = categories
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seg_model = seg_model

        self.df = df
        self.n = len(df)            
        self.nb_iteration = int(np.floor(self.n  / self.batch_size))
        
        self.on_epoch_end()
                    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
   
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        Y = []
        
        # Generate data
        for i, ID in enumerate(index):
            # Read the image
            img = cv2.imread(self.df['full_path'][ID], cv2.IMREAD_GRAYSCALE)
            
            label=[]
            for i in self.categories:
                label.append(self.df[i][ID])
            
            try:
              img  = crop_and_rescale(self.seg_model, img)
            except:
              img = cv2.resize(img, self.dim)/255.0 
                                                                                
            imgray = np.asarray(img)
            
            img = cv2.merge((imgray,imgray,imgray)) 
            
            X.append(np.asarray(img))
            Y.append(np.asarray(label))
            
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if self.augment == True:
          X = self.augmentor(X)
        
        return X, Y#keras.utils.to_categorical(Y, num_classes=2, dtype='uint8')


    def augmentor(self, images):

      'Apply data augmentation'
      sometimes = lambda aug: iaa.Sometimes(0.8, aug)
      seq = iaa.Sequential(
          [
          # apply the following augmenters to most images
          iaa.Fliplr(0.3),  # horizontally flip 50% of all images
          iaa.Flipud(0.3),  # vertically flip 20% of all images
          sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10),  # rotate by -45 to +45 degrees
            shear=(-5, 5),  # shear by -16 to +16 degrees
            order=[0, 1],
            # use nearest neighbour or bilinear interpolation (fast)
            mode=ia.ALL
            # use any of scikit-image's warping modes (see 2nd image from the top for examples)
          )),
          # execute 0 to 5 of the following (less important) augmenters per image
          # don't execute all of them, as that would often be way too strong
          
          ],
          random_order=True
      )
      return seq.augment_images(images)
valid = df_valid
train = df_train

train_gen = Data_gen(df_train, Categories, seg_model)
valid_gen = Data_gen(df_valid, Categories, seg_model)


nb_train_samples = len(train)
nb_validation_samples = len(valid)
a,b = next(iter(train_gen))
plt.imshow(a[2])
b[0]
from keras.applications import ResNet50V2

backbone = ResNet50V2(include_top=False, weights=None, input_tensor=None, input_shape=(256,256,3))

x=backbone.output

prob_map = Conv2D(1, (1, 1), activation='sigmoid', name='classifier')(x)

weight_map = prob_map /  K.sum(prob_map, [2,3], keepdims=True)

feature = x * weight_map

features = K.sum(feature, [2,3], keepdims=True)

a= BatchNormalization()(features)

out1 = Dropout(0.2)(a)

out2 = Conv2D(15, (1, 1), activation='relu', name='classifier_2')(x)

out3 = GlobalAveragePooling2D()(out2)

out = Dense(15, activation='sigmoid')(out3)


Res_model = Model(inputs=backbone.input, outputs=out, name="Resnet34_model")
Res_model.summary()
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#%% compile and train the VGG16 custom model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, SGD

epochs = 20
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.95, nesterov=True)  
Res_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_m,precision_m, recall_m]) 
filepath = Res_model.name + '.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                             save_weights_only=False, save_best_only=True, mode='max', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5,
                              verbose=1, mode='max', min_lr=0.00001)
#tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
callbacks_list = [checkpoint, reduce_lr]

#reset generators
#train_gen.reset()
#validation_gen.reset()

#train the model
history = Res_model.fit_generator(train_gen, steps_per_epoch=nb_train_samples // 32,
                                  epochs=epochs, validation_data=valid_gen,
                                  #class_weight = class_weights,
                                  callbacks=callbacks_list, 
                                  validation_steps=nb_validation_samples // 32, verbose=1)
     
