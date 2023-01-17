import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import cv2
import gc
import random
import tensorflow as tf
import keras.backend as K
import imgaug as ia

from tqdm import tqdm
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras import regularizers
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
root_dir='../input/human-protein-atlas-image-classification'
train_dir=os.path.join(root_dir, "train")
test_dir=os.path.join(root_dir, "test")
train_csv_path=os.path.join(root_dir,"train.csv")
import os
os.listdir("../input/human-protein-atlas-image-classification")
IMAGE_SIZE=224 #256# 512
# read the training csv
train_csv = pd.read_csv(train_csv_path)
print(train_csv.shape)
train_csv.head()
class_names = {
    0:"Nucleoplasm", 1:"Nuclear membrane", 2:"Nucleoli", 3:"Nucleoli fibrillar center", 4:"Nuclear speckles", 
    5:"Nuclear bodies",  6:"Endoplasmic reticulum", 7:"Golgi apparatus", 8:"Peroxisomes", 9:"Endosomes", 
    10:"Lysosomes", 11:"Intermediate filaments", 12:"Actin filaments", 13:"Focal adhesion sites",14:"Microtubules", 
    15:"Microtubule ends", 16:"Cytokinetic bridge", 17:"Mitotic spindle", 18:"Microtubule organizing center", 
    19:"Centrosome", 20:"Lipid droplets", 21:"Plasma membrane", 22:"Cell junctions", 23:"Mitochondria", 
    24:"Aggresome",  25:"Cytosol", 26:"Cytoplasmic bodies", 27:"Rods & rings" 
}
# split the targets in train csv
def split_classes(row):
    for cls_num in row["Target"].split():
        train_csv.loc[row.name, class_names[int(cls_num)]]=1

for cls_num, cls_name in class_names.items():
    train_csv[cls_name]=0

# train_csv["splitted"] = train_csv["Target"].apply(lambda x: i+1 for i in x.split())
train_csv.apply(split_classes, axis=1)
train_csv.head()

#DISABLE BELOW
# train_csv = pd.read_csv(root_dir+"/train_csv.csv")
# train_csv.head()
# load the data batch wise from the disk on the fly
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class BatchDataGenerator(Sequence):
    # 'Generates data for Keras'
    def __init__(self, paths, labels, batch_size, shape, shuffle = True, augment = False, load_green_only=True, 
                 load_3chnls=False, return_paths=False):
         # 'Initialization'
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.augment = augment
        self.load_green_only = load_green_only
        self.load_3chnls = load_3chnls
        self.on_epoch_end()
        self.return_paths = return_paths
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / self.batch_size))
    
    def __getitem__(self, idx):
        # Generate indexes of a batch
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]        
        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        for i, path in enumerate(paths):
            X[i] = self.__load_image(path)
            
        y = self.labels[indexes]
                
        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5), # horizontal flips
                    iaa.Flipud(0.5),
                    iaa.Crop(percent=(0, 0.1)), # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X),  seq.augment_images(X)), 0)#, seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y),0) # y, y), 0)
            
        if self.return_paths:
            return paths, X, y
        else:
            return X, y
    
    def __load_image(self, path):
        if self.load_green_only:
            im = cv2.imread(path + '_green.png')
            cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
            im.resize(IMAGE_SIZE, IMAGE_SIZE, 1)
        elif load_3chnls:
            all_images = np.empty((512,512,3))
            reds = plt.imread(path + '_red.png')
            greens = plt.imread(path + '_green.png')
            blues = plt.imread(path + '_blue.png')
             
            all_images[:,:,0] = reds
            all_images[:,:,1] = greens
            all_images[:,:,2] = blues
            
            im = all_images.reshape(all_images.shape[0], all_images.shape[0], 3)
            im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            all_images = np.empty((512,512,4))
            all_images[:,:,0] = cv2.imread(path + '_red.png')
            all_images[:,:,1] = cv2.imread(path + '_green.png')
            all_images[:,:,2] = cv2.imread(path + '_blue.png')
            all_images[:,:,3] = cv2.imread(path + '_yellow.png')

            # define transformation matrix
            # note that yellow is made usign red and green
            # but you can tune this color conversion yourself
            T = np.array([
                #r g y b
                [1,0,1,0],
                [0,1,1,0],
                [0,0,0,1]])
            
            rgb_image = np.matmul(all_images.reshape(-1, 4), np.transpose(T))
            rgb_image = rgb_image.reshape(all_images.shape[0], all_images.shape[0], 3)
            rgb_image = np.clip(rgb_image, 0, 1)
            
            
            im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        return im
    
    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
# Load the images in train and test
import platform

if platform.system() == 'Windows':
    train_csv["train_paths"] =  train_dir + "\\" + train_csv["Id"].astype(str)
else:
    train_csv["train_paths"] =  train_dir + "/" + train_csv["Id"].astype(str)

gc.collect()
shuffle(train_csv)

MAX_IMG_FOR_MODELING = 31072 #15000 # choosing only this much images to avoid memory error or timeout error
shuffle(train_csv)
subset_data=train_csv[:MAX_IMG_FOR_MODELING]

train_paths, valid_paths, train_labels, valid_labels = train_test_split(subset_data["train_paths"].values, 
                                                                    subset_data[list(class_names.values())].values, 
                                                                    test_size=0.25)
                                                                    #stratify=subset_data[class_names.values()].values)
print(train_paths.shape)
print(train_labels.shape)
print(valid_paths.shape)
print(valid_labels.shape)

print(train_paths[:5])
print(train_labels[:5])
# check class wise train and valid distrib
tc = train_csv.set_index("Id")
train_ids = []
valid_ids = []

def get_distrib(dataset, lst):
    for path in dataset:
        lst.append(path.split(os.path.sep)[-1])
        
    data = train_csv[train_csv["Id"].isin(lst)]
    counts=data[list(class_names.values())].sum()
    counts = counts.to_frame("cnts")
    plt.figure(figsize=(20, 10))
    sns.barplot(counts.index, counts.cnts)
    plt.xticks(rotation=70)
    
    counts.columns = ["counts"]
    print(counts.shape)
    print(counts)
    return counts
        
td = get_distrib(train_paths, train_ids)
plt.show()

vd = get_distrib(valid_paths, valid_ids)
plt.show()

y_integers = np.argmax(train_labels, axis=1)

weights = train_labels.shape[0]/(len(class_names)*td["counts"].values)
class_weights = {}
for cls_num, w in zip(class_names.keys(),weights):
    class_weights[cls_num] = w

print(class_weights)
# Data generation

batch_size=16

load_green_only = False
load_3chnls = True
img_shape = (IMAGE_SIZE, IMAGE_SIZE, 1) if load_green_only else (IMAGE_SIZE, IMAGE_SIZE, 3) \
                                        if load_3chnls else (IMAGE_SIZE, IMAGE_SIZE, 3)
print(img_shape)
train_gen = BatchDataGenerator(train_paths, train_labels, batch_size, img_shape, load_green_only=load_green_only, 
                               load_3chnls=load_3chnls, augment=True, shuffle=False)
val_gen = BatchDataGenerator(valid_paths, valid_labels, batch_size, img_shape, load_green_only=load_green_only, 
                             load_3chnls=load_3chnls, augment=True, shuffle=False)
model_dir = './model'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
checkpoint = ModelCheckpoint(os.path.join(model_dir, 'base.model'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
reduce_LR_on_plateu = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D , BatchNormalization, \
                        Input, GaussianNoise, GlobalMaxPooling2D, GlobalAveragePooling2D
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# Train vs Validation accuracy and loss

def loss_over_epochs(model, loss_type='loss', add_valid=True):
    hist=model.history.history
    plt.plot(list(range(epochs)), hist['loss'], color="blue", label="train")
    if add_valid:
        plt.plot(list(range(epochs)), hist['val_loss'], color="orange", label="valid")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Losses over the Epochs")

# accuracy vs epochs
def acc_over_epochs(model, acc_type='acc', add_valid=True):
    hist=model.history.history
    plt.plot(list(range(epochs)), hist[acc_type], color="blue", label="train")
    if add_valid:
        plt.plot(list(range(epochs)), hist["val_"+acc_type], color="orange", label="valid")
    plt.xlabel("Epochs")
    plt.ylabel(acc_type)
    plt.legend()
    plt.title(acc_type + " over the Epochs")
vgg_model = VGG16(include_top=False, input_shape=img_shape)
vgg_model.summary()
gaus_noise = 0.1
DENSE_COUNT = 1024

vgg_model.trainable = False
in_layer = Input(shape=(img_shape))
noise_layer =  GaussianNoise(0.1)(in_layer)
features_layer = vgg_model(noise_layer)
batch_norm = BatchNormalization()(features_layer)
#gmp_dr = GlobalMaxPooling2D()(batch_norm)
x = GlobalAveragePooling2D()(batch_norm)
x = Dense(DENSE_COUNT, activation = 'relu')(x)
predictions = Dense(len(class_names), activation = 'sigmoid')(x)
model = Model(inputs = [in_layer], outputs = [predictions], name = 'vgg_gnoise_model')
model.summary()
opt = Adam(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy", f1])
gc.collect()
epochs = 30

use_multiprocessing = True # DO NOT COMBINE MULTIPROCESSING WITH CACHE! 
workers = 6 # DO NOT COMBINE MULTIPROCESSING WITH CACHE! 

model.fit_generator(
                    train_gen,
                    steps_per_epoch=len(train_gen), 
                    validation_data=val_gen,
                    validation_steps=50,
                    epochs=epochs,
                    use_multiprocessing=use_multiprocessing,
                    workers=workers,
                    class_weight=class_weights,
                    callbacks=[checkpoint, reduce_LR_on_plateu])
loss_over_epochs(model)
plt.show()
acc_over_epochs(model)
plt.show()
acc_over_epochs(model, acc_type="f1")
plt.show()
acc_over_epochs(model, add_valid=False)
# Get the test data
test_paths = []
sample_test_labels = []

sample_submit = os.path.join(root_dir, 'sample_submission.csv')
data = pd.read_csv(sample_submit)
    
for name in data['Id']:
    y = np.ones(28)
    test_paths.append(os.path.join(test_dir, name))
    sample_test_labels.append(y)

test_paths = np.array(test_paths)
sample_test_labels = np.array(sample_test_labels)

print(test_paths[:3])
print(sample_test_labels[:3])
def load_test_images(load_all=False):
    """
    If load_all==False then it will return the iterator or it will return the all test images in numpy array.
    """
    test_gen = BatchDataGenerator(test_paths, sample_test_labels, 1, img_shape, load_green_only=load_green_only, 
                               load_3chnls=load_3chnls, augment=False, shuffle=False, return_paths=True)
    if load_all:
        images = []
        for img in test_gen:
            images.append(img)
        
        return np.array(images)
    
    return test_gen
full_val_pred = np.empty((0, 28))
for i in tqdm(range(len(val_gen))): 
    im, lbl = val_gen[i]
    scores = model.predict(im)
    full_val_pred = np.append(full_val_pred, scores, axis=0)
print(full_val_pred.shape)
# Take "perc" percentile score of each class a threshold
perc = 85

thresholds = np.empty(len(class_names))
for i in range(len(class_names)):
    thresholds[i] = np.percentile(full_val_pred[:,i], perc)
print('Probability threshold for each class score:')
print(thresholds)
preds = {"Id":[], "Predicted":[]}


test_images = load_test_images()

i=0
for img_data in tqdm(test_images):
    paths, test_imgs, labels = img_data
    score = model.predict(test_imgs)
    tmp = []
    for i in range(len(class_names)):
        if score[0][i] >= thresholds[i]:
            tmp.append(str(i))
    preds["Id"].append(paths[0].split(os.path.sep)[-1])
    preds["Predicted"].append(" ".join(tmp))
submit = pd.DataFrame(preds)
print(submit.shape)
print(submit.head(5))
submit.to_csv("submission1.csv", index=False)