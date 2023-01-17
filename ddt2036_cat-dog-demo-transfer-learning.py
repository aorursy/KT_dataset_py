# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from skimage import io
from tqdm import tqdm
import imgaug as ia
from imgaug import augmenters as iaa

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import random
import os,shutil

src_path="../input"

print(os.listdir(src_path))
#print(os.listdir("../input/test1/test1"))
#constant value
VALID_SPIT=0.2
IMAGE_SIZE=224
BATCH_SIZE=64
CHANNEL_SIZE=3
SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE)
# Any results you write to the current directory are saved as output.
label=[]
data=[]
counter=0
path="../input/train/train"
for file in os.listdir(path):
    data.append(os.path.join(path,file))
    if file.startswith("cat"):
        label.append(0)
    elif file.startswith("dog"):
        label.append(1)
        
    counter+=1
    if counter%1000==0:
        print (counter," image data retreived")

data=np.array(data)
label=np.array(label)
img_dir="../input/train/train"
img_list=os.listdir(img_dir)
img_size=IMAGE_SIZE
sum_r=0
sum_g=0
sum_b=0
count=0

for img_name in img_list:
    img_path=os.path.join(img_dir,img_name)
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(img_size,img_size))
    sum_r=sum_r+img[:,:,0].mean()
    sum_g=sum_g+img[:,:,1].mean()
    sum_b=sum_b+img[:,:,2].mean()
    count=count+1
    if count%1000==0:
        print (count," image data count")
        
sum_r=sum_r/count
sum_g=sum_g/count
sum_b=sum_b/count
img_mean=[sum_r,sum_g,sum_b]
print (img_mean)
img_mean = [124.40483277264002, 115.92854629783018, 106.20628246173563]
from sklearn.model_selection import train_test_split
train_data, valid_data, train_label, valid_label = train_test_split(
    data, label, test_size=0.2, random_state=42)
print(train_data.shape)
print(train_label.shape)
print(valid_data.shape)
print(valid_label.shape)
import seaborn as sns

sns.countplot(train_label)
pd.Series(train_label).value_counts()
sns.countplot(valid_label)
pd.Series(valid_label).value_counts()
import keras
from keras import Sequential
from keras.layers import *
import keras.optimizers as optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from keras.applications import *
from keras import Model
# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class CacheDataGenerator(keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False, augment = False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]
                
        if self.augment == True:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5), # horizontal flips
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

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)
        
        return X, y
    
    def on_epoch_end(self):
        
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        im = np.array(Image.open(path))
        im = cv2.resize(im, (SHAPE[0], SHAPE[1]))
        
        #for dim in range(3):
        #    im[:,:,dim] = im[:,:,dim] - img_mean[dim] 
        #im = np.divide(im, 255)
        return im
base_model = resnet50.ResNet50(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE),
                                           include_top=False, pooling='avg')

x = base_model.output
x = Dropout(0.5)(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()
#optimizers.SGD(lr=1e-3, momentum=0.9)
model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])
train_generator = CacheDataGenerator(train_data, train_label, BATCH_SIZE, SHAPE, use_cache=True, augment = False, shuffle = False)
valid_generator = CacheDataGenerator(valid_data, valid_label, BATCH_SIZE, SHAPE, use_cache=True, shuffle = False)
callack_saver = ModelCheckpoint(
            "model.h5"
            , monitor='val_acc'
            , verbose=0
            , save_weights_only=True
            , mode='auto'
            , save_best_only=True
        )
train_history=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=2,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=[callack_saver])
for layer in base_model.layers:
    layer.trainable = True
model.summary()

opt_sgd = optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=opt_sgd,loss="binary_crossentropy",metrics=["accuracy"])
train_history=model.fit_generator(
        train_generator,
        steps_per_epoch=100,#len(train_generator),
        epochs=10,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=[callack_saver])
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history, 'loss', 'val_loss')
show_train_history(train_history, 'acc', 'val_acc')
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
            "../input/test1"
            , target_size=(IMAGE_SIZE, IMAGE_SIZE)
            , batch_size=100
            , shuffle=False
        )
predicted_labels=model.predict_generator(test_generator, steps=125)
predicted_labels=np.round(predicted_labels,decimals=2)
labels=[1 if value>0.5 else 0 for value in predicted_labels]

#print(len(labels))
id=[os.path.splitext(os.path.basename(filename))[0] for filename in test_generator.filenames]
dataframe_output=pd.DataFrame({"id":id})
dataframe_output["label"]=labels
print(dataframe_output)
dataframe_output.to_csv("submission.csv",index=False)