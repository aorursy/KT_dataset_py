import numpy as np 
import pandas as pd 

import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import imageio
import matplotlib.pyplot as plt

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import binary_accuracy
import shutil
%matplotlib inline
shen_image_list = os.listdir('../input/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png')
mont_image_list = os.listdir('../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png')
image_list = mont_image_list + shen_image_list
df = pd.DataFrame(image_list, columns=['image_id'])

df = df[df['image_id'] != 'Thumbs.db']

df.reset_index(inplace=True, drop=True)
def extract_target(x):
    target = int(x[-5])
    if target == 0:
        return 0
    if target == 1:
        return 1

df['labels'] = df['image_id'].apply(extract_target)

df = shuffle(df)
y = df['labels']

df_train, df_val = train_test_split(df, test_size=0.15, random_state=101, stratify=y)
try:
    base_dir = 'base_dir'
    os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train_dir')
    os.mkdir(os.path.join(base_dir, 'train_dir'))

    val_dir = os.path.join(base_dir, 'val_dir')
    os.mkdir(os.path.join(base_dir, 'val_dir'))

    os.mkdir(os.path.join(train_dir, 'Normal'))
    os.mkdir(os.path.join(train_dir, 'Tuberculosis'))

    os.mkdir(os.path.join(val_dir, 'Normal'))
    os.mkdir(os.path.join(val_dir, 'Tuberculosis'))
except:
    print ("Done")
df.set_index('image_id', inplace=True)
f1 = os.listdir('../input/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png')
f2 = os.listdir('../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png')

def getpath(image_id):
    if image_id in list(df_train['image_id']):
        if df.loc[image_id,'labels']:
            return 'base_dir/train_dir/Tuberculosis/{}'.format(image_id)
        else:
            return 'base_dir/train_dir/Normal/{}'.format(image_id)
    else:
        if df.loc[image_id,'labels']:
            return 'base_dir/val_dir/Tuberculosis/{}'.format(image_id)
        else:
            return 'base_dir/val_dir/Normal/{}'.format(image_id)

IMAGE_RES = 128

for image_id in f1:
    if image_id == 'Thumbs.db':
        continue
    src = '../input/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/{}'.format(image_id)
    dest = getpath(image_id)
    image = cv2.imread(src)
    image = cv2.resize(image, (IMAGE_RES, IMAGE_RES))
#     plt.imshow(image)
    cv2.imwrite(dest, image)

for image_id in f2:
    if image_id == 'Thumbs.db':
        continue
    src = '../input/pulmonary-chest-xray-abnormalities//Montgomery/MontgomerySet/CXR_png/{}'.format(image_id)
    dest = getpath(image_id)
    image = cv2.imread(src)
    image = cv2.resize(image, (IMAGE_RES, IMAGE_RES))
#     plt.imshow(image)
    cv2.imwrite(dest, image)

print(len(os.listdir('base_dir/train_dir/Normal')))
print(len(os.listdir('base_dir/train_dir/Tuberculosis')))

class_list = ['Normal','Tuberculosis']

for item in class_list:
    
    try:
        aug_dir = 'aug_dir'
        os.mkdir(aug_dir)
        img_dir = os.path.join(aug_dir, 'img_dir')
        os.mkdir(img_dir)
    except:
        pass

    img_class = item

    img_list = os.listdir('base_dir/train_dir/' + img_class)

    for fname in img_list:
            src = os.path.join('base_dir/train_dir/' + img_class, fname)
            dst = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst)


    path = aug_dir
    save_path = 'base_dir/train_dir/' + img_class

    datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,save_to_dir=save_path,save_format='png',target_size=(IMAGE_RES,IMAGE_RES),batch_size=batch_size)
    
    num_files = len(os.listdir(img_dir))
    
    num_batches = int(np.ceil((1000-num_files)/batch_size))

    for i in range(0,num_batches):
        imgs, labels = next(aug_datagen)
        
    shutil.rmtree('aug_dir')

train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(train_path,target_size=(IMAGE_RES,IMAGE_RES),batch_size=train_batch_size,class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,target_size=(IMAGE_RES,IMAGE_RES),batch_size=val_batch_size,class_mode='categorical')

test_gen = datagen.flow_from_directory(valid_path,target_size=(IMAGE_RES,IMAGE_RES),batch_size=val_batch_size,class_mode='categorical',shuffle=False)
kernel_size = (3,3)
pool_size= (2,2)

dropout_conv = 0.1
dropout_dense = 0.1

model = Sequential()
model.add(Conv2D(32, kernel_size, activation = 'relu', input_shape = (IMAGE_RES, IMAGE_RES, 3)))
model.add(Conv2D(32, kernel_size, activation = 'relu'))
model.add(Conv2D(32, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(64, kernel_size, activation ='relu'))
model.add(Conv2D(64, kernel_size, activation ='relu'))
model.add(Conv2D(64, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(128, kernel_size, activation ='relu'))
model.add(Conv2D(128, kernel_size, activation ='relu'))
model.add(Conv2D(128, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()
model.compile(Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_gen, steps_per_epoch=train_steps, validation_data=val_gen, validation_steps=val_steps, epochs=100, verbose=1,callbacks=callbacks_list)
val_loss, val_acc = model.evaluate_generator(test_gen, steps=val_steps)

print('val_loss:', val_loss)
print('val_acc:', val_acc)

