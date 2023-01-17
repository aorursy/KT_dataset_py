import cv2
import os
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import io
from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_WIDTH=200
IMAGE_HEIGHT=200
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
BATCH_SIZE=32
EPOCHS=3
filenames_with = os.listdir("../input/dataset-for-mask-detection/dataset/with_mask")
filenames_without = os.listdir("../input/dataset-for-mask-detection/dataset/without_mask")
filenames_list_with = []
filenames_list_without = []
categories_with = []
categories_without = []
for filename in filenames_with:
    filenames_list_with.append("../input/dataset-for-mask-detection/dataset/with_mask/" + filename)
    categories_with.append(str(1))
for filename in filenames_without:
    filenames_list_without.append("../input/dataset-for-mask-detection/dataset/without_mask/" + filename)
    categories_without.append(str(0))
    

df_w = pd.DataFrame({
    'image': filenames_list_with,
    'category': categories_with
})
df_wo = pd.DataFrame({
    'image': filenames_list_without,
    'category': categories_without
})
print(df_w.shape, df_wo.shape)
df = df_w.append(df_wo)
print(df)
#split data into train and valid set
train_df, valid_df = train_test_split(df, test_size = 0.15, stratify = df['category'], random_state = 3)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
total_train = train_df.shape[0]
total_valid = valid_df.shape[0]
print(train_df.shape)
print(valid_df.shape)
#We'll perform individually on train and validation set.
train_datagen = ImageDataGenerator(
                                   zoom_range = 0.3,
                                   rescale=1./255,
#                                    horizontal_flip = True,
                                   )
#fill_mode : 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식

train_gen = train_datagen.flow_from_dataframe(train_df,
                                              x_col = 'image',
                                              y_col = 'category',
                                              target_size = IMAGE_SIZE,
                                              batch_size = BATCH_SIZE,
                                              class_mode='binary',
                                              validate_filenames=False
                                             )

#we do not augment validation data.
validation_datagen = ImageDataGenerator(rescale=1./255)
valid_gen = validation_datagen.flow_from_dataframe(
    valid_df, 
    x_col="image",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    validate_filenames=False
)

def create_model():
    model = Sequential()
    model.add(ResNet152V2(include_top = False, pooling = 'max', weights = 'imagenet'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.layers[0].trainable = False 
    model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')    
    return model

model = create_model()
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
checkpointer = ModelCheckpoint(filepath = 'mask.weights.best.hdf5', save_best_only = True, save_weights_only = True)
callbacks = [learning_rate_reduction]
model.fit_generator(train_gen,
                    epochs = EPOCHS,
                    validation_data = valid_gen,
                    validation_steps=total_valid//BATCH_SIZE,
                    steps_per_epoch=total_train//BATCH_SIZE,
                    callbacks = callbacks)
loss = pd.DataFrame(model.history.history)
loss[['loss', 'val_loss']].plot()
loss[['accuracy', 'val_accuracy']].plot()
# add test data

PATH2= '../input/face-mask-detection/dataset/'
with_mask2 = os.listdir(PATH2+"with_mask")
without_mask2 = os.listdir(PATH2+"without_mask")

filenames_with2 = os.listdir(PATH2 + "with_mask")
filenames_without2 = os.listdir(PATH2 + "without_mask")
filenames_list_with2 = []
filenames_list_without2 = []
category_11 = []
category_22 = []
for filename in filenames_with2:
    filename = PATH2 + 'with_mask/' + filename
    
    image = Image.open(filename)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = np.array(image)
    
    if(image.shape[2] == 4):
        print('rgba detected')
        continue
        
    filenames_list_with2.append(filename)
    category_11.append(str(1))
for filename in filenames_without2:
    filename = PATH2 + 'without_mask/' + filename
    
    image = Image.open(filename)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = np.array(image)
    
    if(image.shape[2] == 4):
        print('rgba detected')
        continue
    filenames_list_without2.append(filename)
    category_22.append(str(0))
    
    

df_w2 = pd.DataFrame({
    'image': filenames_list_with2,
    'category': category_11
})
df_wo2 = pd.DataFrame({
    'image': filenames_list_without2,
    'category': category_22
})

test_df = pd.concat([df_w2, df_wo2], ignore_index=True)

# append 로 진행했을때 index가 duplicate 되는 문제가 있었음.
# test_df = df_w2.append(df_wo2)
print(test_df['image'][0])
print(df_w2.shape)
print('------------------------------')
print(df_wo2.shape)
print('------------------------------')
print(test_df.shape)
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    x_col='image',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validate_filenames=False
)
predict = model.predict_generator(test_generator)
print(predict)
masked_sum = 0
unmasked_sum =0
masked_misjudge = 0
unmasked_misjudge = 0
masked_image_but_misjudge = []
unmasked_image_but_misjudge = []

for (idx,val) in enumerate(predict):
    if(idx < 220):
        if(val > 0.85):
            masked_sum += 1
            print('mask', val, idx)
        else:
            masked_image_but_misjudge.append(idx)
            masked_misjudge += 1
            print('mask_misjudge', val, idx)
            
    else:
        if(val < 0.85):
            unmasked_sum += 1
            print('unmask', val, idx)
        else:
            unmasked_misjudge += 1
            unmasked_image_but_misjudge.append(idx)
            print('unmask_misjudge', val, idx)
# 마스크를 쓰지 않았지만 마스크를 썼다고 잘못 판단된 이미지들을 보여줍니다.

f=plt.figure(figsize=(10, 10))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    try:
        f.add_subplot(rows, columns, i)
        num = unmasked_image_but_misjudge.pop()
        image = mpimg.imread(test_df['image'][num])
        plt.imshow(image)
    except IndexError:
        break
plt.show()
# 마스크를 썼지만 마스크를 쓰지 않았다고 잘못 판단된 이미지들을 보여줍니다.

f=plt.figure(figsize=(10, 10))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    try:
        f.add_subplot(rows, columns, i)
        num = masked_image_but_misjudge.pop()
        image = mpimg.imread(test_df['image'][num])
        plt.imshow(image)
    except IndexError:
        break
plt.show()
# face detection with opencv
def face_detection(img):
    
    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
        
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x-20,y-20),((x+w)+20,(y+h)+20),(255,0,0),2)
        img = img[y-20:y+h+20, x-20:x+w+20] # for cropping
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv_rgb
# img = mpimg.imread(test_df['image'][1])
plt.figure(figsize=(5,5))
img = cv2.imread("../input/dataset-for-mask-detection/dataset/with_mask/100.jpeg")
c=face_detection(img)
plt.imshow(c)
