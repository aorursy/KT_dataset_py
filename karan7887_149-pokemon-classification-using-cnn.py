import numpy as np
import pandas as pd
import os,shutil
# shutil.rmtree('/kaggle/working/train_images') 
# shutil.rmtree('/kaggle/working/val_images')
parent = '../input/pokemon-generation-one/dataset'
folders = os.listdir(parent)
new_folders = []
for i in folders:
    if i=='dataset':
        continue
    new_folders.append(i)
len(new_folders)
lis = ['train_images','val_images','train_new_images']
pat = '/kaggle/working'
for f in lis:
    if not os.path.isdir(os.path.join(pat,f)):
        os.mkdir(os.path.join(pat,f))
os.listdir('/kaggle/working')
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
new_folders

for c in new_folders:
    pat = os.path.join('/kaggle/working/train_images',c)
    if not os.path.isdir(pat):
        os.mkdir(pat)
    pat = os.path.join('/kaggle/working/val_images',c)
    if not os.path.isdir(pat):
        os.mkdir(pat)
    pat = os.path.join('/kaggle/working/train_new_images',c)
    if not os.path.isdir(pat):
        os.mkdir(pat)
len(os.listdir('/kaggle/working/train_new_images'))
train_path = os.path.join('/kaggle/working','train_images')
val_path = os.path.join('/kaggle/working','val_images')
split = 0.7
for image_folder in new_folders:
    p = os.path.join(parent,image_folder)
    pat_train = os.path.join(train_path,image_folder)
    pat_val = os.path.join(val_path,image_folder)
    if not os.path.join(train_path,image_folder):
        os.mkdir(os.path.join(train_path,image_folder))
    if not os.path.join(val_path,image_folder):
        os.mkdir(os.path.join(val_path,image_folder))
    images = os.listdir(p)
    length = int(split*(len(images)))
    train_images = images[:length]
    val_images = images[length:]
    for proper_img in train_images:
        sub = str(proper_img[-4:])
        if sub == '.jpg' or sub == '.png' or sub == 'jpeg':
            src = os.path.join(p,proper_img)
            des = os.path.join(pat_train,proper_img)
            shutil.copy(src,des)
    for proper_img in val_images:
        sub = str(proper_img[-4:])
        if sub == '.jpg' or sub == '.png' or sub == 'jpeg':
            src = os.path.join(p,proper_img)
            des = os.path.join(pat_val,proper_img)
            shutil.copy(src,des)
des = os.path.join('/kaggle/working/train_images')
lis = os.listdir(des)
print(len(lis))
des = os.path.join('/kaggle/working/val_images')
lis = os.listdir(des)
print(len(lis))
lis = os.listdir('/kaggle/working/train_images')
count = []
for folder in lis:
    p = os.path.join('/kaggle/working/train_images',folder)
    count.append(len(os.listdir(p)))
    print(str(folder) + ' count is :'+str(len(os.listdir(p))))
print(min(count))
datagen = ImageDataGenerator(rescale=1.0/255,
                            width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             rotation_range=40,
                             fill_mode='nearest'
                            )
from PIL import Image
import math
train_path = '/kaggle/working/train_images'
for folders in os.listdir(train_path):
#     defining the image path or the path of folder in which images are present
    images_path = os.path.join(train_path,folders)
#     counting the numvber of image si particular folder
    img_count = len(os.listdir(images_path))
    if(img_count <= 107):
        img_arr = os.listdir(images_path)
        
        for img in img_arr:
            
          
            
            
            
            img_ = image.load_img(os.path.join(images_path,img),target_size=(240,240))
            img_ = image.img_to_array(img_)
            img_ = img_.reshape(1,240,240,3)
            
            limit = np.floor(213/img_count)
           
            i = 0
            for x in datagen.flow(img_,batch_size=1,save_to_dir = images_path,save_prefix = folders,save_format = 'jpg'):
                i += 1
                x = x.reshape(240,240,3)
                img = Image.fromarray(x,'RGB')
                pathii = os.path.join(images_path,'save.png')
                img.save(pathii)
                if i>=limit:
                    break
        
        
train_path = '/kaggle/working/train_images'
for folder in os.listdir(train_path):
    train_new_path = os.path.join(train_path,folder)
    images = os.listdir(train_new_path)
#     split = 100
    images = images[:100]
    for img in images:
        src = os.path.join(train_new_path,img)
        d = os.path.join('/kaggle/working/train_new_images',folder)
        des = os.path.join(d,img)
        shutil.move(src,des)
count = []
lis = os.listdir('/kaggle/working/train_new_images')
for folder in lis:
    p = os.path.join('/kaggle/working/train_new_images',folder)
    count.append(len(os.listdir(p)))
    print(str(folder) + ' count is :'+str(len(os.listdir(p))))
print(min(count))
train_generator = datagen.flow_from_directory(
                    '/kaggle/working/train_new_images',
    
                    class_mode = 'categorical',
                    batch_size = 128,
                    shuffle = True,
                    target_size = (240,240),
                    )
val_generator = datagen.flow_from_directory(
  
                    directory='/kaggle/working/val_images',
                    class_mode = 'categorical',
                    batch_size = 128,
                    shuffle = True,
                    target_size = (240,240),
                    )
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import *
model = ResNet50(include_top=False,weights='imagenet',input_shape = (240,240,3))
model.summary()
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras import Model
layer1 = GlobalAveragePooling2D()(model.output)
layer2 = Dense(1000,activation = 'relu')(layer1)
layer3 = Dense(500,activation='relu')(layer2)
layer_out = Dense(149,activation='softmax')(layer3)

model_new = Model(inputs=model.input,outputs = layer_out)


model_new.summary()
model_new.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# hist = model_new.fit_generator?
# hist = model_new.fit_generator
hist = model_new.fit_generator(train_generator,epochs=20,validation_data=val_generator)
hist = model_new.fit_generator(train_generator,epochs=1,validation_data=val_generator)