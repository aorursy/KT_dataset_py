# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

#---------------------------------------------------------------------------------
# Here We will set all parameters for the Jupyter notebook:
#---------------------------------------------------------------------------------
min_object_N = 500

test_validation_split = 0.8 # The training/validation splitting value

scale = 128 # The scale that will be used to resize all images

# The following are the training parameters
epochs = 80 # The Number of Epochs to run for the training
batch_size = 32 # The Batchsize for training
steps_per_epoch = 4000
validation_steps = 1000 #
#---------------------------------------------------------------------------------

# Directory of the Test sets
dir_list = os.listdir("../input/voctest_06-nov-2007/VOCdevkit/VOC2007/ImageSets/Main/")
print(dir_list, len(dir_list))
# This cleans up the directory on the Kaggle computing node
! rm -r test/*
! rm -r validation/*
! rm create_subfolders.sh
! rmdir test
! rmdir validation
! rm *.hdf5
! ls
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil


# The root directory for the data files
root_directory = '../input/voctest_06-nov-2007/VOCdevkit/VOC2007/ImageSets/Main/'
img_directory = '../input/voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'

# The object list
object_list = ['person','dog','cat','car']

# The specific files containing the lists that 
person_set ='person_test.txt'
dog_set = 'dog_test.txt'
cat_set = 'cat_test.txt'
car_set = 'car_test.txt'

#--------------------------------------------------------------------------------------------------------------------------------
# Create a pandas data frame for each of the test files
#--------------------------------------------------------------------------------------------------------------------------------
person_data = pd.read_csv(root_directory+person_set, header = None, sep=r"\s*",engine='python', dtype=str)

dog_data = pd.read_csv(root_directory+dog_set, header = None, sep=r"\s*",engine='python', dtype=str)

cat_data = pd.read_csv(root_directory+cat_set, header = None, sep=r"\s*",engine='python', dtype=str)

car_data = pd.read_csv(root_directory+car_set, header = None, sep=r"\s*",engine='python', dtype=str)

#--------------------------------------------------------------------------------------------------------------------------------
# Create the columns for all of the categories
#--------------------------------------------------------------------------------------------------------------------------------
person_data.columns = ["img_ID", "is_person"]
dog_data.columns = ["img_ID", "is_dog"]
cat_data.columns = ["img_ID", "is_cat"]
car_data.columns = ["img_ID", "is_car"]
#--------------------------------------------------------------------------------------------------------------------------------

# Convert the dataset columns to integers
person_data.is_person = person_data.is_person.astype(int)
dog_data.is_dog = dog_data.is_dog.astype(int)
cat_data.is_cat = cat_data.is_cat.astype(int)
car_data.is_car = car_data.is_car.astype(int)


# Now we join the data according to the ID column
df = pd.merge(person_data, dog_data, on='img_ID')
df = pd.merge(df, cat_data, on='img_ID')
df = pd.merge(df, car_data, on='img_ID')


#====================================================================================================================================================
# Now we will drop all columns that are not mutually exclusive
keep_indx = []

for k in range(0,len(df)):

    if(df['is_person'].loc[k]==1 and df['is_dog'].loc[k]==-1 and df['is_cat'].loc[k]==-1 and df['is_car'].loc[k]==-1):
        keep_indx.append(k)
    
    if(df['is_person'].loc[k]==-1 and df['is_dog'].loc[k]==1 and df['is_cat'].loc[k]==-1 and df['is_car'].loc[k]==-1):
        keep_indx.append(k)
    
    if(df['is_person'].loc[k]==-1 and df['is_dog'].loc[k]==-1 and df['is_cat'].loc[k]==1 and df['is_car'].loc[k]==-1):
        keep_indx.append(k)
    
    if(df['is_person'].loc[k]==-1 and df['is_dog'].loc[k]==-1 and df['is_cat'].loc[k]==-1 and df['is_car'].loc[k]==1):
        keep_indx.append(k)
    
# Copy the mutually exclusive pictures, then shuffle the data and reset the index
df = df.loc[keep_indx]
df = df.reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())
#====================================================================================================================================================

#==================================================================================================
# Count the number of objects in each category
#==================================================================================================
total_N = len(df)
#other_N = df[df.is_other == 1].is_other.sum()
person_N = df[df.is_person == 1].is_person.sum()
dog_N = df[df.is_dog == 1].is_dog.sum()
cat_N = df[df.is_cat == 1].is_cat.sum()
car_N = df[df.is_car == 1].is_car.sum()

print('')
print('The total number of images: ', total_N)
print('Number of Persons: ',person_N)  
print('Number of Dogs: ',dog_N)  
print('Number of Cats: ',cat_N)  
print('Number of Cars: ',car_N)    
print('')

#==================================================================================================
# Find the catefory with the minimum objects
#==================================================================================================
#min_object_N = min(total_N,person_N, dog_N,cat_N,car_N,other_N)

# Given all the data, we now split the set into a training and validation set.
# 1. Randomly shuffle all of the data
df = df.sample(frac=1).reset_index(drop=True)


# 1.5 Remove a large number of objects in the categories
dropped_persons =0
dropped_dogs =0
dropped_cats = 0
dropped_cars=0

drop_indx = []
for k in range(0,len(df)):

    if(df['is_person'].loc[k]==1 and dropped_persons < (person_N-min_object_N) and k not in drop_indx ):
        drop_indx.append(k)
        dropped_persons+=1
        
        
    if(df['is_cat'].loc[k]==1 and dropped_cats< (cat_N-min_object_N) and k not in drop_indx ):
        drop_indx.append(k)
        dropped_cats+=1
        
    if(df['is_dog'].loc[k]==1 and dropped_dogs < (dog_N-min_object_N) and k not in drop_indx ):
        drop_indx.append(k)
        dropped_dogs+=1
        
    if(df['is_car'].loc[k]==1 and dropped_cars < (car_N-min_object_N) and k not in drop_indx ):
        drop_indx.append(k)
        dropped_cars+=1

# Drop the desired data and shuffle the indices 
df.drop(df.index[drop_indx], inplace=True)
df = df.sample(frac=1).reset_index(drop=True)


person_N = df[df.is_person == 1].is_person.sum()
dog_N = df[df.is_dog == 1].is_dog.sum()
cat_N = df[df.is_cat == 1].is_cat.sum()
car_N = df[df.is_car == 1].is_car.sum()

print('=========================================')
print('dropped: ', len(drop_indx))
print('new person number: ', person_N)
print('new dog number: ', dog_N)
print('new cat number: ', cat_N)
print('new car number: ', car_N)
print('New Dataframe',len(df))
print(df.head())
print('=========================================')


# 2. Create a test and validation data frame
#    The first N*f are the training set, afterwards, validations
msk = []
N_train = int(test_validation_split*len(df))

for i in range(len(df)):
    
    if(i<=N_train):
        msk.append(True)
    else:
        msk.append(False)
        
# Convert to the Numpy array 
msk =np.asarray(msk)
 

# Split the data_frames into test and train
df_train = df[msk]
df_train = df_train.reset_index(drop=True)

df_validation = df[~msk]
df_validation = df_validation.reset_index(drop=True)

print('Training set size: ', len(df_train))
print('Validation set size: ', len(df_validation))

#==========================================================================================
# 3. Generate subfolders with the class names for the test and validation sets
f = open("create_subfolders.sh", "a")
commands ='''
#!/bin/sh
'''
commands+= 'for object in '

for object in object_list:
    commands+= object+' '
commands+='''
do
    mkdir 'test/'$object
    mkdir 'validation/'$object
done
'''
f.write(commands)
f.close()

# Run the newly generated bash file
! mkdir test
! mkdir validation
! bash create_subfolders.sh
#==========================================================================================

# 4. Iterate over the training and validation data frames:
# * For each 1, add the image to the corresponding subfolder
for k in range(0,len(df_train)):
    
    img_name = df_train.iloc[k,:][0]
    
    is_dog = (df_train['is_dog'].iloc[k] == 1)
    is_cat = (df_train['is_cat'].iloc[k] == 1)
    is_car = (df_train['is_car'].iloc[k] == 1)
    is_person = (df_train['is_person'].iloc[k] == 1)
    
    if(is_dog== True):
        folder = 'dog/'
        shutil.copyfile(img_directory+img_name+'.jpg','test/'+folder+img_name+'.jpg')
        
    if(is_person== True):
        folder = 'person/'
        shutil.copyfile(img_directory+img_name+'.jpg','test/'+folder+img_name+'.jpg')
    
    if(is_cat== True):
        folder = 'cat/'
        shutil.copyfile(img_directory+img_name+'.jpg','test/'+folder+img_name+'.jpg')
        
    if(is_car== True):
        folder = 'car/'
        shutil.copyfile(img_directory+img_name+'.jpg','test/'+folder+img_name+'.jpg')


for k in range(0,len(df_validation)):
    
    img_name = df_validation.iloc[k,:][0]
    
    is_dog = (df_validation['is_dog'].iloc[k] == 1)
    is_cat = (df_validation['is_cat'].iloc[k] == 1)
    is_car = (df_validation['is_car'].iloc[k] == 1)
    is_person = (df_validation['is_person'].iloc[k] == 1)
    
    if(is_dog== True):
        folder = 'dog/'
        shutil.copyfile(img_directory+img_name+'.jpg','validation/'+folder+img_name+'.jpg')
    
    if(is_person== True):
        folder = 'person/'
        shutil.copyfile(img_directory+img_name+'.jpg','validation/'+folder+img_name+'.jpg')
    
    if(is_cat== True):
        folder = 'cat/'
        shutil.copyfile(img_directory+img_name+'.jpg','validation/'+folder+img_name+'.jpg')
        
    if(is_car== True):
        folder = 'car/'
        shutil.copyfile(img_directory+img_name+'.jpg','validation/'+folder+img_name+'.jpg')
    
'''
Here we build the model that will classify the images
'''
from keras.utils import plot_model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images vary in size
img_width, img_height = scale,scale 

# Here we specify the directory for our training and validation images
train_data_dir = 'test'
validation_data_dir = 'validation'

# The number of training and validation samples
nb_train_samples = len(df_train)
nb_validation_samples = len(df_validation)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    chanDim=-1
else:
    input_shape = (img_width, img_height, 3)
    chanDim =1

    
#========================================================================================
#
# The Defined Baseline model
#
#========================================================================================
model0 = Sequential()
model0.add(Conv2D(32, (3, 3), input_shape=input_shape))
model0.add(Activation('relu'))
model0.add(MaxPooling2D(pool_size=(2, 2)))
model0.add(Flatten())
model0.add(Dense(len(object_list)))
model0.add(Activation('softmax'))

model0.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])
#========================================================================================
model1 = Sequential()
model1.add(Conv2D(32, (3, 3), input_shape=input_shape))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(32, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(64))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(len(object_list)))
model1.add(Activation('softmax'))

model1.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

#========================================================================================

model2 = Sequential()
model2.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
 
model2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
 
model2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
 
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(len(object_list), activation='softmax'))

model2.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

# Here we plot all of the models and save them as PDF figures
plot_model(model0,to_file='model0_plot.pdf', show_shapes=True, show_layer_names=True)
plot_model(model1,to_file='model1_plot.pdf', show_shapes=True, show_layer_names=True)
plot_model(model2,to_file='model2_plot.pdf', show_shapes=True, show_layer_names=True)
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback

# Here we fit the model, takes some time.
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rotation_range=50.,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale=1. / scale,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip = True
)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# The data augmentation for the validation set
test_datagen = ImageDataGenerator(rescale=1. / scale)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


history0 = model0.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps  // batch_size)
model0.save_weights('model0.h5') 

history1 = model1.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps  // batch_size)
model1.save_weights('model1.h5') 

history2 = model2.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps  // batch_size)
model2.save_weights('model2.h5')
# summarize history for accuracy
plt.plot(history0.history['acc'],'-o',label="Model 0: Training",color="b")
plt.plot(history0.history['val_acc'],'--',color="b")
plt.plot(history1.history['acc'],'-o',label="Model 1: Training",color="r")
plt.plot(history1.history['val_acc'],'--',color="r")
plt.plot(history2.history['acc'],'-o',label="Model 2: Training",color="green")
plt.plot(history2.history['val_acc'],'--',color="green")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.savefig('accuracy_vs_epoch.pdf',bboxes='tight')
plt.show()


print("Testing accuracies: ",np.mean(history0.history['acc'][-4:]),np.mean(history1.history['acc'][-4:]),np.mean(history2.history['acc'][-4:]))
print("Validation accuracies: ",np.mean(history0.history['val_acc'][-4:]),np.mean(history1.history['val_acc'][-4:]),np.mean(history2.history['val_acc'][-4:]))

# summarize history for loss
plt.plot(history0.history['loss'],'-o',color="b",label="Model 0: Training")
plt.plot(history0.history['val_loss'],'--',color="b")
plt.plot(history1.history['loss'],'-o',color="r",label="Model 1: Training")
plt.plot(history1.history['val_loss'],'--',color="r")
plt.plot(history2.history['loss'],'-o',color="green",label="Model 2: Training")
plt.plot(history2.history['val_loss'],'--',color="green")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend( loc='upper left')
plt.savefig('loss_vs_epoch.pdf',bboxes='tight')
plt.show()

print("Testing losses: ",np.mean(history0.history['loss'][-4:]),np.mean(history1.history['loss'][-4:]),np.mean(history2.history['loss'][-4:]))
print("Validation losses: ",np.mean(history0.history['val_loss'][-4:]),np.mean(history1.history['val_loss'][-4:]),np.mean(history2.history['val_loss'][-4:]))
! rm -r test/*
! rm -r validation/*
! rm create_subfolders.sh
! rmdir test
! rmdir validation
! rm *.hdf5
! ls
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image

# Make sure it matches the 
img_width, img_height = scale,scale
# Let us now make a prediction 
img_directory = '../input/voctest_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    chanDim=-1
else:
    input_shape = (img_width, img_height, 3)
    chanDim =1

#=============================================
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
 
model2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
 
model2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
 
model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(len(object_list), activation='softmax'))

model2.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])
#=============================================

print(train_generator.class_indices.keys())


model2.load_weights('model2.h5')

img = image.load_img(img_directory+'006936.jpg',False,target_size=(scale,scale))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model0.predict_classes(x)
prob = model0.predict_proba(x)
print(preds, prob)
plt.imshow(img)

