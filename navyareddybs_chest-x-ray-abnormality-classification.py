# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/sample/sample_labels.csv')
df.head()
df1 = df["Finding Labels"].replace({"No Finding": "Normal"}, inplace=False)
df1.head()
for i in df1:
    if(i!="Normal"):
        df1.replace({i: "AbNormal"}, inplace=True)
        
df1.count()
df['BinaryClassification']=df1
df.columns
import glob
img_directory = sorted(glob.glob(os.path.join("/kaggle/input/sample/sample", "sample/images","*.png")))
df = pd.read_csv(os.path.join("/kaggle/input/sample/sample", "sample_labels.csv"))
df1 = pd.DataFrame()
df1['Id'] = df['Image Index'].copy()
df1.loc[df['Finding Labels'] == "No Finding",'Label'] = "Normal"
df1.loc[df['Finding Labels'] != "No Finding", 'Label'] = "Abnormal"
Labels = ['Normal', 'Abnormal']
columnsIDs = random.sample(range(len(img_directory)), 25)
columnsIDs = random.sample(columnsIDs, 25)
rows = math.ceil(np.sqrt(25))
cols = math.ceil(25/rows)
for i in range(25):
            img = cv2.imread(img_directory[columnsIDs[i]])
            plt.subplot(rows, cols, i+1)
            #if display_label:
            plt.gca().set_title(df1['Label'][columnsIDs[i]],wrap=True)
            plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            fig = plt.figure(figsize=(20,20))
df1.columns
from sklearn.model_selection import train_test_split
train, val = train_test_split(df1, test_size=0.3, random_state=42, shuffle = True, stratify=df1['Label'])

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

print("Found {} samples for training".format(len(train)))
print("Found {} samples for validation".format(len(val)))
from keras.preprocessing.image import ImageDataGenerator
traindata = ImageDataGenerator(rescale=1./255, horizontal_flip = True, vertical_flip = False, height_shift_range= 0.05, width_shift_range=0.1,rotation_range=5, 
                                   shear_range = 0.1, fill_mode = 'reflect',zoom_range=0.15)

validateData = ImageDataGenerator(rescale=1./255)

trainGen = traindata.flow_from_dataframe(dataframe=df1, directory='/kaggle/input/sample/sample/sample/images', x_col="Id", y_col="Label",batch_size=64,
        target_size=(128,128),class_mode='binary')

validateGenr = validateData.flow_from_dataframe(dataframe=val,directory='/kaggle/input/sample/sample/sample/images', x_col="Id",y_col="Label", batch_size=64,
        target_size=(128,128),class_mode='binary')
plt.imshow(trainGen[0][0][5])
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD, Adam
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(128, 128, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2, 2)) 
model.add(Flatten())
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

#define optimizer
opt = Adam(lr=1e-3)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks = [EarlyStopping(patience=5, verbose=1),ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.0001, verbose=1),]

train_steps = int(len(train)/64)
val_steps = int(len(val)/64)
model.fit_generator(trainGen, steps_per_epoch=train_steps,epochs=10,callbacks=callbacks, validation_data=validateGenr,validation_steps=val_steps,verbose=1)
history = model.history

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from keras.applications.inception_resnet_v2 import InceptionResNetV2
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for layer in base_model.layers:
    layer.trainable = False
    
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for i, layer in enumerate(model.layers[-11:]):
    print(i, layer.name)
    
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_steps = int(len(train)/64)
val_steps = int(len(val)/64)
model.fit_generator(trainGen, steps_per_epoch=train_steps,epochs=5,callbacks=callbacks,validation_data=validateGenr,validation_steps=val_steps,verbose=1)
history = model.history

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
import glob
img_directory = sorted(glob.glob(os.path.join("/kaggle/input/sample/sample", "sample/images","*.png")))
df = pd.read_csv(os.path.join("/kaggle/input/sample/sample", "sample_labels.csv"))
df1 = pd.DataFrame()
df1['Id'] = df['Image Index'].copy()
df1['Label'] = df['Finding Labels'].apply(lambda val: val.split('|'))
Labels = ['Cardiomegaly','Emphysema','Effusion','Hernia','Nodule','Pneumothorax','Atelectasis','Pleural_Thickening','Mass','Edema','Consolidation','Infiltration','Fibrosis','Pneumonia', 'No Finding']
df1['Label']
import random
columnsIDs = random.sample(range(len(img_directory)), 25)
columnsIDs = random.sample(columnsIDs, 25)
import math
rows = math.ceil(np.sqrt(25))
cols = math.ceil(25/rows)
import cv2
import matplotlib.pyplot as plt
for i in range(25):
            img = cv2.imread(img_directory[columnsIDs[i]])
            plt.subplot(rows, cols, i+1)
            #if display_label:
            plt.gca().set_title(df1['Label'][columnsIDs[i]],wrap=True)
            plt.axis('off')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            fig = plt.figure(figsize=(20,20))
from sklearn.model_selection import train_test_split
train, val = train_test_split(df1, test_size=0.2, random_state=42, shuffle = True)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

print("Found {} samples for training".format(len(train)))
print("Found {} samples for validation".format(len(val)))
val.columns
from keras.preprocessing.image import ImageDataGenerator
trainDataGen = ImageDataGenerator(rescale=1./255,horizontal_flip = True,vertical_flip = False,height_shift_range= 0.05, 
                                   width_shift_range=0.1,rotation_range=5, shear_range = 0.1,fill_mode = 'reflect',zoom_range=0.15)

valDataGen = ImageDataGenerator(rescale=1./255)

trainGen = trainDataGen.flow_from_dataframe(
        dataframe=df1,
        directory='/kaggle/input/sample/sample/sample/images',
        x_col="Id",y_col="Label",batch_size=64,target_size=(128,128),classes = Labels,class_mode='categorical')

valGen = valDataGen.flow_from_dataframe(dataframe=val,
        directory='/kaggle/input/sample/sample/sample/images',x_col="Id",y_col="Label",batch_size=64,target_size=(128,128),classes = Labels,class_mode='categorical')
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential

base_mobilenet_model = MobileNet(input_shape = trainGen[0][0].shape[1:], 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(Labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
multi_disease_model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.0001, verbose=1),
]

train_steps = int(len(train)/64)
val_steps = int(len(val)/64)
multi_disease_model.fit_generator(trainGen,
        steps_per_epoch=train_steps,
        epochs=5,
        callbacks=callbacks,
        validation_data=valGen,
        validation_steps=val_steps,
        verbose=1)
history = multi_disease_model.history

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
        dataframe=val,
        directory='/kaggle/input/sample/sample/sample/images',
        x_col="Id",
        batch_size=1,
        target_size=(128,128),
        shuffle=False,
        class_mode=None)

test_generator.reset()
pred = multi_disease_model.predict_generator(test_generator, steps=len(val), verbose=1)
df2 = val.copy()
        
onehot_arr = np.zeros((len(val),len(Labels)))
for i in range(len(df2)):
    for element in df2['Label'][i]:
        onehot_arr[i,Labels.index(element)] = 1
        
print(onehot_arr.shape == pred.shape)

sickest_idx = np.argsort(np.sum(onehot_arr, 1)<1)

fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    img = plt.imread("/kaggle/input/sample/sample/sample/images/" + val['Id'][idx])
    c_ax.imshow(img, cmap = 'bone')
    stat_str = [n_class[:6] for n_class, n_score in zip(Labels,onehot_arr[idx]) if n_score>0.5]
    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(Labels, 
                onehot_arr[idx], pred[idx]) if (n_score>0.5) or (p_score>0.5)]
    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
    c_ax.axis('off')
sample_weights = df1['Label'].map(lambda x: len(x)).values + 4e-2
sample_weights /= sample_weights.sum()
res = df1.sample(3000, weights=sample_weights)
train, val = train_test_split(df1, test_size=0.2, random_state=42, shuffle = True)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

print("Found {} samples for training".format(len(train)))
print("Found {} samples for validation".format(len(val)))
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip = True, 
                                   vertical_flip = False, 
                                   height_shift_range= 0.05, 
                                   width_shift_range=0.1, 
                                   rotation_range=5, 
                                   shear_range = 0.1,
                                   fill_mode = 'reflect',
                                   zoom_range=0.15)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=df1,
        directory='/kaggle/input/sample/sample/sample/images',
        x_col="Id",
        y_col="Label",
        batch_size=64,
        target_size=(128,128),
        classes = Labels,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val,
        directory='/kaggle/input/sample/sample/sample/images',
        x_col="Id",
        y_col="Label",
        batch_size=64,
        target_size=(128,128),
        classes = Labels,
        class_mode='categorical')
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential

base_mobilenet_model = MobileNet(input_shape = train_generator[0][0].shape[1:], 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(Labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae'])
multi_disease_model.summary()
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
        dataframe=val,
        directory='/kaggle/input/sample/sample/sample/images',
        x_col="Id",
        batch_size=1,
        target_size=(128,128),
        shuffle=False,
        class_mode=None)

test_generator.reset()
pred = multi_disease_model.predict_generator(test_generator, steps=len(val), verbose=1)

df1 = val.copy()
        
onehot_arr = np.zeros((len(val),len(Labels)))
for i in range(len(df1)):
    for element in df1['Label'][i]:
        onehot_arr[i,Labels.index(element)] = 1
        
print(onehot_arr.shape == pred.shape)
sickest_idx = np.argsort(np.sum(onehot_arr, 1)<1)

fig, m_axs = plt.subplots(4, 2, figsize = (16, 32))
for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    img = plt.imread("/kaggle/input/sample/sample/sample/images/" + val['Id'][idx])
    c_ax.imshow(img, cmap = 'bone')
    stat_str = [n_class[:6] for n_class, n_score in zip(Labels,onehot_arr[idx]) if n_score>0.5]
    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(Labels, 
                onehot_arr[idx], pred[idx]) if (n_score>0.5) or (p_score>0.5)]
    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
    c_ax.axis('off')
