import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.models import Model
from PIL import Image
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Dense, Dropout,Flatten, AveragePooling2D, GlobalAveragePooling2D, Concatenate

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

all_data_df = pd.read_csv('../input/data/Data_Entry_2017.csv')
data_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input', 'data', 'images*', '*', '*.png'))}
print('Scans found:', len(data_image_paths), ', Total Headers', all_data_df.shape[0])
all_data_df['path'] = all_data_df['Image Index'].map(data_image_paths.get)
all_data_df["labels"]=all_data_df["Finding Labels"].apply(lambda x:x.split("|"))
all_data_df['Patient Age'] = all_data_df['Patient Age'].map(lambda x: int(x))
all_data_df.sample(3)
all_data_df['Finding Labels'] = all_data_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
from itertools import chain
all_labels = np.unique(list(chain(*all_data_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print(all_labels)
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_data_df[c_label] = all_data_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_data_df.sample(3)        
#  keep at least 1000 cases
# MIN_CASES = 1000
# all_labels = [c_label for c_label in all_labels if all_data_df[c_label].sum()>MIN_CASES]
# print('Clean Labels ({})'.format(len(all_labels)), 
#       [(c_label,int(all_data_df[c_label].sum())) for c_label in all_labels])
all_data_df['disease_vec'] = all_data_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
all_data_df.head(10)
dataset_path = '../input/data'
train_valid_images_path = os.path.join(dataset_path, "train_val_list.txt")
test_images_path = os.path.join(dataset_path, "test_list.txt")
dataset_df_path = os.path.join(dataset_path, 'Data_Entry_2017.csv')

with open(train_valid_images_path, 'r') as the_file:
    train_valid_images = the_file.read().splitlines()
    
with open(test_images_path, 'r') as the_file: 
    test_images = the_file.read().splitlines()

    
print(f"We have {len(train_valid_images)} training and validation images.")
print(f"We have {len(test_images)} testing images.")
train_valid_images[:9]
train_valid_df = all_data_df[all_data_df['Image Index'].isin(train_valid_images)]
train_valid_df = train_valid_df.sample(40000)

test_df = all_data_df[all_data_df['Image Index'].isin(test_images)]
test_df = test_df.sample(10000)

print("length of training", len(train_valid_df))
print("length of testing", len(test_df))


from sklearn.model_selection import train_test_split

train_patients_ids, valid_patients_ids = train_test_split(train_valid_df['Patient ID'].unique(), test_size=0.25, random_state=1993)
train_df = train_valid_df[train_valid_df['Patient ID'].isin(train_patients_ids)]
valid_df = train_valid_df[train_valid_df['Patient ID'].isin(valid_patients_ids)]
print("length of training", len(train_df))
print("length of validation", len(valid_df))
def plot_disease_distribtion(all_data_df):
    all_data_df['labels'].value_counts().plot(kind="bar")
    
plot_disease_distribtion(all_data_df.explode('labels'))
plot_disease_distribtion(train_df.explode('labels'))
plot_disease_distribtion(valid_df.explode('labels'))
plot_disease_distribtion(test_df.explode('labels'))
#from tensorflow.keras.applications.densenet import DenseNet121
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
# base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
# base_model.summary()
# model = Sequential()
# model.add(mobilenet_model)
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.5))
# model.add(Dense(512))
# model.add(Dropout(0.5))
# model.add(Dense(224))
# model.add(Dropout(0.5))
# model.add(Dense(15 , activation = 'sigmoid'))
# model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
# model.summary()

from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
base_mobilenet_model = MobileNet(input_shape=(128, 128, 3), 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_mobilenet_model)
multi_disease_model.add(GlobalAveragePooling2D())
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(512))
multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(len(all_labels), activation = 'sigmoid'))
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['accuracy'])
multi_disease_model.summary()
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
datagen=ImageDataGenerator(rescale=1 / 255,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True,
                                  horizontal_flip=True,
                                  vertical_flip=False,
                                  height_shift_range=0.05,
                                  width_shift_range=0.1,
                                  rotation_range=5,
                                  shear_range=0.1,
                                  fill_mode='reflect',
                                  zoom_range=0.15)
test_datagen=ImageDataGenerator(rescale=1./255.)

train_gen = flow_from_dataframe(datagen, train_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size=(128, 128),
                            batch_size = 32)

valid_gen = flow_from_dataframe(datagen, valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size=(128, 128),
                            batch_size = 256)
history=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=30,
        validation_data=valid_generator,
        validation_steps=len(valid_generator))
model.save('model_savemobilenet.h5')

model.save_weights('model_save_weightsmobilenet.h5')

model_json = model.to_json()
with open('F:/private/project/Final Project/model_savemobilenet.json', 'w') as json_file:
    json_file.write(model_json)
test_generator = (flow_from_dataframe(datagen, 
                               test_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                            batch_size = 1024))

model.evaluate_generator(test_generator, 100)
# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
  
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
import cv2
img = cv2.imread("F:/private/project/Data sets/images/00023026_008.png")
print(img.shape)
img = cv2.resize(img,(128,128))
print(img.shape)
#img = img.transpose((2,0,1))
#print(img.shape)
img = img.astype('float32')
img = img/255
img = np.expand_dims(img,axis=0)
print(img.shape)
pred = model.predict(img)
print(pred)

y_pred = np.array([1 if pred[0,i]>=0.3  else 0 for i in range(pred.shape[1])])
print(y_pred)
#classes=['Cardiomegaly','Emphysema', 'Effusion', 'No Finding', 'Hernia','Infiltration','Mass','Nodule','Atelectasis', 'Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation']
classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Infiltration','No Finding', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
[classes[i] for i in range(14) if y_pred[i]==1 ] 