import glob
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout, merge, UpSampling2D, Input
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
IMAGE_SIZE = (224,224)
DIR_NAME = '/kaggle/input/covid-chest-xray/'
IMAGE_DIR = DIR_NAME + 'images/'
ANNOTATIONS_DIR = DIR_NAME + 'annotations/'
for dirname, _, filenames in os.walk(DIR_NAME):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv(DIR_NAME + 'metadata.csv')
df.head()
df = df.sample(frac=1).reset_index(drop=True)
df.head()
df.shape
len(os.listdir(IMAGE_DIR))
df.columns
def count_na(df, col):
    return df[col].isna().sum()
count_na(df,'filename')
files = os.listdir(IMAGE_DIR)
files
check_file_present = []
for f in df['filename']:
    check_file_present.append(f in files)
len(check_file_present)
c = 0
for cfp in check_file_present:
    if cfp:
        c+=1
c, len(check_file_present)-c
df = df[check_file_present]
df.shape
df =df.drop(['patientid', 'offset', 'survival', 'intubated',
       'intubation_present', 'went_icu', 'in_icu', 'needed_supplemental_O2',
       'extubated', 'temperature', 'pO2_saturation', 'leukocyte_count',
       'neutrophil_count', 'lymphocyte_count', 'date', 'doi', 'license', 'other_notes', 'Unnamed: 28'], axis=1)
df.head()
df = df.drop(['url', 'folder'], axis=1)
df.head()
count_na(df,'finding')
df['finding'] = df['finding'].astype('category')
df['label'] = df['finding'].cat.codes
df.head(20)
finding_to_label = {}
label_to_finding = {}
for _,row in df.iterrows():
    finding_to_label[row['finding']] = row['label']
    label_to_finding[row['label']] = row['finding']
finding_to_label
label_to_finding
df = df.drop(['finding'], axis=1)
count_na(df,'sex')
df['view'] = df['view'].astype('category')
df['modality'] = df['modality'].astype('category')
df['view'] = df['view'].cat.codes
df['modality'] = df['modality'].cat.codes
df.head()
df['sex'] = df['sex'].fillna('M')
df['sex']
count_na(df,'sex')
df['sex'] = df['sex'].astype('category').cat.codes
df.head()
df = df.drop(['location'], axis=1)
df.head()
count_na(df,'clinical_notes')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
df['clinical_notes'] = df['clinical_notes'].fillna("")
df['clinical_notes'].values
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(np.array(df['clinical_notes'].values))
integer_encoded.shape
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded
onehot_encoded.shape
# invert first example
inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
print(inverted)
X_clinical_notes = onehot_encoded
X_clinical_notes.shape
Y = df['label'].values
Y.shape
df = df.drop(['clinical_notes'], axis = 1)
df.head()
df = df.drop(['label'], axis = 1)
df.head()
X_sex = df['sex'].values
X_age = df['age'].values
X_view = df['view'].values
X_modality = df['modality'].values
X_sex.shape, X_age.shape, X_modality.shape, X_view.shape
df = df.drop(['sex', 'view', 'age', 'modality'], axis = 1)
df.head()
images = df['filename'].values
X_images = []
for i in range(0,len(images)):
    imgName = images[i]
    oriimg = cv2.imread(IMAGE_DIR+imgName)
    img = cv2.resize(oriimg, IMAGE_SIZE)
    print(i,img.shape)
    X_images.append(img)
X_images = np.array(X_images)
X_images.shape
X_sex = np.reshape(X_sex, (X_sex.shape[0],1))
X_age = np.reshape(X_age, (X_age.shape[0],1))
X_view = np.reshape(X_view, (X_view.shape[0],1))
X_modality = np.reshape(X_modality, (X_modality.shape[0],1))
X_images.shape,X_clinical_notes.shape,X_sex.shape,X_age.shape,X_view.shape,X_modality.shape,Y.shape
HEIGHT=224
WIDTH=224
CHANNEL=3
import math
split_num = math.ceil(len(X_images)*0.3)
split_num
X_non_image_data = np.concatenate((X_clinical_notes,X_sex,X_age,X_view,X_modality), axis=1)
X_non_image_data.shape
X_images_train = X_images[split_num:]
X_images_test = X_images[:split_num]
X_images_train.shape, X_images_test.shape
X_non_image_data_train = X_non_image_data[split_num:]
X_non_image_data_test = X_non_image_data[:split_num]
X_non_image_data_train.shape, X_non_image_data_test.shape
y_train = Y[split_num:]
y_test = Y[:split_num:]
y_train.shape, y_test.shape
#encoder
model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(HEIGHT, WIDTH, CHANNEL)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
#decoder
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu'))
model.add(UpSampling2D((4, 4)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((4, 4)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((7, 7)))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
model.summary()
model.layers[0].trainable = False
model.summary()
model.compile(optimizer='adadelta', loss='mse')
model.fit(X_images_train,X_images_train, epochs=50,
                batch_size=35, validation_data=(X_images_test, X_images_test)) 
f = model.predict(X_images_train)
X_images_train1 = f
X_images_train1.shape
X_images_train.shape
X_images_test1 = model.predict(X_images_test)
X_images_test1.shape
X_images_test.shape
X_images_trainf = np.reshape(X_images_train1, (X_images_train1.shape[0], HEIGHT*WIDTH*CHANNEL))
X_images_trainf.shape
X_images_testf = np.reshape(X_images_test1, (X_images_test1.shape[0], HEIGHT*WIDTH*CHANNEL))
X_images_testf.shape
X_non_image_data_train.shape
X_non_image_data_test.shape
X_train = np.concatenate((X_non_image_data_train,X_images_trainf), axis=1)
X_train.shape
X_test = np.concatenate((X_non_image_data_test,X_images_testf), axis=1)
X_test.shape
y_train.shape
y_test.shape
np.savez_compressed('dataset.npz',X_train, X_test, y_train, y_test)
finding_to_label
label_to_finding
