import keras
from keras.models import Sequential
from keras.layers import MaxPool2D,Conv2D,Flatten,Dropout,Dense,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import plotly
print(os.listdir("../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/"))
meta_data = pd.read_csv("../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")
meta_data.sample(10)
train_data = meta_data[meta_data['Dataset_type'] == 'TRAIN']
test_data = meta_data[meta_data['Dataset_type'] == 'TEST']

print(f"Shape of training dataset: {train_data.shape}")
print(f"Shape of testing dataset: {test_data.shape}")

print("-----_________------Test data------_________--------")
test_data.sample(10)
#Null Values
print(f"Null values in train data:\n{train_data.isna().sum()}")

print("======================================================")

print(f"Null values in test data:\n{test_data.isna().sum()}")
train_fill = train_data.fillna('unknown')
train_fill.sample(10)
test_fill = test_data.fillna('unknown')
#labels = ['Label','Label_2_Virus_category','Label_1_Virus_category']
#With Unknown 
fig,ax = plt.subplots(3, 2, figsize=(20, 10))

plt.style.use('seaborn')

#print(labels[0])
sns.countplot('Label',data=train_fill,ax=ax[0,0])
sns.countplot('Label_2_Virus_category',data=train_fill,ax=ax[0,1])
sns.countplot('Label_1_Virus_category',data=train_fill,ax=ax[1,0])
sns.countplot('Label',data=test_fill,ax=ax[1,1])
sns.countplot('Label_2_Virus_category',data=test_fill,ax=ax[2,0])
sns.countplot('Label_1_Virus_category',data=test_fill,ax=ax[2,1])
fig.show()
#Without Unknown
fig,ax = plt.subplots(2, 2, figsize=(20, 10))

plt.style.use('seaborn')

#print(labels[0])
sns.countplot('Label',data=train_data,ax=ax[0,0])
sns.countplot('Label_2_Virus_category',data=train_data,ax=ax[0,1])
sns.countplot('Label_1_Virus_category',data=train_data,ax=ax[1,0])
#sns.countplot('Label',data=test_data,ax=ax[1,1])
#sns.countplot('Label_2_Virus_category',data=test_data,ax=ax[2,0])
#sns.countplot('Label_1_Virus_category',data=test_data,ax=ax[2,1])
fig.show()
TEST_FOLDER = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
TRAIN_FOLDER = '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
sample_train_imgs = list(os.walk(TRAIN_FOLDER))[0][2][:6]
sample_train_imgs
sample_train_imgs = list(map(lambda x: os.path.join(TRAIN_FOLDER, x), sample_train_imgs))
sample_test_imgs = list(os.walk(TEST_FOLDER))[0][2][:6]
sample_test_imgs
sample_test_imgs = list(map(lambda x: os.path.join(TEST_FOLDER, x), sample_test_imgs))
from PIL import Image
plt.figure(figsize=(20,20))

for iterator, filename in enumerate(sample_train_imgs):
    image = Image.open(filename)
    plt.subplot(4, 2, iterator+1)
    plt.axis('off')
    plt.imshow(image)


plt.tight_layout()
plt.figure(figsize=(20,20))

for i in range(len(sample_test_imgs)):
    image = Image.open(sample_test_imgs[i])
    plt.subplot(3,2,i+1)
    plt.axis("off")
    plt.imshow(image)
train_data.shape
final_train_data = train_data[(train_data['Label'] == 'Normal') | 
                              ((train_data['Label'] == 'Pnemonia') & (train_data['Label_2_Virus_category'] == 'COVID-19'))]
final_train_data['target'] = ['negative' if holder == 'Normal' else 'positive' for holder in final_train_data['Label']]
from sklearn.utils import shuffle 
final_train_data = shuffle(final_train_data, random_state=1)

final_validation_data = final_train_data.iloc[1000:, :]
final_train_data = final_train_data.iloc[:1000, :]

print(f"Final train data shape : {final_train_data.shape}")
final_train_data.sample(10)

train_image_generator = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=90,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=[0.9, 1.25],
    brightness_range=[0.5, 1.5]
)

test_image_generator = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_image_generator.flow_from_dataframe(
    dataframe=final_train_data,
    directory=TRAIN_FOLDER,
    x_col='X_ray_image_name',
    y_col='target',
    target_size=(224, 224),
    batch_size=8,
    seed=2020,
    shuffle=True,
    class_mode='binary'
)


validation_generator = train_image_generator.flow_from_dataframe(
    dataframe=final_validation_data,
    directory=TRAIN_FOLDER,
    x_col='X_ray_image_name',
    y_col='target',
    target_size=(224, 224),
    batch_size=8,
    seed=2020,
    shuffle=True,
    class_mode='binary'
)

test_generator = test_image_generator.flow_from_dataframe(
    dataframe=test_data,
    directory=TEST_FOLDER,
    x_col='X_ray_image_name',
    target_size=(224, 224),
    shuffle=False,
    batch_size=16,
    class_mode=None
)
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(224,224,3),activation='relu'))
model.add(MaxPool2D((3,3)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((3,3)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((3,3)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
             metrics=['accuracy'])
history = model.fit_generator(train_generator,validation_data=validation_generator,epochs=3)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.lineplot(x=np.arange(1, 4), y=history.history.get('loss'), ax=ax[0, 0])
sns.lineplot(x=np.arange(1, 4), y=history.history.get('accuracy'), ax=ax[0, 1])
sns.lineplot(x=np.arange(1, 4), y=history.history.get('val_loss'), ax=ax[1, 0])
sns.lineplot(x=np.arange(1, 4), y=history.history.get('val_accuracy'), ax=ax[1, 1])
ax[0, 0].set_title('Training Loss vs Epochs')
ax[0, 1].set_title('Training Accuracy vs Epochs')
ax[1, 0].set_title('Validation Loss vs Epochs')
ax[1, 1].set_title('Validation Accu vs Epochs')
fig.suptitle('Base CNN model', size=16)
plt.show()
