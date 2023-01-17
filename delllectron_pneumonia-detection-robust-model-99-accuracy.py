import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from PIL import Image

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
BASE_URL = '../input/chest-xray-pneumonia/chest_xray'
FOLDERS = ['test', 'val', 'train']
CATEGORIES = ['NORMAL', 'PNEUMONIA']
#get all the image filenames on normal class and create a dataframe
class_normal = []

for f in FOLDERS:
    folder = os.path.join(BASE_URL, f)

    for path in os.listdir(os.path.join(folder, 'NORMAL')):
        class_normal.append([os.path.join(folder, 'NORMAL/'+path), 0])
normal_df = pd.DataFrame(class_normal, columns=['filepaths', 'labels'])
#get all the image filenames on pneumonia class and create a dataframe

class_pneumonia = []

for f in FOLDERS:
    folder = os.path.join(BASE_URL, f)

    for path in os.listdir(os.path.join(folder, 'PNEUMONIA')):
        class_pneumonia.append([os.path.join(folder, 'PNEUMONIA/'+path), 1])
pneumonia_df = pd.DataFrame(class_pneumonia, columns=['filepaths', 'labels'])
#concatenate the two dataframes we created above
df = pd.concat([normal_df, pneumonia_df], axis=0).reset_index()
df.drop('index', axis=1, inplace=True)
df.head()
#show dataframe shape
print('DATAFRAME SHAPE: ',df.shape)
print(df.labels.value_counts())
#show countplot
plt.style.use('ggplot')
plt.figure(figsize=(10,5))
sns.countplot(df.labels);
#function to load images and convert them to array
def read_img(path, target_size):
    img = image.load_img(path, target_size=target_size)
    img = image.img_to_array(img) /255.
    return img
#show sample image from normal_class

fig, ax = plt.subplots(1,6,figsize=(14,3));
plt.suptitle('XRAY IMAGES FROM NORMAL CLASS')

for i,path in enumerate(normal_df.filepaths[:6].values):
    ax[i].imshow(read_img(path, (255,255)))

#show sample image from pneumonia_class

fig, ax = plt.subplots(1,6,figsize=(14,3));
plt.suptitle('XRAY IMAGES FROM PNEUMONIA CLASS')

for i,path in enumerate(pneumonia_df.filepaths[:6].values):
    ax[i].imshow(read_img(path, (255,255)))

#create a imagegenerator
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2
)

#load a sample image
sample_image = (read_img(normal_df.filepaths[0], (255,255)))

plt.figure(figsize=(10,10))
plt.suptitle('SAMPLE AUGMENTATION', fontsize=25)

i = 0

#generate and show
for batch in datagen.flow(tf.expand_dims(sample_image,0), batch_size=32):
    plt.subplot(3,3, i+1)
    plt.grid(False)
    plt.imshow(tf.squeeze(batch, 0));
    
    if i == 8:
        break
    i = i+1
plt.show();
augmented_img = []

#function for augmentation
def augment(path):
    #load images then transform
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 255.
    img = tf.image.resize(img, (150,150))
    i=0
    for batch in datagen.flow(tf.expand_dims(img, 0), batch_size=32):
        augmented_img.append(tf.squeeze(batch, 0))
        
        if i == 2:
            break
        i = i+1

#apply the augmentation function
normal_df['filepaths'].apply(augment)
#convert the generated images as tensors
normal_tensor =  tf.convert_to_tensor(augmented_img)
normal_tensor.shape
#same function but without augmentation
pneumonia_tensor = []
IMAGE_SIZE = 150
def map_fn(path):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 255.
    img = tf.image.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    pneumonia_tensor.append(img)
pneumonia_df.filepaths.apply(map_fn)
#convert to tensor
pneumonia_tensor = tf.convert_to_tensor(pneumonia_tensor)
pneumonia_tensor.shape
#generate labels
normal_labels = tf.zeros(len(normal_tensor), dtype = np.int64)
pneumonia_labels = tf.ones(len(pneumonia_tensor), dtype=np.int64)
tensor_dataset = tf.data.Dataset.from_tensor_slices( (tf.concat([normal_tensor, pneumonia_tensor], axis=0),
                                                      tf.concat([normal_labels, pneumonia_labels], axis=0)))
# TRAIN_SIZE = int(len(tensor_dataset) *0.7)
#BUFFER_SIZE = 101
#tensor_dataset = tensor_dataset.shuffle(BUFFER_SIZE)
# train_data = tensor_dataset.take(TRAIN_SIZE)
# test_data = tensor_dataset.skip(TRAIN_SIZE)

def is_test(x, y):
    return x % 4 == 0

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x,y: y

test_dataset = tensor_dataset.enumerate() \
                    .filter(is_test) \
                    .map(recover)

train_dataset = tensor_dataset.enumerate() \
                    .filter(is_train) \
                    .map(recover)
#just storing the labels for future use
train_labels = []
for i,l in train_dataset.take(-1):
    train_labels.append(l)
    
test_labels = []
for i,l in test_dataset.take(-1):
    test_labels.append(l)
#value counts on datasets
print('TRAIN DATA VALUE COUNTS: ',np.bincount(np.array(test_labels)))
print('TRAIN DATA VALUE COUNTS: ',np.bincount(np.array(train_labels)))
#Looks good! the data is balanced now
#you can try 16, but it will slow down the training
BATCH_SIZE = 32

#shuffle the train data
train_data = train_dataset.shuffle(10000).batch(BATCH_SIZE)
test_data = test_dataset.batch(BATCH_SIZE)
#check the labels per batch if they are truly shuffled, haha i have trust issues though
for i,l in train_data.take(20):
    print(l)
#check the shape per batch
for i,l in train_data.take(1):
    print(i.shape)
#define the input shape
INPUT_SHAPE = (150,150,3)

#i'll use mobilenetv2 and you try anything you want.
base_model = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE,
                                            include_top=False,
                                            weights='imagenet')

#freeze the weights of convolutional layer
base_model.trainable = False
base_model.summary()
#check the output shape of convolutional layer
for i, l in train_data.take(1):
    print(base_model(i).shape)
model = Sequential()
model.add(base_model)
#you can try Flatten instead of GlobalAveragePooling
model.add(Layers.GlobalAveragePooling2D())
model.add(Layers.Dense(128, activation='relu'))
model.add(Layers.Dropout(0.2))
model.add(Layers.Dense(1, activation = 'sigmoid'))
model.summary()
#set a callback
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

#compile the model
model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#Throw the data
history = model.fit(train_data, epochs=4, validation_data= test_data, callbacks=[callbacks])
#Owww Bingo!
#show model training accuracy and loss
fig, ax = plt.subplots(1,2, figsize=(14,5))
ax[0].set_title('MODEL ACCURACY')
ax[1].set_title('MODEL TRANING LOSS')
ax[0].plot(history.history['accuracy'], color= 'steelblue', lw=2);
ax[1].plot(history.history['loss'], color='salmon');
#get the predictions
predictions = model.predict_classes(test_data)
#print a classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(test_labels, predictions))
#here is our favorite, the confusion matrix!
conf_mat = confusion_matrix(np.array(test_labels), predictions)
plt.figure(figsize=(8,8))
plt.title('CONFUSION MATRIX')
sns.heatmap(conf_mat, annot=True, 
            yticklabels=['Normal', 'PNEUMONIA'],
            xticklabels=['Normal', 'PNEUMONIA'],
            square=True, cmap='magma');
for img, label in test_data.take(1):
    sample_img = img[1]
    img_true_label = label[1]
pred = np.array(model.predict_classes(tf.expand_dims(sample_img, 0))).flatten()[0]
plt.title(CATEGORIES[pred])
plt.imshow(sample_img);
#show image true label
print('IMAGE TRUE LABEL: ', CATEGORIES[img_true_label])