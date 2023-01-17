import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set_style('whitegrid')



import tensorflow as tf

from tensorflow.keras.models import Sequential

import tensorflow.keras.layers as Layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

from PIL import Image



import os

from tqdm import tqdm

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')

df.head()
#show sum of nullvalues per feature

df.isnull().sum()
#show datatypes

df.info()
#impute unknown to null data points, we don't wanna see those ugly null values

df.fillna('unknown', inplace=True)

df.isnull().sum()
print(df['Label_1_Virus_category'].value_counts())

print('='*50)

print(df['Label_2_Virus_category'].value_counts())
#separate train data and test data

train_data = df[df['Dataset_type']=='TRAIN']

test_data = df[df['Dataset_type']=='TEST']

print('Train shape: ',train_data.shape)

print('Test Shape: ',test_data.shape)
#show a countplot

plt.figure(figsize=(10,5))

sns.countplot(train_data['Label_2_Virus_category']);
#get the path of train and test folders

train_img_path = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'

test_img_path = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
#show sample image

samp_img1 = Image.open(os.path.join(train_img_path, train_data['X_ray_image_name'][0]))

samp_img2 = Image.open(os.path.join(train_img_path, train_data['X_ray_image_name'][22]))

fig, ax =plt.subplots(1,2, figsize=(10,5))

ax[0].imshow(samp_img1, cmap='gray');

ax[1].imshow(samp_img2, cmap='gray');
#sample x-ray image of person with covid-19

with_covid = train_data[train_data['Label_2_Virus_category'] == 'COVID-19']



#show sample image

samp_img1 = Image.open(os.path.join(train_img_path, with_covid['X_ray_image_name'].iloc[8]))

samp_img2 = Image.open(os.path.join(train_img_path, with_covid['X_ray_image_name'].iloc[15]))

fig, ax =plt.subplots(1,2, figsize=(10,5))

ax[0].imshow(samp_img1);

ax[1].imshow(samp_img2);
#remove Pnuemonia with unknown value

final_train_data = train_data[(train_data['Label'] == 'Normal') | 

                              ((train_data['Label'] == 'Pnemonia') &

                               (train_data['Label_2_Virus_category'] == 'COVID-19'))]
# add a target and class feature

final_train_data['class'] = final_train_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')

test_data['class'] = test_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')



final_train_data['target'] = final_train_data.Label.apply(lambda x: 0 if x=='Normal' else 1)

test_data['target'] = test_data.Label.apply(lambda x: 0 if x=='Normal' else 1)
#get the important features

final_train_data = final_train_data[['X_ray_image_name', 'class', 'target', 'Label_2_Virus_category']]

final_test_data = test_data[['X_ray_image_name', 'class', 'target']]
test_data['Label'].value_counts()




#create a imagegenerator for for augmentation

datagen =  ImageDataGenerator(

  shear_range=0.2,

  zoom_range=0.2,

)



# function to convert image to array

def read_img(filename, size, path):

    img = image.load_img(os.path.join(path, filename), target_size=size)

    #convert image to array

    img = image.img_to_array(img) / 255

    return img
#read a sample image

samp_img = read_img(final_train_data['X_ray_image_name'][0],

                                 (255,255),

                                 train_img_path)



plt.figure(figsize=(10,10))

plt.suptitle('Data Augmentation', fontsize=28)



i = 0



#show augmented images

for batch in datagen.flow(tf.expand_dims(samp_img,0), batch_size=6):

    plt.subplot(3, 3, i+1)

    plt.grid(False)

    plt.imshow(batch.reshape(255, 255, 3));

    

    if i == 8:

        break

    i += 1

    

plt.show();
#augment the images labeled with covid-19 to balance the data



corona_df = final_train_data[final_train_data['Label_2_Virus_category'] == 'COVID-19']

with_corona_augmented = []



#create a function for augmentation

def augment(name):

    img = read_img(name, (255,255), train_img_path)

    i = 0

    for batch in tqdm(datagen.flow(tf.expand_dims(img, 0), batch_size=32)):

        with_corona_augmented.append(tf.squeeze(batch).numpy())

        if i == 20:

            break

        i =i+1



#apply the function

corona_df['X_ray_image_name'].apply(augment)
# extract the image from traing data and test data, then convert them as array

train_arrays = [] 

final_train_data['X_ray_image_name'].apply(lambda x: train_arrays.append(read_img(x, (255,255), train_img_path)))

test_arrays = []

final_test_data['X_ray_image_name'].apply(lambda x: test_arrays.append(read_img(x, (255,255), test_img_path)))
print(len(train_arrays))

print(len(test_arrays))
#concatenate the training data labels and the labels for augmented images

y_train = np.concatenate((np.int64(final_train_data['target'].values), np.ones(len(with_corona_augmented), dtype=np.int64)))
train_tensors = tf.convert_to_tensor(np.concatenate((np.array(train_arrays), np.array(with_corona_augmented))))

test_tensors  = tf.convert_to_tensor(np.array(test_arrays))

y_train_tensor = tf.convert_to_tensor(y_train)

y_test_tensor = tf.convert_to_tensor(final_test_data['target'].values)
train_dataset = tf.data.Dataset.from_tensor_slices((train_tensors, y_train_tensor))

test_dataset = tf.data.Dataset.from_tensor_slices((test_tensors, y_test_tensor))
for i,l in train_dataset.take(1):

    plt.imshow(i);
BATCH_SIZE = 16

BUFFER = 1000



train_batches = train_dataset.shuffle(BUFFER).batch(BATCH_SIZE)

test_batches = test_dataset.batch(BATCH_SIZE)



for i,l in train_batches.take(1):

    print('Train Shape per Batch: ',i.shape);

for i,l in test_batches.take(1):

    print('Test Shape per Batch: ',i.shape);
#define input shape

INPUT_SHAPE = (255,255,3) 



#get the pretrained model

base_model = tf.keras.applications.ResNet50(input_shape= INPUT_SHAPE,

                                               include_top=False,

                                               weights='imagenet')



#set the trainable method of covolution layer as false

# why set to false?? because we don't want to mess up the pretrained weights of the model!!

base_model.trainable = False

base_model.summary()
#let's try to pass an image to the model to verify the output shape

for i,l in train_batches.take(1):

    pass

base_model(i).shape
model = Sequential()

model.add(base_model)

model.add(Layers.GlobalAveragePooling2D())

model.add(Layers.Dense(128))

model.add(Layers.Dropout(0.2))

model.add(Layers.Dense(1, activation = 'sigmoid'))

model.summary()
#add a earlystopping callback to stop the training if the model is not learning anymore

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)



#let's just choose adam as our optimizer, we all love adam anyway.

model.compile(optimizer='adam',

              loss = 'binary_crossentropy',

              metrics=['accuracy'])
model.fit(train_batches, epochs=10, validation_data=test_batches, callbacks=[callbacks])
#predict the test data

pred = model.predict_classes(np.array(test_arrays))
#let's print a classification report

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(test_data['target'], pred.flatten()))
### ohhh not that bad

### lets plot confusion matrix to make it look professional



con_mat = confusion_matrix(test_data['target'], pred.flatten())

plt.figure(figsize = (10,10))

plt.title('CONFUSION MATRIX')

sns.heatmap(con_mat, cmap='cividis',

            yticklabels=['Negative', 'Positive'],

            xticklabels=['Negative', 'Positive'],

            annot=True);