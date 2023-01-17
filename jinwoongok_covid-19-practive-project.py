# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

sns.set()

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.applications import DenseNet121, VGG19, ResNet50



import PIL.Image

import matplotlib.image as mpimg



import os



from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from tensorflow.keras.preprocessing import image



from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")



from sklearn.utils import shuffle



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv')

train_df.shape

train_df.head(10)
train_df.info()
missing_vals = train_df.isnull().sum()

missing_vals.plot(kind = 'bar')
train_df.dropna(how='all')

train_df.isnull().sum()
train_df.fillna('unknown', inplace=True)

train_df.isnull().sum()
train_data = train_df[train_df['Dataset_type'] == 'TRAIN']

test_data = train_df[train_df['Dataset_type'] == 'TEST']

assert train_data.shape[0] + test_data.shape[0] == train_df.shape[0]



print(f"Shape of train data : {train_data.shape}")

print(f"Shape of test data : {test_data.shape}") #f-string 처음 알게 됨.

test_data.sample(10)
print((train_df['Label_1_Virus_category']).value_counts())
print((train_df['Label_2_Virus_category']).value_counts())
plt.figure(figsize=(15,10))

sns.countplot(train_data['Label_2_Virus_category']);
test_img_dir= '/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'

train_img_dir ='/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'



sample_train_images = list(os.walk(train_img_dir))[0][2][:8]

sample_train_images = list(map(lambda x: os.path.join(train_img_dir, x), sample_train_images)) #??



sample_test_images = list(os.walk(test_img_dir))[0][2][:8]

sample_test_images = list(map(lambda x: os.path.join(test_img_dir, x), sample_test_images))



#glob공부

plt.figure(figsize= (10,10))

for iterator, filename in enumerate(sample_train_images):

    image = PIL.Image.open(filename)

    plt.subplot(4,2, iterator+1)

    plt.imshow(image, cmap= plt.cm.bone)

    

plt.tight_layout()
fig, ax = plt.subplots(4,2,figsize=(15,10))



covid_path = train_data[train_data['Label_2_Virus_category']=='COVID-19']['X_ray_image_name'].values



sample_covid_path = covid_path[:4]

sample_covid_path = list(map(lambda x: os.path.join(train_img_dir, x), sample_covid_path))



for row, file in enumerate(sample_covid_path):

    image = plt.imread(file)

    ax[row, 0].imshow(image, cmap=plt.cm.bone)

    ax[row, 1].hist(image.ravel(), 256, [0,256])

    ax[row, 0].axis('off')

    if row==0:

        ax[row,0].set_title('Images')

        ax[row, 1].set_title('Histograms')

fig.suptitle('Label 2 Virus Category = COVID-19', size=16)

plt.show()
fig, ax = plt.subplots(4,2,figsize=(15,10))



normal_path = train_data[train_data['Label']=='Normal']['X_ray_image_name'].values



sample_normal_path = normal_path[:4]

sample_normal_path = list(map(lambda x:os.path.join(train_img_dir, x), sample_normal_path))



for row, file in enumerate(sample_normal_path):

    image = plt.imread(file)

    ax[row,0].imshow(image, cmap=plt.cm.bone)

    ax[row,1].hist(image.ravel(), 256, [0,256])

    ax[row,0].axis('off')

    if row ==0:

        ax[row, 0].set_title('Images')

        ax[row, 1].set_title('Histograms')

fig.suptitle('Label = NORMAL', size=16)

plt.show()
#remove Pnuemonia with unknown value

final_train_data = train_data[(train_data['Label']=='Normal') |

                             ((train_data['Label'] =='Pnemonia') &

                              (train_data['Label_2_Virus_category'] ==

                               'COVID-19'))]



#add a target and class feature



final_train_data['class'] = final_train_data.Label.apply(lambda x : 'negative' if x=='Normal' else 'positive')

test_data['class'] = test_data.Label.apply(lambda x: 'negative' if x=='Normal' else 'positive')



final_train_data['target'] = final_train_data.Label.apply(lambda x: 0 if x=='Normal' else 1)

test_data['target'] = test_data.Label.apply(lambda x: 0 if x=='Normal' else 1)



# get the important features

final_train_data = final_train_data[['X_ray_image_name', 'class', 'target', 'Label_2_Virus_category']]

final_test_data = test_data[['X_ray_image_name', 'class', 'target']]



test_data['Label'].value_counts()
#create a imagegenerator for for augmentation

datagen =  ImageDataGenerator(

  shear_range=0.2,

  zoom_range=0.2,

)

def read_img(filename, size, path):

    img = load_img(os.path.join(path, filename), target_size=size)

    #convert image to array

    img = img_to_array(img) / 255

    return img
#augment the images labeled with covid-19 to balance the data



corona_df = final_train_data[final_train_data['Label_2_Virus_category'] == 'COVID-19']

with_corona_augmented = []



#create a function for augmentation

def augment(name):

    img = read_img(name, (255,255), train_img_dir)

    i = 0

    for batch in tqdm(datagen.flow(tf.expand_dims(img, 0), batch_size=32)):

        with_corona_augmented.append(tf.squeeze(batch).numpy())

        if i == 20:

            break

        i =i+1



#apply the function

corona_df['X_ray_image_name'].apply(augment)





#extract the image from training data and test data, then convert them as array



train_arrays=[]

final_train_data['X_ray_image_name'].apply(lambda x: train_arrays.append(read_img(x, (255,255), train_img_dir)))

test_arrays=[]

final_test_data['X_ray_image_name'].apply(lambda x: test_arrays.append(read_img(x, (255,255), test_img_dir)))
#concatenate(korean:연결짓다.) the training data labels and the labels for augmented images



y_train = np.concatenate((np.int64(final_train_data['target'].values), np.ones(len(with_corona_augmented), dtype=np.int64)))
#Converting Data to tensors

train_tensors = tf.convert_to_tensor(np.concatenate((np.array(train_arrays), np.array(with_corona_augmented))))

test_tensors = tf.convert_to_tensor(np.array(test_arrays))

y_train_tensor = tf.convert_to_tensor(y_train)

y_test_tensor = tf.convert_to_tensor(final_test_data['target'].values)



train_dataset = tf.data.Dataset.from_tensor_slices((train_tensors, y_train_tensor))

test_dataset = tf.data.Dataset.from_tensor_slices((test_tensors, y_test_tensor))
BATCH_SIZE = 16

BUFFER = 1000



train_batches = train_dataset.shuffle(BUFFER).batch(BATCH_SIZE)

test_batches = test_dataset.batch(BATCH_SIZE)
#define input shape

INPUT_SHAPE = (255,255,3) 



#get the pretrained model

base_model = tf.keras.applications.ResNet50(input_shape= INPUT_SHAPE,

                                               include_top=False,

                                               weights='imagenet')



#set the trainable method of covolution layer as false

# why set to false?? because we don't want to mess up the pretrained weights of the model!!

base_model.trainable = False
#let's try to pass an image to the model to verify the output shape

for i,l in train_batches.take(1):

    pass

base_model(i).shape
model= Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(128))

model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid'))
#add a earlystopping callback to stop the training if the model is not learning anymore



callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)



# Let's just choose adam as our optimizer, we all love adam anyway.

model.compile(optimizer='adam', 

             loss='binary_crossentropy',

             metrics=['accuracy'])
model.fit(train_batches,epochs=10, validation_data=test_batches, callbacks=[callbacks] )
#predict the test

pred = model.predict_classes(np.array(test_arrays))

#Let's print a classification report

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(test_data['target'], pred.flatten()))
con_mat = confusion_matrix(test_data['target'], pred.flatten())

plt.figure(figsize=(10,10))

plt.title('CONFUSION MATRIX')

sns.heatmap(con_mat, cmap='coolwarm',

          yticklabels=['Negative', 'Positive'],

          xticklabels=['Negative', 'Positive'],

          annot=True);