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
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
#print(os.listdir("../input"))

import zipfile
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")
    
with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")
 # detect and init the TPU
import tensorflow as tf
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
    
    # instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
filenames = os.listdir("../working/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df.head()
df.tail()
df['category']=df['category'].replace({1:'dog',0:'cat'})
df['category'].value_counts().plot.bar()
#1-dog
#2-cat
#viewing a random image
sample=random.choice(filenames)
image=load_img("../working/train/"+sample)
plt.imshow(image)
#Building a CNN Model using keras

IMAGE_WIDTH=120
IMAGE_HEIGHT=120
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
#with tpu_strategy.scope():
model=Sequential()
model.add(Conv2D(32, (3, 3), 
                     activation='relu',
                     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(60, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

    
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(300,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
    
    
model.add(Dense(2, activation='softmax'))
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    
        
     


model.summary()
#using earlystooping to prevent overfitting and learning rate reduction 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop=EarlyStopping(patience=10)
learning_rate_reduction=ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000010)
callbacks=[earlystop,learning_rate_reduction]
#preproccesing data for ImageDataGenerator
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
train_total=train_df.shape[0]
validate_total=train_df.shape[0]
batch_size=32

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../working/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
validation_datagen = ImageDataGenerator(rescale=1./255)
validate_df['category']=validate_df['category'].replace({1:'dog',0:'cat'})
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../working/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "../working/train/", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()
#fitting our model
epochs=35
history=model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validate_total//batch_size,
    steps_per_epoch=train_total//batch_size,
    callbacks=callbacks
)
model.save_weights('model.h5')
#plotting out the training

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,12))

ax1.plot(history.history['loss'],color='b',label='training loss')
ax1.plot(history.history['val_loss'],color='r',label='validation loss')
ax1.set_xticks(np.arange(1,epochs,1))


ax2.plot(history.history['accuracy'],color='b',label='training accuracy')
ax2.plot(history.history['val_accuracy'],color='r',label='validation accuracy')
ax2.set_xticks(np.arange(1,epochs,1))


legend=plt.legend(loc='best',shadow=True)
plt.tight_layout()
plt.show()
#preprocessing training data
test_filenames = os.listdir("../working/test1")
#print(test_filenames)
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

#creating testing generator
test_gen=ImageDataGenerator(rescale=1./255)
test_generator=test_gen.flow_from_dataframe(
    test_df,
"../working/test1/",
x_col="filename",
y_col=None,
class_mode=None,
target_size=IMAGE_SIZE,
batch_size=batch_size,
shuffle=False)
prediction=model.predict_generator(test_generator,steps=np.ceil(nb_samples/batch_size))
test_df['category']=np.argmax(prediction,axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
#test_df['category']=test_df['category'].replace({'dog':1,'cat':0})
test_df['category'].value_counts().plot.bar()


#test_df['category']=test_df['category'].replace({0:'cat',1:'dog'})
sample_test=test_df.head(18)
sample_test.head()

sample_test.tail()
#sampling first 30 images

sample_test=test_df.head(36)
sample_test.head()
plt.figure(figsize=(18,36))
for i,r in sample_test.iterrows():
    filename=r['filename']
    category=r['category']
    img=load_img("../working/test1/"+filename,target_size=IMAGE_SIZE)
    plt.subplot(6,6,i+1)
    plt.imshow(img)
    plt.xlabel(filename+" "+format(category))
plt.tight_layout()
plt.show()
#sampling last 30 images
sample_test=test_df.tail(36)
sample_test.head()
plt.figure(figsize=(12,24))
for i,r in sample_test.iterrows():
    filename=r['filename']
    category=r['category']
    img=load_img("../working/test1/"+filename,target_size=IMAGE_SIZE)
    plt.subplot(6,6,12500-i)
    plt.imshow(img)
    plt.xlabel(filename+" "+format(category))
plt.tight_layout()
plt.show()
