import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from kaggle_datasets import KaggleDatasets

import tensorflow as tf

from tensorflow.keras.layers import Input,Dropout,GlobalAveragePooling2D,Dense,Conv2D

from tensorflow.keras.models import Model,load_model

from tensorflow.keras.optimizers import Adam,Adamax

from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

from sklearn.model_selection import train_test_split
main_path='../input/plant-pathology-2020-fgvc7/'

train_df=pd.read_csv(os.path.join(main_path,'train.csv'))

test_df=pd.read_csv(os.path.join(main_path,'test.csv'))
DIMS=(512,512,3)

EPOCHS=50
print('Train Datashape: ',train_df.shape)

train_df.head(10)
sns.distplot(train_df['healthy'])
sns.distplot(train_df['rust'])
sns.distplot(train_df['scab'])
#Plotting some images

fig=plt.figure(figsize=(10,10))

rows,cols=3,3

for i in range(1,rows*cols+1):

    img=cv2.imread(os.path.join(main_path,'images',train_df.loc[i,'image_id']+'.jpg'))

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    fig.add_subplot(rows,cols,i)

    plt.imshow(img)

    plt.title([train_df.loc[i,'healthy'],train_df.loc[i,'multiple_diseases'],

              train_df.loc[i,'rust'],train_df.loc[i,'scab']])

plt.show()
print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

print('Running on TPU ', tpu.master())

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



strategy = tf.distribute.experimental.TPUStrategy(tpu)

print("REPLICAS: ", strategy.num_replicas_in_sync)



BATCH_SIZE = 16 * strategy.num_replicas_in_sync

gcs_path = KaggleDatasets().get_gcs_path()
def format_path(st):

    return gcs_path + '/images/' + st + '.jpg'



train_data,val_data=train_test_split(train_df,test_size=0.2)



train_paths = train_data.image_id.apply(format_path).values

val_paths = val_data.image_id.apply(format_path).values



train_labels = train_data.loc[:, 'healthy':].values

val_labels = val_data.loc[:, 'healthy':].values
def decode_image(filename,label=None,image_size=(DIMS[0],DIMS[1])):

    bits=tf.io.read_file(filename)

    img=tf.image.decode_jpeg(bits,channels=3)

    img=tf.cast(img,tf.float32)/255.0

    img=tf.image.resize(img,image_size)

    if label is None:

        return img

    else:

        return img, label

    

def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.adjust_brightness(image,0.2)

    image = tf.image.rot90(image)

    



    if label is None:

        return image

    else:

        return image, label
train_dataset=(tf.data.Dataset.from_tensor_slices((train_paths,train_labels)).map(decode_image,num_parallel_calls=AUTO)

               .map(data_augment,num_parallel_calls=AUTO).repeat()

              .shuffle(13)

              .batch(BATCH_SIZE).prefetch(AUTO))



val_dataset=(tf.data.Dataset.from_tensor_slices((val_paths,val_labels))

             .map(decode_image,num_parallel_calls=AUTO)

             .shuffle(13)

             .batch(BATCH_SIZE)

             .cache()

             .prefetch(AUTO))
with strategy.scope():

    inp=Input(shape=DIMS)

    base_feat=tf.keras.applications.DenseNet121(weights='imagenet',include_top=False)(inp)

    x=GlobalAveragePooling2D()(base_feat)

    out=Dense(4,activation='sigmoid')(x)

    model=Model(inp,out)

    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['acc'])

STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE

mc=ModelCheckpoint('classifier.h5',monitor='val_loss',save_best_only=True,verbose=1,period=1)

rop=ReduceLROnPlateau(monitor='val_loss',min_lr=0.000001,patience=3,mode='min')
history=model.fit(train_dataset,epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_data=val_dataset,

                 callbacks=[mc,rop])
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b',color='red', label='Training acc')

plt.plot(epochs, val_acc, 'b',color='blue', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', color='red', label='Training loss')

plt.plot(epochs, val_loss, 'b',color='blue', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
model=load_model('classifier.h5')

test_paths = test_df.image_id.apply(format_path).values

test_dataset=(tf.data.Dataset.from_tensor_slices(test_paths)

             .map(decode_image,num_parallel_calls=AUTO)

             .batch(BATCH_SIZE))

preds=model.predict(test_dataset,verbose=1)
sample_sub=pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')

sample_sub.loc[:, 'healthy':] = preds

sample_sub.to_csv('submission.csv', index=False)

sample_sub.head()