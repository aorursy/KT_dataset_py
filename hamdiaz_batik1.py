!ls /kaggle/input/shopeeproductdetectiondataset/
import numpy as np

import pandas as pd



import os

import matplotlib.pyplot as plt
import keras

from keras import backend as K

from keras.models import Sequential

from keras.models import Model

from keras.layers import Activation, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers.core import Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
!pip install git+https://github.com/qubvel/efficientnet

# from keras.applications.nasnet import NASNetMobile

from efficientnet.keras import EfficientNetB4 as efn
batch_size = 128

epochs = 10

epochs_range = range(epochs)



# changed to 224 because it is forced for NASNetMobile

IMG_HEIGHT = 32

IMG_WIDTH = 32
# train = pd.read_csv('/kaggle/input/shopeeproductdetectiondataset/train.csv')

test = pd.read_csv('/kaggle/input/shopeeproductdetectiondataset/test.csv')
len(os.listdir('/kaggle/input/shopeeproductdetectiondataset/test/test'))
path = '/kaggle/input/shopeeproductdetectiondataset/train/train/'

classes = os.listdir(path)
arr_train = []

arr_val = []

for c in classes:

    arr = os.listdir(path+c)

    limit = len(arr)//5

    for i in range(limit):

        arr_val.append([c+'/'+arr[i], c])

    for i in range(limit, len(arr)):

        arr_train.append([c+'/'+arr[i], c])
train = pd.DataFrame(arr_train, columns=['file', 'class'])

val = pd.DataFrame(arr_val, columns=['file', 'class'])



train.shape, val.shape
image_gen_train = ImageDataGenerator(

                    rescale=1./255,

                    rotation_range=45,

                    width_shift_range=.15,

                    height_shift_range=.15,

                    horizontal_flip=True,

                    zoom_range=0.5)



image_gen_val = ImageDataGenerator(rescale=1./255)



image_gen_test = ImageDataGenerator(rescale=1./255)
# https://www.kaggle.com/kmkarakaya/transfer-learning-for-image-classification

# https://keras.io/api/applications/
train.head()
train_dir = '/kaggle/input/shopeeproductdetectiondataset/train/train/'

test_dir = '/kaggle/input/shopeeproductdetectiondataset/test/'

# classes = ['00', '01']

# classes = None

classes = os.listdir(train_dir)
train_data_gen = image_gen_train.flow_from_dataframe(batch_size=batch_size,

                                                     dataframe=train,

                                                     directory=train_dir,

                                                     shuffle=True,

                                                     x_col="file",

                                                     y_col="class",

                                                     classes=classes,

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),

#                                                      subset='training'

                                                    )



validation_data_gen = image_gen_val.flow_from_dataframe(batch_size=batch_size,

                                                     dataframe=val,

                                                     directory=train_dir,

                                                     shuffle=True,

                                                     x_col="file",

                                                     y_col="class",

                                                     classes=classes,

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),

#                                                      subset='validation'

                                                       )
test_data_gen = image_gen_test.flow_from_directory(batch_size=batch_size,

                                                   directory=test_dir,

                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                   shuffle= False)
total_train = train_data_gen.samples + (batch_size-1)

total_val = validation_data_gen.samples + (batch_size-1)

total_test = test_data_gen.samples + (batch_size-1)
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)
len(train_data_gen.class_indices)
# NS_WEIGHTS_PATH = 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/'

# model_name = 'efficientnet-b4'

# weights = 'noisy-student'



# file_name = "{}_{}.h5".format(model_name, weights)

# file_hash = 'c000bfa03bf3c93557851b4e1fe18f51' #NS_WEIGHTS_HASHES[model_name][0]

# weights_path = keras.utils.get_file(

#     file_name,

#     NS_WEIGHTS_PATH + file_name,

#     cache_subdir='../input/',

#     file_hash=file_hash,

# )
# base_model = Xception(weights=None,

#                  include_top=False,

#                  input_shape=(IMG_HEIGHT, IMG_WIDTH ,3))



base_model = efn(weights='imagenet',

                 include_top=False,

                 input_shape=(IMG_HEIGHT, IMG_WIDTH ,3))



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

prediction = Dense(len(train_data_gen.class_indices), activation='softmax')(x)



model = Model(input=base_model.input, output=prediction)
# model.load_weights('/kaggle/input/modelx1/cobaBatik1.h5')
# base_model = NASNetMobile(weights='imagenet',

#                           include_top=False,

#                           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3))



# base_model.trainable = False
model.compile(optimizer='adam',

#               loss=keras.losses.BinaryCrossentropy(from_logits=True),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



# model.summary()
# checkpoint_path = 'training_1/cp-{epoch:04d}.ckpt'

# checkpoint_dir = os.path.dirname(checkpoint_path)



# cp_callback =  keras.callbacks.ModelCheckpoint(checkpoint_path,

#                                                  verbose=1,

#                                                  save_weights_only=True,

#                                                  period=1)
history = model.fit_generator(

    train_data_gen,

    steps_per_epoch=total_train // batch_size,

    epochs=epochs,

    validation_data=validation_data_gen,

    validation_steps=total_val // batch_size

)
model.save_weights("/kaggle/working/cobaBatike1.h5")
# !nvidia-smi
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
preds = model.predict_generator(test_data_gen,

                                steps=total_test // batch_size,

                                verbose=1)
class_name = [None]*len(train_data_gen.class_indices)

for key, val in train_data_gen.class_indices.items():

    class_name[val] = key

    

predClass = [class_name[idx] for idx in np.argmax(preds,axis=1)]

names = [name.split('/')[1] for name in test_data_gen.filenames]



print(len(predClass), len(names))
predDict = {}

for key, val in zip(names, predClass):

    predDict[key] = val

    

test['category'] = test['filename'].map(predDict)

test['category'].value_counts()
test[test['filename'].isin(['c7fd77508a8c355eaab0d4e10efd6b15.jpg',

                            '0b29fd3063e2265ef2bbd430b64cad32.jpg'])]

# index:1 result is (probably) "mobil2an"
# https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
test.shape
create_download_link(test)