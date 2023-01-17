from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



import pandas as pd

import numpy as np

from tqdm import tqdm



import tensorflow as tf

np.random.seed(0)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

INPUT_SOURCE = '/kaggle/input/dogs-vs-cats'

FAST_RUN = False

IMAGE_WIDTH=224

IMAGE_HEIGHT=224

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS=3

FILE_PATH = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'
!rm -rf "./train"

!unzip -q "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
# Preparing the data

filenames = os.listdir("./train")

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
df.shape
# Distribution of count classes

sns.set(style="white")

sns.countplot(df["category"])
sample = np.random.choice(df['filename'])

image = load_img("./train/" + sample)

# Each image is of different shapes and has 3 channel for RGB

plt.imshow(image)

plt.show()
df['category'] = df["category"].replace({0: 'cat', 1: 'dog'}) 
# Splitting the data

train_df, val_df = train_test_split(df, test_size=.2, stratify=df["category"], random_state=42)

train_df = train_df.reset_index()

val_df = val_df.reset_index()
batch_size=32

total_train = train_df.shape[0]

total_validate = val_df.shape[0]
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

    "./train/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    val_df, 

    "./train/", 

    x_col='filename',

    y_col='category',

    target_size=IMAGE_SIZE,

    class_mode='categorical',

    batch_size=batch_size

)
class ConvBlock(tf.keras.Model):

    def __init__(self, filters, kernel, strides, padding):

        '''

        Khởi tạo Convolution Block với các tham số đầu vào

        

        Parameters

        ----------

        filters: int

            số lượng filter

        kernel: int

            kích thước kernel

        strides: int

            stride của convolution layer

        padding: str

            Loại padding của convolution layer

        

        '''

        

        super(ConvBlock, self).__init__()

        # Tạo layer Conv2D

        self.cnn = tf.keras.layers.Conv2D(filters, kernel,  strides=strides,

                                          activation='relu',

                                          kernel_initializer='he_normal',

                                          kernel_regularizer=tf.keras.regularizers.l2(1e-3),

                                          padding=padding)



        # Tạo layer MaxPool2D

        self.pool = tf.keras.layers.MaxPool2D((2, 2))

        

        

    def call(self, inputs):

        '''

        Hàm này sẽ được gọi trong quá trình forwarding của mạng

        

        Parameters

        ----------

        inputs: tensor đầu vào

        

        Returns

        -------

        tensor

            giá trị đầu ra của mạng

        '''

        

        x = inputs

        

        x = self.cnn(x)

        x = self.pool(x)



        return x
class CNN(tf.keras.Model):

    def __init__(self, num_classes):

        

        super(CNN, self).__init__()

        

        self.block1 = ConvBlock(32, (3,3), (1,1), 'same')

        self.block2 = ConvBlock(64, (3,3), (1,1), 'same')

        self.block3 = ConvBlock(64, (3,3), (1,1), 'same')

        self.block4 = ConvBlock(64, (3,3), (1,1), 'same')

        self.block5 = ConvBlock(128, (3,3), (1,1), 'same')

        self.block6 = ConvBlock(128, (3,3), (1,1), 'same')

        

        self.flatten = tf.keras.layers.Flatten()

        

        self.dropout = tf.keras.layers.Dropout(.25)

        

        self.dense2 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')

        

        self.dense3 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')

        

        self.dense1 = tf.keras.layers.Dense(num_classes)

        



    def call(self, inputs):

        

        x = inputs

        

        x = self.block1(x)

        x = self.block2(x)

        x = self.dropout(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.dropout(x)

        x = self.block5(x)

        x = self.block6(x)

        x = self.dropout(x)

        

        x = self.flatten(x)

        x = self.dense2(x)

        x = self.dropout(x)

        x = self.dense3(x)

        x = self.dense1(x)

        

        with tf.device('/CPU:0'):

            output = tf.nn.softmax(x)



        return output
from tensorflow.python.client import device_lib



def get_available_gpus():

    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos if x.device_type == 'GPU']



get_available_gpus()
device = '/GPU:0' if len(get_available_gpus()) > 0 else '/CPU:0'

print(device)

batch_size = 32

epochs = 100

num_classes = 2

history = None



with tf.device(device):

    # Khởi tạo model

    model = CNN(num_classes)

    

    # Tạo callback để lưu model có accuracy trên tập validation tốt nhất

    mcp = tf.keras.callbacks.ModelCheckpoint("model_CNN_v1.h5", monitor="val_accuracy", verbose=2,

                      save_best_only=True, save_weights_only=True)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    

    # Compile model

    learning_rate = 1e-3

    decay_rate = learning_rate / epochs

    optimizer = tf.keras.optimizers.Adam(learning_rate, decay_rate)

    model.compile(optimizer=optimizer, loss=tf.keras.backend.categorical_crossentropy,

                   metrics=['accuracy'])

    

    steps_per_epoch = total_train // batch_size

    validation_steps = total_validate // batch_size



    history = model.fit_generator(train_generator, 

                      steps_per_epoch = steps_per_epoch,

                      epochs=epochs,

                      validation_data=validation_generator, 

                      validation_steps=validation_steps,

                      verbose=1,

                      callbacks=[es, mcp])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

ax1.plot(history.history['loss'], color='b', label="Training loss")

ax1.plot(history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(1, epochs, 1))

ax1.set_yticks(np.arange(0, 1, 0.1))



ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(1, epochs, 1))



legend = plt.legend(loc='best', shadow=True)

plt.tight_layout()

plt.show()
!rm -rf "./test"

!unzip -q "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"
test_filenames = os.listdir("./test")

test_df = pd.DataFrame({

    'filename': test_filenames

})

nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    "./test/", 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    shuffle=False

)
# Load best model

model = CNN(num_classes)



# Thiết lập kích thước input cho model

dummy_x = tf.zeros((1, 224, 224, 3))

model._set_inputs(dummy_x)



# Load model đã lưu trước đó trong quá trình huấn luyện

model.load_weights('model_CNN_v1.h5')

print("Model đã được load")
# Save model

model.save("DogVsCatModelv2")
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })
# Distribution of count classes

sns.set(style="white")

sns.countplot(test_df["category"])
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)
# Dọn dẹp các file hình

import shutil

shutil.rmtree("./train")

shutil.rmtree("./test")