import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import f1_score



import tensorflow as tf



from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import Sequential, Model

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.utils import np_utils

from keras import optimizers

from keras.optimizers import SGD, RMSprop



from IPython.core.display import display, HTML

from PIL import Image

from io import BytesIO

import base64



plt.style.use('ggplot')



%matplotlib inline



print(tf.__version__)
base_folder = '../input/celeba-dataset/'

images_folder = base_folder + 'img_align_celeba/img_align_celeba/'



TEST_PIC = images_folder + '000426.jpg'



TRAINING_SAMPLES = 10000

VALIDATION_SAMPLES = 2000

TEST_SAMPLES = 2000

IMG_WIDTH = 178

IMG_HEIGHT = 218

BATCH_SIZE = 16

NUM_EPOCHS = 20

#my_data = pd.read_csv(f"{base_folder}/list_attr_celeba.csv")

#my_data
df_attr = pd.read_csv(base_folder + 'list_attr_celeba.csv')

df_attr.set_index('image_id', inplace=True)

df_attr.replace(to_replace=-1, value=0, inplace=True)



for i, j in enumerate(df_attr.columns):

    print(i, j)
img = load_img(TEST_PIC)

plt.grid(False)

plt.imshow(img)

df_attr.loc[TEST_PIC.split('/')[-1]][['Smiling', 'Male', 'Young']]
sns.countplot(df_attr['Male'])

plt.show()
df_partition = pd.read_csv(base_folder + 'list_eval_partition.csv')

df_partition.head()
df_partition['partition'].value_counts().sort_index()
df_partition.set_index('image_id', inplace=True)

df_par_attr = df_partition.join(df_attr['Male'], how='inner')

df_par_attr.head()
def load_reshape_img(fname):

    img = load_img(fname)

    x = img_to_array(img)/255.

    x = x.reshape((1,) + x.shape)

    

    return x



def generate_df(partition, attr, num_samples):

    df_ = df_par_attr[(df_par_attr['partition'] == partition) 

                      & (df_par_attr[attr] == 0)].sample(int(num_samples/2))

    df_ = pd.concat([df_, df_par_attr[(df_par_attr['partition'] == partition)

                                     & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])

    if partition != 2:

        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])

        x_ = x_.reshape(x_.shape[0], 218, 178, 3)

        y_ = np_utils.to_categorical(df_[attr], 2)

    

    else:

        x_ = []

        y_ = []

        

        for index, target in df_.iterrows():

            im = cv2.imread(images_folder + index)

            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32)/255.0

            im = np.expand_dims(im, axis=0)

            x_.append(im)

            y_.append(target[attr])

    return x_, y_
image_datagen = ImageDataGenerator(

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

)



img=load_img(TEST_PIC)

x = img_to_array(img)/255.

x=x.reshape((1,) + x.shape)



plt.figure(figsize=(20, 10))

plt.suptitle('Data Augmentation', fontsize=28)



i=0

for batch in image_datagen.flow(x, batch_size=1):

    plt.subplot(3, 5, i+1)

    plt.grid(False)

    plt.imshow(batch.reshape(218, 178, 3))

    if i == 9:

        break

    i+= 1

plt.show()
x_train, y_train = generate_df(0, 'Male', TRAINING_SAMPLES)



train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                  rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



train_datagen.fit(x_train)



train_generator=train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
x_valid, y_valid = generate_df(1, 'Male', VALIDATION_SAMPLES)
inc_model=InceptionV3(weights='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',

                     include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

print('Number of layers: ', len(inc_model.layers))
x = inc_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(512, activation="relu")(x)

predictions = Dense(2, activation="softmax")(x)
model_ = Model(inputs=inc_model.input, outputs=predictions)



for layer in model_.layers[:52]:

    layer.trainable=False

    

model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='weights.best.inc.male.hdf5', verbose=1, save_best_only=True)
hist = model_.fit_generator(train_generator, 

                            validation_data=(x_valid, y_valid),

                            steps_per_epoch= TRAINING_SAMPLES/BATCH_SIZE,

                            epochs= NUM_EPOCHS,

                            callbacks=[checkpointer],

                            verbose= 1

                           )
plt.figure(figsize=(18, 4))

plt.plot(hist.history['loss'], label = 'train')

plt.plot(hist.history['val_loss'], label='valid')

plt.legend()

plt.title('Loss')

plt.show()
plt.figure(figsize=(18, 4))

plt.plot(hist.history['accuracy'], label = 'train')

plt.plot(hist.history['val_accuracy'], label='valid')

plt.legend()

plt.title('Accuracy')

plt.show()
model_.load_weights('weights.best.inc.male.hdf5')
x_test, y_test = generate_df(2, 'Male', TEST_SAMPLES)

model_predictions = [np.argmax(model_.predict(feature)) for feature in x_test]

test_accuracy = 100*np.sum(np.array(model_predictions)==y_test)/len(model_predictions)

print('Model Evaluation')

print('Test Accuracy: %.4f%%' % test_accuracy)

print('f1_score: ', f1_score(y_test, model_predictions))