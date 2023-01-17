import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import cv2

import seaborn as sns

from sklearn.metrics import f1_score



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")



from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Dropout, Dense, Flatten, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.utils import np_utils

from keras.optimizers import SGD
from IPython.core.display import display, HTML

from PIL import Image

from io import BytesIO

import base64

plt.style.use("ggplot")

%matplotlib inline



import tensorflow as tf

print(tf.__version__)
main_folder = "../input/celeba-dataset/"

images_folder = main_folder + "img_align_celeba/img_align_celeba/"



example_pic = images_folder + "000506.jpg"



training_sample = 10000

validation_sample = 2000

test_sample = 2000

img_width = 178

img_height = 218

batch_size = 16

num_epochs = 5
df_attr = pd.read_csv(main_folder + 'list_attr_celeba.csv')

df_attr.set_index('image_id', inplace=True)

df_attr.replace(to_replace=-1, value=0, inplace=True)
df_attr.head(5)
df_attr.describe()
df_attr.columns
df_attr.isnull().sum()
df_attr.shape
for i,j in enumerate(df_attr.columns):

    print(i+1, j)
# load a example image



img = load_img(example_pic)

plt.grid(False)

plt.imshow(img)

df_attr.loc[example_pic.split('/')[-1]][['Smiling','Male',"Young"]]
sns.countplot(df_attr["Male"])

plt.show()
df_partition = pd.read_csv(main_folder + "list_eval_partition.csv")

df_partition.head(5)
df_partition.sample(100)
df_partition["partition"].value_counts().sort_index()
df_partition.set_index('image_id', inplace=True)

df_par_attr = df_partition.join(df_attr["Male"], how="inner")



df_par_attr.head(5)
df_par_attr.shape
def load_reshape_img(fname):

    img = load_img(fname)

    x = img_to_array(img)/255.

    x = x.reshape((1,)+x.shape)

    return x
def generate_df(partition, attr, num_samples):

    

    df_ = df_par_attr[(df_par_attr['partition'] == partition) 

                           & (df_par_attr[attr] == 0)].sample(int(num_samples/2))

    df_ = pd.concat([df_,

                      df_par_attr[(df_par_attr['partition'] == partition) 

                                  & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])



    # for Train and Validation

    if partition != 2:

        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])

        x_ = x_.reshape(x_.shape[0], 218, 178, 3)

        y_ = np_utils.to_categorical(df_[attr],2)

        

    # for Test

    else:

        x_ = []

        y_ = []



        for index, target in df_.iterrows():

            im = cv2.imread(images_folder + index)

            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (img_width, img_height)).astype(np.float32) / 255.0

            im = np.expand_dims(im, axis =0)

            x_.append(im)

            y_.append(target[attr])



    return x_, y_
# generate image generator for data augmentation



datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



# load one image and reshape



img = load_img(example_pic)

x = img_to_array(img)/255.

x = x.reshape((1,) + x.shape)



# plot 10 augmented images of the loaded image



plt.figure(figsize=(20,10))

plt.suptitle("Data augmentation", fontsize=28)



i = 0



for batch in datagen.flow(x, batch_size=1):

    plt.subplot(3,5,i+1)

    plt.grid(False)

    plt.imshow(batch.reshape(218,178, 3))

    

    if i==9:

        break

    i = i+1

    

plt.show()
# build data generators



# train data



x_train, y_train = generate_df(0, "Male", training_sample)



train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



train_datagen.fit(x_train)



train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
# validation data



x_valid, y_valid = generate_df(1, "Male", validation_sample)
# import inceptionv3 model



inc_model = InceptionV3(weights="../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_shape=(img_height,img_width,3))



print("number of layers in the model : ", len(inc_model.layers))
x = inc_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(512, activation='relu')(x)

predictions = Dense(2, activation='softmax')(x)
# creating the final model



model_ = Model(inputs=inc_model.input, outputs=predictions)



# lock initial layers to not to be trained



for layer in model_.layers[:52]:

    layer.trainable = False

    

# compile the model



model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
# train the model



checkpointer = ModelCheckpoint(filepath='weights.best.inc.male.hdf5', verbose=1, save_best_only=True)
hist = model_.fit_generator(train_generator, validation_data=(x_valid, y_valid), steps_per_epoch=training_sample/batch_size, epochs=num_epochs, callbacks=[checkpointer], verbose=1)
# plot loss with epochs



plt.figure(figsize=(18,4))

plt.plot(hist.history['loss'], label='train')

plt.plot(hist.history['val_loss'], label='validation')

plt.legend()

plt.title('loss function')

plt.show()
# Plot accuracy through epochs

plt.figure(figsize=(18, 4))

plt.plot(hist.history['acc'], label = 'train')

plt.plot(hist.history['val_acc'], label = 'valid')

plt.legend()

plt.title('Accuracy')

plt.show()
# load the best model



model_.load_weights('weights.best.inc.male.hdf5')
# test data



x_test, y_test = generate_df(2, 'Male', test_sample)



# generate predictions



model_prediction = [np.argmax(model_.predict(feature)) for feature in x_test]



# report test accuracy



test_accuracy = 100 * (np.sum(np.array(model_prediction)==y_test)/len(model_prediction))

print('model evaluation')

print("test accuracy : ", test_accuracy)

print('f1 score : ', f1_score(y_test, model_prediction))