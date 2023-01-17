import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 

import seaborn as sns

from matplotlib import pyplot as plt



from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras import optimizers

from keras.models import Sequential, Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.utils import np_utils

from keras.optimizers import SGD



from IPython.core.display import display, HTML

from PIL import Image

from io import BytesIO

import base64



plt.style.use('ggplot')



%matplotlib inline
# set variables 

main_folder = '../input/celeba-dataset/'

images_folder = main_folder + 'img_align_celeba/img_align_celeba/'



EXAMPLE_PIC = images_folder + '000506.jpg'

# import the data set that include the attribute for each picture

df_attr = pd.read_csv(main_folder + 'list_attr_celeba.csv')

df_attr.set_index('image_id', inplace=True)

df_attr.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0

df_attr.shape
df_attr.info()
gender = df_attr["Male"]

# In gender array 0-Female while 1-Male

gender.head(5)
plt.title('Female or Male')

sns.countplot(y='Male', data=df_attr, color="c")

plt.show()
df_partition = pd.read_csv(main_folder + 'list_eval_partition.csv')

df_partition.head()
# display counter by partition

# 0 -> TRAINING

# 1 -> VALIDATION

# 2 -> TEST

df_partition['partition'].value_counts().sort_index()
df_partition.set_index('image_id', inplace=True)

df_par_attr = df_partition.join(gender, how='inner')

df_par_attr.head()
def load_reshape_img(fname):

    img = load_img(fname)

    x = img_to_array(img)/255.

    x = x.reshape((1,) + x.shape)



    return x





def generate_df(partition, attr, num_samples):

    '''

    partition

        0 -> train

        1 -> validation

        2 -> test

    

    '''

    

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

            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0

            im = np.expand_dims(im, axis =0)

            x_.append(im)

            y_.append(target[attr])



    return x_, y_
TRAINING_SAMPLES = 10000

VALIDATION_SAMPLES = 2000



x_train, y_train = generate_df(0, 'Male', TRAINING_SAMPLES)

x_valid, y_valid = generate_df(1, 'Male', VALIDATION_SAMPLES)
IMG_HEIGHT = 218

IMG_WIDTH = 178

BATCH_SIZE = 100

NUM_EPOCHS = 5



inc_model = InceptionV3(weights= 'imagenet',

                        include_top=False,

                        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))



print("number of layers:", len(inc_model.layers))
#Adding custom Layers

x = inc_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(512, activation="relu")(x)

predictions = Dense(2, activation="softmax")(x)
model_ = Model(inputs=inc_model.input, outputs=predictions)



# Lock initial layers to do not be trained

for layer in model_.layers[:52]:

    layer.trainable = False



# compile the model

model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9)

                    , loss='categorical_crossentropy'

                    , metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='weights.best.inc.male.hdf5', 

                               verbose=1, save_best_only=True)
train_datagen =  ImageDataGenerator(

  preprocessing_function=preprocess_input,

  rotation_range=30,

  width_shift_range=0.2,

  height_shift_range=0.2,

  shear_range=0.2,

  zoom_range=0.2,

  horizontal_flip=True,

)



train_datagen.fit(x_train)



train_generator = train_datagen.flow(

x_train, y_train,

batch_size=BATCH_SIZE,

)
hist = model_.fit_generator(train_generator

                     , validation_data = (x_valid, y_valid)

                      , steps_per_epoch= TRAINING_SAMPLES/BATCH_SIZE

                      , epochs= NUM_EPOCHS

                      , callbacks=[checkpointer]

                      , verbose=1

                    )
plt.figure(figsize=(18, 4))

plt.plot(hist.history['loss'], label = 'train')

plt.plot(hist.history['val_loss'], label = 'valid')

plt.legend()

plt.title('Loss Function')

plt.show()
print (hist.history)
plt.figure(figsize=(18, 4))

plt.plot(hist.history['accuracy'], label = 'train')

plt.plot(hist.history['val_accuracy'], label = 'valid')

plt.legend()

plt.title('Accuracy')

plt.show()
from sklearn.metrics import f1_score

TEST_SAMPLES = 2000

model_.load_weights('weights.best.inc.male.hdf5')



x_test, y_test = generate_df(2, 'Male', TEST_SAMPLES)



# generate prediction

model_predictions = [np.argmax(model_.predict(feature)) for feature in x_test ]



# report test accuracy

test_accuracy = 100 * np.sum(np.array(model_predictions)==y_test) / len(model_predictions)

print('Model Evaluation')

print('Test accuracy: %.4f%%' % test_accuracy)

print('f1_score:', f1_score(y_test, model_predictions))