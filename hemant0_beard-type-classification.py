import pandas as pd
import numpy as np
import cv2    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

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

import tensorflow as tf
print(tf.__version__)
celeb_path = '../input/celeba-dataset/list_attr_celeba.csv'
celeb_data = pd.read_csv(celeb_path) 
celeb_data.set_index('image_id', inplace=True)
celeb_data.replace(to_replace=-1, value=0, inplace=True)
print(celeb_data.columns)

# Drop all females
indexNames = celeb_data[celeb_data['Male'] == 0 ].index 
celeb_data.drop(indexNames , inplace=True)
celeb_data.describe()
celeb_features = ['Goatee', 'Mustache', 'No_Beard']
celeb_data = celeb_data[celeb_features]
# celeb_data.head()

celeb_data.loc[(celeb_data['Goatee'] == 0) & (celeb_data['No_Beard'] == 1) & (celeb_data['Mustache'] == 1)]
# set variables 
main_folder = '../input/celeba-dataset/'
images_folder = main_folder + 'img_align_celeba/img_align_celeba/'

EXAMPLE_PIC = images_folder + '000931.jpg'

TRAINING_SAMPLES = 10000
VALIDATION_SAMPLES = 2000
TEST_SAMPLES = 2000
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 20

img = load_img(EXAMPLE_PIC)
plt.grid(False)
plt.imshow(img)
# Distribution

plt.title('Without Beard')
sns.countplot(y='No_Beard', data=celeb_data, color="c")
plt.show()
df_partition = pd.read_csv(main_folder + 'list_eval_partition.csv')
df_partition['partition'].value_counts().sort_index()
df_partition.set_index('image_id', inplace=True)
df_par_attr = df_partition.join(celeb_data, how='inner')
df_par_attr.describe()
def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)
    return x

def generate_df(partition, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
    
    '''
#     df_ = df_par_attr[(df_par_attr['partition'] == partition) 
#                            & (df_par_attr[attr] == 0)].sample(int(num_samples/2))
#     df_ = pd.concat([df_,
#                       df_par_attr[(df_par_attr['partition'] == partition) 
#                                   & (df_par_attr[attr] == 1)].sample(int(num_samples/2))])

    df_ = df_par_attr[(df_par_attr['partition'] == partition)].sample(num_samples)

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = df_[celeb_features].values.astype('int32')
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[celeb_features])

    return x_, y_


# Generate image generator for data augmentation
datagen =  ImageDataGenerator(
  #preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
# load one image and reshape
img = load_img(EXAMPLE_PIC)
x = img_to_array(img)/255.
x = x.reshape((1,) + x.shape)

# plot 10 augmented images of the loaded iamge
plt.figure(figsize=(20,10))
plt.suptitle('Data Augmentation', fontsize=28)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(3, 5, i+1)
    plt.grid(False)
    plt.imshow( batch.reshape(218, 178, 3))
    
    if i == 9:
        break
    i += 1
    
plt.show()
# Train data
x_train, y_train = generate_df(0, TRAINING_SAMPLES)

# Train - Data Preparation - Data Augmentation with generators
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
y_train

# Validation Data
x_valid, y_valid = generate_df(1, VALIDATION_SAMPLES)
# Import InceptionV3 Model
inc_model = InceptionV3(weights='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

print("number of layers:", len(inc_model.layers))
#inc_model.summary()
#Adding custom Layers
x = inc_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)
# creating the final model 
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
hist = model_.fit_generator(train_generator
                     , validation_data = (x_valid, y_valid)
                      , steps_per_epoch= TRAINING_SAMPLES/BATCH_SIZE
                      , epochs= NUM_EPOCHS
                      , callbacks=[checkpointer]
                      , verbose=1
                    )
#load the best model
model_.load_weights('weights.best.inc.male.hdf5')
# Test Data
x_test, y_test = generate_df(2, TEST_SAMPLES)

# generate prediction
model_predictions = [np.argmax(model_.predict(feature)) for feature in x_test ]
print(model_predictions)
# report test accuracy
test_accuracy = 100 * np.sum(np.array(model_predictions)==y_test) / len(model_predictions)
print('Model Evaluation')
print('Test accuracy: %.4f%%' % test_accuracy)
print('f1_score:', f1_score(y_test, model_predictions))