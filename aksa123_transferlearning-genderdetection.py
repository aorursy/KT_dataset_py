# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg16 import VGG16
#from keras.applications.ResNet152V2 import preprocess_input
#from keras.applications.vgg16 import preprocess_input
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD
import tensorflow as tf
from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64

plt.style.use('ggplot')

%matplotlib inline
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


df = pd.read_csv("../input/celeba-dataset/list_attr_celeba.csv")
df = reduce_mem_usage(df)
df.reset_index(inplace = True)
df.set_index('image_id', inplace=True)
df.drop(columns=['index'], inplace=True)
df.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0
df.shape

filename = df.index[567]
image_folder = "../input/celeba-dataset/img_align_celeba/img_align_celeba/"
imagepath = image_folder + filename
img = cv2.imread(imagepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(imagepath)
print(f"Smiling: {df.loc[filename]['Smiling']}, Young: {df.loc[filename]['Young']}, Straight Hair: {df.loc[filename]['Straight_Hair']}")
plt.imshow(img)
plt.show()
import gc
del img
gc.collect()
#plt.title('Female or Male')
#sns.countplot(y='Male', data=df, color="c")
#plt.show()
tvtsplit = pd.read_csv("../input/celeba-dataset/list_eval_partition.csv")
tvtsplit = reduce_mem_usage(tvtsplit)
#tvtsplit
tvtsplit['partition'].value_counts().sort_index()
tvtsplit.reset_index(inplace=True)
tvtsplit.set_index("image_id", inplace=True)
tvtsplit.drop(columns=['index'], inplace=True)
tvtsplit.head()
tvtmerge = pd.merge(tvtsplit, df['Male'], on="image_id", how="inner")
tvtmerge = reduce_mem_usage(tvtmerge)
del tvtsplit
gc.collect()
IMG_WIDTH = 224
IMG_HEIGHT = 224
TRAINING_SAMPLES =5000
VALIDATION_SAMPLES = 1500
TEST_SAMPLES = 1500
'01610sdsds9.jpg' in tvtmerge.index
def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = cv2.resize(x, (224, 224))
    #x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
    
    '''
    
    df_ = tvtmerge[(tvtmerge['partition'] == partition) 
                           & (tvtmerge[attr] == 0)].sample(int(num_samples/2))
    
    #df_ = df_.append(tvtmerge[(tvtmerge['partition'] == partition) & (tvtmerge[attr] == 1)].sample(int(num_samples/2)))
    df_ = pd.concat([df_,
                      tvtmerge[(tvtmerge['partition'] == partition) 
                                  & (tvtmerge[attr] == 1)].sample(int(num_samples/2))])
    df_ = reduce_mem_usage(df_)
    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(image_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 224, 224, 3)
        y_ = np.array(df_[attr])
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(image_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
            #im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])
    del df_
    gc.collect()
    return x_, y_
# Train data
x_train, y_train = generate_df(0, 'Male', TRAINING_SAMPLES)

# Train - Data Preparation - Data Augmentation with generators
train_datagen =  ImageDataGenerator(
  #preprocessing_function=preprocess_input,
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
batch_size=32
)
del x_train, y_train
gc.collect()
x_valid, y_valid = generate_df(1, 'Male', VALIDATION_SAMPLES)
#vgg = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)
resnet = tf.keras.applications.ResNet152V2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224,3),
)
# don't train existing weights
for layer in resnet.layers:
  layer.trainable = False
x = Flatten()(resnet.output)
x = Dense(1000, activation="relu")(x)
#x = Dropout(0.5)(x)
x = Dense(400, activation="relu")(x)
#x = Dropout(0.5)(x)
x = Dense(400, activation="relu")(x)
#x = Dropout(0.5)(x)
prediction = Dense(1, activation='sigmoid')(x)
# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

# view the structure of the model
model.summary()
# tell the model what cost and optimization method to use
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
  loss='binary_crossentropy',
  optimizer=opt,
  metrics=['accuracy']
)
filepath="model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy',save_weights_only=False, verbose=1, save_best_only=True, mode='max')
#del r 
#gc.collect()
r = model.fit_generator(
  train_generator,
  validation_data=(x_valid, y_valid),
  epochs=10,
  verbose =1,
  steps_per_epoch=(TRAINING_SAMPLES//32 ),
  callbacks = [checkpoint]
)
#del r
gc.collect()
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
del plt
gc.collect()
import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
del plt
gc.collect()
import tensorflow as tf

from keras.models import load_model

best_model = load_model('./model.h5')

#del x_test, y_test
#gc.collect()
x_test, y_test = generate_df(2, 'Male', TEST_SAMPLES)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test.shape
### START CODE HERE ### (1 line)
preds = model.evaluate(x_test,y_test)
### END CODE HERE ###
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
Y_pred = best_model.predict(x_test)
predict_labels = Y_pred
predict_labels[predict_labels<0.5] = 0
predict_labels[predict_labels>=0.5] = 1
from sklearn.metrics import f1_score
print('f1_score:', f1_score(y_test, predict_labels))
import matplotlib.pyplot as plt
a = predict_labels.reshape(-1)
#fnames = test_set.filenames
path = '../input/celeba-dataset/img_align_celeba/img_align_celeba/'
b = x_test[np.where(a!=y_test)[0]]
print(f"Prediction: {predict_labels[np.where(a!=y_test)[0]][7]}")
print(f"Original: {y_test[np.where(a!=y_test)[0]][7]}")
plt.imshow(b[7])
plt.show()
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, predict_labels)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Female', 'Male'], fontsize=16)
plt.yticks(range(2), ['Female', 'Male'], fontsize=16)
plt.show()