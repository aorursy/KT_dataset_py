import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=UserWarning)



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import time

import cv2

import gc



from keras import backend as K

from keras import losses

from keras.applications.resnet50 import preprocess_input, ResNet50

#from keras.applications.densenet import DenseNet121, preprocess_input

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ProgbarLogger

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, BatchNormalization, Input, Flatten, LeakyReLU

from keras.models import load_model, Model, Sequential

from keras.optimizers import Adam, SGD, RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split
SEED = 575

IMAGE_SIZE = 256

NUM_CLASSES = 5

BATCH_SIZE = 32



test_image_directory = '../input/aptos2019-blindness-detection/test_images/'

test_data_file = '../input/aptos2019-blindness-detection/test.csv'

#weights_file = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

best_weight_file = '../input/weights6/aptos_best_weights(1).h5'

submission_file = 'submission.csv'

sample_file = '../input/aptos2019-blindness-detection/sample_submission.csv'
#os.listdir('../input')
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def preprocess_image(img):

    height, width = img.shape[0], img.shape[1]

    ratio = height / width

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = crop_image_from_gray(img)

    newheight, newwidth = int(IMAGE_SIZE * ratio), IMAGE_SIZE

    #print('Resizing from ({},{}) to ({},{})'.format(width, height, newwidth, newheight))

    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), newheight/30) ,-4 ,128)

    img = cv2.resize(img, (newwidth, newheight), interpolation=cv2.INTER_AREA)

    img = preprocess_input(img)

    return img
def build_model():

    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    base_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)

    output_tensor = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model
model = build_model()
model.load_weights(best_weight_file)
test_generator = ImageDataGenerator(preprocessing_function=preprocess_image)
df_test = pd.read_csv(test_data_file)

df_sample = pd.read_csv(sample_file)

df_test['filename'] = df_test['id_code'].apply(lambda i : "{}.png".format(i))
test_flow = test_generator.flow_from_dataframe(directory=test_image_directory, dataframe=df_test, x_col='filename', batch_size=BATCH_SIZE, max_queue_size=128, class_mode=None)
y_pred = model.predict_generator(

                    generator = test_flow,

                    steps = (test_flow.n // test_flow.batch_size) + 1,

                    verbose=1,

                    max_queue_size = 75,

                    workers=4

                )
df_test['diagnosis'] = np.argmax(y_pred[0:len(df_test)], axis=-1).astype('uint8')
df_test.head(10)
df_test.groupby('diagnosis').count()
df_sample['id_code'] = df_test['id_code']

df_sample['diagnosis'] = df_test['diagnosis']
df_sample.to_csv(submission_file, index=False)
df_sample.head(10)
from IPython.display import FileLink

FileLink(submission_file)