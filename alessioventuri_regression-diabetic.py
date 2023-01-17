# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install git+https://github.com/keras-team/keras-preprocessing.git

from keras_preprocessing.image import ImageDataGenerator

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

TRAIN_IMG_PATH = '/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train/'





import pandas as pd

from sklearn.model_selection import train_test_split

trainData = pd.read_csv("../input/diabetic-retinopathy-resized/trainLabels.csv")

trainLabels_cropped = pd.read_csv("../input/diabetic-retinopathy-resized/trainLabels_cropped.csv")



trainData = trainData[['image','level']]

trainData['image'] = trainData['image'].astype(str) + '.jpeg'



df = trainData[:30000]

test_df = trainData[30000:]

train_df,val_df = train_test_split(df, test_size = 0.07)







print(train_df.shape)

print(test_df.shape)

print(val_df.shape)

import tensorflow as tf

IMG_WIDTH = 299

IMG_HEIGHT = 299

CHANNELS = 3

BATCH_SIZE = 64

im_size = 299



seed = 11

np.random.seed(seed)

tf.random.set_seed(seed)

def crop_image_from_gray(img, tol=7):

    """

    Applies masks to the orignal image and 

    returns the a preprocessed image with 

    3 channels

    """

    # If for some reason we only have two channels

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    # If we have a normal RGB images

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

    

# Make all images circular (possible data loss)

def circle_crop(img):   

    """

    Create circular crop around image centre    

    """    

        

    img = crop_image_from_gray(img)

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    return img 

def circle_crop_test(img):   

    """

    Create circular crop around image centre    

    """    

        

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    return img 





def preprocess_image(image):

    """

    The whole preprocessing pipeline:

    1. Add Gaussian noise to increase Robustness

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_WIDTH,IMG_HEIGHT))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,10), -4, 128)

    return image



def preprocess_image_train(image):

    """

    The whole preprocessing pipeline:

    1. Add Gaussian noise to increase Robustness

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_WIDTH,IMG_HEIGHT))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,10), -4, 128)

    return image



train_datagen = ImageDataGenerator(#rotation_range=90,

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                   zoom_range=(0.7,1),

                                   fill_mode= 'constant',

                                   brightness_range=(0.5,2),

                                   cval = 0,

                                   preprocessing_function=preprocess_image_train,

                                   rescale=1./255

                                  )



val_datagen = ImageDataGenerator(rescale=1./255,

                                 preprocessing_function=preprocess_image)



test_datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_image)

                               

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import RMSprop,Adam

from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,GlobalAveragePooling2D

from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense



inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT,3),name='inputs')

layer = Conv2D(256,(3,3),activation = 'relu',padding = 'same')(inputs)

layer = Conv2D(128,(3,3),activation = 'relu',padding = 'same')(layer)

layer = GlobalAveragePooling2D(name = 'features')(layer)

layer = Dense(256, activation ='relu')(layer)

layer = Dropout(0.5)(layer)

layer = BatchNormalization()(layer)

layer = Dense(5,activation='relu')(layer)

layer = Dense(1,activation='relu')(layer)



model = Model(inputs = inputs , outputs = layer)

model.summary()


import keras



train_generator = train_datagen.flow_from_dataframe(train_df, 

                                                    x_col='image', 

                                                    y_col='level',

                                                    directory = TRAIN_IMG_PATH,

                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),

                                                    batch_size=BATCH_SIZE,

                                                    class_mode='other',

                                                    shuffle = True,

                                                    seed=seed)



val_generator = val_datagen.flow_from_dataframe(val_df, 

                                                  x_col='image', 

                                                  y_col='level',

                                                  directory = TRAIN_IMG_PATH,

                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),

                                                  batch_size=BATCH_SIZE,

                                                  class_mode='other',

                                                  shuffle= True,

                                                  seed=seed)

tes_generator = test_datagen.flow_from_dataframe(test_df,

                                                  x_col='image', 

                                                  y_col='level',

                                                  directory = TRAIN_IMG_PATH,

                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),

                                                  batch_size=BATCH_SIZE,

                                                  class_mode='other',

                                                  shuffle= False,

                                                  seed=seed)
tes_generator = test_datagen.flow_from_dataframe(test_df,

                                                  x_col='image', 

                                                  y_col='level',

                                                  directory = TRAIN_IMG_PATH,

                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),

                                                  batch_size=BATCH_SIZE,

                                                  class_mode='other',

                                                  shuffle= False,

                                                  seed=seed)
from skimage import io



def imshow(im):

    io.imshow(im)

    io.show()

    

x1,y1 = train_generator[0]

print(y1[0])

x2,y2 = val_generator[0]

x3,y3 = tes_generator[0]

print(y2[0])

print(y3[0])

imshow(x1[0])

imshow(x2[0])

imshow(x3[0])


from keras import backend as K



from keras.activations import elu

from keras.optimizers import adam, Optimizer

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau

from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import cohen_kappa_score

from keras.applications import Xception

IMG_WIDTH = 299

IMG_HEIGHT = 299

def fine():

    conv_base = Xception(weights='imagenet',

                      include_top=False,

                      input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    model = Sequential()

    model.add(conv_base)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(256,activation=elu))

    model.add(Dropout(0.5))

    model.add(BatchNormalization())

    model.add(Dense(1,activation='relu'))



    model.summary()

    return model



model = fine()

    
def rmse(y_true, y_pred):

    from keras import backend

    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))



# mean squared error (mse) for regression

def mse(y_true, y_pred):

    from keras import backend

    return backend.mean(backend.square(y_pred - y_true), axis=-1)



# coefficient of determination (R^2) for regression

def r_square(y_true, y_pred):

    from keras import backend as K

    SS_res =  K.sum(K.square(y_true - y_pred)) 

    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 

    return (1 - SS_res/(SS_tot + K.epsilon()))



def r_square_loss(y_true, y_pred):

    from keras import backend as K

    SS_res =  K.sum(K.square(y_true - y_pred)) 

    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 

    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))
model.compile(loss='mse',

             optimizer = adam(1e-6,decay= 1e-6),

             metrics=['acc',r_square,rmse])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

STEP_SIZE_TEST=tes_generator.n//tes_generator.batch_size

print()
history = model.fit(train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    epochs= 1,

                    validation_data = val_generator,

                    validation_steps = STEP_SIZE_VALID)
print(tf.__version__)

model.compile(loss='mse',

             optimizer = adam(1e-4,decay= 1e-4),

             metrics=['acc',r_square,rmse])
history = model.fit_generator(train_generator,

                    steps_per_epoch=train_generator.samples// BATCH_SIZE,

                    epochs= 15,

                    validation_data = val_generator,

                    validation_steps = val_generator.samples // BATCH_SIZE)
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("modelXception5epo.h5")

print("Saved model to disk")
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

epochs = range(len(acc))

loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot( acc, 'r', label='Training acc')

plt.plot( val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.legend()



plt.show()



plt.plot(loss, 'r', label='Training loss')

plt.plot( val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
y_pred = model.predict_generator(tes_generator)
predicted_class_indices=np.argmax(y_pred,axis=1)

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames=tes_generator.filenames

results=pd.DataFrame({"Filename":filenames,

                      "Predictions":predictions})

results.to_csv("rexsults.csv",index=False)
print(y_pred)

print(y_pred[77])


import sklearn

import math

# plot training curve for R^2 (beware of scale, starts very low negative)

plt.plot(history.history['val_r_square'])

plt.plot(history.history['r_square'])

plt.title('model R^2')

plt.ylabel('R^2')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

           

# plot training curve for rmse

plt.plot(history.history['rmse'])

plt.plot(history.history['val_rmse'])

plt.title('rmse')

plt.ylabel('rmse')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()







#-----------------------------------------------------------------------------

# print statistical figures of merit

#-----------------------------------------------------------------------------







print("\n")

print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(test_df['level'],y_pred))

print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(test_df['level'],y_pred))

print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(test_df['level'],y_pred)))

print("R square (R^2):                 %f" % sklearn.metrics.r2_score(test_df['level'],y_pred))
history = model.fit_generator(train_generator,

                    steps_per_epoch=train_generator.samples// BATCH_SIZE,

                    epochs= 10,

                    validation_data = val_generator,

                    validation_steps = val_generator.samples // BATCH_SIZE)
class OptimizedRounder(object):

    """

    An optimizer for rounding thresholds

    to maximize Quadratic Weighted Kappa score

    """

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        """

        Get loss according to

        using current coefficients

        """

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = cohen_kappa_score(y, X_p, weights='quadratic')

        return -ll



    def fit(self, X, y):

        """

        Optimize rounding thresholds

        """

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        """

        Make predictions with specified thresholds

        """

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']
def get_preds_and_labels(model, generator):

    """

    Get predictions and labels from the generator

    """

    generator.reset()

    preds = []

    labels = []

    for _ in range(int(np.ceil(generator.samples / 32))):

        x, y = next(generator)

        preds.append(model.predict(x))

        labels.append(y)

    # Flatten list of numpy arrays

    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()
from sklearn.metrics import cohen_kappa_score

from functools import partial

import scipy as sp



y_val_preds, val_labels = get_preds_and_labels(model, tes_generator)

print(y_val_preds.shape, val_labels.shape)

optR = OptimizedRounder()

optR.fit(y_val_preds, val_labels)

coefficients = optR.coefficients()

opt_val_predictions = optR.predict(y_val_preds, coefficients)

new_val_score = cohen_kappa_score(val_labels, opt_val_predictions, weights="quadratic")

print(new_val_score)
print(opt_val_predictions)
#predicted_class_indices=np.argmax(y_pred,axis=1)

labels = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

labels = dict((v,k) for k,v in labels.items())

print(labels)

predictions = [labels[k] for k in opt_val_predictions]

print(len(predictions))

print(len(filenames))

filenames=tes_generator.filenames

results=pd.DataFrame({"Filename":filenames,

                      "Predictions":predictions})

results.to_csv("rexsults.csv",index=False)
results['Predictions'].value_counts().sort_index().plot(kind="bar", 

                                                       figsize=(12,5), 

                                                       rot=0)
history = model.fit_generator(train_generator,

                    steps_per_epoch=train_generator.samples// BATCH_SIZE,

                    epochs= 15,

                    validation_data = val_generator,

                    validation_steps = val_generator.samples // BATCH_SIZE)
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model26epo.h5")

print("Saved model to disk")
history = model.fit_generator(train_generator,

                    steps_per_epoch=train_generator.samples// BATCH_SIZE,

                    epochs= 5,

                    validation_data = val_generator,

                    validation_steps = val_generator.samples // BATCH_SIZE)