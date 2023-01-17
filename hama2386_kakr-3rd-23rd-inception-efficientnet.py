DATA_PATH = '../input/2019-3rd-ml-month-with-kakr/'
import os

import tensorflow as tf

import keras

import numpy as np

import pandas as pd



TRAIN_IMG_PATH = '../input/kakr3cropdata/cropdata/cropdata/train_crop/'

TEST_IMG_PATH = '../input/kakr3cropdata/cropdata/cropdata/test_crop/'
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))



print("The number of the target class : {}".format(df_class.shape[0]))

print("The number of the target class of the Train data : {}".format(df_train['class'].nunique()))
from keras import backend as K



def get_steps(num_samples, batch_size):

  if (num_samples % batch_size) > 0 :

    return (num_samples // batch_size) + 1

  else :

    return num_samples // batch_size



def recall_m(y_target, y_pred):

        true_positives = K.sum(K.round(K.clip(y_target * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_target, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_target, y_pred):

        true_positives = K.sum(K.round(K.clip(y_target * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_target, y_pred):

    precision = precision_m(y_target, y_pred)

    recall = recall_m(y_target, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
!pip install keras-adabound

!pip install efficientnet
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, GlobalAveragePooling2D, ELU

from keras.layers.merge import concatenate

from keras.layers.normalization import BatchNormalization

from keras.applications.xception import Xception

from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3

from keras.optimizers import Adam, SGD, RMSprop

from efficientnet import EfficientNetB5

from efficientnet import EfficientNetB7

from keras_adabound import AdaBound

import tensorflow as tf

from keras import backend as K



def swish(x):

    return (K.sigmoid(x) * x)



def build_model(model_name,learning_rate):

    

    if model_name == 'x':

        pretrained_model = Xception(include_top = False, input_shape = (299,299,3), weights = 'imagenet')

    elif model_name == 'e':

        pretrained_model = EfficientNetB5(include_top = False, input_shape = (224,224,3), weights = 'imagenet')

    elif model_name == 'i':

        pretrained_model = InceptionV3(include_top = False, input_shape = (299,299,3), weights = 'imagenet')

    

    model = Sequential()

    model.add(pretrained_model)

    

    model.add(GlobalAveragePooling2D())

    

    model.add(Dense(1024,activation='relu'))

    model.add(Dropout(0.4))

    

    model.add(Dense(1024,activation='relu'))

    model.add(Dropout(0.4))

    

    model.add(Dense(196,activation='softmax'))

    

    model.summary()

    

    #opt = Adam(lr=0.001)

    opt = AdaBound(lr=learning_rate, final_lr=0.1)

    #opt = SGD(lr=0.001)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])



    return model
img_size = (299,299)

#img_size = (224,224)

#img_size = (299,299)

epochs = 100

batch_size = 16

n_fold = 5

model_name = 'i'

learning_rate = 0.00001
from sklearn.model_selection import StratifiedKFold, KFold

from keras.applications.xception import preprocess_input

from keras.preprocessing.image import ImageDataGenerator



nb_test_samples = len(df_test)



df_train["class"] = df_train["class"].astype('str')



df_train = df_train[['img_file','class']]

df_test = df_test[['img_file']]



skf = StratifiedKFold(n_splits=n_fold, random_state=42)

#skf = KFold(n_splits=n_fold, shuffle = False, random_state=42)

valid_accuracy = []



X = df_train['img_file']

Y = df_train['class']



train_datagen = ImageDataGenerator(

    rescale = 1./255,

    horizontal_flip = True,

    vertical_flip = False,

    zoom_range = 0.10,

    rotation_range=40,

    fill_mode='nearest'

)

val_datagen = ImageDataGenerator(

    rescale = 1./255

)

test_datagen = ImageDataGenerator(

    rescale = 1./255,

    horizontal_flip = True,

    rotation_range=10,

    fill_mode='nearest'

)



test_generator = test_datagen.flow_from_dataframe(

    dataframe = df_test,

    directory = TEST_IMG_PATH,

    x_col = 'img_file',

    y_col = None,

    target_size = img_size,

    color_mode = 'rgb',

    class_mode = None,

    batch_size = batch_size,

    shuffle = False

)
train_generator = train_datagen.flow_from_dataframe(

            dataframe = df_train,

            directory = TRAIN_IMG_PATH,

            x_col = 'img_file',

            y_col = 'class',

            target_size = img_size,

            color_mode = 'rgb',

            class_mode = 'categorical',

            batch_size = batch_size,

            shuffle = True,

            seed=42

        )



test_generator_f = test_datagen.flow_from_dataframe(

    dataframe = df_test,

    directory = TEST_IMG_PATH,

    x_col = 'img_file',

    y_col = None,

    target_size = (224,224),

    color_mode = 'rgb',

    class_mode = None,

    batch_size = batch_size,

    shuffle = False

)

test_generator_n = test_datagen.flow_from_dataframe(

    dataframe = df_test,

    directory = TEST_IMG_PATH,

    x_col = 'img_file',

    y_col = None,

    target_size = (299,299),

    color_mode = 'rgb',

    class_mode = None,

    batch_size = batch_size,

    shuffle = False

)



model_result = []

inception_file_path = ["../input/kakr3model/kakr3model/kakr3model/saved-inception-model-"+str(i)+".hdf5" for i in range(1,n_fold+1)]

effinet_file_path = ["../input/kakr3model/kakr3model/kakr3model/saved-efficientnet-model-"+str(i)+".hdf5" for i in range(1,n_fold+1)]



inception_weight_path = ["../input/kakr3model/kakr3modelweights/kakr3modelweights/saved-inception-model-weights-"+str(i)+".h5" for i in range(1,n_fold+1)]

effinet_weight_path = ["../input/kakr3model/kakr3modelweights/kakr3modelweights/saved-efficientnet-model-weights-"+str(i)+".h5" for i in range(1,n_fold+1)]
model = keras.models.load_model(inception_file_path[0],custom_objects={'f1_m': f1_m,'precision_m':precision_m,'recall_m':recall_m, 'AdaBound':AdaBound})



for i in range(n_fold):

    print("testing inception model",i)

    

    model.load_weights(inception_weight_path[i])

    

    for j in range(0,5):

        test_generator_n.reset()

        prediction = model.predict_generator(

            generator = test_generator_n,

            steps = get_steps(nb_test_samples, batch_size),

            verbose=1

        )

        

        model_result.append(prediction)
model = keras.models.load_model(effinet_file_path[0],custom_objects={'f1_m': f1_m,'precision_m':precision_m,'recall_m':recall_m, 'AdaBound':AdaBound})



for i in range(n_fold):

    print("testing effinet model",i)

    

    model.load_weights(effinet_weight_path[i])

    

    for j in range(0,5):

        test_generator_f.reset()

        prediction = model.predict_generator(

            generator = test_generator_f,

            steps = get_steps(nb_test_samples, batch_size),

            verbose=1

        )

        

        model_result.append(prediction)
mean_prediction_result = np.mean(model_result, axis=0)
predicted_class_indices = np.argmax(mean_prediction_result, axis=1)



labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission["class"] = predictions

submission.to_csv("submission.csv",index=False)

submission.head()