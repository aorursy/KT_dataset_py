# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install efficientnet

import datetime

starttime = datetime.datetime.now()



import os

import sys

import cv2

import shutil

import random

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import multiprocessing as mp

import matplotlib.pyplot as plt



from keras.activations import elu



from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from keras import backend as K

from keras.models import Model

from keras.utils import to_categorical

from keras import optimizers, applications

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler,ModelCheckpoint



#from keras import load_weights

from sklearn.metrics import classification_report

from imgaug import augmenters as iaa







def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed) 

seed = 2020

seed_everything(seed)





import sys

# Repository source: https://github.com/qubvel/efficientnet

#sys.path.append(os.path.abspath('../input/efficientnetb0b7-keras-weights/efficientnet-master/efficientnet-master/'))

#from efficientnet import EfficientNetB0





def cosine_decay_with_warmup(global_step,

                             learning_rate_base,

                             total_steps,

                             warmup_learning_rate=0.0,

                             warmup_steps=0,

                             hold_base_rate_steps=0):

    """

    Cosine decay schedule with warm up period.

    In this schedule, the learning rate grows linearly from warmup_learning_rate

    to learning_rate_base for warmup_steps, then transitions to a cosine decay

    schedule.

    :param global_step {int}: global step.

    :param learning_rate_base {float}: base learning rate.

    :param total_steps {int}: total number of training steps.

    :param warmup_learning_rate {float}: initial learning rate for warm up. (default: {0.0}).

    :param warmup_steps {int}: number of warmup steps. (default: {0}).

    :param hold_base_rate_steps {int}: Optional number of steps to hold base learning rate before decaying. (default: {0}).

    :param global_step {int}: global step.

    :Returns : a float representing learning rate.

    :Raises ValueError: if warmup_learning_rate is larger than learning_rate_base, or if warmup_steps is larger than total_steps.

    """



    if total_steps < warmup_steps:

        raise ValueError('total_steps must be larger or equal to warmup_steps.')

    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(

        np.pi *

        (global_step - warmup_steps - hold_base_rate_steps

         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))

    if hold_base_rate_steps > 0:

        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,

                                 learning_rate, learning_rate_base)

    if warmup_steps > 0:

        if learning_rate_base < warmup_learning_rate:

            raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')

        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps

        warmup_rate = slope * global_step + warmup_learning_rate

        learning_rate = np.where(global_step < warmup_steps, warmup_rate,

                                 learning_rate)

    return np.where(global_step > total_steps, 0.0, learning_rate)





class WarmUpCosineDecayScheduler(Callback):

    """Cosine decay with warmup learning rate scheduler"""



    def __init__(self,

                 learning_rate_base,

                 total_steps,

                 global_step_init=0,

                 warmup_learning_rate=0.0,

                 warmup_steps=0,

                 hold_base_rate_steps=0,

                 verbose=0):

        """

        Constructor for cosine decay with warmup learning rate scheduler.

        :param learning_rate_base {float}: base learning rate.

        :param total_steps {int}: total number of training steps.

        :param global_step_init {int}: initial global step, e.g. from previous checkpoint.

        :param warmup_learning_rate {float}: initial learning rate for warm up. (default: {0.0}).

        :param warmup_steps {int}: number of warmup steps. (default: {0}).

        :param hold_base_rate_steps {int}: Optional number of steps to hold base learning rate before decaying. (default: {0}).

        :param verbose {int}: quiet, 1: update messages. (default: {0}).

        """



        super(WarmUpCosineDecayScheduler, self).__init__()

        self.learning_rate_base = learning_rate_base

        self.total_steps = total_steps

        self.global_step = global_step_init

        self.warmup_learning_rate = warmup_learning_rate

        self.warmup_steps = warmup_steps

        self.hold_base_rate_steps = hold_base_rate_steps

        self.verbose = verbose

        self.learning_rates = []



    def on_batch_end(self, batch, logs=None):

        self.global_step = self.global_step + 1

        lr = K.get_value(self.model.optimizer.lr)

        self.learning_rates.append(lr)



    def on_batch_begin(self, batch, logs=None):

        lr = cosine_decay_with_warmup(global_step=self.global_step,

                                      learning_rate_base=self.learning_rate_base,

                                      total_steps=self.total_steps,

                                      warmup_learning_rate=self.warmup_learning_rate,

                                      warmup_steps=self.warmup_steps,

                                      hold_base_rate_steps=self.hold_base_rate_steps)

        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:

            print('\nBatch %02d: setting learning rate to %s.' % (self.global_step + 1, lr))

            

            








# Model parameters

HEIGHT = 224

WIDTH = 224

CHANNELS = 3

TTA_STEPS = 5



weights_path_list = ['../input/regular5foldpreprocess/nopreprocess_effnetb5_224_fold0.h5',

                     

                     '../input/regular5foldpreprocess/nopreprocess_effnetb5_224_fold1.h5',

                     '../input/regular5foldpreprocess/nopreprocess_effnetb5_224_fold2.h5',

                     '../input/regular5foldpreprocess/nopreprocess_effnetb5_224_fold3.h5',

                     '../input/regular5foldpreprocess/nopreprocess_effnetb5_224_fold4.h5'

            ]







#The address here is very important: regular_train  regular-test

train_data = '../input/regular-deepdrid/Regular_DeepDRiD/regular_train/'

valid_data = '../input/regular-deepdrid/Regular_DeepDRiD/regular_valid/'

test_data = '../input/regular-deepdrid/Regular_DeepDRiD/regular-test/'







train_df = pd.read_csv('../input/regular-deepdrid/DR_label/DR_label/regular-fundus-training.csv')

valid_df = pd.read_csv('../input/regular-deepdrid/DR_label/DR_label/regular-fundus-validation.csv')

test_df = pd.read_csv('../input/regular-deepdrid/DR_label/DR_label/Challenge1_upload.csv')



train_df['image_id'] = train_df['image_id'] + ".jpg"# Two meathods add jpg

valid_df['image_id'] = valid_df['image_id'] + ".jpg"# Two meathods add jpg

test_df['image_id'] = test_df['image_id'] + ".jpg"# Two meathods add jpg





#add diagnosis：https://www.cnblogs.com/guxh/p/9420610.html

#df_left.insert(2, 'diagnosis', 0)

train_df['diagnosis']=None

for i in range(len(train_df)):

    if 'r' in train_df['image_id'][i]:

        train_df['diagnosis'][i]=train_df['right_eye_DR_Level'][i]

    else:

        train_df['diagnosis'][i]=train_df['left_eye_DR_Level'][i]

        



#df_left.insert(2, 'diagnosis', 0)

valid_df['diagnosis']=None



for i in range(len(valid_df)):

    if 'r' in train_df['image_id'][i]:

        valid_df['diagnosis'][i]=valid_df['right_eye_DR_Level'][i]

    else:

        valid_df['diagnosis'][i]=valid_df['left_eye_DR_Level'][i]



#valid_df['image_id'] = valid_df['image_id'] + ".jpg"

display(train_df.head())

display(valid_df.head())

display(test_df.head())





print('Number of train samples: ', train_df.shape[0])

print('Number of valid samples: ', valid_df.shape[0])
# Model parameters

FACTOR = 4

BATCH_SIZE = 16

EPOCHS = 20

WARMUP_EPOCHS = 5

LEARNING_RATE = 1e-4 * FACTOR

WARMUP_LEARNING_RATE = 1e-3 * FACTOR

HEIGHT = 224

WIDTH = 224

CHANNELS = 3

TTA_STEPS = 5

ES_PATIENCE = 5

RLROP_PATIENCE = 3

DECAY_DROP = 0.5

LR_WARMUP_EPOCHS_1st = 2

LR_WARMUP_EPOCHS_2nd = 5



def classify(x):

    if x < 0.5:

        return 0

    elif x < 1.5:

        return 1

    elif x < 2.5:

        return 2

    elif x < 3.5:

        return 3

    return 4





def ensemble_preds(model_list, generator):

    preds_ensemble = []

    for model in model_list:

        generator.reset()

        preds = model.predict_generator(generator, steps=generator.n)

        preds_ensemble.append(preds)



    return np.mean(preds_ensemble, axis=0)



def apply_tta(model, generator, steps=5):

    step_size = generator.n//generator.batch_size

    preds_tta = []

    for i in range(steps):

        generator.reset()

        preds = model.predict_generator(generator, steps=step_size)

        preds_tta.append(preds)



    return np.mean(preds_tta, axis=0)



def test_ensemble_preds(model_list, generator, steps=5):

    preds_ensemble = []

    for model in model_list:

        preds = apply_tta(model, generator, steps)

        preds_ensemble.append(preds)



    return np.mean(preds_ensemble, axis=0)



import cv2

def preprocess_image(image, sigmaX=10):

    """

    The whole preprocessing pipeline:

    1. Read in image

    2. Apply masks

    3. Resize image to desired size

    4. Add Gaussian noise to increase Robustness

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #image = crop_image_from_gray(image)

    image = cv2.resize(image, (WIDTH, HEIGHT))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)

    return image







datagen=ImageDataGenerator(rescale=1./255, 

                           rotation_range=360,

                           horizontal_flip=True,

                           vertical_flip=True,

                          

                          #preprocessing_function=preprocess_image

                          )





train_generator=datagen.flow_from_dataframe(

                        dataframe=train_df,

                        directory=train_data,

                        x_col="image_id",

                        y_col="diagnosis",

                        class_mode="raw",

                        batch_size=BATCH_SIZE,

                        target_size=(HEIGHT, WIDTH),

                       

                        seed=seed)



valid_generator=datagen.flow_from_dataframe(

                        dataframe=valid_df,

                        directory=valid_data,

                        x_col="image_id",

                        y_col="diagnosis",

                        class_mode="raw",

                        batch_size=BATCH_SIZE,

                        target_size=(HEIGHT, WIDTH),

                       

                        seed=seed)



test_generator=datagen.flow_from_dataframe(  

                       dataframe=test_df,

                       directory=test_data,

                       x_col="image_id",

                       batch_size=1,

                       class_mode=None,

                       shuffle=False,

                       target_size=(HEIGHT, WIDTH),

                       seed=seed)





import efficientnet.keras as efn 



def create_model(input_shape, weights_path):

    input_tensor = Input(shape=input_shape)

    base_model = efn.EfficientNetB5(weights=None, 

                                include_top=False,

                                input_tensor=input_tensor)



    x = GlobalAveragePooling2D()(base_model.output)

    final_output = Dense(1, activation='linear', name='final_output')(x)

    model = Model(input_tensor, final_output)

    model.load_weights(weights_path)

    

    return model





model_list = []



for weights_path in weights_path_list:

    model_list.append(create_model(input_shape=(HEIGHT, WIDTH, CHANNELS), weights_path=weights_path))

    







preds = test_ensemble_preds(model_list, test_generator, TTA_STEPS)

predictions = [classify(x) for x in preds]



predictions[:20]


results = pd.DataFrame({'image_id':test_df['image_id'], 'DR_Level':predictions})

results['image_id'] = results['image_id'].map(lambda x: str(x)[:-4])





results.to_csv('Challenge1_upload.csv', index=False)

display(results.head())
results['DR_Level'].value_counts()