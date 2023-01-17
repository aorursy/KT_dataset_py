import pandas as pd

import os



import time

# For keeping time. GPU limit for this competition is set to ± 9 hours.

t_start = time.time()



data_dir = '../regular-deepdrid/Regular_DeepDRiD'

#The address here is very important: regular_train  regular-test

train_data = '../input/ultradeepdrid/ultra/ultra-widefield-train/'

valid_data = '../input/ultradeepdrid/ultra/ultra-widefield-valid/'

test_data = '../input/ultradeepdrid/ultra/ultra-widefield-test/'



train_df = pd.read_csv('../input/ultradeepdrid/ultra/ultra-widefield-training.csv')

valid_df = pd.read_csv('../input/ultradeepdrid/ultra/ultra-widefield-validation.csv')

test_df = pd.read_csv('../input/ultradeepdrid/ultra/Challenge3_upload.csv')



train_df['image_id'] = train_df['image_id'] + ".jpg"# Two meathods add jpg

valid_df['image_id'] = valid_df['image_id'] + ".jpg"# Two meathods add jpg

test_df["image_id"] = test_df["image_id"].apply(lambda x: x + ".jpg")







#valid_df['image_id'] = valid_df['image_id'] + ".jpg"

display(train_df.head())

display(valid_df.head())

display(test_df.head())



print('Number of train samples: ', train_df.shape[0])

print('Number of valid samples: ', valid_df.shape[0])

print('Number of test samples: ', test_df.shape[0])
for i in range(len(train_df)):

    if train_df['DR_level'][i]==5:

        train_df['DR_level'][i]=4
#数据EDA

import cv2

import seaborn as sns

import matplotlib.pyplot as plt



def add_counts_to_bars(ax):

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x() + p.get_width(), height,

                '%d' % int(height),

                ha='center', va='bottom')

    return ax

plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

ax1 = sns.countplot('DR_level', data=train_df)

ax1 = add_counts_to_bars(ax1)

plt.ylabel("Count")

plt.xlabel("Diagnosis")

plt.title('Training Set')



plt.subplot(1,2,2)

ax2 = sns.countplot('DR_level', data=valid_df)

ax2 = add_counts_to_bars(ax2)

plt.ylabel("Count")

plt.xlabel("Diagnosis")

plt.title(' valid Set')

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

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler





from sklearn.metrics import classification_report

from imgaug import augmenters as iaa







def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed) 

seed = 2020

seed_everything(seed)



# Model parameters

FACTOR = 4



BATCH_SIZE = 8 * FACTOR



EPOCHS = 30

WARMUP_EPOCHS = 5

LEARNING_RATE = 1e-4 * FACTOR

WARMUP_LEARNING_RATE = 1e-3 * FACTOR

HEIGHT = 299

WIDTH = 299

CHANNELS = 3

TTA_STEPS = 5

ES_PATIENCE = 5

RLROP_PATIENCE = 3

DECAY_DROP = 0.5



LR_WARMUP_EPOCHS_1st = 2

LR_WARMUP_EPOCHS_2nd = 5



STEP_SIZE = len(train_df) // BATCH_SIZE

TOTAL_STEPS_1st = WARMUP_EPOCHS * STEP_SIZE

TOTAL_STEPS_2nd = EPOCHS * STEP_SIZE

WARMUP_STEPS_1st = LR_WARMUP_EPOCHS_1st * STEP_SIZE

WARMUP_STEPS_2nd = LR_WARMUP_EPOCHS_2nd * STEP_SIZE





datagen=ImageDataGenerator(rescale=1./255, 

                           rotation_range=20,

                           #horizontal_flip=True,

                           #vertical_flip=True,

                           #preprocessing_function=augment

                          )



train_generator=datagen.flow_from_dataframe(

                        dataframe=train_df,

                        directory=train_data,

                        x_col="image_id",

                        y_col="DR_level",

                        class_mode="raw",

                        batch_size=BATCH_SIZE,

                        target_size=(HEIGHT, WIDTH),

                       

                        seed=seed)



valid_generator=datagen.flow_from_dataframe(

                        dataframe=valid_df,

                        directory=valid_data,

                        x_col="image_id",

                        y_col="DR_level",

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
from keras.preprocessing import image

x,y=train_generator.next()

for i in range(0,2):

    image=x[i]

    label=y[i]

    print(label)

    plt.imshow(image)

    plt.show()
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
def create_model(input_shape):

    input_tensor = Input(shape=input_shape)

    base_model = applications.InceptionV3(weights=None, 

                                include_top=False,

                                input_tensor=input_tensor)

    base_model.load_weights('../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')





    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)



    x = Dense(5, activation=elu)(x)



    final_output = Dense(1, activation='linear', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
model = create_model(input_shape=(HEIGHT, WIDTH, CHANNELS))

for layer in model.layers:

    layer.trainable = False



for i in range(-4, 0):

    model.layers[i].trainable = True



cosine_lr_1st = WarmUpCosineDecayScheduler(learning_rate_base=WARMUP_LEARNING_RATE,

                                           total_steps=TOTAL_STEPS_1st,

                                           warmup_learning_rate=0.0,

                                           warmup_steps=WARMUP_STEPS_1st,

                                           hold_base_rate_steps=(2 * STEP_SIZE))



metric_list = ["accuracy"]

callback_list = [cosine_lr_1st]

optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metric_list)

#model.summary()



STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size



history_warmup = model.fit_generator(generator=train_generator,

                                     steps_per_epoch=STEP_SIZE_TRAIN,

                                     validation_data=valid_generator,

                                     validation_steps=STEP_SIZE_VALID,

                                     epochs=WARMUP_EPOCHS,

                                     callbacks=callback_list,

                                     verbose=2).history
for layer in model.layers:

    layer.trainable = True



#es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)





es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=12)





cosine_lr_2nd = WarmUpCosineDecayScheduler(learning_rate_base=LEARNING_RATE,

                                           total_steps=TOTAL_STEPS_2nd,

                                           warmup_learning_rate=0.0,

                                           warmup_steps=WARMUP_STEPS_2nd,

                                           hold_base_rate_steps=(3 * STEP_SIZE))



callback_list = [es, cosine_lr_2nd]



optimizer = optimizers.Adam(lr=LEARNING_RATE)



model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metric_list)

#model.summary()



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              epochs=EPOCHS,

                              callbacks=callback_list,

                              verbose=2).history
# Create empty arays to keep the predictions and labels

df_preds = pd.DataFrame(columns=['label', 'pred', 'set'])

train_generator.reset()

valid_generator.reset()



# Add train predictions and labels

for i in range(STEP_SIZE_TRAIN + 1):

    im, lbl = next(train_generator)

    preds = model.predict(im, batch_size=train_generator.batch_size)

    for index in range(len(preds)):

        df_preds.loc[len(df_preds)] = [lbl[index], preds[index][0], 'train']



# Add validation predictions and labels

for i in range(STEP_SIZE_VALID + 1):

    im, lbl = next(valid_generator)

    preds = model.predict(im, batch_size=valid_generator.batch_size)

    for index in range(len(preds)):

        df_preds.loc[len(df_preds)] = [lbl[index], preds[index][0], 'validation']



df_preds['label'] = df_preds['label'].astype('int')





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



# Classify predictions

df_preds['predictions'] = df_preds['pred'].apply(lambda x: classify(x))



train_preds = df_preds[df_preds['set'] == 'train']

validation_preds = df_preds[df_preds['set'] == 'validation']





labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']

def plot_confusion_matrix(train, validation, labels=labels):

    train_labels, train_preds = train

    validation_labels, validation_preds = validation

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(24, 7))

    train_cnf_matrix = confusion_matrix(train_labels, train_preds)

    validation_cnf_matrix = confusion_matrix(validation_labels, validation_preds)



    train_cnf_matrix_norm = train_cnf_matrix.astype('float') / train_cnf_matrix.sum(axis=1)[:, np.newaxis]

    validation_cnf_matrix_norm = validation_cnf_matrix.astype('float') / validation_cnf_matrix.sum(axis=1)[:, np.newaxis]



    train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=labels, columns=labels)

    validation_df_cm = pd.DataFrame(validation_cnf_matrix_norm, index=labels, columns=labels)



    sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues",ax=ax1).set_title('Train')

    sns.heatmap(validation_df_cm, annot=True, fmt='.2f', cmap=sns.cubehelix_palette(8),ax=ax2).set_title('Validation')

    plt.show()



plot_confusion_matrix((train_preds['label'], train_preds['predictions']), (validation_preds['label'], validation_preds['predictions']))



def evaluate_model(train, validation):

    train_labels, train_preds = train

    validation_labels, validation_preds = validation

    print("Train        Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds, train_labels, weights='quadratic'))

    print("Validation   Cohen Kappa score: %.3f" % cohen_kappa_score(validation_preds, validation_labels, weights='quadratic'))

    print("Complete set Cohen Kappa score: %.3f" % cohen_kappa_score(np.append(train_preds, validation_preds), np.append(train_labels, validation_labels), weights='quadratic'))

    

evaluate_model((train_preds['label'], train_preds['predictions']), (validation_preds['label'], validation_preds['predictions']))
from sklearn.metrics import classification_report

print(classification_report(validation_preds['label'], validation_preds['predictions'],digits=4))
def apply_tta(model, generator, steps=1):

    step_size = generator.n//generator.batch_size

    preds_tta = []

    for i in range(steps):

        generator.reset()

        preds = model.predict_generator(generator, steps=step_size)

        preds_tta.append(preds)



    return np.mean(preds_tta, axis=0)



preds = apply_tta(model, test_generator, TTA_STEPS)

predictions = [classify(x) for x in preds]



results = pd.DataFrame({'image_id':test_df['image_id'], 'DR_level':predictions})

results['image_id'] = results['image_id'].map(lambda x: str(x)[:-4])

results.to_csv('submission_ultra.csv', index=False)

display(results.head())



model.save_weights('../working/inceptionv3-20200307_ultra.h5')



# Check kernels run-time. GPU limit for this competition is set to ± 9 hours.

t_finish = time.time()

total_time = round((t_finish-t_start) / 3600, 4)

print('Kernel runtime = {} hours ({} minutes)'.format(total_time, 

                                                      int(total_time*60)))