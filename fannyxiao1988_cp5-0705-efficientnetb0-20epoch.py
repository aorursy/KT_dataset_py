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

            
train_data ='../input/cataract/'

#valid_data ='../input/cpfive/validation'

#test_data  ='../input/cpfive/test'



# Model parameters

N_CLASSES=4



FACTOR = 4

BATCH_SIZE = 16

EPOCHS = 20

WARMUP_EPOCHS = 2

LEARNING_RATE = 1e-4 * FACTOR

WARMUP_LEARNING_RATE = 1e-3 * FACTOR

HEIGHT =224

WIDTH = 224

CHANNELS = 3

TTA_STEPS = 5

ES_PATIENCE = 10

RLROP_PATIENCE = 3

DECAY_DROP = 0.5

LR_WARMUP_EPOCHS_1st = 2

LR_WARMUP_EPOCHS_2nd = 5
datagen=ImageDataGenerator(rescale=1./255, 

                           rotation_range=360,

                           horizontal_flip=True,

                           vertical_flip=True,

                           width_shift_range=0.1,

                           height_shift_range=0.1,

                           validation_split=0.15,

                           zoom_range=0.1)





train_generator = datagen.flow_from_directory(

    train_data,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    #color_mode='grayscale',

    class_mode='categorical',

    subset='training',

    shuffle=True,

    seed=seed

    )



valid_generator= datagen.flow_from_directory(

    train_data,

    target_size=(HEIGHT, WIDTH),

    batch_size=BATCH_SIZE,

    #color_mode='grayscale',

    class_mode='categorical',

    subset='validation',

    shuffle=True,

    seed=seed

    )



from keras.preprocessing import image

x,y=train_generator.next()

for i in range(0,4):

    image=x[i]

    label=y[i]

    print(label)

    plt.imshow(image)

    plt.show()
STEP_SIZE = train_generator.n // BATCH_SIZE

TOTAL_STEPS_1st = WARMUP_EPOCHS * STEP_SIZE

TOTAL_STEPS_2nd = EPOCHS * STEP_SIZE

WARMUP_STEPS_1st = LR_WARMUP_EPOCHS_1st * STEP_SIZE

WARMUP_STEPS_2nd = LR_WARMUP_EPOCHS_2nd * STEP_SIZE



print(len(train_generator))

print(train_generator.n)

print(train_generator.batch_size)

print(train_generator.n//train_generator.batch_size)

print(train_generator.n//BATCH_SIZE)



import keras

#import efficientnet.keras as efn 

import efficientnet.keras as efn 





def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = efn.EfficientNetB0(weights='imagenet', 

                                include_top=False,

                                input_tensor=input_tensor)

    #base_model.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5')



    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)

    x = Dropout(0.25)(x)

    final_output = Dense(n_out, activation='softmax', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model



model = create_model(input_shape=(HEIGHT, WIDTH, CHANNELS), n_out=N_CLASSES)
for layer in model.layers:

    layer.trainable = False



for i in range(-5, 0):

    model.layers[i].trainable = True



cosine_lr_1st = WarmUpCosineDecayScheduler(learning_rate_base=WARMUP_LEARNING_RATE,

                                           total_steps=TOTAL_STEPS_1st,

                                           warmup_learning_rate=0.0,

                                           warmup_steps=WARMUP_STEPS_1st,

                                           hold_base_rate_steps=(2 * STEP_SIZE))

    

#class_weights = class_weight.compute_class_weight('balanced', np.unique(X_train['diagnosis'].astype('int').values), X_train['diagnosis'].astype('int').values)

callback_list = [cosine_lr_1st]

metric_list = ["accuracy"]

optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metric_list)

#model.summary()

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size



history_warmup = model.fit_generator(generator=train_generator,

                                     steps_per_epoch=STEP_SIZE_TRAIN,

                                     validation_data=valid_generator,

                                     validation_steps=STEP_SIZE_VALID,

                                     #epochs=1,

                                     epochs=WARMUP_EPOCHS,

                                     callbacks=callback_list,

                                     class_weight='auto',

                                     verbose=1).history





for layer in model.layers:

    layer.trainable = True



# lrstep = LearningRateScheduler(step_decay)

es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)



cosine_lr_2nd = WarmUpCosineDecayScheduler(learning_rate_base=LEARNING_RATE,

                                           total_steps=TOTAL_STEPS_2nd,

                                           warmup_learning_rate=0.0,

                                           warmup_steps=WARMUP_STEPS_2nd,

                                           hold_base_rate_steps=(3 * STEP_SIZE))



callback_list = [es, cosine_lr_2nd]



optimizer = optimizers.Adam(lr=LEARNING_RATE)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metric_list)

#model.summary()



history_finetunning = model.fit_generator(generator=train_generator,

                                          steps_per_epoch=STEP_SIZE_TRAIN,

                                          validation_data=valid_generator,

                                          validation_steps=STEP_SIZE_VALID,

                                          #epochs=1,

                                          epochs=int(EPOCHS*0.8),

                                          callbacks=callback_list,

                                          class_weight='auto',

                                          verbose=1).history





optimizer = optimizers.SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metric_list)



history_finetunning_2 = model.fit_generator(generator=train_generator,

                                          steps_per_epoch=STEP_SIZE_TRAIN,

                                          validation_data=valid_generator,

                                          validation_steps=STEP_SIZE_VALID,

                                          #epochs=1,

                                          epochs=int(EPOCHS*0.2),

                                          callbacks=callback_list,

                                          class_weight='auto',

                                          verbose=1).history



model.save_weights('../working/effnetb3.h5')
label_map = (train_generator.class_indices)

label_map
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

#STEP_SIZE_TEST = test_generator.n//test_generator.batch_size



# Create empty arays to keep the predictions and labels

#lastFullTrainPred = np.empty((0, N_CLASSES))

#lastFullTrainLabels = np.empty((0, N_CLASSES))

lastFullValPred = np.empty((0, N_CLASSES))

lastFullValLabels = np.empty((0, N_CLASSES))





#lastFulltestPred = np.empty((0, N_CLASSES))

#lastFulltestLabels = np.empty((0, N_CLASSES))

'''

# Add train predictions and labels

for i in range(STEP_SIZE_TRAIN+1):

    im, lbl = next(train_generator)

    scores = model.predict(im, batch_size=train_generator.batch_size)

    lastFullTrainPred = np.append(lastFullTrainPred, scores, axis=0)

    lastFullTrainLabels = np.append(lastFullTrainLabels, lbl, axis=0)

'''





# Add validation predictions and labels

for i in range(STEP_SIZE_VALID+1):

    im, lbl = next(valid_generator)

    scores = model.predict(im, batch_size=valid_generator.batch_size)

    lastFullValPred = np.append(lastFullValPred, scores, axis=0)

    lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)

    

# Add validation predictions and labels

#for i in range(STEP_SIZE_TEST+1):

#    im, lbl = next(test_generator)

#    scores = model.predict(im, batch_size=test_generator.batch_size)

#    lastFulltestPred = np.append(lastFulltestPred, scores, axis=0)

#    lastFulltestLabels = np.append(lastFulltestLabels, lbl, axis=0)

    

validation_preds = [np.argmax(pred) for pred in lastFullValPred]

validation_labels = [np.argmax(label) for label in lastFullValLabels]



#test_preds = [np.argmax(pred) for pred in lastFulltestPred]

#test_labels = [np.argmax(label) for label in lastFulltestLabels]



#print(test_preds)

#print(test_labels)
from sklearn.metrics import classification_report

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import confusion_matrix

#cm=confusion_matrix(test_labels, test_preds)

#print(confusion_matrix(test_labels, test_preds))



print(classification_report(validation_labels, validation_preds,digits=4))

#print(classification_report(test_labels, test_preds,digits=4))



print("validation Cohen Kappa score: %.4f" % cohen_kappa_score(validation_labels, validation_preds, weights='quadratic'))

#print("Test Cohen Kappa score: %.4f" % cohen_kappa_score(test_labels, test_preds, weights='quadratic'))
# -*- coding: utf-8 -*-

"""

plot a pretty confusion matrix with seaborn

Created on Mon Jun 25 14:17:37 2018

@author: Wagner Cipriano - wagnerbhbr - gmail - CEFETMG / MMC

REFerences:

  https://www.mathworks.com/help/nnet/ref/plotconfusion.html

  https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report

  https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python

  https://www.programcreek.com/python/example/96197/seaborn.heatmap

  https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/31720054

  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

"""



#imports

from pandas import DataFrame

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

from matplotlib.collections import QuadMesh

import seaborn as sn





def get_new_fig(fn, figsize=[9,9]):

    """ Init graphics """

    fig1 = plt.figure(fn, figsize)

    ax1 = fig1.gca()   #Get Current Axis

    ax1.cla() # clear existing plot

    return fig1, ax1

#



def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):

    """

      config cell text and colors

      and return text elements to add and to dell

      @TODO: use fmt

    """

    text_add = []; text_del = [];

    cell_val = array_df[lin][col]

    tot_all = array_df[-1][-1]

    per = (float(cell_val) / tot_all) * 100

    curr_column = array_df[:,col]

    ccl = len(curr_column)



    #last line  and/or last column

    if(col == (ccl - 1)) or (lin == (ccl - 1)):

        #tots and percents

        if(cell_val != 0):

            if(col == ccl - 1) and (lin == ccl - 1):

                tot_rig = 0

                for i in range(array_df.shape[0] - 1):

                    tot_rig += array_df[i][i]

                per_ok = (float(tot_rig) / cell_val) * 100

            elif(col == ccl - 1):

                tot_rig = array_df[lin][lin]

                per_ok = (float(tot_rig) / cell_val) * 100

            elif(lin == ccl - 1):

                tot_rig = array_df[col][col]

                per_ok = (float(tot_rig) / cell_val) * 100

            per_err = 100 - per_ok

        else:

            per_ok = per_err = 0



        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]



        #text to DEL

        text_del.append(oText)



        #text to ADD

        font_prop = fm.FontProperties(weight='bold', size=16)

        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)

        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]

        lis_kwa = [text_kwargs]

        dic = text_kwargs.copy(); dic['color'] = 'y'; lis_kwa.append(dic);

        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);

        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]

        for i in range(len(lis_txt)):

            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])

            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)

            text_add.append(newText)

        #print '\n'



        #set background color for sum cells (last line and last column)

        carr = [0.27, 0.30, 0.27, 1.0]

        if(col == ccl - 1) and (lin == ccl - 1):

            carr = [0.17, 0.20, 0.17, 1.0]

        facecolors[posi] = carr



    else:

        if(per > 0):

            txt = '%s\n%.2f%%' %(cell_val, per)

        else:

            if(show_null_values == 0):

                txt = ''

            elif(show_null_values == 1):

                txt = '0'

            else:

                txt = '0\n0.0%'

        oText.set_text(txt)



        #main diagonal

        if(col == lin):

            #set color of the textin the diagonal to white

            oText.set_color('w')

            # set background color in the diagonal to blue

            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]

        else:

            oText.set_color('r')



    return text_add, text_del

#



def insert_totals(df_cm):

    """ insert total column and line (the last ones) """

    sum_col = []

    for c in df_cm.columns:

        sum_col.append( df_cm[c].sum() )

    sum_lin = []

    for item_line in df_cm.iterrows():

        sum_lin.append( item_line[1].sum() )

    df_cm[''] = sum_lin

    sum_col.append(np.sum(sum_lin))

    df_cm.loc[''] = sum_col

    #print ('\ndf_cm:\n', df_cm, '\n\b\n')

#



def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=16,

      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):

    """

      print conf matrix with default layout (like matlab)

      params:

        df_cm          dataframe (pandas) without totals

        annot          print text in each cell

        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:

        fz             fontsize

        lw             linewidth

        pred_val_axis  where to show the prediction values (x or y axis)

                        'col' or 'x': show predicted values in columns (x axis) instead lines

                        'lin' or 'y': show predicted values in lines   (y axis)

    """

    if(pred_val_axis in ('col', 'x')):

        xlbl = 'Predicted label'   #修改标签

        ylbl = 'True label'

    else:

        xlbl = 'True label'

        ylbl = 'Predicted label'

        df_cm = df_cm.T



    # create "Total" column

    insert_totals(df_cm)



    #this is for print allways in the same window

    fig, ax1 = get_new_fig('Conf matrix default', figsize)



    #thanks for seaborn

    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,

                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)



    #set ticklabels rotation

    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 16,horizontalalignment="center") # 横轴是否旋转

    ax.set_yticklabels(ax.get_yticklabels(), rotation = 90, fontsize = 16, Verticalalignment="center") #纵轴

#https://stackoverflow.com/questions/43520322/matplotlib-move-x-axis-tick-labels-one-position-to-left

#根据标签的长短调整坐标轴 ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 16,ha="right") ha="right"\

#主要是这两个函数调整

#Verticalalignment="center"

#horizontalalignment=left,right 

    # Turn off all the ticks

    for t in ax.xaxis.get_major_ticks():

        t.tick1On = False

        t.tick2On = False

    for t in ax.yaxis.get_major_ticks():

        t.tick1On = False

        t.tick2On = False



    #face colors list

    quadmesh = ax.findobj(QuadMesh)[0]

    facecolors = quadmesh.get_facecolors()



    #iter in text elements

    array_df = np.array( df_cm.to_records(index=False).tolist() )

    text_add = []; text_del = [];

    posi = -1 #from left to right, bottom to top.

    for t in ax.collections[0].axes.texts: #ax.texts:

        pos = np.array( t.get_position()) - [0.5,0.5]

        lin = int(pos[1]); col = int(pos[0]);

        posi += 1

        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))



        #set text

        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)



        text_add.extend(txt_res[0])

        text_del.extend(txt_res[1])



    #remove the old ones

    for item in text_del:

        item.remove()

    #append the new ones

    for item in text_add:

        ax.text(item['x'], item['y'], item['text'], **item['kw'])

        

      #这就是所谓的第一种情况哦

    font = {'family' : 'Times New Roman',

       # 'color'  : 'darkred',

        'weight' : 'normal',

        'size'   : 24,

        }





    #titles and legends

    ax.set_title('Confusion matrix (FPN)',font)

    ax.set_xlabel(xlbl,font)

    ax.set_ylabel(ylbl,font)

    plt.tight_layout()  #set layout slim

    plt.show()

#



def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",

      fmt='.2f', fz=16, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):

    """

        plot confusion matrix function with y_test (actual values) and predictions (predic),

        whitout a confusion matrix yet

    """

    from sklearn.metrics import confusion_matrix

    from pandas import DataFrame



    #data

    if(not columns):

        #labels axis integer:

        ##columns = range(1, len(np.unique(y_test))+1)

        #labels axis string:

        from string import ascii_uppercase

        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]



    confm = confusion_matrix(y_test, predictions)

    labels = ['Normal', 'MP', 'BP', 'VP', 'COVID-19']

    



   



    cmap = 'Oranges';

    fz = 20;# 修改表格中数字的大小

    figsize=[8,8];#修改每个格子的大小

    show_null_values = 2

    df_cm = DataFrame(confm, index=labels, columns=labels)

    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)

#







#

#TEST functions

#

def _test_cm():

    #test function with confusion matrix done

    array = cm

    #get pandas dataframe

    df_cm = DataFrame(array, index=range(0,5), columns=range(0,5))#这里修改参数几分类就是n-1

    #colormap: see this and choose your more dear

    cmap = 'PuRd'





    pretty_plot_confusion_matrix(df_cm, cmap=cmap)

#



def _test_data_class():

    """ test function with y_test (actual values) and predictions (predic) """

    #data

    y_test = test_labels        # 测试标签

    predic =test_preds                # 预测值

    """

      Examples to validate output (confusion matrix plot)

        actual: 5 and prediction 1   >>  3

        actual: 2 and prediction 4   >>  1

        actual: 3 and prediction 4   >>  10

    """

    columns = []

    annot = True;

    cmap = 'Oranges';

    fmt = '.2f'

    lw = 0.5

    cbar = False

    show_null_values = 2

    pred_val_axis = 'y'

    #size::

    fz = 20;

    figsize = [6,6];

    if(len(y_test) > 10):

        fz=16; figsize=[14,14];

    plot_confusion_matrix_from_data(y_test, predic, columns,

      annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)

#





#

#MAIN function

#

if(__name__ == '__main__'):

    print('__main__')

    print('_test_cm: test function with confusion matrix done\nand pause')

    _test_cm()

    plt.pause(5)

    print('_test_data_class: test function with y_test (actual values) and predictions (predic)')

    _test_data_class()