# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import matplotlib.pyplot as plt

import datetime

from pathlib import Path



from sklearn.metrics import roc_auc_score



import numpy as np

import os

import cv2

import warnings

warnings.filterwarnings("ignore")



# IMPORT KERAS LIBRARY

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, Input, GlobalAveragePooling2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

from keras.callbacks import ModelCheckpoint

from keras import backend as K

import tensorflow as tf



%load_ext autoreload

%autoreload
model_path='.'

path='../input/chexpert/chexp'

train_folder=f'{path}train'

test_folder=f'{path}test'

train_lbl=f'{path}train_labels.csv'
chestxrays_root = Path(path)

data_path = chestxrays_root
!ls '../input'
full_train_df = pd.read_csv(data_path/'CheXpert-v1.0-small/train.csv')

full_valid_df = pd.read_csv(data_path/'CheXpert-v1.0-small/valid.csv')
chexnet_targets = ['No Finding',

       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',

       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',

       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',

       'Support Devices']



chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
full_train_df.head()
u_one_features = ['Atelectasis', 'Edema']

u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
def feature_string(row):

    feature_list = []

    for feature in u_one_features:

        if row[feature] in [-1,1]:

            feature_list.append(feature)

            

    for feature in u_zero_features:

        if row[feature] == 1:

            feature_list.append(feature)

            

    return ';'.join(feature_list)

            

     
full_train_df['train_valid'] = False

full_valid_df['train_valid'] = True
full_train_df['patient'] = full_train_df.Path.str.split('/',3,True)[2]

full_train_df  ['study'] = full_train_df.Path.str.split('/',4,True)[3]



full_valid_df['patient'] = full_valid_df.Path.str.split('/',3,True)[2]

full_valid_df  ['study'] = full_valid_df.Path.str.split('/',4,True)[3]
full_df = pd.concat([full_train_df, full_valid_df])

full_df.head()
full_df['feature_string'] = full_df.apply(feature_string,axis = 1).fillna('')

full_df['feature_string'] =full_df['feature_string'] .apply(lambda x:x.split(";"))

full_df.head()
#get the first 5 whale images

paths =  full_df.Path[:5]

labels = full_df.feature_string[:5]



fig, m_axs = plt.subplots(1, len(labels), figsize = (20, 10))

#show the images and label them

for ii, c_ax in enumerate(m_axs):

    c_ax.imshow(cv2.imread(os.path.join(data_path,paths[ii])))

    c_ax.set_title(labels[ii])
from collections import Counter



labels_count = Counter(label for chexpert_targets in full_df['feature_string'] for label in chexpert_targets)

#plt.bar(chexpert_targets, labels_count.values(), align='center', alpha=0.5)

#plt.show

x_pos = np.arange(len(labels_count.values()))

#Plot the data:

my_colors = 'rgbkymc'

lbls = list.copy(chexpert_targets)

lbls.insert(0,'')

plt.bar(x_pos, labels_count.values(), align='center', alpha=0.5 , color=my_colors)

plt.xticks(x_pos, lbls, rotation='vertical')

sample_perc = 0.00

train_only_df = full_df[~full_df.train_valid]

valid_only_df = full_df[full_df.train_valid]

unique_patients = train_only_df.patient.unique()

mask = np.random.rand(len(unique_patients)) <= sample_perc

sample_patients = unique_patients[mask]



dev_df = train_only_df[full_train_df.patient.isin(sample_patients)]

train_df = train_only_df[~full_train_df.patient.isin(sample_patients)]



print(valid_only_df.Path.size)

print(train_df.Path.size)
datagen=image.ImageDataGenerator(rescale=1./255, 

                                 featurewise_center=True,

                                 featurewise_std_normalization=True,

                                 rotation_range=5,

                                 width_shift_range=0.2,

                                 height_shift_range=0.2,

                                 horizontal_flip=True,

                                 validation_split = 0.1)

test_datagen=image.ImageDataGenerator(rescale=1./255)
def generate_datasets(image_size = 224):



    train_generator=datagen.flow_from_dataframe(dataframe=train_df, directory=data_path, 

                                                x_col="Path", y_col="feature_string", has_ext=True, seed = 42, #classes = chexpert_targets,

                                                class_mode="categorical", target_size=(image_size,image_size), batch_size=32, subset = "training")



    validation_generator = datagen.flow_from_dataframe(dataframe=train_df, directory=data_path, 

                                                       x_col="Path", y_col="feature_string", has_ext=True, seed = 42, #classes = chexpert_targets,

                                                       class_mode="categorical", target_size=(image_size,image_size), batch_size=32, subset = "validation")



    test_generator = test_datagen.flow_from_dataframe(dataframe=valid_only_df, directory=data_path, 

                                                      target_size=(image_size,image_size),class_mode='categorical',

                                                      batch_size=1, shuffle=False, #classes = chexpert_targets,

                                                      x_col="Path", y_col="feature_string")

    

    return [train_generator,validation_generator,test_generator]
from keras.callbacks import *



class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.

    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```

    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """



    def __init__(self, base_lr=0.001, max_lr=0.01, step_size=2000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()



        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}



        self._reset()



    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

        

    def on_train_begin(self, logs={}):

        logs = logs or {}



        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

            

    def on_batch_end(self, epoch, logs=None):

        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
from keras.layers import Conv2D

from keras.optimizers import Adam

def build_model(image_size = 224):

    base_model = InceptionResNetV2(include_top= False, input_shape=(image_size,image_size,3), weights='imagenet')

    

    # IRNV2 is already trained. We want to preserve such weights and train our custom layers.

    for layer in base_model.layers:

        layer.trainable = True



    # this is the model we will train

    

    ########################### CUSTOM 

    

    x = base_model.output

    x = Conv2D(500, kernel_size = (3,3), activation='relu', input_shape = (image_size,image_size,3))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    x = GlobalAveragePooling2D(input_shape=(1024,1,1))(x)

    

    # Add a fully connected layer .

    # x = Flatten()(x)

    

    x = Dense(700, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    

    predictions = Dense(6, activation='sigmoid')(x)

    ############### END CUSTOM

    

    model = Model(inputs=base_model.input, outputs=predictions)

    

    model.load_weights('../input/chexpert-inceptionresv2-clean/weights.hdf5')



    adamOptimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0, amsgrad=False)

    # compile the model (should be done *after* setting layers to non-trainable)

    model.compile(optimizer=adamOptimizer, loss='categorical_crossentropy', metrics=['accuracy', auc])

    return model
def train_model(clr_triangular, model , datasets, epochs=1, image_size = 224):

    

    checkpointer = ModelCheckpoint(filepath='weights.hdf5', 

                                   verbose=1, save_best_only=True)

    

    train_generator,validation_generator,test_generator = datasets

    

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

    STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

    print(STEP_SIZE_TRAIN)

    print(STEP_SIZE_VALID)



    return model.fit_generator(generator=train_generator,

                        steps_per_epoch=STEP_SIZE_TRAIN,

                        validation_data=validation_generator,

                        validation_steps=STEP_SIZE_VALID,

                        epochs=epochs, callbacks = [clr_triangular, checkpointer])
image_size_input = 224

model = build_model(image_size = image_size_input)
#from keras.utils import plot_model

#plot_model(model, to_file='model.png')
datasets = generate_datasets(image_size = image_size_input)

train_generator,validation_generator,test_generator = datasets
clr_triangular = CyclicLR(mode='exp_range')

history = train_model(clr_triangular, model , datasets, epochs=3, image_size = image_size_input)
clr_triangular.history
print("LR Range : ", min(clr_triangular.history['lr']), max(clr_triangular.history['lr']))

print("Momentum Range : ", min(clr_triangular.history['momentum']), max(clr_triangular.history['momentum']))
plt.xlabel('Training Iterations')

plt.ylabel('Learning Rate')

plt.title("CLR")

plt.plot(clr_triangular.history['lr'])

plt.show()
plt.xlabel('Training Iterations')

plt.ylabel('Momentum')

plt.title("CLR")

plt.plot(clr_triangular.history['momentum'])

plt.show()
history.history
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.preprocessing import MultiLabelBinarizer

test = pd.Series(test_generator.labels)

mlb = MultiLabelBinarizer()

y_labels = mlb.fit_transform(test)
test_generator.reset()

y_pred_keras = model.predict_generator(test_generator,verbose = 1,steps=test_generator.n)
from sklearn.metrics import roc_curve

from sklearn.metrics import auc



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')



for ii in range(1, y_pred_keras.shape[1]):

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_labels[:,ii], y_pred_keras[:,ii])

    auc_keras = auc(fpr_keras, tpr_keras)

    plt.plot(fpr_keras, tpr_keras, label=chexpert_targets[ii-1] + '(area = {:.3f})'.format(auc_keras))

    

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()

    


