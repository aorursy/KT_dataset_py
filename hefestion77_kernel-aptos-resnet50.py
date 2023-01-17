#from IPython.core.display import display, HTML
#toggle_code_str = '''
#<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Toggle Code"></form>
#'''
#
#toggle_code_prepare_str = '''
#    <script>
#    function code_toggle() {
#        if ($('div.cell.code_cell.rendered.selected div.input').css('display')!='none'){
#            $('div.cell.code_cell.rendered.selected div.input').hide();
#        } else {
#            $('div.cell.code_cell.rendered.selected div.input').show();
#        }
#    }
#    </script>
#
#'''
#
#display(HTML(toggle_code_prepare_str + toggle_code_str))
#
#def toggle_code():
#    display(HTML(toggle_code_str))
from warnings import filterwarnings
filterwarnings("ignore")

import os
import sys
import glob
from time import time
import numpy as np
import pandas as pd
import imageio as io
import cv2 as cv
import matplotlib.pyplot as plt
from IPython.display import SVG, HTML
from keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam, SGD, RMSprop, Nadam, Optimizer
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import plot_model, to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras.utils.vis_utils import model_to_dot
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

#!pip install -U graphviz
#!pip install -U --pre efficientnet
#!pip install -U git+https://github.com/qubvel/efficientnet
#from efficientnet.keras import EfficientNetB7, preprocess_input
sys.path.append(os.path.abspath('/kaggle/input/efficientnetb7-keras-model-weights'))
sys.path.append(os.path.abspath('/kaggle/input/ranger-optimizer-rectified-adam-lookahead'))
#from efficientnet import EfficientNetB7, preprocess_input
from RAdam import RAdam
from Lookahead import Lookahead

#sys.path.append(os.path.abspath('../input/ranger-optimizer-rectified-adam-lookahead'))
#import ranger_optimizer

plt.style.use('seaborn-paper')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

print('Tensorflow version:', tf.__version__)

print('\nSetup complete!')
#%load '../input/efficientnetb7-keras-model-weights/efficientnet.py'
def convert_seconds_to_time(seconds):
    """
    (float -> str)
    
    Converts seconds (float) in days, hours, minutes and seconds and returns a string with the result.  
    """
    if seconds < float(86400) and seconds >= float(3600):
        h, sec = divmod(int(round(seconds)), 3600)
        m, sec = divmod(int(sec), 60)
        return f'{int(h)} hours, {int(m)} minutes and {round(sec)} seconds'
    
    elif seconds < float(86400) and seconds < float(3600):
        if seconds >= float(60):
            m, sec = divmod(int(round(seconds)), 60)
            return f'{int(m)} minutes and {round(sec)} seconds'
        else:
            return f'{round(seconds)} seconds'
    else:
        d, sec = divmod(int(round(seconds)), 86400)
        return f'{int(d)} days, {convert_seconds(float(sec))}'
        
def diab_retin(prediction):
    """
    (int -> str)
    
    Returns a string with information of the type of diabetic retinopathy, if present, 
    according to an integer which is the prediction given by the model.
    """
    if prediction == 0:
        return 'No diabetic retinopathy'
    elif prediction == 1:
        return 'Mild nonproliferative diabetic retinopathy'
    elif prediction == 2:
        return 'Moderate nonproliferative diabetic retinopathy'
    elif prediction == 3:
        return 'Severe nonproliferative diabetic retinopathy'
    elif prediction == 4:
        return 'Proliferative diabetic retinopathy'
    else:
        raise ValueError('The argument should be an integer from 0 to 4, both included.')
# PREPARING CUSTOM METRICS (Cohen's kappa coefficient):        
#def cohens_kappa(y_true, y_pred):
#    y_true_classes = K.argmax(y_true, axis = 1)
#    y_pred_classes = K.argmax(y_pred, axis = 1)
#    return tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, 5)[1] # Returns update_op: Operation that increments po, pe_row and pe_col variables appropriately and whose value matches kappa.
# PREPARING CUSTOM ACTIVATION FUNCTION (MISH(x) = x * tanh(ln(1+e^x))) 
#-------------------------------------------------------------------
class Mish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'mish'
        
def mish_activation(z):
    """
    Returns new Mish activation of z (Mish = z * tanh(ln(1+e^z))) as a tensor
    """
    return z * K.tanh(K.softplus(z))

get_custom_objects().update({'mish': Mish(mish_activation)})
url_resnet50 = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'   
base_model = ResNet50(include_top = False, input_shape = (256,256,3), pooling = 'avg', weights = url_resnet50)

#base_model = EfficientNetB7(weights = None, include_top = False, input_shape = (256, 256, 3))
#base_model.load_weights('../input/efficientnetb7-keras-model-weights/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')


hidden_layers = 1
dropout = 0.05
lambda2 = 0.
units = 250
#for idx, layer in enumerate(base_model.layers):
#        print(idx, layer.name)
def my_model(hidden_layers = hidden_layers,
             dropout = dropout,
             lambda2 = lambda2,
             base_model = base_model,
             units = units, 
             lr = 0.03, 
             pool = 'avg', 
             classes = 5):
     
    model = Sequential(name = 'APTOS_model')
    model.add(base_model)
        
    #model.add(GlobalAveragePooling2D())

    for num in range(hidden_layers):
        #dense_layer_name = 'FC_' + str(num + 1)
        model.add(Dense(units, kernel_regularizer = l2(lambda2)))
        model.add(BatchNormalization())
        model.add(Activation('mish'))
        if dropout > 0:
            model.add(Dropout(dropout))
   
    model.add(Dense(classes, activation = 'softmax', name = 'Predictions', kernel_regularizer = l2(lambda2)))
    
    for layer in base_model.layers:
        layer.trainable = False
    
    optim = RAdam(beta_1 = 0.95, beta_2 = 0.999, learning_rate = lr)
    Ranger = Lookahead(optimizer = optim, k = 5, alpha = 0.5) # # Implement RAdam with LookAhead
    model.compile(optimizer = Ranger,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model
# In case we do not use model.fit_generator but model.fit with validation_data: (x_val, y_val) or model.evaluate. Create an instance of the class as a callback.

#class Metrics(Callback):
#    
#    def __init__(self, classification, classes):
#        self.classification = classification
#        if self.classification.lower() == 'binary':
#            classes = 2
#        self.classes = classes
#        
#    def on_train_begin(self, logs={}):
#        #self.confusion = []
#        #self.precision = []
#        #self.recall = []
#        #self.f1s = []
#        #self.auc = []
#        self.kappa = []
#
#    def on_epoch_end(self, epoch, logs={}):
#        if self.classification.lower() == 'binary':
#            score = np.asarray(self.model.predict(self.validation_data[0]))
#            predict = np.round(score).astype(int)
#            targ = self.validation_data[1]
#        elif self.classification.lower() == 'multilabel' or self.classification.lower() == 'categorical':
#            score = np.asarray(self.model.predict(self.validation_data[0]))
#            predict = np.argmax(score, axis = 1)
#            predict_categ = to_categorical(predict, self.classes)
#            # we assume here that we have one-hot encoded labels:
#            targ_categ = self.validation_data[1]
#            targ = np.argmax(targ_categ, axis = 1)
#            # in case we have ground truth labels, not one-hot encoded labels:
#            #targ = self.validation_data[1]
#            #targ_categ = to_categorical(targ, self.classes)
#            
#        #val_auc = roc_auc_score(targ, score)
#        #val_confusion = confusion_matrix(targ, predict)
#        #val_precision = precision_score(targ, predict)
#        #val_recall = recall_score(targ, predict)
#        #val_f1 = f1_score(targ, predict, average = 'macro')
#        val_kappa = cohen_kappa_score(targ, predict)
#        
#        #self.auc.append(val_auc)
#        #self.confusion.append(val_confusion)
#        #self.precision.append(val_precision)
#        #self.recall.append(val_recall)
#        #self.f1s.append(val_f1)
#        self.kappa.append(val_kappa)
#        
#        print(f'Validation kappa score: {val_kappa:.4f}\n')
model = my_model()
print('MODEL SUMMARY BEFORE FINE-TUNING', '\n')
print(model.summary(), '\n')
plot_model(model, to_file = 'model_APTOS.png', show_shapes = False)
#SVG(model_to_dot(model).create(prog = 'dot', format = 'svg'))
#%%time
timea = time()
    
with np.load('../input/kernel-aptos-img-to-arrays-one-hot-and-split/train_set.npz') as traindata:
    print('Data in train_set.npz:', traindata.files, '.....', end = '')
    X_train = traindata['X_train']
    Y_train = traindata['Y_train']
    print('Retrieved!')

with np.load('../input/kernel-aptos-img-to-arrays-one-hot-and-split/val_set.npz') as valdata:
    print('Data in val_set.npz:', valdata.files, '.....', end = '')
    X_val = valdata['X_val']
    Y_val = valdata['Y_val']
    print('Retrieved!')

timeb = time()
total_time = timeb - timea

print(f'All preprocessed data retrieved in {convert_seconds_to_time(total_time)}\n')
#
print('Shape of the array containing preprocessed training images', X_train.shape)
print('Shape of the array containing one-hot encoded labels of the training subset', Y_train.shape)
print('Shape of the array containing preprocessed cross-validation images', X_val.shape)
print('Shape of the array containing one-hot encoded labels of the cross-validation subset', Y_val.shape,'\n')
#GRID SEARCH CV
#--------------

#model = KerasClassifier(build_fn = my_model)
#params = {'dropout': [0, 0.15, 0.3],
#          'hidden_layers': [1, 2, 3]}
#
#grid = GridSearchCV(estimator = model,
#                    param_grid = params,
#                    cv = 2)
#
#timea = time()
#grid = grid.fit(X_train, Y_train)
#best_params = grid.best_params_
#best_score = grid.best_score_
#timeb = time()
#
#total_time = timeb - timea
#
#print(f'\nGrid Search completed in {convert_seconds_to_time(total_time)}')
#print(best_params)
#print(best_score)
#print(f'\nThe best score is {best_score:.4f} with the following parameters: {best_params}')
#DATA AUGMENTATION - TRANSFER LEARNING
#---------------------------------------
#timea = time()
#
#df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
#df_train['id_code'] = df_train['id_code'].apply(lambda x: x + '.png')
#df_train['diagnosis'] = df_train['diagnosis'].astype('str')
#
#url_train = '../input/aptos2019-blindness-detection/train_images/'
#train_datagen = ImageDataGenerator(rescale = 1./255,
#                                   preprocessing_function = preprocess_input,
#                                   horizontal_flip = True,
#                                   vertical_flip = True,
#                                   rotation_range = 8,
#                                   shear_range = 0.3,
#                                   zoom_range = 0.3,
#                                   validation_split = 0.2,
#                                   channel_shift_range = 0.3)
#
#train_generator  = train_datagen.flow_from_dataframe(dataframe = df_train, 
#                                                     directory = url_train, 
#                                                     x_col = 'id_code',
#                                                     y_col = 'diagnosis',
#                                                     target_size = (600, 600),
#                                                     subset = 'training',
#                                                     class_mode = 'categorical',
#                                                     batch_size = 16, 
#                                                     shuffle = True)
#
#val_generator = train_datagen.flow_from_dataframe(dataframe = df_train, 
#                                                  directory = url_train, 
#                                                  x_col = 'id_code',
#                                                  y_col = 'diagnosis',
#                                                  target_size = (600, 600),
#                                                  subset = 'validation',
#                                                  class_mode = 'categorical',
#                                                  batch_size = 16,
#                                                  shuffle = True)
#
#num_minibatches_train = train_generator.samples // train_generator.batch_size # number of training images // batch size (16 in this case)
#num_minibatches_cv = val_generator.samples // val_generator.batch_size # number of cross-validation images // batch size (16 in this case)
#
#K.get_session().run(tf.local_variables_initializer()) # We need to initialize tf variables before training to use cohens_kappa function we defined before.
#
#epochs_pre = 2
##my_metrics = Metrics(classification = 'categorical', classes = 5)
#modelAPTOS_pretuned_hist = model.fit_generator(train_generator, 
#                                               steps_per_epoch = num_minibatches_train, 
#                                               epochs = epochs_pre,
#                                               verbose = 1, 
#                                               validation_data = val_generator,
#                                               validation_steps = num_minibatches_cv)
#
#timeb = time()
#total_time = timeb - timea
#print(f'\nData augmentation and training with cross-validation after {epochs_pre} epochs completed in {convert_seconds_to_time(total_time)}', '\n')
#DATA AUGMENTATION - TRANSFER LEARNING
#---------------------------------------

gen_train = ImageDataGenerator(horizontal_flip = True,
                               vertical_flip = True,
                               rotation_range = 10,
                               shear_range = 0.3,
                               zoom_range = 0.3,
                               preprocessing_function = preprocess_input,
                               channel_shift_range = 0.3)
#
gen_cv = ImageDataGenerator(preprocessing_function = preprocess_input)
#
train_generator = gen_train.flow(X_train, Y_train, batch_size = 16, shuffle = True)
val_generator = gen_cv.flow(X_val, Y_val, batch_size = 16, shuffle = True)
#%%time
timea = time()

#num_minibatches_train = train_generator.n // train_generator.batch_size # number of training images // batch size (16 in this case)
#num_minibatches_cv = val_generator.n // val_generator.batch_size # number of cross-validation images // batch size (16 in this case)

#K.get_session().run(tf.local_variables_initializer()) # We need to initialize tf variables before training to use cohens_kappa function we defined before.

epochs_pre = 2
#my_metrics = Metrics(classification = 'categorical', classes = 5)
modelAPTOS_pretuned_hist = model.fit_generator(train_generator, 
                                               steps_per_epoch = train_generator.n, 
                                               epochs = epochs_pre,
                                               verbose = 1, 
                                               validation_data = val_generator,
                                               validation_steps = val_generator.n)
timeb = time()
total_time = timeb - timea
#
#pre_tune_history = modelAPTOS_pretuned_hist.history
#model.save('modelAPTOS_PRETUNED.h5')
print(f'\nData augmentation and pre-training with cross-validation after {epochs_pre} epochs completed in {convert_seconds_to_time(total_time)}', '\n')
#print('Accuracy of the pre-tuned model on the training subset:', pre_tune_history['accuracy'][-1])
#print('Accuracy of the pre-tuned model on the cross-validation subset:', pre_tune_history['val_accuracy'][-1], '\n')
#FINE TUNING
#---------

timea = time()

#-------------------------
#FINE TUNING FOR RESNET50:

for layer in base_model.layers[:143]:
    layer.trainable = False
for layer in base_model.layers[143:]:
    layer.trainable = True

#--------------------------
#FINE TUNING FOR EFFICIENTNETB7

#841

#for layer in base_model.layers[:841]:
#    layer.trainable = False
#for layer in base_model.layers[841:]:
#    layer.trainable = True  
#-----------------------------

optim = RAdam(beta_1 = 0.95, beta_2 = 0.999, learning_rate = 3e-5) # Low learning rate for RAdam for fine tuning...
Ranger = Lookahead(optimizer = optim, k = 5, alpha = 0.5) # # Implement RAdam with LookAhead again.

model.compile(optimizer = Ranger,
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

#K.get_session().run(tf.local_variables_initializer()) # We need to initialize tf variables before training to use cohens_kappa function we defined before.

print('MODEL SUMMARY AFTER FINE-TUNING', '\n')   
print(model.summary(), '\n')

epochs_post = 42
#my_metrics = Metrics(classification = 'categorical', classes = 5)
early_stop = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 10, verbose = 1, mode = 'max', baseline = None, restore_best_weights = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.1, patience = 8, mode = 'max', min_lr = 1e-8)

modelAPTOS_tuned_hist = model.fit_generator(train_generator, 
                                            steps_per_epoch = train_generator.n, 
                                            epochs = epochs_post,
                                            verbose = 2, 
                                            validation_data = val_generator,
                                            validation_steps = val_generator.n,
                                            callbacks = [early_stop, reduce_lr])

timeb = time()
total_time = timeb - timea

post_tune_history = modelAPTOS_tuned_hist.history
ind_best_epoch = post_tune_history['val_accuracy'].index(early_stop.best)
model.save('modelAPTOS_TUNED.h5')
print(f'\nFine tuning with early stopping of the model after {early_stop.stopped_epoch +1} epochs completed in {convert_seconds_to_time(total_time)}', '\n')
print(f'Accuracy of the tuned model on the training subset (from epoch {ind_best_epoch + 1}):', post_tune_history['accuracy'][ind_best_epoch])
print(f'Accuracy of the tuned model on the cross-validation subset (from epoch {ind_best_epoch + 1}):', early_stop.best, '\n')
#print(f'\nFine tuning of the model after {epochs_post} epochs completed in {convert_seconds_to_time(total_time)}', '\n')
#print('Accuracy of the tuned model on the training subset:', post_tune_history['accuracy'][-1])
#print('Accuracy of the tuned model on the cross-validation subset:', post_tune_history['val_accuracy'][-1], '\n')
#plot_model(model, to_file = 'model_APTOS.png', show_shapes = True)
#SVG(model_to_dot(model).set_size('4x48').create(prog = 'dot', format = 'svg'))
#model.evaluate_generator(batch_cv, steps = batch_cv.n, verbose = 2)
%reset_selective -f X_train
%reset_selective -f Y_train
%reset_selective -f X_val
%reset_selective -f Y_val
#timea = time()
#with np.load('../input/kernel-aptos-img-to-arrays-one-hot-and-split/test_set.npz') as testdata:
#    print('Data in test_set.npz:', testdata.files, '.....', end = '')
#    X_test = testdata['test_im']
#    print('Retrieved!')
#    
#timeb = time()
#total_time = timeb - timea
#
#print(f'Array containing test images retrieved in {convert_seconds_to_time(total_time)}\n')
#    
#print('Shape of the array containing preprocessed test images', X_test.shape)
#gen_test = ImageDataGenerator(preprocessing_function = preprocess_input)
#
#batch_test = gen_test.flow(X_test, y = None, batch_size = 1, shuffle = False)

#batch_test.reset()

#STEP_SIZE_TEST = batch_test.n // batch_test.batch_size # number of test images // batch size (1 in this case)
#predictions = model.predict_generator(batch_test, 
#                                      steps = batch_test.n, 
#                                      verbose = 2)
#predictions = np.argmax(predictions, axis = 1)
#url = r'../input/aptos2019-blindness-detection/test.csv'
#test = pd.read_csv(url)
#test_ids = test['id_code'].tolist()
#submission = pd.DataFrame({'id_code': test_ids, 'diagnosis': predictions})
#submission.to_csv('submission.csv', index = False)
#
#print('PREDICTIONS READY!')
#epochs_used = len(post_tune_history['acc'])
#
#plt.figure(figsize = (12,6))

#ax[0].plot(range(1, epochs_pre + 1), [x * 100 for x in pre_tune_history['acc']], 'o-b', label = 'Training')
#ax[0].plot(range(1, epochs_pre + 1), [x * 100 for x in pre_tune_history['val_acc']], 'o-r', label = 'Cross validation')
#ax[0].set_xlabel('Epochs')
#ax[0].set_ylabel('Accuracy (%)')
#ax[0].set_yticks(range(40,105,10))
#ax[0].set_xticks(range(1, epochs_pre + 1, 1))
#ax[0].set_title(f'PERFORMANCE OF TRANSFER LEARNING\n WITH RESNET-50 AFTER {epochs_pre} EPOCHS', loc = 'center')
#ax[0].legend(loc = 'upper left')

#plt.plot(range(1, epochs_used + 1), [x * 100 for x in post_tune_history['acc']], 'o-b', label = 'Training')
#plt.plot(range(1, epochs_used + 1), [x * 100 for x in post_tune_history['val_acc']], 'o-r', label = 'Cross validation')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy (%)')
#plt.yticks(range(40,110,10))
#plt.xticks(range(1, epochs_used + 1))
#plt.title(f'PERFORMANCE OF FINE TUNING ON RESNET-50 AFTER {epochs_used} EPOCHS WITH EARLY STOPPING')
#plt.legend(loc = 'upper left')
#
#plt.show()
#plt.figure(figsize = (12,6))

#plt.plot(range(1, epochs_used + 1), my_metrics.kappa, 'o-b')
#plt.xlabel('Epochs')
#plt.ylabel('Kappa score for the cross validation set')
#plt.yticks(range(-1, 2))
#plt.xticks(range(1, epochs_used + 1))
#plt.title(f'PERFORMANCE OF FINE TUNING ON RESNET-50 AFTER {epochs_used} EPOCHS WITH EARLY STOPPING')
#
#plt.show()
#WITH EARLYSTOPPING:

epochs_used = ind_best_epoch + 1
plt.figure(figsize = (12,6))

plt.plot(range(1, epochs_used + 1), [x * 100 for x in post_tune_history['accuracy'][:ind_best_epoch + 1]], 'o-b', label = 'Training')
plt.plot(range(1, epochs_used + 1), [x * 100 for x in post_tune_history['val_accuracy'][:ind_best_epoch + 1]], 'o-r', label = 'Cross validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.yticks(range(40,110,10))
plt.xticks(range(1, epochs_used + 1))
plt.title(f'PERFORMANCE OF FINE TUNING WITH EARLY STOPPING ON RESNET-50 AFTER {epochs_used} EPOCHS')
plt.legend(loc = 'upper left')

plt.show()
#WITHOUT EARLYSTOPPING:

#epochs_used = len(post_tune_history['accuracy'])
#plt.figure(figsize = (12,6))
#
#plt.plot(range(1, epochs_used + 1), [x * 100 for x in post_tune_history['accuracy']], 'o-b', label = 'Training')
#plt.plot(range(1, epochs_used + 1), [x * 100 for x in post_tune_history['val_accuracy']], 'o-r', label = 'Cross validation')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy (%)')
#plt.yticks(range(40,110,10))
#plt.xticks(range(1, epochs_used + 1))
#plt.title(f'PERFORMANCE OF FINE TUNING ON RESNET-50 AFTER {epochs_used} EPOCHS')
#plt.legend(loc = 'upper left')
#
#plt.show()
#epochs_used = len(post_tune_history['accuracy'])
#
#fig, ax = plt.subplots(1, 2, figsize = (24, 6))
#fig.suptitle(f'PERFORMANCE OF TRANSFER LEARNING WITH RESNET-50 AFTER {epochs_used} EPOCHS', fontsize = 18)
#fig.suptitle(f'PERFORMANCE OF TRANSFER LEARNING WITH EFFICIENTNET AFTER {epochs_used} EPOCHS', fontsize = 18)
#
#ax[0].plot(range(1, epochs_used + 1), [x * 100 for x in post_tune_history['accuracy']], 'o-b', label = 'Training')
#ax[0].plot(range(1, epochs_used + 1), [x * 100 for x in post_tune_history['val_accuracy']], 'o-r', label = 'Cross validation')
#ax[0].set_xlabel('Epochs')
#ax[0].set_ylabel('Accuracy (%)')
#ax[0].set_yticks(range(40,105,10))
#ax[0].set_xticks(range(1, epochs_used + 1))
#ax[0].set_title('ACCURACY FOR THE TRAINING AND CROSS VALIDATIONS SETS', loc = 'center')
#ax[0].legend(loc = 'upper left')
#
#ax[1].plot(range(1, epochs_used + 1), post_tune_history['cohens_kappa'], 'o-b', label = 'Training')
#ax[1].plot(range(1, epochs_used + 1), post_tune_history['val_cohens_kappa'], 'o-r', label = 'Cross validation')
#ax[1].set_xlabel('Epochs')
#ax[1].set_ylabel("Cohen's kappa coefficient")
#ax[1].set_yticks(range(-1, 2))
#ax[1].set_xticks(range(1, epochs_used + 1))
#ax[1].set_title("COHEN'S KAPPA COEFFICIENT FOR THE TRAINING AND CROSS VALIDATIONS SETS", loc = 'center')
#ax[1].legend(loc = 'upper left')

#ax[1].plot(range(1, epochs_used + 1), my_metrics.kappa, 'o-g')
#ax[1].set_xlabel('Epochs')
#ax[1].set_ylabel('Quadratic kappa score')
#ax[1].set_yticks(range(-1, 2))
#ax[1].set_xticks(range(1, epochs_used + 1))
#ax[1].set_title("COHEN'S KAPPA SCORE FOR THE CROSS VALIDATION SET", loc = 'center')
                
#plt.show()
url_test = r"/kaggle/input/aptos2019-blindness-detection/test.csv"
url_sample = r"/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"
test = pd.read_csv(url_test)
sample = pd.read_csv(url_sample)

test_id_codes = test["id_code"].tolist()
#%reset_selective -f test

index = 0
predictions_list = []
#gen_test = ImageDataGenerator(preprocessing_function = preprocess_input)

timec = time()
print("Processing and resizing images from the test set...")
for im_test in test_id_codes:
    uri = glob.glob("/kaggle/input/aptos2019-blindness-detection/test_images/" + im_test + ".*")
    image = io.imread(uri[0])
    image = cv.resize(image, (256, 256), interpolation = cv.INTER_AREA) / 255 #Normalising...
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    predictions_list.append(prediction)
    %reset_selective -f image
    index += 1
    if index % 500 == 0:
        print(f"\t{index} images")      

%reset_selective -f test_id_codes, test
timed = time()
total_time = timed - timec        
print(f"All images from the test set have been processed in {convert_seconds_to_time(total_time)}\n")
print("Predictions ready!")
print('Distribution of the predicted diagnoses')
print('---------------------------------------\n')

num_diagnosis_0 = predictions_list.count(0)
num_diagnosis_1 = predictions_list.count(1)
num_diagnosis_2 = predictions_list.count(2)
num_diagnosis_3 = predictions_list.count(3)
num_diagnosis_4 = predictions_list.count(4)

sample['diagnosis'] = predictions_list
#%reset_selective -f predictions_list

print(f"  0 - No DR:              {num_diagnosis_0} examples")
print(f"  1 - Mild:               {num_diagnosis_1} examples")
print(f"  2 - Moderate:           {num_diagnosis_2} examples")
print(f"  3 - Severe:             {num_diagnosis_3} examples")
print(f"  4 - Proliferative DR:   {num_diagnosis_4} examples\n")

print(f"  Total number of test examples = {num_diagnosis_0 + num_diagnosis_1 + num_diagnosis_2 + num_diagnosis_3 + num_diagnosis_4}")
sample.head()
with np.load('../input/kernel-aptos-img-to-arrays-one-hot-and-split/test_set.npz') as testdata:
    print('Data in test_set.npz:', testdata.files, '.....', end = '')
    X_test = testdata['test_im']
    print('Retrieved!')
rand_samples = np.random.randint(0, X_test.shape[0] + 1, size = 3)

fig, ax = plt.subplots(1,len(rand_samples), figsize = (18, 9))
plt.suptitle('Predictions for 3 random examples from the provided test set of images', fontsize = 20)

for i in range(len(rand_samples)):
    ax[i].imshow(X_test[rand_samples[i]])
    ax[i].set_title(diab_retin(predictions_list[rand_samples[i]]))
    ax[i].axis('off')

plt.show()
%reset_selective -f X_test
sample.to_csv('submission.csv', index = False)
print('Submission ready!')