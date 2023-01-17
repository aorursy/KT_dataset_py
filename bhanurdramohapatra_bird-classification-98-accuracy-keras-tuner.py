import os, imp

try:

    imp.find_module('kerastuner')

    print('Kerastuner module is already installed')

except:

    os.system('pip install keras-tuner')

    print('Keras tuner was not available but now it is installed')
import numpy as np

import pandas as pd

import datetime

import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.applications import InceptionV3  # as the input size is 224 x 224

from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D, Input, Lambda

from tensorflow.keras.losses import categorical_crossentropy  # multiple classification

from tensorflow.keras.activations import softmax, relu, sigmoid

from tensorflow.keras.preprocessing import image

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from kerastuner.tuners import RandomSearch

import tensorflow as tf
rand_seed = 4

model_path = r'/kaggle/working/best_model.hdf5'
def inp_path(rand_seed):

    train_path = r'/kaggle/input/100-bird-species/train'

    test_path = r'/kaggle/input/100-bird-species/test'

    val_path = r'/kaggle/input/100-bird-species/valid'

    cons_path = r'/kaggle/input/100-bird-species/consolidated'

    birds_cat = os.listdir(cons_path)

    all_path = [train_path, test_path, val_path]

    return [train_path, test_path, val_path, birds_cat, all_path]
def create_label(rand_seed):

    bird = []

    count = []

    folder = []

    for fold in range(len(inp_path(rand_seed)[4])):

        for cat in range(len(inp_path(rand_seed)[3])):

            folder.append(inp_path(rand_seed)[4][fold])

            bird.append(inp_path(rand_seed)[3][cat])

            count.append(len([name for name in os.listdir(inp_path(rand_seed)[4][fold] + '/' + inp_path(rand_seed)[3][cat])]))

    return [folder, bird, count]
def inp_chk(rand_seed):

    folder, bird, count = create_label(rand_seed)

    df = pd.DataFrame({'Folder': create_label(rand_seed)[0], 'Bird': create_label(rand_seed)[1], 'Count': create_label(rand_seed)[2]})

    print('Number of different Bird species in each folder\n')

    print(df.groupby(['Folder']).Bird.count())

    print('\nNumber of images in each folder\n')

    print(df.groupby(['Folder']).Count.sum())

    print('\nCount of each Bird species images in Training folder\n')

    print(df[df.Folder.str.contains('train')])

    print('\n\n')
def sample_disp(rand_seed):

    gen = image.ImageDataGenerator().flow_from_directory(inp_path(rand_seed)[0], class_mode='binary', target_size=(224, 224))

    x, y = gen.next()

    idx = gen.class_indices.items()

    img = []

    label = []

    for i in range(x.shape[0]):

        img.append(image.array_to_img(x[i]))

        for item in idx:

            if item[1] == y[i]:

                label.append(item[0])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), dpi=100)

    for pic, axes in enumerate(ax.flat):

        axes.imshow(img[pic])

        axes.set_title(label[pic])
def data_gen(rand_seed):

    train_gen = image.ImageDataGenerator(rescale=1. / 255,

                                         preprocessing_function=tf.keras.applications.inception_v3.preprocess_input).flow_from_directory(

        inp_path(rand_seed)[0], target_size=(224, 224),

        class_mode='categorical', batch_size=32)

    test_gen = image.ImageDataGenerator(rescale=1. / 255,

                                        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input).flow_from_directory(

        inp_path(rand_seed)[1], target_size=(224, 224),

        class_mode='categorical', batch_size=32)

    val_gen = image.ImageDataGenerator(rescale=1. / 255,

                                       preprocessing_function=tf.keras.applications.inception_v3.preprocess_input).flow_from_directory(

        inp_path(rand_seed)[2], target_size=(224, 224),

        class_mode='categorical', batch_size=32)

    return [train_gen, val_gen, test_gen]
def plot(trained):



    epoch = []

    tacc = []

    vacc = []

    tloss = []

    vloss = []

    train = True

    tf.keras.backend.clear_session()

    while train:        

        history = trained

        for t_acc  in history.history['acc']:

            tacc.append(t_acc)

        for v_acc in history.history['val_acc']:

            vacc.append(v_acc)

        for t_loss in history.history['loss']:

            tloss.append(t_loss)

        for v_loss in history.history['val_loss']:

            vloss.append(v_loss)

        train=False

    epoch_count = len(tacc)

    for i in range(epoch_count):

        epoch.append(i+1)

    plt.style.use('fivethirtyeight')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    axes[0].plot(np.array(epoch), np.array(tloss), 'r', label='Training Loss')

    axes[0].plot(np.array(epoch), np.array(vloss), 'g', label='Validation Loss')

    axes[0].set_title('Training Vs Validation Loss')

    axes[0].set_xlabel('Epochs')

    axes[0].set_ylabel('Loss')

    axes[0].legend()

    axes[1].plot(np.array(epoch), np.array(tacc), 'r', label='Training Accuracy')

    axes[1].plot(np.array(epoch), np.array(vacc), 'g', label='Validation Accuracy')

    axes[1].set_title('Training Vs Validation Accuracy')

    axes[1].set_xlabel('Epochs')

    axes[1].set_ylabel('Accuracy')

    axes[1].legend()

    plt.tight_layout()

    plt.show()
def prediction(best_model):

    pred_model = load_model(best_model)

    print('Evaluating Model now\n')

    evaluation = pred_model.evaluate(data_gen(rand_seed)[2], verbose=0)

    print('\nThe Model accuracy is: {}%'.format(round(evaluation[1]*100, 2)))

    #pred_gen = pred_model.predict(data_gen(rand_seed)[2])

    #print(pred_gen)

def tuner_model(hp):

    init_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))

    flat_1 = Flatten()(init_model.output)

    for i in range(hp.Int('num_layers', 1, 6)):

        model_in = Dense(units=hp.Int('units', min_value=128, max_value=1024, step=32, default=512), activation=relu,

                         name='First_FC_Layer')(flat_1)   



    output_layer = Dense(units=200, activation=softmax, name='Output_Layer')(model_in)

    tune_model = Model(inputs=init_model.input, outputs=output_layer)

    print("Number of Layers Built:", len(tune_model.layers))

    tune_model.compile(

        optimizer=SGD(hp.Choice('lr', values=[18e-4, 185e-4, 19e-4, 195e-4, 20e-4, 15e-4], default=15e-4),

                      momentum=hp.Choice('momentum', [0.4, 0.6, 0.7, 0.8, 0.9],

                                         default=0.8)),

        loss=categorical_crossentropy, metrics=['acc'])

    return tune_model
def best_mod(rand_seed):

    tf.keras.backend.clear_session()

    start_time = datetime.datetime.now()

    print('Tuning start time: ', start_time.strftime('%d-%b-%Y %I:%M:%S%p'))

    tuner = RandomSearch(tuner_model, objective='val_acc', max_trials=2, executions_per_trial=2, project_name='ML_git', seed=rand_seed)

    tuner.search(data_gen(rand_seed)[0], epochs=5, validation_data=data_gen(rand_seed)[1], steps_per_epoch=len(data_gen(rand_seed)[0]), verbose=0)

    params = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values

    model = tuner.get_best_models()[0]

    stop_time = datetime.datetime.now()

    print('Time taken to complete the tuning is:', str((stop_time - start_time).seconds//3600) + ' Hours ' + str((stop_time - start_time).seconds//60) +' Mins '

    + str((stop_time - start_time).seconds - ((stop_time - start_time).seconds//60) * 60) + ' Sec' )

    return model
model = best_mod(rand_seed)
tf.keras.callbacks.CSVLogger(r'/kaggle/working/training.log', append=False, separator=',')

# This is optional whether you would like to save the verbose/ non verbose log details of the model training.
class epoch_average_print(tf.keras.callbacks.Callback):

    

    def on_epoch_begin(self, epoch, logs=None):

        print('Epoch {:d}, learning rate: {:.7f}'.format(epoch + 1, tf.keras.backend.get_value(self.model.optimizer.lr)),'\n')

    

    

        

    def on_epoch_end(self, epoch, logs=None): # function keywords  are predefined as per keras. visit 'ref' link below to know more

        print('Params value at end of epoch, training loss: {:.7f}, val loss: {:.7f}, train acc: {:.7f}, val acc: {:.7f}.'

              .format(logs['loss'], logs['val_loss'], logs['acc'], logs['val_acc']), '\n')

        

        # Taking the average values to display as it would help to understand whether there's big increase or decrease in the values.

        # Note - Don't need to use average function as the inbuilt function stores the average value in the parameters 

        # ref - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/BaseLogger
class stopearly(tf.keras.callbacks.Callback):

    

    def __init__(self, patience = 2):

        # patience defines the number of epoch with no improvement seen, after which the training will be stopped. 

        # I would like to give the training one buffer epoch to improve its accuracy and losses ;) 

        self.patience = patience



        

    def on_train_begin(self, logs=None):

        self.wait = 0 # The number of epochs passed when the loss is not declining

        self.stopped_epoch = 0 # The epoch at which the training is stopped

        self.best_loss = np.inf # setting the loss to infinity as it would help us to compare with training loss

    

    

    def on_epoch_end(self, epoch, logs=None):

        if logs.get('loss') < self.best_loss:

            self.best_loss = logs.get('loss')

            self.wait = 0

        else:

            self.wait = self.wait + 1

            if self.wait > self.patience:

                self.stopped_epoch = epoch

                self.model.stop_training = True # stopping the training 

                print('Model Training stopped as the training loss is plateaued. The model is saved with best weights before the plateauing of loss')

    

    

    def on_train_end(self, logs=None):

        if self.stopped_epoch > 0:

            print('The model is stopped at epoch: {}'.format(self.stopped_epoch + 1))
def modelcheck():

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode='max')

    return checkpoint
class adjustlr(tf.keras.callbacks.Callback):

    tr_high_acc = 0 # setting 0 to highest training accuracy

    tr_low_loss = np.inf # setting lowest training loss to infinity

    val_high_acc = 0 # setting 0 to highest validation accuracy

    low_v_loss = np.inf # setting lowest validation loss to infinity 

    epochs = 0

    model = model

    best_weights = None

    best_model = None

    int_model = None

    lr = float(tf.keras.backend.get_value(model.optimizer.lr))

    

    

    def __init__(self):

        super(adjustlr, self).__init__()

        self.tr_high_acc = 0

        self.low_v_loss = np.inf

        self.model = model

        try:

            self.best_weights = self.model.get_weights()

        except:

            self.best_weights = None

        self.int_model = None

        self.epochs = 0

        self.tr_low_loss = np.inf

        self.lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))



        

    def on_epoch_end(self, epoch, logs=None): # Function naming format is available in the tensorflow site. If you use anyother format, it won't recognise the function.

        # as per tf.keras documentation, epochs return as dict of parameters' average  value per particular epoch. 

        val_loss = logs.get('val_loss') # getting the average validation loss for this epoch 

        tr_loss = logs.get('loss') # getting the average validation loss for this epoch

        val_acc = logs.get('val_acc') # getting the average validation accuracy for this epoch

        tr_acc = logs.get('acc') # getting the average training accuracy for this epoch

        adjustlr.lr = float(tf.keras.backend.get_value(self.model.optimizer.lr)) # fetching the lr value used in this epoch

        adjustlr.epochs = adjustlr.epochs + 1

        # checking whether the current epoch's training accuracy is better than previous training accuracy

        if adjustlr.tr_high_acc < tr_acc: 

            adjustlr.tr_high_acc = tr_acc 

        # checking whether the current epoch's validation accuracy is better than previous validation accuracy

        if adjustlr.val_high_acc < val_acc:

            adjustlr.val_high_acc = val_acc

        # checking whether the current epoch's validation loss is better than previous validation loss

        if adjustlr.low_v_loss > val_loss:

            adjustlr.low_v_loss = val_loss

        # checking whether the current epoch's training loss is better than previous validation loss

        if adjustlr.tr_low_loss > tr_loss:

            adjustlr.tr_low_loss = tr_loss

        # LR Adjustment -1: checks if the training accuracy is less than 0.95 then adjust the LR to improve it  

        if tr_acc <= 0.95 and tr_acc < adjustlr.tr_high_acc:

            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

            adjusted_lr = lr * 0.8 * (tr_acc / adjustlr.tr_high_acc)

            tf.keras.backend.set_value(self.model.optimizer.lr, adjusted_lr) 

            print("The current Training accuracy {:.7f} is lower than previous Training accuracy {:.7f}, hence reducing the LR to {:.7f}\n".format(tr_acc, adjustlr.tr_high_acc, adjusted_lr))

            self.model.set_weights(adjustlr.best_weights)

            print('Loading back the previous best model\n')

        # LR Adjustment -2 : if the Validation loss increases, adjust LR to improve the validation loss - Avoid Overfitting

        elif tr_loss > adjustlr.tr_low_loss and tr_acc <= 0.95:

            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

            adjusted_lr = lr * 0.8 * (adjustlr.tr_low_loss / tr_loss)

            tf.keras.backend.set_value(self.model.optimizer.lr, adjusted_lr)

            print("The current Training loss {:.7f} is higher than previous Training Loss {:.7f}, hence reducing the LR to {:.7f}\n".format(tr_loss, adjustlr.tr_low_loss, adjusted_lr))

            self.model.set_weights(adjustlr.best_weights)

            print('Loading back the previous best model\n')

        # LR Adjustment -3 : if the Training loss increases, adjust LR to improve the loss - Avoid Underfitting

        elif val_loss > adjustlr.low_v_loss and tr_acc <= 0.95:

            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

            adjusted_lr = lr * 0.8 * (adjustlr.low_v_loss / val_loss)

            tf.keras.backend.set_value(self.model.optimizer.lr, adjusted_lr)

            print("The current Validation loss {:.7f} is higher than previous validation loss {:.7f}, hence reducing the LR to {:.7f}\n".format(val_loss, adjustlr.low_v_loss, adjusted_lr))

            self.model.set_weights(adjustlr.best_weights)             

            print('Loading back the previous best model\n')

            

        else:

            adjustlr.best_weights = self.model.get_weights()
def train_model(rand_seed):

    tf.keras.backend.clear_session()

    start_time = datetime.datetime.now()

    print('Training start time: ', start_time.strftime('%d-%b-%Y %I:%M:%S%p'))

    train = model.fit(x = data_gen(rand_seed)[0], validation_data = data_gen(rand_seed)[1], steps_per_epoch = len(data_gen(rand_seed)[0]), epochs=20, 

                               validation_steps = len(data_gen(rand_seed)[1]), verbose=0, callbacks = [epoch_average_print(), adjustlr(), stopearly(), modelcheck()])

    stop_time = datetime.datetime.now()

    print('Time taken to complete the training is: ', str((stop_time - start_time).seconds//3600) + ' Hours ' + str((stop_time - start_time).seconds//60) +' Mins '

    + str((stop_time - start_time).seconds - ((stop_time - start_time).seconds//60) * 60) + ' Sec')

    plot(train)

    prediction(model_path)

    
train_model(rand_seed)