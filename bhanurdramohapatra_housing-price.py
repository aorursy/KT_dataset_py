import os, imp

try:

    imp.find_module('kerastuner')

    print('Kerastuner module is already installed')

except:

    os.system('pip install keras-tuner')

    print('Keras tuner was not available but now it is installed')
import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model, load_model

from kerastuner import RandomSearch

from tensorflow.keras.activations import relu, linear

from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

import tensorflow as tf

import datetime

pd.options.display.width=None
train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

price = train_data.SalePrice

train_data.drop(columns=['SalePrice', 'Id'], inplace=True)

test = test_data.drop(columns=['Id'])

data = pd.concat([train_data, test], ignore_index=True)
model_path = r'/kaggle/working/best_model.hdf5'
print(train_data.shape, test.shape, data.shape)
def process():

    missing_data_cols = []

    for cols in range(len(data.columns)):

        if int(data[data.columns[cols]].describe()[0]) < len(data):

            missing_data_cols.append(data.columns[cols])

    copy = data

    cat = copy.select_dtypes(include='object', exclude='number').isnull()

    copy = copy.astype(str).apply(LabelEncoder().fit_transform)

    encoded_data = copy.where(~cat, data)

    df = encoded_data.astype('float64')

    for i in missing_data_cols:

        df[i].fillna(df.groupby(['MSSubClass'])[i].transform('mean'), inplace=True)

    drop_cols = []

    for i in range(len(df.columns)):

        if int(df[df.columns[i]].describe()[0]) < len(df):

            drop_cols.append(i)

    df = df.drop(columns=df.columns[drop_cols])

    return df
def train_test():

    train = process()[:1460]

    test = process()[1460:]

    return train, test
def tuner_model(hp):

    t_model = Sequential()

    t_model.add(Dense(units=128, input_shape=(train_test()[0].shape[1],), activation=relu, name='1st_layer'))

    for i in range(hp.Int('num_layers', 1, 10)):

        t_model.add(Dense(hp.Int('units', min_value=128, max_value=1024, step=64, default=256), activation=relu))

    t_model.add(Dense(units=1, activation=linear, name='Output_Layer'))

    print('Number of Layers Built: ', len(t_model.layers))

    # t_model.compile(optimizer=Adam(), loss=mean_squared_error, metrics=[mean_squared_error])

    t_model.compile(optimizer=Adam(hp.Choice('lr', values=[1e-4, 11e-4, 12e-4], default=1e-4)), loss=mean_absolute_error,

                    metrics=[mean_absolute_error])

    return t_model
def best_model():

    tf.keras.backend.clear_session()

    start_time = datetime.datetime.now()

    print('Tuning start time: ', start_time.strftime('%d-%b-%Y %I:%M:%S%p'))

    tuner = RandomSearch(tuner_model, objective='val_mean_absolute_error', max_trials=3, executions_per_trial=2, project_name='house')

    tuner.search(train_test()[0], price, epochs=len(train_test()[0]), validation_split=0.2, verbose=0)

    stop_time = datetime.datetime.now()

    print('Time taken to tune: ', str((stop_time - start_time).seconds // 3600) + ' Hours ' +

          str((stop_time - start_time).seconds // 60) + ' Mins ' + str(

        (stop_time - start_time).seconds - ((stop_time - start_time).seconds // 60) * 60) + ' Sec')

    best_mod = tuner.get_best_models()[0]

    return best_mod
model = best_model()
def plot(trained):



    epoch = []

    tmae = []

    vmae = []

    tloss = []

    vloss = []

    train = True

    tf.keras.backend.clear_session()

    while train:        

        history = trained

        for t_acc  in history.history['mean_absolute_error']:

            tmae.append(t_acc)

        for v_acc in history.history['val_mean_absolute_error']:

            vmae.append(v_acc)

        for t_loss in history.history['loss']:

            tloss.append(t_loss)

        for v_loss in history.history['val_loss']:

            vloss.append(v_loss)

        train=False

    epoch_count = len(tmae)

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

    axes[1].plot(np.array(epoch), np.array(tmae), 'r', label='Training Error')

    axes[1].plot(np.array(epoch), np.array(vmae), 'g', label='Validation Error')

    axes[1].set_title('Training Vs Validation Error')

    axes[1].set_xlabel('Epochs')

    axes[1].set_ylabel('Error')

    axes[1].legend()

    plt.tight_layout()

    plt.show()
def prediction(best_model):

    pred_model = load_model(best_model)

    print('Predicting House Price now\n')

    pred_gen = pred_model.predict(train_test()[1])

    return pred_gen
class epoch_avg_print(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):

        print('Epoch {:d}, learning rate {:.7f}'.format(epoch + 1, tf.keras.backend.get_value(self.model.optimizer.lr)),

              '\n')

        



    def on_epoch_end(self, epoch, logs=None):

        print('Param values at the end of epoch, train_mae: {:.4f}, val_mae: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}'.format(

            logs[list(logs)[1]], logs[list(logs)[3]], logs[list(logs)[0]], logs[list(logs)[2]]), '\n')

        '''print(logs[list(logs)[0]], logs[list(logs)[1]], logs[list(logs)[2]], logs[list(logs)[3]])

        print(logs)'''
class earlystop(tf.keras.callbacks.Callback):

    def __init__(self, patience = 20):

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

        
class adjustlr(tf.keras.callbacks.Callback):

    tr_low_mae = 0 # setting 0 to highest training accuracy

    tr_low_loss = np.inf # setting lowest training loss to infinity

    val_low_mae = 0 # setting 0 to highest validation accuracy

    val_low_loss = np.inf # setting lowest validation loss to infinity 

    epochs = 0

    model = model

    best_weights = None

    best_model = None

    int_model = None

    lr = float(tf.keras.backend.get_value(model.optimizer.lr))

    

    

    def __init__(self):

        super(adjustlr, self).__init__()

        self.tr_low_mae = 0

        self.val_low_loss = np.inf

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

        val_mae = logs.get('val_mean_absolute_error') # getting the average validation accuracy for this epoch

        tr_mae = logs.get('mean_absolute_error') # getting the average training accuracy for this epoch

        adjustlr.lr = float(tf.keras.backend.get_value(self.model.optimizer.lr)) # fetching the lr value used in this epoch

        adjustlr.epochs = adjustlr.epochs + 1

        # checking whether the current epoch's training accuracy is better than previous training accuracy

        if adjustlr.tr_low_mae < tr_mae: 

            adjustlr.tr_low_mae = tr_mae 

        # checking whether the current epoch's validation accuracy is better than previous validation accuracy

        if adjustlr.val_low_mae < val_mae:

            adjustlr.val_low_mae = val_mae

        # checking whether the current epoch's validation loss is better than previous validation loss

        if adjustlr.val_low_loss > val_loss:

            adjustlr.val_low_loss = val_loss

        # checking whether the current epoch's training loss is better than previous validation loss

        if adjustlr.tr_low_loss > tr_loss:

            adjustlr.tr_low_loss = tr_loss

        # LR Adjustment -1 : if the Validation loss increases, adjust LR to improve the validation loss - Avoid Overfitting

        if tr_loss > adjustlr.tr_low_loss:

            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

            adjusted_lr = lr * 0.9 * (adjustlr.tr_low_loss / tr_loss)

            tf.keras.backend.set_value(self.model.optimizer.lr, adjusted_lr)

            print("The current Training loss {:.7f} is higher than previous Training Loss {:.7f}, hence reducing the LR to {:.7f}\n".format(tr_loss, adjustlr.tr_low_loss, adjusted_lr))

            self.model.set_weights(adjustlr.best_weights)

            print('Loading back the previous best model\n')

        # LR Adjustment -2 : if the Training loss increases, adjust LR to improve the loss - Avoid Underfitting

        elif val_loss > adjustlr.val_low_loss:

            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

            adjusted_lr = lr * 0.9 * (adjustlr.val_low_loss / val_loss)

            tf.keras.backend.set_value(self.model.optimizer.lr, adjusted_lr)

            print("The current Validation loss {:.7f} is higher than previous validation loss {:.7f}, hence reducing the LR to {:.7f}\n".format(val_loss, adjustlr.val_low_loss, adjusted_lr))

            self.model.set_weights(adjustlr.best_weights)             

            print('Loading back the previous best model\n')

            

        else:

            adjustlr.best_weights = self.model.get_weights()
def modelcheck():

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min')

    return checkpoint
def train_model():

    tf.keras.backend.clear_session()

    start_time = datetime.datetime.now()

    print('Training start time: ', start_time.strftime('%d-%b-%Y %I:%M:%S%p'))

    train = model.fit(train_test()[0].to_numpy(), price.to_numpy(), epochs=300, validation_split=0.2, batch_size=32,

                      steps_per_epoch=(len(train_test()[0]) * 0.8) / 32, verbose=0, callbacks=[epoch_avg_print(), adjustlr(), modelcheck(), earlystop()])

    stop_time = datetime.datetime.now()

    print('Time taken to complete the training is: ', str((stop_time - start_time).seconds//3600) + ' Hours ' + str((stop_time - start_time).seconds//60) +' Mins '

    + str((stop_time - start_time).seconds - ((stop_time - start_time).seconds//60) * 60) + ' Sec')

    plot(train)

    
train_model()
prediction(model_path)
submission_df = pd.DataFrame({'Id': test_data.Id, 'SalePrice': np.array(prediction(model_path)).reshape(len(test_data))})

submission_df.head(5)
submission_df.to_csv('house_price_prediction.csv', index=False)