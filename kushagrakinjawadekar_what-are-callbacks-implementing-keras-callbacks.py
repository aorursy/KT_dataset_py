import tensorflow as tf
import tf.keras

"""Callbacks are passed to the model via the callbacks argument in fit, which takes a list of

callbacks. You can pass any number of callbacks."""



callbacks_list = [

        tf.keras.callbacks.EarlyStopping(  #Interrupts training when improvement stops

        monitor='acc',                     #Monitor's the model Validation Accuracy

        patience=1,),                      #Interrupts training when accuracy nas stopped improving for more than one epoch.

    

        tf.keras.callbacks.ModelCheckpoint(#Saves the current weight after every epoch

        filepath='my_model.h5',            #Path to destination model file

        monitor='val_loss',                #These two arguments mean you won’t overwrite the model file unless val_loss has improved, 

        save_best_only=True)               #which allows you to keep the best model seen during training.

                 ]                                    

model.compile(optimizer='rmsprop',         

              loss='binary_crossentropy',

              metrics=['acc'])             #You monitor accuracy, so it should be part of the model’s metrics.

model.fit(x, y,

          epochs=10,

          batch_size=32,

          callbacks=callbacks_list,

          validation_data=(x_val, y_val))  #Note that because the callback will monitor validation loss and validation accuracy,

                                           #you need to pass validation_data to the call to fit.
callbacks_list = [

                 tf.keras.callbacks.ReduceLROnPlateau(

                    monitor='val_loss',  #Monitor's the model Validation Accuracy

                    factor=0.1,         #Divides the learning rate by 10 when triggered

                    patience=10)        #Interrupts training when accuracy nas stopped improving for more than one epoch.

                 ]



model.fit(x, y,

          epochs=10,

          batch_size=32,

          callbacks=callbacks_list,

          validation_data=(x_val, y_val))  #Note that because the callback will monitor validation loss and validation accuracy,

                                           #you need to pass validation_data to the call to fit.
tf.keras.callbacks.CSVLogger(filename, separator=",", append=False)
csv_logger = CSVLogger('training.log')

model.fit(X_train, Y_train, callbacks=[csv_logger])

# Here’s a simple example of a custom callback that saves to disk (as Numpy arrays) the

# activations of every layer of the model at the end of every epoch, computed on the

# first sample of the validation set:

import keras

import numpy as np

class ActivationLogger(keras.callbacks.Callback):

    def set_model(self, model):

        self.model = model

        layer_outputs = [layer.output for layer in model.layers]

        self.activations_model = keras.models.Model(model.input,

        layer_outputs)

    def on_epoch_end(self, epoch, logs=None):

        if self.validation_data is None:

            raise RuntimeError('Requires validation_data.')

        validation_sample = self.validation_data[0][0:1]

        activations = self.activations_model.predict(validation_sample)

        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')

        np.savez(f, activations)

        f.close()