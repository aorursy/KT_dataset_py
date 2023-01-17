import tensorflow as tf

print(tf.__version__)
# Load the diabetes dataset

from sklearn.datasets import load_diabetes



diabetes_dataset = load_diabetes()

print(diabetes_dataset['DESCR'])
# Save the input and target variables



data =  diabetes_dataset["data"]

targets = diabetes_dataset["target"]


targets.shape
# Normalise the target data (this will make clearer training curves)

targets =  (targets - targets.mean(axis = 0))/targets.std()

targets
# Split the data into train and test sets



from sklearn.model_selection import train_test_split

train_data, test_data , train_targets , test_targets = train_test_split(data,targets,test_size = 0.1)



print(train_data.shape,train_targets.shape)

print(test_data.shape,test_targets.shape)
train_data.shape[1]
# Build the model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten,Dense



def get_model():

    model = Sequential([

    Dense(128,activation = 'relu',input_shape = (train_data.shape[1],)),

    Dense(128,activation = 'relu'),

    Dense(128,activation = 'relu'),

    Dense(128,activation = 'relu'),

    Dense(128,activation = 'relu'),

    Dense(128,activation = 'relu'),

    Dense(1)

])

    

    return(model)



model = get_model()

# Print the model summary



model.summary()
# Compile the model

opt = tf.keras.optimizers.Adam()

mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer = opt,

             loss = 'mse',

             metrics = [mae])

# Train the model, with some of the data reserved for validation

history = model.fit(train_data,train_targets,epochs = 10, batch_size =64,validation_split = 0.15,verbose = False)

# Evaluate the model on the test set



model.evaluate(test_data,test_targets,verbose = 2)
import matplotlib.pyplot as plt

%matplotlib inline
# Plot the training and validation loss



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss vs. epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()
from tensorflow.keras.layers import Dropout

from tensorflow.keras import regularizers
def get_regularised_model(wd, rate):

    model = Sequential([

        Dense(128, activation="relu", kernel_regularizer = regularizers.l2(wd),input_shape=(train_data.shape[1],)),

        Dropout(rate),

        Dense(128, activation="relu",kernel_regularizer = regularizers.l2(wd)),

         Dropout(rate),

        Dense(128, activation="relu",kernel_regularizer = regularizers.l2(wd)),

         Dropout(rate),

        Dense(128, activation="relu",kernel_regularizer = regularizers.l2(wd)),

         Dropout(rate),

        Dense(128, activation="relu",kernel_regularizer = regularizers.l2(wd)),

         Dropout(rate),

        Dense(128, activation="relu",kernel_regularizer = regularizers.l2(wd)),

         Dropout(rate),

        Dense(1)

    ])

    return model
# Re-build the model with weight decay and dropout layers



model = get_regularised_model(1e-5,0.3)

# Compile the model

opt = tf.keras.optimizers.Adam()

mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer = opt,

             loss = 'mse',

             metrics = [mae])
# Train the model, with some of the data reserved for validation

history = model.fit(train_data,train_targets,epochs = 10, batch_size =64,validation_split = 0.15,verbose = False)



# Evaluate the model on the test set

model.evaluate(test_data,test_targets,verbose = 2)

# Plot the training and validation loss



import matplotlib.pyplot as plt



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss vs. epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()
# Write a custom callback

from tensorflow.keras.callbacks import Callback



class TrainingCallback(Callback):

    

    def on_train_begin(self,logs = None):

        print("On Begin of Train")

        

    def on_train_end(self,logs = None):

        print("On End of Train")

        

    def on_epoch_begin(self,epoch, logs=None):

        print(f'Begin Epoch is {epoch}')

        

    def on_epoch_end(self,epoch, logs=None):

        print(f'End Epoch is {epoch}')

    

    def on_train_batch_begin(self,batch, logs=None):

        print(f'Begin Batch is {batch}')

        

    def on_train_batch_end(self,batch, logs=None):

        print(f'End Batch is {batch}')

        



# Write a custom callback

from tensorflow.keras.callbacks import Callback



class TestingCallback(Callback):

    

    def on_test_begin(self,logs = None):

        print("On Begin of Test")

        

    def on_test_end(self,logs = None):

        print("On End of Test")

        

    def on_test_batch_begin(self,batch, logs=None):

        print(f'Begin Test Batch is {batch}')

        

    def on_test_batch_end(self,batch, logs=None):

        print(f'End Test Batch is {batch}')
# Write a custom callback

from tensorflow.keras.callbacks import Callback



class PredictionCallback(Callback):

    

    def on_predict_begin(self,logs = None):

        print("On Begin of Prediction")

        

    def on_predict_end(self,logs = None):

        print("On End of Prediction")

        

    def on_predict_batch_begin(self,batch, logs=None):

        print(f'Begin Prediction Batch is {batch}')

        

    def on_predict_batch_end(self,batch, logs=None):

        print(f'End Prediction Batch is {batch}')
# Re-build the model



model = get_regularised_model(1e-5,0.3)
# Compile the model

opt = tf.keras.optimizers.Adam()

mae = tf.keras.metrics.MeanAbsoluteError()

model.compile(optimizer = opt,

             loss = 'mse',

             metrics = [mae])

# Train the model, with some of the data reserved for validation



# Train the model, with some of the data reserved for validation

history = model.fit(train_data,train_targets,epochs = 10, batch_size =64,validation_split = 0.15,verbose = False, 

                    callbacks = [TrainingCallback()])

# Evaluate the model

# Evaluate the model on the test set

model.evaluate(test_data,test_targets,verbose = 2,callbacks = [TestingCallback()])
# Make predictions with the model

model.predict(test_data,verbose = 2,callbacks = [PredictionCallback()])

# Re-train the unregularised model

unregularised_model = get_model()



opt = tf.keras.optimizers.Adam()

mae = tf.keras.metrics.MeanAbsoluteError()

unregularised_model.compile(optimizer = opt,

             loss = 'mse',

             metrics = [mae])



unreg_history = unregularised_model.fit(train_data,train_targets,epochs = 10, 

                                        batch_size =64,validation_split = 0.15,verbose = False, 

                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5)])
# Evaluate the model on the test set

unregularised_model.evaluate(test_data,test_targets,verbose = 2)
# Re-train the regularised model

regularised_model = get_regularised_model(1e-8,0.2)



opt = tf.keras.optimizers.Adam()

mae = tf.keras.metrics.MeanAbsoluteError()

regularised_model.compile(optimizer = opt,

             loss = 'mse',

             metrics = [mae])



reg_history = unregularised_model.fit(train_data,train_targets,epochs = 10, 

                                        batch_size =64,validation_split = 0.15,verbose = False, 

                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5)])

# Evaluate the model on the test set

unregularised_model.evaluate(test_data,test_targets,verbose = 2)
# Plot the training and validation loss



import matplotlib.pyplot as plt



fig = plt.figure(figsize=(12, 5))



fig.add_subplot(121)



plt.plot(unreg_history.history['loss'])

plt.plot(unreg_history.history['val_loss'])

plt.title('Unregularised model: loss vs. epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training', 'Validation'], loc='upper right')



fig.add_subplot(122)



plt.plot(reg_history.history['loss'])

plt.plot(reg_history.history['val_loss'])

plt.title('Regularised model: loss vs. epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training', 'Validation'], loc='upper right')



plt.show()