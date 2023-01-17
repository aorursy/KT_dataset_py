import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras

from sklearn.datasets import load_files

data_dir = '../input/flowers/flowers'

data = load_files(data_dir)
X = np.array(data['filenames'])
y = np.array(data['target'])
labels = np.array(data['target_names'])

# How the arrays look like?
print('Data files - ',X)
print('Target labels - ',y) 
# numbers are corresponding to class label. We need to change them to a vector of 5 elements.

# Remove .pyc or .py files
pyc_file_pos = (np.where(file==X) for file in X if file.endswith(('.pyc','.py')))
for pos in pyc_file_pos:
    X = np.delete(X,pos)
    y = np.delete(y,pos)
    
print('Number of training files : ', X.shape[0])
print('Number of training targets : ', y.shape[0])

from keras.preprocessing.image import img_to_array, load_img

def convert_img_to_arr(file_path_list):
    arr = []
    for file_path in file_path_list:
        img = load_img(file_path, target_size = (224,224))
        img = img_to_array(img)
        arr.append(img)
    return arr

X = np.array(convert_img_to_arr(X))
print(X.shape) 
X = X.astype('float32')/255
no_of_classes = len(np.unique(y))
from keras.utils import np_utils
y = np.array(np_utils.to_categorical(y,no_of_classes))
y[0]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_test.shape[0])
X_test,X_valid, y_test, y_valid = train_test_split(X_test,y_test, test_size = 0.5)
print(X_valid.shape[0])
# Fine-tuning
from keras.models import Model
from keras import optimizers
from keras import applications
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import GlobalAveragePooling2D,Dense,Flatten,Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

#load the Xception model without the final layers(include_top=False)
base_model = applications.Xception(weights='imagenet', include_top=False)
print('Loaded model!')

#Total of 132 layers, we want to train only the last 15 layers. 
#Let's freeze the first 132-15=117 layers 
for layer in base_model.layers[:117]:
    layer.trainable = False
    
base_model.summary()
for layer in base_model.layers:
    print(layer,layer.trainable)
top_model = Sequential()  
top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(no_of_classes, activation='softmax')) 
top_model.summary()
model = Sequential()
model.add(base_model)
model.add(top_model)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
epochs = 15
batch_size=32
best_model_finetuned_path = 'best_finetuned_model.hdf5'

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train,y_train,
    batch_size=batch_size)

validation_generator = test_datagen.flow(
    X_valid,y_valid,
    batch_size=batch_size)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs= epochs ,
    validation_data=validation_generator,
    validation_steps=len(X_valid) // batch_size,
    callbacks=[ModelCheckpoint(best_model_finetuned_path,save_best_only = True,verbose = 1)])
model.load_weights(best_model_finetuned_path)  
   
(eval_loss, eval_accuracy) = model.evaluate(  
     X_test, y_test, batch_size=batch_size, verbose=1)

print("Accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("Loss: {}".format(eval_loss)) 
import matplotlib.pyplot as plt 
# Let's visualize the loss and accuracy wrt epochs
def plot(history):
    plt.figure(1)  

     # summarize history for accuracy  

    plt.subplot(211)  
    plt.plot(history.history['acc'])  
    plt.plot(history.history['val_acc'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  

plot(history)
from keras.optimizers import SGD
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.5
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model_time = Sequential()
model_time.add(base_model)
model_time.add(top_model)
model_time.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
history_time_based_decay = model_time.fit(X_train, y_train, 
   validation_data=(X_test, y_test), 
   epochs=epochs, 
   batch_size=batch_size, 
   callbacks=[ModelCheckpoint(best_model_finetuned_path,save_best_only = True,verbose = 1)], 
   verbose=2)
model_time.load_weights(best_model_finetuned_path)  
   
(eval_loss, eval_accuracy) = model_time.evaluate(  
     X_test, y_test, batch_size=batch_size, verbose=1)

print("Accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("Loss: {}".format(eval_loss)) 
plot(history_time_based_decay)
from keras.callbacks import LearningRateScheduler
import math

model_step = Sequential()
model_step.add(base_model)
model_step.add(top_model)
momentum = 0.5
sgd = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False) 
model_step.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

# learning schedule callback
loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate, ModelCheckpoint(best_model_finetuned_path,save_best_only = True,verbose = 1)]

# fit the model
history_step_based_decay = model_step.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     epochs=epochs, 
                     batch_size=batch_size, 
                     callbacks=callbacks_list, 
                     verbose=2)
model_step.load_weights(best_model_finetuned_path)  
   
(eval_loss, eval_accuracy) = model_step.evaluate(  
     X_test, y_test, batch_size=batch_size, verbose=1)

print("Accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("Loss: {}".format(eval_loss)) 
plot(history_step_based_decay)
fig = plt.figure()
plt.plot(range(1,epochs+1),loss_history.lr,label='learning rate')
plt.xlabel("epoch")
plt.xlim([1,epochs+1])
plt.ylabel("learning rate")
plt.legend(loc=0)
plt.grid(True)
plt.title("Learning rate")
plt.show()
momentum = 0.8
sgd = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False)

model_exp = Sequential()
model_exp.add(base_model)
model_exp.add(top_model)

model_exp.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(len(self.losses)))
        print('lr:', exp_decay(len(self.losses)))

def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate

# learning schedule callback
loss_history = LossHistory()
lrate = LearningRateScheduler(exp_decay)
callbacks_list = [loss_history, lrate,ModelCheckpoint(best_model_finetuned_path,save_best_only = True,verbose = 1)]

# fit the model
history_exponential_based_decay = model_exp.fit(X_train, y_train, 
                     validation_data=(X_test, y_test), 
                     epochs=epochs, 
                     batch_size=batch_size, 
                     callbacks=callbacks_list, 
                     verbose=2)
model_exp.load_weights(best_model_finetuned_path)  
   
(eval_loss, eval_accuracy) = model_exp.evaluate(  
     X_test, y_test, batch_size=batch_size, verbose=1)

print("Accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("Loss: {}".format(eval_loss)) 
plot(history_exponential_based_decay)
fig = plt.figure()
plt.plot(range(1,epochs+1),loss_history.lr,label='learning rate')
plt.xlabel("epoch")
plt.xlim([1,epochs+1])
plt.ylabel("learning rate")
plt.legend(loc=0)
plt.grid(True)
plt.title("Learning rate")
plt.show()
fig = plt.figure()
plt.plot(range(1,epochs+1),history.history['val_acc'],label='Constant lr')
plt.plot(range(1,epochs+1),history_time_based_decay.history['val_acc'],label='Time based lr')
plt.plot(range(1,epochs+1),history_step_based_decay.history['val_acc'],label='Step based lr')
plt.plot(range(1,epochs+1),history_exponential_based_decay.history['val_acc'],label='Exponential lr')
plt.legend(loc=0)
plt.xlabel('epochs')
plt.xlim([0,epochs])
plt.ylabel('accuracy on validation set')
plt.grid(True)
plt.title("Comparing Model Accuracy")
plt.show()
plt.show()
