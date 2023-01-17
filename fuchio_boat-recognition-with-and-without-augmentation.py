import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  os # data 

# Image processing
from PIL import Image, ImageFile
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Plotting
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
path, dirs, files = next(os.walk("../input/boats"))
dirs
boat_types = ['cruise ship', 'gondola', 'kayak', 'sailboat']
import glob

i = 0

X_data = []
Y_data = []

for boat in boat_types:
    files = glob.glob ("../input/boats/" + str(boat) + "/*.jpg")
    
    for myFile in files:
      img = Image.open(myFile)
      #img.thumbnail((width, height), Image.ANTIALIAS) # resizes image in-place keeps ratio
      img = img.resize((128,128), Image.ANTIALIAS) # resizes image without ratio
      img = np.array(img)

      if img.shape == (128, 128, 3):
        # Add the numpy image to matrix with all data
        X_data.append (img)
        Y_data.append (i)
        
    i += 1
X = np.array(X_data)
Y = np.array(Y_data)
# Print shapes to see if they are correct
print(X.shape)
print(Y.shape)
from keras.utils.np_utils import to_categorical
X = X.astype('float32') / 255.0
y_cat = to_categorical(Y_data, len(boat_types))
boats = []
number_of_boats = []

path, dirs, files = next(os.walk("../input/boats"))  

for dir in dirs:
  path2, dirs2, files2 = next(os.walk("../input/boats/" + dir))  
  boats.append(dir)
  number_of_boats.append(len(files2))

 
df = pd.DataFrame({'Boat Types':boats, 'N':number_of_boats})
df = df.sort_values(['N'], ascending=False)

df_actual = df.set_index('Boat Types')
df_actual = df_actual.loc[boat_types]
df_actual = df_actual.sort_values(['N'], ascending=False)
fig, axes = plt.subplots(2,2, figsize=(14,14))  # 1 row, 2 columns
df.plot('Boat Types', ax=axes[0,0], kind='bar', legend=False, color=[plt.cm.Paired(np.arange(len(df)))], width=0.95)
df_actual.plot(kind='bar', ax=axes[0,1], legend=False, color=[plt.cm.Paired(np.arange(len(df)))], width=0.95)
df.plot('Boat Types', 'N', kind='pie', labels=df['Boat Types'], ax=axes[1,0])
df_actual.plot('N', kind='pie', ax=axes[1,1], subplots=True)
plt.tight_layout()
plt.close('all')
plt.figure(figsize=(12, 12))

for i in range(16):
  # Plot the images in a 4x4 grid
  plt.subplot(4, 4, i+1)

  # Plot image [i]
  plt.imshow(X[i])
  
  # Turn off axis lines
  cur_axes = plt.gca()
  cur_axes.axes.get_xaxis().set_visible(False)
  cur_axes.axes.get_yaxis().set_visible(False)
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
import keras.backend as K
def load_CNN(output_size):
  K.clear_session()
  model = Sequential()
  model.add(Conv2D(128, (5, 5),
               input_shape=(128, 128, 3),
               activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  #model.add(BatchNormalization())

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  #model.add(BatchNormalization())

  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  #model.add(BatchNormalization())

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dense(output_size, activation='softmax'))
  return model
early_stop_loss = EarlyStopping(monitor='loss', patience=3, verbose=1)
early_stop_val_acc = EarlyStopping(monitor='val_acc', patience=3, verbose=1)
model_callbacks=[early_stop_loss, early_stop_val_acc]
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.layers import Activation, Dense
from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD, Adam, Adagrad, RMSprop
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)
print("The model has " + str(len(X_train)) + " inputs")
model = load_CNN(4)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0005),
              metrics=['accuracy'])

weights = model.get_weights()
batch_sizes = [4, 8, 16, 32, 64, 128]

histories_acc = []
histories_val = []
for batch_size in batch_sizes:
  model.set_weights(weights)
  h = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=25,
                verbose=0,
                callbacks=[early_stop_loss],
                shuffle=True,
                validation_data=(X_test, y_test))

  histories_acc.append(h.history['acc'])
  histories_val.append(h.history['val_acc'])
histories_acc = np.array(histories_acc)
histories_val = np.array(histories_val)
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
lrsHistories_acc = []
lrsHistories_val = []

for lr in learning_rates:

    model=load_CNN(4)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, 
                  batch_size=16, 
                  epochs=25, 
                  verbose=0, 
                  callbacks=[early_stop_loss],
                  shuffle=True,
                  validation_data=(X_test, y_test))

    lrsHistories_acc.append(h.history['acc'])
    lrsHistories_val.append(h.history['val_acc'])
lrsHistories_acc = np.array(lrsHistories_acc)
lrsHistories_val = np.array(lrsHistories_val)
optimizers = ['SGD(lr=0.0001)',
              'SGD(lr=0.0001, momentum=0.3)',
              'SGD(lr=0.0001, momentum=0.3, nesterov=True)',  
              'Adam(lr=0.0001)',
              'Adagrad(lr=0.0001)',
              'RMSprop(lr=0.0001)']

optimizeList_acc = []
optimizeList_val = []

for opt_name in optimizers:

    model=load_CNN(4)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=eval(opt_name),
                  metrics=['accuracy'])
    h = model.fit(X_train, y_train, 
                  batch_size=16, 
                  epochs=25, 
                  verbose=0, 
                  callbacks=[early_stop_loss],
                  shuffle=True,
                  validation_data=(X_test, y_test))

    optimizeList_acc.append(h.history['acc'])
    optimizeList_val.append(h.history['val_acc'])
optimizeList_acc = np.array(optimizeList_acc)
optimizeList_val = np.array(optimizeList_val)
import random
image_number = random.randint(0,len(X_train))
print(image_number)
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_train[image_number].reshape(1,128,128,3))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(12,12))
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)    
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
plt.figure(figsize=(8, 8))
plt.imshow(X_train[image_number])
display_activation(activations, 4, 4, 4)
acc_lr = lrsHistories_acc
val_lr = lrsHistories_val

acc_bs = histories_acc
val_bs = histories_val

acc_opt = optimizeList_acc
val_opt = optimizeList_val
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for b in acc_bs:
  plt.plot(b)
  plt.title('Accuracy for different batch sizes')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['4', '8', '16', '32', '64', '128'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

plt.subplot(212)
for z in val_bs:
  plt.plot(z)
  plt.title('Validation accuracy for different batch sizes')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['8', '16', '32', '64', '128', '256'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for x in acc_lr:
  plt.plot(x)
  plt.title('Accuracy for different learning rates')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['0.01', '0.005', '0.001', '0.0005', '0.0001'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

plt.subplot(212)
for y in val_lr:
  plt.plot(y)
  plt.title('Validation accuracy for different learning rates')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['0.01', '0.005', '0.001', '0.0005', '0.0001'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for x in acc_opt:
  plt.plot(x)
  plt.title('Accuracy for different optimizers')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['SGD(lr=0.001)',
              'SGD(lr=0.001, momentum=0.3)',
              'SGD(lr=0.001, momentum=0.3, nesterov=True)',  
              'Adam(lr=0.001)',
              'Adagrad(lr=0.001)',
              'RMSprop(lr=0.001)'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

plt.subplot(212)
for y in val_opt:
  plt.plot(y)
  plt.title('Validation accuracy for different optimizers')
  #plt.ylim(0, 1.01)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['SGD(lr=0.001)',
              'SGD(lr=0.001, momentum=0.3)',
              'SGD(lr=0.001, momentum=0.3, nesterov=True)',  
              'Adam(lr=0.001)',
              'Adagrad(lr=0.001)',
              'RMSprop(lr=0.001)'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top
from sklearn.metrics import confusion_matrix
import seaborn as sns
model=load_CNN(4)
    
model.compile(loss='binary_crossentropy',
              optimizer=Adagrad(0.0005),
              metrics=['accuracy'])
h = model.fit(X_train, y_train, 
              batch_size=8, 
              epochs=25, 
              verbose=0, 
              callbacks=[early_stop_loss],
              shuffle=True,
              validation_data=(X_test, y_test))
y_pred = model.predict(X_test) 
y_pred_classes = np.argmax(y_pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1)
plt.plot(h.history['acc'], label='accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['acc',
            'val_acc'], loc='best')
con_matrix = confusion_matrix(y_true, y_pred_classes, labels=[0,1,2,3]) 

plt.figure(figsize=(10,10))
plt.title('Prediction of boat types')
sns.heatmap(con_matrix, annot=True, fmt="d", linewidths=.5)
# Create a data generation variable with characteristics
datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
train_generator = datagen.flow(X_train, y_train, batch_size=256)
validation_generator = datagen.flow(X_test, y_test, batch_size=256)
plt.close('all')
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=16):
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(X_batch[i])
        
        # Turn off axis lines
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
    break
model = load_CNN(4)
    
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

weights_aug = model.get_weights()
epochs = 25
batch_sizes = [8, 16, 32, 64, 128, 256]

data_augmentation_bs_acc = []
data_augmentation_bs_val = []
for batch_size in batch_sizes:
  
  train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
  validation_generator = datagen.flow(X_test, y_test, batch_size=batch_size)
  
  model.set_weights(weights_aug)
  h = model.fit_generator(train_generator,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=10,
                    callbacks=[early_stop_loss],
                    verbose=0)

  data_augmentation_bs_acc.append(h.history['acc'])
  data_augmentation_bs_val.append(h.history['val_acc'])
data_augmentation_bs_acc = np.array(data_augmentation_bs_acc)
data_augmentation_bs_val = np.array(data_augmentation_bs_val)
data_augmentation_lr_acc = []
data_augmentation_lr_val = []

train_generator = datagen.flow(X_train, y_train, batch_size=16)
validation_generator = datagen.flow(X_test, y_test, batch_size=16)

for lr in learning_rates:

    model=load_CNN(4)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    h = model.fit_generator(train_generator,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=epochs,
                    validation_data=validation_generator,
                    callbacks=[early_stop_loss],
                    verbose=0,
                    validation_steps=10)

    data_augmentation_lr_acc.append(h.history['acc'])
    data_augmentation_lr_val.append(h.history['val_acc'])
data_augmentation_lr_acc = np.array(data_augmentation_lr_acc)
data_augmentation_lr_val = np.array(data_augmentation_lr_val)
data_augmentation_opt_acc = []
data_augmentation_opt_val = []

for opt_name in optimizers:

    model=load_CNN(4)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=eval(opt_name),
                  metrics=['accuracy'])
    h = model.fit_generator(train_generator,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=epochs,
                    validation_data=validation_generator,
                    callbacks=[early_stop_loss],
                    verbose=0,
                    validation_steps=10)

    data_augmentation_opt_acc.append(h.history['acc'])
    data_augmentation_opt_val.append(h.history['val_acc'])
data_augmentation_opt_acc = np.array(data_augmentation_opt_acc)
data_augmentation_opt_val = np.array(data_augmentation_opt_val)
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for b in data_augmentation_bs_acc:
  plt.plot(b)
  plt.title('Accuracy for different batch sizes')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['8', '16', '32', '64', '128', '256'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

plt.subplot(212)
for z in data_augmentation_bs_val:
  plt.plot(z)
  plt.title('Validation accuracy for different batch sizes')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['8', '16', '32', '64', '128', '256'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for x in data_augmentation_lr_acc:
  plt.plot(x)
  plt.title('Accuracy for different learning rates')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['0.01', '0.005', '0.001', '0.0005', '0.0001'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom, 1.01)   # set the ylim to bottom, top

plt.subplot(212)
for y in data_augmentation_lr_val:
  plt.plot(y)
  plt.title('Validation accuracy for different learning rates')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['0.01', '0.005', '0.001', '0.0005', '0.0001'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top
plt.subplots(2,1,figsize=(12,12))
plt.subplot(211)
for x in data_augmentation_opt_acc:
  plt.plot(x)
  plt.title('Data augmentation accuracy for different optimizers')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['SGD(lr=0.001)',
              'SGD(lr=0.001, momentum=0.3)',
              'SGD(lr=0.001, momentum=0.3, nesterov=True)',  
              'Adam(lr=0.001)',
              'Adagrad(lr=0.001)',
              'RMSprop(lr=0.001)'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim(bottom-0.01, 1.01)   # set the ylim to bottom, top

plt.subplot(212)
for y in data_augmentation_opt_val:
  plt.plot(y)
  plt.title('Data augmentation validation accuracy for different optimizers')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(['SGD(lr=0.001)',
              'SGD(lr=0.001, momentum=0.3)',
              'SGD(lr=0.001, momentum=0.3, nesterov=True)',  
              'Adam(lr=0.001)',
              'Adagrad(lr=0.001)',
              'RMSprop(lr=0.001)'], loc='best')
  bottom, top = plt.ylim()  # return the current ylim
  plt.ylim((bottom - bottom*0.03), (top + top*0.03))   # set the ylim to bottom, top
from sklearn.metrics import confusion_matrix
import seaborn as sns
model=load_CNN(4)
    
model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.0002),
              metrics=['accuracy'])
h = model.fit_generator(train_generator,
                steps_per_epoch=len(X_train) / 32,
                epochs=150,
                validation_data=validation_generator,
                #callbacks=[early_stop_loss],
                verbose=0,
                validation_steps=10)
y_pred = model.predict(X_test) 
y_pred_classes = np.argmax(y_pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1)
plt.plot(h.history['acc'], label='accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['acc',
            'val_acc'], loc='best')

con_matrix = confusion_matrix(y_true, y_pred_classes, labels=[0,1,2,3]) 

plt.figure(figsize=(10,10))
plt.title('Prediction of boat types')
sns.heatmap(con_matrix, annot=True, fmt="d", linewidths=.5)

