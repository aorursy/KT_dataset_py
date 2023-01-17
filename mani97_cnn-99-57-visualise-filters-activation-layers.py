import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Basic

import numpy as np

import pandas as pd



# Set random seed

np.random.seed(1)



# Plottiing

import matplotlib.pyplot as plt

import seaborn as sns



# Label encoding

from keras.utils.np_utils import to_categorical



# Train val split

from sklearn.model_selection import train_test_split



# Modelling

from keras import models

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping, ModelCheckpoint



# Evaluation

from sklearn.metrics import confusion_matrix
X_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print('Shape of the training data: ', X_train.shape)

print('Shape of the test data: ', X_test.shape)
X_train.head(1)
# Extracting out y_train



y_train = X_train['label']



# Dropping the label from X_train



X_train.drop(labels = ['label'], axis=1, inplace=True)
#---- CHECK FOR NULL VALUES ----#



print('Null values in training data: ',X_train.isna().any().sum() == 'True')

print('Null values in test data: ',X_test.isna().any().sum() == 'True')





#---- NORMALIZATION ----#



X_train = X_train / 255.0

X_test = X_test / 255.0





#---- VISUALISE IMAGES ----#



x = X_train[:10]

x = x.values.reshape(x.shape[0], 28, 28)



for i in range(10):

    plt.subplot(1,10,i+1)

    plt.axis('off')

    plt.imshow(x[i],cmap=plt.cm.binary)



    

#---- RESHAPE DATA ----#



X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)





#---- CLASS IMBALANCE ----#



print('\nCount for each class')

print(y_train.value_counts())

sns.countplot(y_train)

#---- LABEL ENCODING ----#



y_train = to_categorical(y_train, num_classes=10)





#---- TRAIN-VALIDATION SPLIT ----#



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1)
#---- CNN MODEL ----#





model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



model.summary()
#---- OPTIMIZER ----#



optimizer = Adam(learning_rate=0.001, epsilon=1e-07)



#---- COMPILING ----#



model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])



#---- LEARNING RaTE REDUCTION ----#



earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='auto')

mcp = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='auto')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='auto')



datagen = ImageDataGenerator(

    rotation_range=5,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=5,

    zoom_range=0.1)



datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),

                              epochs = 100, 

                              validation_data = (X_val,y_val),

                              verbose = 1, 

                              steps_per_epoch=X_train.shape[0]//64, 

                             callbacks = [earlyStopping, mcp, reduce_lr_loss])
plt.figure(figsize=(6, 4))

plt.plot(history.history['loss'], color='b', label="Training loss")

plt.plot(history.history['val_loss'], color='r', label="validation loss")

plt.legend(loc='best')

plt.title('Training loss and Validation Loss across epochs')

plt.show()



plt.figure(figsize=(6, 4))

plt.plot(history.history['accuracy'], color='b', label="Training accuracy")

plt.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

plt.legend(loc='best')

plt.title('Training loss and Validation Accuracy across epochs')

plt.show()
y_pred = model.predict(X_val)

y_pred = np.argmax(y_pred, axis=1)

y_val_test = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_val_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g', cbar=False)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion Matrix for predicted and true labels')

plt.show()
model.load_weights(filepath = '.mdl_wts.hdf5')
scores = model.evaluate(X_val, y_val, callbacks = [earlyStopping, mcp, reduce_lr_loss])
rows = 2

cols = 10



plt.figure(figsize=(16, 4))

for i in range(0,20):

    ax = plt.subplot(rows, cols, i+1)

    plt.axis('off')

    ax.set_xticks([])

    ax.set_yticks([])

    plt.imshow(X_val[i].reshape(28, 28), cmap=plt.cm.binary)

    plt.title('Predicted: {}\nTrue: {}'.format(y_pred[i], y_val_test[i]))

#---- FINDING OUT THE IMAGES THAT WERE INCORRECTLY PREDICTED ----#



ls = np.array(y_pred - y_val_test) # Subtract true values from predicted --> All non-zero results were incorrectly predicted



nonzero_pred = np.nonzero(ls)[0]



rows = 3

cols = (int)(np.ceil(nonzero_pred.size/3))



plt.figure(figsize=(10, 7))

for i in range(0, nonzero_pred.size):

    ax = plt.subplot(rows, cols, i+1)

    plt.axis('off')

    ax.set_xticks([])

    ax.set_yticks([])

    plt.imshow(X_val[nonzero_pred[i]].reshape(28, 28), cmap=plt.cm.binary)

    plt.title('Predicted: {}\nTrue: {}'.format(y_pred[nonzero_pred[i]], y_val_test[nonzero_pred[i]]))
test = X_test[0:10,:,:,:]

test_pred = np.argmax(model.predict(X_test), axis=1)



plt.figure(figsize=(18,1))

for i in range(0,10):

    ax = plt.subplot(1, 10, i+1)

    plt.axis('off')

    ax.set_xticks([])

    ax.set_yticks([])

    plt.imshow(test[i].reshape(28, 28), cmap=plt.cm.binary)

    plt.title('Predicted: {}'.format(test_pred[i]))
img_tensor = X_test[5].reshape(-1, 28, 28, 1)
for layer in model.layers:

    if'conv' in layer.name:

        filters, biases = layer.get_weights()

        

        print('Layer: ', layer.name, filters.shape)

        

        f_min, f_max = filters.min(), filters.max()

        filters = (filters - f_min) / (f_max - f_min)

        

        print('Filter size: (', filters.shape[0], ',', filters.shape[1], ')')

        print('Channels in this layer: ', filters.shape[2])

        print('Number of filters: ', filters.shape[3])

        

        count = 1

        plt.figure(figsize = (18, 4))

        

        # Plotting the first channel of every filter

        for i in range(filters.shape[3]):

            

            ax= plt.subplot(4, filters.shape[3]/4, count)

            ax.set_xticks([])

            ax.set_yticks([])

            plt.imshow(filters[:,:,0, i], cmap=plt.cm.binary)

            count+=1

            

        plt.show()

        
# Extract outputs of top 8 layers

layer_outputs = [layer.output for layer in model.layers[0:8]]



# Create a model that will return these outputs given the model input

activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]

print(first_layer_activation.shape)
ax = plt.subplot(1, 2, 1)

ax.set_xticks([])

ax.set_yticks([])

plt.imshow(first_layer_activation[0,:,:,0], cmap = plt.cm.binary)

plt.title('1st Filter')



ax = plt.subplot(1, 2, 2)

ax.set_xticks([])

ax.set_yticks([])

plt.imshow(first_layer_activation[0,:,:,22], cmap = plt.cm.binary)

plt.title('23rd Filter')

    

plt.show()
layer_names = []

for layer in model.layers[:8]:

    layer_names.append(layer.name)



for activation_layer, layer_name in zip(activations, layer_names):



    n_features = activation_layer.shape[3]

    feat_per_row = 16

    rows = n_features//feat_per_row

    size = activation_layer.shape[1]

    

    print(layer_name)

    plt.figure(figsize=(20, rows))

    for i in range(n_features):

        ax = plt.subplot(rows, feat_per_row, i+1)

        ax.set_xticks([])

        ax.set_yticks([])

        plt.imshow(activation_layer[:,:,:,i].reshape(size, size))#, cmap = plt.cm.binary)

    

    plt.show()
# predict the results



results = np.argmax(model.predict(X_test), axis=1)



results = pd.Series(results, name = "Label")



results.head(2)
# Submission 



submission = pd.concat([pd.Series(range(1, 28001), name = "ImageId"), results], axis = 1)



submission.head(2)
submission.to_csv("CNN_MNIST_results.csv",index=False)