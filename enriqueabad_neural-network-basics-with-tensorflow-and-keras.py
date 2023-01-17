# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
!pip install py7zr

from py7zr import unpack_7zarchive
import shutil
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)
shutil.unpack_archive('/kaggle/input/cifar-10/train.7z', '/kaggle/working')

train_path = '/kaggle/working/train/'

train_list = listdir(train_path)

X_list = [0] * len(train_list)

for i_image in train_list:
    num_image = int(i_image[:-4]) -1
    imagen = Image.open(train_path + i_image)
    imagen_numpy = np.uint8(imagen)
    X_list[num_image] = imagen_numpy

X = np.array(X_list)

print(X.shape)
train_data = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
onehot_encoder = OneHotEncoder()
y = onehot_encoder.fit_transform(train_data[['label']])
y = y.toarray()
labels_df = train_data.groupby(['label']).count()
labels_matrix = y.sum(axis=0)
print(labels_df)
print(labels_matrix)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
X_train_normalized = X_train/255.0
X_train_normalized = X_train_normalized.astype('float32')

neural_network = keras.Sequential([
    keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(32,32,3)),
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

neural_network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

neural_network.fit(X_train_normalized,y_train, epochs=50)

nn_loss, nn_score = neural_network.evaluate(X_test,y_test,verbose=2)

print("The accuracy of the neural network on the test data is {0:4.3f}".format(nn_score,))
def create_neural_network(train_array_iterator,test_array_iterator,num_filters_input,num_neurons_input,num_conv_input,num_maxpool_input,overfitting,label=None,verbosity=0):

    # Cleanup inputs

    num_filters = round(num_filters_input)
    num_neurons = round(num_neurons_input)
    num_conv = round(num_conv_input)
    num_maxpool = round(num_maxpool_input)
    if overfitting == 'l1':
        regularizer = l1(0.01)
    elif overfitting == 'l2':
        regularizer = l2(0.01)
    else:
        regularizer = None

    # Build the neural network

    neural_network = Sequential()
    neural_network.add(Conv2D(num_filters,(3,3),activation='relu',input_shape=(32,32,3)))
    for i_maxpool in range(num_maxpool):
        try:
            for i_conv in range(num_conv):
                neural_network.add(Conv2D(num_filters, (3, 3), activation='relu', padding='same',
                                          kernel_regularizer=regularizer, bias_regularizer=regularizer))
            neural_network.add(MaxPooling2D(pool_size=(2,2)))
            if overfitting == 'Dropout':
                neural_network.add(Dropout(0.25))
            elif overfitting == 'Batch_Normalization':
                neural_network.add(BatchNormalization(momentum=0.99))
        except:
            break

    neural_network.add(keras.layers.Flatten())
    neural_network.add(keras.layers.Dense(num_neurons,activation='relu'))
    neural_network.add(keras.layers.Dense(round(num_neurons/2),activation='relu'))
    neural_network.add(keras.layers.Dense(10,activation='softmax'))
    neural_network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Created a neural network with the following structure:')
    print(neural_network.summary())
    
    # Callbacks

    rlr = keras.callbacks.ReduceLROnPlateau(patience=15, verbose=1)
    es = keras.callbacks.EarlyStopping(patience=35, restore_best_weights=True, verbose=1)
    print('Fitting the neural network')
    historia = neural_network.fit_generator(train_array_iterator, validation_data=test_array_iterator, epochs=50, verbose=verbosity, callbacks=[rlr,es])
    
    # Plot
    
    #plt.figure(figsize=[15,15])
    plt.figure()
    if label is None:
        plt.plot(historia.history['accuracy'], label='Train accuracy')
        plt.plot(historia.history['val_accuracy'], label='Test accuracy')
    else:
        plt.plot(historia.history['accuracy'], label='Train accuracy ({0})'.format(label))
        plt.plot(historia.history['val_accuracy'], label='Test accuracy ({0})'.format(label))
    plt.legend()
    
    return neural_network

train_preprocessor = ImageDataGenerator(rescale=1.0/255.0)
test_preprocessor = ImageDataGenerator(rescale=1.0/255.0)

train_iterator = train_preprocessor.flow(X_train,y_train,batch_size=256)
test_iterator = test_preprocessor.flow(X_test,y_test,batch_size=256)

red_neuronal = create_neural_network(train_iterator,test_iterator,64,32,2,2,None)

train_preprocessor = ImageDataGenerator(rescale=1.0/255.0)
test_preprocessor = ImageDataGenerator(rescale=1.0/255.0)

train_iterator = train_preprocessor.flow(X_train,y_train,batch_size=256)
test_iterator = test_preprocessor.flow(X_test,y_test,batch_size=256)

normalization = {}

for i_normalizer in [None,'l1','l2','Dropout','Batch_normalization']:
#    train_preprocessor = ImageDataGenerator(rescale=1.0 / 255.0)
#    test_preprocessor = ImageDataGenerator(rescale=1.0 / 255.0)
#    train_iterator = train_preprocessor.flow(X_train, y_train, batch_size=256)
#    test_iterator = test_preprocessor.flow(X_test, y_test, batch_size=256)

    this_neural_network = create_neural_network(train_iterator,test_iterator,64,32,2,2,i_normalizer,i_normalizer)

#    test_preprocessor = ImageDataGenerator(rescale=1.0 / 255.0)
#    test_iterator = test_preprocessor.flow(X_test, y_test, batch_size=256)
    nn_loss, nn_score = this_neural_network.evaluate_generator(test_iterator,verbose=2)
    normalization[i_normalizer] = nn_score

print(normalization)
test_nn = {}
test_score = {}

for i_normalizer in [None, 'Dropout']:
    for i_convolution in range(1,4):
        for i_maxpool in [1,2]:
            for i_neurons in [8,32]:
                for i_filters in [64,128]:
                    indice = '{0}, {1}, {2}, {3}, {4}'.format(i_normalizer,i_convolution,i_maxpool,i_neurons,i_filters)
                    texto = 'Creating the neural network for parameters: ' + indice
                    print(texto)
                    this_neural_network = create_neural_network(train_iterator, test_iterator, i_filters, i_neurons,
                                                                i_convolution, i_maxpool, i_normalizer, indice)
                    nn_loss, nn_score = this_neural_network.evaluate_generator(test_iterator, verbose=2)
                    print('Test score: {0}'.format(nn_score))
                    test_score[indice] = nn_score
                    test_nn[indice] = this_neural_network


best_nn = sorted(test_score.items(), key=lambda x : x[1], reverse=True)

print(best_nn)

best_nn_indices = [ x[0] for x in best_nn ]

best_5_nn_indices = best_nn_indices[0:5]
the_input = keras.Input(shape=(32,32,3))

nn_legs = []
for i_key in best_5_nn_indices:
    output_i = test_nn[i_key](the_input)
    nn_legs.append(output_i)

x = Concatenate()(nn_legs)
x = Dense(1024,activation='relu')(x)
the_output = Dense(10,activation='softmax')(x)

merged_model = Model(inputs=the_input,outputs=the_output)
merged_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rlr = keras.callbacks.ReduceLROnPlateau(patience=15, verbose=1)
es = keras.callbacks.EarlyStopping(patience=35, restore_best_weights=True, verbose=1)
merged_model.fit_generator(train_iterator, validation_data=test_iterator, epochs=50, callbacks=[rlr,es])
shutil.unpack_archive('/kaggle/input/cifar-10/test.7z', '/tmp/testfiles')
! ls /tmp/testfiles/test | wc -l
predict_preprocessing = ImageDataGenerator(rescale=1.0 / 255.0)
predict_iterator = predict_preprocessing.flow_from_directory(directory='/tmp/testfiles',target_size=(32,32),color_mode='rgb',class_mode=None,batch_size=256,shuffle=False)

prediccion = merged_model.predict_generator(predict_iterator)

label_prediction = onehot_encoder.inverse_transform(prediccion)
prediction_df = pd.DataFrame(label_prediction)

# In order to get the indices right, we must take into account that flow_from_directory reads files on a alphabetical order. This means that after '1' comes '10' and not two.
# For that reason we need to play around a little bit with the indices, to sort them alphabetically.
labels_text_sorted = sorted([ str(x) for x in range(1, prediction_df.shape[0] + 1)])
labels_as_integers = [int(x) for x in labels_text_sorted]
prediction_df.insert(0,'Indice',labels_as_integers)
prediction_df = prediction_df.sort_values(by=['Indice'])

prediction_df.to_csv('/kaggle/working/prediction.csv',
                     header=['id','label'],index=False)