!pip install tensorflow==1.14.0
import numpy as np
import matplotlib.pyplot as plt
import time


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from keras.layers import Activation, Dropout, BatchNormalization
from keras.utils import plot_model
from keras import optimizers

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current sessio
!pip install tflearn
import tflearn.datasets.oxflower17 as oxflower17
train_x, train_y = oxflower17.load_data(one_hot=True)
# Analisis de los datos 

 

 #The oxflower17 dataset consists of 1360 colour images (224 pixels high and 224 pixes width) of flowers in 17 classes, with 80 images per class. All images will be used for training. Before running the model, it will be indicated the ratio of samples that will be used for validation.


#The 17 classes are:
 
#| index | class name |
#| --- | --- |
#| 0 | Daffodil|
#| 1 | Snowdrop|
#| 2 | Daisy|    
#| 3 | ColtsFoot|										
#| 4 | Dandelion|										
#| 5 | Cowslip|
#| 6 | Buttercup|   
#| 7 | Windflower|										
#| 8 | Pansy|										
#| 9 | LilyValley|										
#|10 | Bluebell |										
#|11 | Crocus|
#|12 | Iris|										
#|13 | Tigerlily|										
#|14 | Tulip|										
#|15 | Fritillary|
#|16 | Sunflower|										       
dic = {0: 'Daffodil', 1: 'Snowdrop', 2: 'Daisy', 3: 'ColtsFoot', 4: 'Dandelion', \
       5: 'Cowslip', 6: 'Buttercup', 7: 'Windflower', 8: 'Pansy', 9:'LilyValley', \
       10: 'Bluebell', 11: 'Crocus', 12: 'Iris', 13: 'Tigerlily', 14:'Tulip', \
       15: 'Fritillary', 16: 'Sunflower'}
# Plotting the content of a sample
import matplotlib.pyplot as plt

sample = 72

plt.imshow(train_x[sample]);
print('y =',  np.squeeze(train_y[sample]))

for i in [i for i,x in enumerate(train_y[sample]) if x == 1]:
    print('')

print('y =',  i, ';', 'the sample', sample, 'corresponds to a(an)', dic[i])

print('the shape is', train_x.shape)
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
print(train_y[0])
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)
# Creating a Sequential Model and adding the layers

def architecture(batch_normalization, dropout, input_shape, activation):
    
    # Creating a sequential model
    model = Sequential()
    
    # 1st Convolutional layer
    model.add(Conv2D(filters=96, activation=activation, input_shape=input_shape,\
      kernel_size=(11,11), strides=(4,4), padding='valid', kernel_initializer='he_uniform'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')) 
    if batch_normalization: 
        model.add(BatchNormalization())  

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, activation=activation, kernel_size=(11,11), strides=(1,1), padding='valid'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    if batch_normalization: 
        model.add(BatchNormalization())  
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, activation=activation, kernel_size=(3,3), strides=(1,1), padding='valid'))
    if batch_normalization: 
        model.add(BatchNormalization())  
    
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, activation=activation, kernel_size=(3,3), strides=(1,1), padding='valid'))
    if batch_normalization: 
        model.add(BatchNormalization())   
    
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, activation=activation, kernel_size=(3,3), strides=(1,1), padding='valid'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    if batch_normalization: 
        model.add(BatchNormalization())   

    # Passing it to a dense layer
    model.add(Flatten())
    
    # 1st Dense Layer
    model.add(Dense(4096, activation=activation, input_shape=(224*224*3,), kernel_initializer = 'he_uniform'))
    # Add Dropout to prevent overfitting
    if dropout:
        model.add(Dropout(0.4))
    if batch_normalization: 
        model.add(BatchNormalization())   
    
    # 2nd Dense Layer
    model.add(Dense(4096, activation=activation, kernel_initializer = 'he_uniform'))
    model.add(Activation('relu'))
    # Add Dropout
    if dropout:
        model.add(Dropout(0.4))
    if batch_normalization: 
        model.add(BatchNormalization())   

    # 3rd Dense Layer
    model.add(Dense(1000, activation=activation, kernel_initializer = 'he_uniform' ))
    # Add Dropout
    if dropout:
        model.add(Dropout(0.4))
    if batch_normalization: 
        model.add(BatchNormalization())   

    # Output Layer
    model.add(Dense(17, activation='softmax'))
              
    return model
            
# Generating the model using the defined architecture

batch_normalization=True
dropout=True
one_image = (224, 224, 3)
activation = 'relu'

oxflower17_model = architecture(batch_normalization, dropout, one_image, activation)

plot_model(oxflower17_model, to_file='oxflower17_model.png', show_shapes=True, show_layer_names=True)

oxflower17_model.summary()
#Compiling the model using Adam as optimizer

lr = 0.001  # Learning rate

oxflower17_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], \
optimizer=optimizers.Adam(lr,beta_1=0.9, beta_2=0.999, amsgrad=False))

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
start_time = time.time()

history = oxflower17_model.fit(train_x, train_y, batch_size=64, epochs=50, verbose=1, validation_split=0.2, shuffle=True)

end_time = time.time()
print("Time for training: {:10.4f}s".format(end_time - start_time))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Lr = 0.001, loss_train: 0.1894, \n loss_val: 1.5591, BatchNorm=True \n Dropout = 0.4')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.ylim(top=13)    # The instruction is used to limit the upper value of the loss function 
plt.ylim(bottom=0)  # The instruction is used to limit the lower value of the loss function
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Lr = 0.001, Acc_train: 0.9404, \n Acc_val: 0.6544 BatchNorm=True \n Dropout = 0.4')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()
