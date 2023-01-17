

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



from PIL import Image

from keras import layers, models, optimizers

from keras.preprocessing.image import ImageDataGenerator



def display_grid(data, path, w =10, h =10, columns = 4, rows = 5):

    fig=plt.figure(figsize=(12, 8))

    for i in range(1, columns*rows +1):

        file = data[i]

        file = os.path.join(path, file)

        img = Image.open(file)

        fig.add_subplot(rows, columns, i)

        imshow(img)

    plt.show()

    

def plot_results(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(1, len(acc) + 1)



    plt.figure(figsize = (24, 6))

    plt.subplot(1,2,1)

    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')

    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')

    plt.grid(True)

    plt.legend()

    plt.xlabel('Epoch')

    





    plt.subplot(1,2,2)

    plt.plot(epochs, loss, 'b', label = 'Training Loss')

    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')

    plt.grid(True)

    plt.legend()

    plt.xlabel('Epoch')

    plt.show()

 

def get_best_epcoh(history):

    valid_acc = history.history['val_acc']

    best_epoch = valid_acc.index(max(valid_acc)) + 1

    best_acc =  max(valid_acc)

    print('Best Validation Accuracy Score {:0.5f}, is for epoch {}'.format( best_acc, best_epoch))

    return best_epoch
base_dir = '/kaggle/input/cars-wagonr-swift/data/'

train_swift = os.listdir(os.path.join(base_dir, 'train/swift') )

val_swift  = os.listdir(os.path.join(base_dir, 'validation/swift') )

test_swift  =  os.listdir(os.path.join(base_dir, 'test/swift') )

print('Instances for Class Swift: Train {}, Validation {} Test {}'.format(len(train_swift), len(val_swift), len(test_swift)))
#Sanity checks: no overlaping bteween train test and validation sets

val_train = [x for x in val_swift if x in train_swift]

test_train = [x for x in test_swift if x in train_swift]

val_test =  [x for x in test_swift if x in val_swift]

len(val_train), len(test_train), len(val_test)
display_grid(data = train_swift, path = os.path.join(base_dir, 'train/swift'), w =10, h =10, columns = 8, rows = 5)
train_wr = os.listdir(os.path.join(base_dir, 'train/wagonr') )

val_wr  = os.listdir(os.path.join(base_dir, 'validation/wagonr') )

test_wr  =  os.listdir(os.path.join(base_dir, 'test/wagonr') )

print('Instances for Class Wagonr: Train {}, Validation {} Test {}'.format(len(train_swift), len(val_swift), len(test_swift)))
#Sanity checks: no overlaping bteween train test and validation sets

val_train = [x for x in val_wr if x in train_wr]

test_train = [x for x in test_wr if x in train_wr]

val_test =  [x for x in test_wr if x in val_wr]

len(val_train), len(test_train), len(val_test)
display_grid(data = train_wr, path = os.path.join(base_dir, 'train/wagonr'), w =10, h =10, columns = 8, rows = 5)
def build_cnn(display_summary =False):

    model = models.Sequential()

    model.add( layers.Conv2D(32, (3,3),  activation= 'relu', input_shape = (150, 150, 3)) )

    model.add(layers.MaxPooling2D((2,2)))



    model.add( layers.Conv2D(64, (3,3),  activation= 'relu') )

    model.add(layers.MaxPooling2D((2,2)))



    model.add( layers.Conv2D(128, (3,3),  activation= 'relu') )

    model.add(layers.MaxPooling2D((2,2)))



    model.add( layers.Conv2D(128, (3,3),  activation= 'relu') )

    model.add(layers.MaxPooling2D((2,2)))



    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation= 'relu'))

    model.add(layers.Dense(1, activation= 'sigmoid'))



    model.compile(loss = 'binary_crossentropy',

                  optimizer = optimizers.RMSprop(lr = 1e-4),

                  metrics = ['acc']

                  )

    if display_summary:

       model.summary()

    return model


train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation' )





train_datagen = ImageDataGenerator(rescale= 1./255)

test_datagen = ImageDataGenerator(rescale= 1./255)



train_generator = train_datagen.flow_from_directory(

                                                   train_dir,              

                                                   target_size = (150,150), #Resize images to 150 X 150

                                                   batch_size  = 20,

                                                   class_mode = 'binary'

                                                   )

validation_generator = test_datagen.flow_from_directory(

                                                   validation_dir,              

                                                   target_size = (150,150), #Resize images to 150 X 150

                                                   batch_size  = 20,

                                                   class_mode = 'binary'

                                                   )



for data_batch, labels_batch, in train_generator:

    print('Data Batch shape:', data_batch.shape)

    print('Labels Batch shape:', labels_batch.shape)

    break
model = build_cnn(display_summary = True)

%%time

history = model.fit_generator(

                            train_generator,

                            steps_per_epoch = 120,  # = num_train_images/batch size(2400/20)

                            epochs = 50,

                            validation_data = validation_generator,

                            validation_steps = 40  # = num_valid_images/batch_size

                             )

model.save('cat_and_dogs.h5')
plot_results(history)

best_epoch =get_best_epcoh(history)