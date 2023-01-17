from keras import layers
from keras import models

import numpy as np
import pandas as pd

from keras.utils import to_categorical

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

%matplotlib notebook

from keras.models import load_model

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import imageio
import random

import PIL

from keras import models
from keras import layers
from keras import optimizers
base_data_dir = "../input/galen-cats-dogs/cats_and_dogs_small/cats_and_dogs_small/"
#base_data_dir = "./cats_and_dogs_small/"
import keras 

print("keras version is:", keras.__version__)

def display_random_image(data_dir = "./"):
    '''
    display a random image from data_dir
    
    input:
    - data_dir: location of images
    
    output: for a randomly selected image
    - image pixel dimensions
    - min, max values
    
    returns:
    - PIL.Image
    '''
    
    # load a random animal
    animal_filename = random.choice(os.listdir(data_dir))

    print(animal_filename)
    animal = imageio.read(data_dir+animal_filename)
    #df =

    data = animal.get_data(0)

    # 374 x 500 x 3 colors
    print("image shape, min, max values:")
    print(data.shape)
    print(data.min())
    print(data.max())

    return(PIL.Image.fromarray(data))
display_random_image(base_data_dir + 'train/cats/')
display_random_image(base_data_dir + 'train/dogs/')
def setup_data_generators(base_data_dir = './', 
                          rescale_factor = 255, 
                          target_width = 150, 
                          target_height = 150, 
                          batch_size = 20,
                          class_mode = 'binary'):
    '''
    create keras ImageDataGenerator (see keras for further documentation)
    - train
    - validation
    - test
    
    inputs:
      base_data_dir = location of data
      rescale_factor = how much to downscale pixel values
      target_width = output image width
      target_height = output image height
      batch_size = number of images in each batch
      class_mode = number of output classes
    
    
    returns: 3 tuple of generators
    (train_generator, validation_generator, test_generator)
    '''
    
    train_dir = base_data_dir + 'train/'
    validation_dir = base_data_dir + 'validation/'
    test_dir = base_data_dir + "test/"

    # scale the data values
    train_datagen = ImageDataGenerator(rescale=1./rescale_factor)
    test_datagen = ImageDataGenerator(rescale=1./rescale_factor)

    # the data generator from data files
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(target_height, target_width),
        batch_size= batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(target_height, target_width),
        batch_size=batch_size,
        class_mode='binary')

    # the generator for unclassified test data  Note settings for class_mode and batch_size
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(target_height, target_width),
        batch_size=1, 
        class_mode = None, 
        shuffle = False)
    
    
    return(train_generator, validation_generator, test_generator)

(train_generator, validation_generator, test_generator) = setup_data_generators(base_data_dir)
# # load a prepackaged keras image network
# from keras.applications import VGG16

# conv_base = VGG16(weights='imagenet',
#                   include_top=False,
#                   input_shape=(150, 150, 3))  #input shape
conv_base = models.load_model("../input/vgg16-pretrained/VGG16.h5")
#conv_base = models.load_model("./VGG16.h5") 
conv_base.summary()
base_dir = base_data_dir

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
def extract_features(directory, sample_count):
    '''
    use pre-trained network to obtain features from images
    features can be passed into classifier
    '''
    
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
        
    for inputs_batch, labels_batch in generator:

        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
            
    return features, labels
# calculate features using pre-trained convolutional network

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
# reshape for passing to dense classifier

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
# pass features to classifier and fit

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
# plot the results

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend();
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
def print_best_validation(val_acc, val_loss):

    print('max validation accuracy', np.max(val_acc), 'occurs at', np.argmax(val_acc))
    print('min validation loss', np.min(val_loss), 'occurs at', np.argmin(val_loss))
print_best_validation(val_acc, val_loss)
def build_compile_model():
    '''
    build and compile keras convolutional model
    
    inputs:
    
    returns:
    keras convolutional model
    '''
    
    model = models.Sequential()

    # the layers that build up a hiearchy toward higher-level features
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # prepare to pass to classifier
    model.add(layers.Flatten())

    # the classifier with one hidden layer
    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    
    return(model)
model = build_compile_model()
model.summary()
# print the data batch shape

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
# load the model and history instead of training (just for speed)

# this was trained in this notebook on kaggle

#model_dir = "../input/dogs-and-cats/"
#model_dir = "./"

#from keras.models import load_model
#model = load_model(model_dir + 'cats_and_dogs_small_1.h5')

#history_dir = "../working/"
# history_dir = "./"

# history = pd.read_pickle(history_dir + "cats_and_dogs_small_1_history.pkl")
#if using kaggle or a GPU architecture, run here
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

#save the model and training history
model.save('cats_and_dogs_small_1.h5')

#also save the model as pickle to be sure
pd.to_pickle(obj = model, path = "cats_and_dogs_small_1_model.pkl")

pd.to_pickle(obj=history, path="cats_and_dogs_small_1_history.pkl")
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
print_best_validation(val_acc, val_loss)
# use this saved model or a model in memory
saved_model_name = 'cats_and_dogs_small_1.h5'
model = load_model(model_dir + saved_model_name)


test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)

# get the labels
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
outfolder = "./"

filenames=test_generator.filenames

results=pd.DataFrame({#"Filename":filenames,
                      "Predictions":predictions})

results.to_csv(outfolder + "results.csv",index=True)
def create_predictions(trained_model, test_generator, train_generator):
    '''
    create a dataframe of predictions using data from test_generator
    
    inputs:
    keras model (already trained)
    keras imagedatagenerators used during training and testing
    
    returns:
    pandas Dataframe with columns:
    Filenames: the image filename
    predictions: predicted classes
    '''
    
    test_generator.reset()
    pred=trained_model.predict_generator(test_generator,verbose=1)

    # get the labels
    predicted_class_indices=np.argmax(pred,axis=1)
    
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    
    filenames=test_generator.filenames

    resultsDF=pd.DataFrame({"Filename":filenames,
                          "Predictions":predictions})

    return(resultsDF)

def save_predictions(outfilename, predictionsDF):
    '''
    save only index and prediction to .csv file
    
    input:
    pandas Dataframe with columns:
    Filenames: the image filename
    predictions: predicted classes
    
    output:
    .csv file saved to outfilename
    '''
    
    series = predictionsDF['predictions']
    
    series.to_csv(outfilename)
predictionsDF = create_predictions(model, test_generator, train_generator)

save_predictions(outfolder + "results.csv", predictionsDF)