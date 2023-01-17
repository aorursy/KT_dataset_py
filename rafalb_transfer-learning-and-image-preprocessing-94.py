# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from pandas import HDFStore, DataFrame
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py # library for processing h5 file
from keras.utils import to_categorical  
from sklearn.model_selection import train_test_split

np.random.seed(105208) # for reproducibility
# import basic Keras function  
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.initializers import Initializer
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator  #Easy to use api for generate images on the run
from keras.applications.xception import Xception  # Pre trained model


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
%matplotlib inline
# read and look on head of images
df = pd.read_csv("../input/traditional-decor-patterns/decor.csv")
df.head()
# create new column with Country-pattern it can by useful for recogintion task
df['country_decor'] = df[['country', 'decor']].apply(lambda x: '-'.join(x), axis=1)
df['type_country_decor'] = pd.factorize(df.country_decor)[0]

country = df.country.unique()
decor = df.decor.unique()
typeOf = df.type.unique()
productDf = df[df['type'] == 'product']
coutryDecor = df.country_decor.unique()

def lookAt(data):
    return ", ".join(str(x) for x in data)
        

print("Country: " + lookAt(country))
print("Decor: " + lookAt(decor))
print("Type: " + lookAt(typeOf))
print("Decor with Country: " + ", ".join(str(x) for x in coutryDecor))
def createHist(data, xlabel = None, ylabel = None, title = None, grid = True, size = (5, 5)):
    plt.figure(figsize=size)
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
    
createHist(df.country, xlabel = 'Country', ylabel = "Count", title = "Country Distribution", grid = True)
createHist(df.decor, xlabel = 'Decor', ylabel = "Count", title = "Decor Distribution", grid = True, size = (12, 10))
createHist(df.country_decor, xlabel = 'Country with decor', ylabel = "Count", title = "Decor grouped by coutry", grid = True, size = (20, 10))

# read h5 file 
f = h5py.File('../input/traditional-decor-patterns/DecorColorImages.h5', 'r')
keys = list(f.keys())
keys
images = np.array(f[keys[2]])

fig = plt.figure(figsize=(10, 10))
for idx in range(25):
    num = 100
    if(idx % 5 == 0):
        num += 50
    plt.subplot(5,5, idx + 1)
    plt.imshow(images[num + idx])
    
plt.tight_layout()
# Create tensors and targets
countries = np.array(f[keys[0]])
decors = np.array(f[keys[1]])
types = np.array(f[keys[3]])

print ('Country shape:', countries.shape)
print ('Decor shape', decors.shape)
print ('Image shape:', images.shape)
print ('Type shape', types.shape)
# Normalize the images
images = images.astype('float32') / 255
cat_countries = to_categorical(np.array(countries-1), 4)
cat_decors = to_categorical(np.array(decors-1), 7)
targets = np.concatenate((cat_countries, cat_decors), axis=1)
concatTargets = np.concatenate((countries, decors))

cat_countries.shape, cat_decors.shape, targets.shape
img_rows, img_cols = 150, 150
X_train, X_test, y_train, y_test = train_test_split(images, cat_countries, test_size=0.2, random_state=42)
input_shape = (img_rows, img_cols, 3)
num_classes = y_test.shape[1]
# draw learing curve to avoid overfitting
def draw_learning_curve(history, key='acc', ylim=(0, 1.01)):
    plt.figure(figsize=(15,15))
    plt.plot(history.history[key])
    plt.plot(history.history['val_' + key])
    plt.title('Learning Curve')
    plt.ylabel(key.title())
    plt.xlabel('Epoch')
    plt.ylim(ylim)
    plt.legend(['train', 'test'], loc='best')
    plt.show()
# create CNN model and first check only on our data
def get_simple_cnn():
    return Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPool2D(pool_size=(2, 2)),
      
        Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer = 'glorot_normal'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.5),
        
        Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer = 'glorot_normal'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.5),
        
        Flatten(), #<= bridge between conv layers and full connected layers
        
        Dense(128, activation='relu', kernel_initializer = 'glorot_normal'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])

get_simple_cnn().summary()
def trainModel(model, X_train, y_train, X_test, y_test, batch_size, epochs, optimizer = 'Adam'):
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("Network Error: %.2f%%" % (100-score[1]*100))
    draw_learning_curve(history, 'acc')
    return history
trainModel(get_simple_cnn(), X_train, y_train, X_test, y_test, 32, 50, optimizer = 'Adam')
# Make more data with Image Generator api from Keras. 
# Images generate on the run so we don't have to save any of them.
datagen = ImageDataGenerator(
        featurewise_center=False,
        rotation_range=60,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2, 
        height_shift_range=0.2)

def modelWithGenData(model, datagen, X_train, y_train, X_test, y_test, batch_size = 32, steps_per_epoch = len(X_train), epochs = 20):

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # create more data on the run
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=(X_test, y_test))
    model.save('CNN_model.h5')

modelWithGenData(get_simple_cnn(), datagen, X_train, y_train, X_test, y_test, batch_size = 32, epochs = 50)
# reshape images from (150 x 150 x 3) to (197 x 197 x 3) 
transferImage = np.zeros((485, 197, 197, 3))
transferImage[:images.shape[0], :images.shape[1], :images.shape[2], :images.shape[3]] = images
#Generate new X, y test / train set
X_train, X_test, y_train, y_test = train_test_split(transferImage, targets, test_size=0.25, random_state=42)
num_classes = y_test.shape[1]
# Base model with Transfer Learning 
baseModel = Xception(weights="../input/keras-pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5", 
                        include_top=False,
                       input_shape = (197, 197, 3))
for layer in baseModel.layers:
    layer.trainable = False

transferModel = Sequential([
    baseModel,
    
    Flatten(), #<= bridge between conv layers and full connected layers
        
    Dense(128, activation='relu'),
    Dropout(0.7),
    Dense(num_classes, activation='sigmoid')
    
])

optimizer = Adam(0.0005, decay=0.0005)
transferModel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = transferModel.fit(X_train, y_train,
          batch_size=8,
          epochs=5,
          verbose=1,
          validation_data=(X_test, y_test))
#unFreeze 10 layers 
for layer in baseModel.layers[-10:]:
    layer.trainable = True

for it, layer in enumerate(baseModel.layers):
    print(it, layer.name, layer.trainable)
# Images generate on the run so we don't have to save any of them.
datagen = ImageDataGenerator(
        featurewise_center=False,
        rotation_range=60,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2, 
        height_shift_range=0.2)

datagen.fit(X_train)

transferModel = Sequential([
    baseModel,
    
    Flatten(), #<= bridge between conv layers and full connected layers
        
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(num_classes, activation='sigmoid')
    
])

optimizer = Adam(0.00001, decay=0.00001, amsgrad=True)
transferModel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = transferModel.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
                        steps_per_epoch=len(X_train),
                        epochs=15,
                        validation_data=(X_test, y_test))

transferModel.save('general_weights.h5')
draw_learning_curve(history)
#load pretrained weights
from keras.models import load_model

new_model = load_model('general_weights.h5')
def predictImage(imagePath, fromData = True):
    # split path for image
    data = df[df.file == imagePath.split("/")[-1]]
    index = int(data.index[0])
    img = images[index]
    plt.imshow(img)
    # reshape images from (150 x 150 x 3) to (197 x 197 x 3) 
    reshapeImage = np.zeros((197, 197, 3))
    reshapeImage[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    reshapeImage = np.reshape(reshapeImage, (1, 197, 197, 3))
    #predict Value for specyfic Image
    predict = new_model.predict(reshapeImage)
    predictIndex = predict.argsort()[0][-2:]
    #print answer
    print("Predict Country: " + country[min(predictIndex)] + ", predict decor: " + decor[max(predictIndex) - 4])
    print("Real Country: " + data.country.to_string(index = False) + ", real decor: " + data.decor.to_string(index = False))
    print("Type of Image: " + data.type.to_string(index = False))
    countryTrue = data.country.to_string(index = False) == country[min(predictIndex)]
    decorTrue = decor[max(predictIndex) - 4] == data.decor.to_string(index = False)
    print("Predict of country: " + str(countryTrue) + '. Predict of decor: ' + str(decorTrue))

#check our model
path = '../input/traditional-decor-patterns/decor/01_01_2_041.png'
predictImage(path)
