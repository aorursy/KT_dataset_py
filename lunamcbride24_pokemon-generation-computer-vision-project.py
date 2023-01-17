# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import tensorflow as tf #Import tensorflow in order to use Keras

from keras import backend as K #Used for trying to clear memory, but I could still not clear enough RAM



from tensorflow.keras.preprocessing.sequence import pad_sequences #Add padding to help the Keras Sequencing

import tensorflow.keras.layers as L #Import the layers as L for quicker typing

from tensorflow.keras.optimizers import Adam #Pull the adam optimizer for usage



from tensorflow.keras.losses import SparseCategoricalCrossentropy #Loss function being used

from sklearn.model_selection import train_test_split #Train Test Split

from tensorflow.keras.preprocessing import image #Add image handling, because I am looking at images

from PIL import Image #Pillow images, as that is the format Keras uses



from keras.models import Sequential #Sequential

from keras.layers import Conv2D, MaxPooling2D #Load 2d layers

from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization #Load important layers



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pokemon = pd.read_csv("../input/pokemon-images-and-types/pokemon.csv") #Load the Pokemon

pokemon.head() #Take a peek at the pokemon
print(pokemon.isnull().any()) #Print if there are any nulls
#DualTypeMaker: Makes a pokemon a dual type of the same type if their second type is null

#Input: The two types it is

#Output: the type type2 will become

def dualTypeMaker(type1, type2):

    if pd.isnull(type2): #If the second type is null

        return type1 #Make the second type the same as the first

    return type2 #Leave the already dual type alone



pokemon["Type2"] = pokemon.apply(lambda x: dualTypeMaker(x["Type1"], x["Type2"]), axis = 1) #Set the second type of the pokemon

pokemon.head() #Take a peek at the dataset
gen = [] #A list to hold the generation numbers

length = len(pokemon) #Get the number of pokemon in the dataset



#A for loop to put the generation into the list, going based on the list above -1 because the index starts at 0

for i in range(0, length):

    if i<=150:

        gen.append(1)

    elif i<=250:

        gen.append(2)

    elif i<=385:

        gen.append(3)

    elif i<=492:

        gen.append(4)

    elif i<=648:

        gen.append(5)

    elif i<=720:

        gen.append(6)

    else:

        gen.append(7)

        

#Get the pokemon that boarder each generation so we can make sure there were no mistakes

boarderPokemon = ["mew", "chikorita", "celebi", "treecko", "deoxys-normal", "turtwig", "arceus", "victini", "genesect", "chespin",

                 "volcanion", "rowlet", "melmetal"]

        

pokemon["Gen"] = gen #Insert the generation numbers into the dataset



pokemon["Gen"] = pokemon["Gen"].astype(str) #Change the gen to strings



print("Name, Generation\n") #Print the format of the pokemon prints



#For each pokemon, find the boarder pokemon and print their name and generation number

for i in range(0, length):

    if pokemon["Name"][i] in boarderPokemon: #If the pokemon is a boarder pokemon

        print(pokemon["Name"][i], ", ", pokemon["Gen"][i]) #Print the name and generation of that pokemon



pokemon.head() #Take a peek at the dataset
pokemonImages = [] #A list to hold the image data, which will then be put into the dataframe

datagen = image.ImageDataGenerator()



#For each pokemon, get and convert their image

for i in range(0, length):

    pokemonName = pokemon["Name"][i] #Get the pokemon name

    

    try: #Try to load the image assuming it is a png

        path = "../input/pokemon-images-and-types/images/images/{}.png".format(pokemonName) #Get the image based on the pokemon

        img = image.load_img(path, target_size = None, interpolation = "nearest") #Load the image of the pokemon

    except: #Catch the issue if the image is not a png, try a jpg

        path = "../input/pokemon-images-and-types/images/images/{}.jpg".format(pokemonName) #Get the image based on the pokemon

        img = image.load_img(path, target_size = None, interpolation = "nearest") #Load the image of the pokemon

    

    pokemonImages.append(path) #Append the image to the image list

    

pokemon["Image"] = pokemonImages #Put the image list into a new column "Image"

pokemon.head() #Take a peek at the dataset
#Split the images and gens into train and test

imageTrain, imageTest, genTrain, genTest = train_test_split(pokemon["Image"], pokemon["Gen"], test_size = 0.33)
trainDf = pd.DataFrame(imageTrain) #Put the training images into a dataframe

trainDf["Gen"] = genTrain #Add the generation to the dataframe

train = datagen.flow_from_dataframe(trainDf, x_col = "Image", y_col = "Gen") #Make a flow variable for keras
testDf = pd.DataFrame(imageTest) #Put the testing images into a dataframe

testDf["Gen"] = genTest #Add the generation to the dataframe

test = datagen.flow_from_dataframe(testDf, x_col = "Image", y_col = "Gen") #Make a flow variable for keras
tf.keras.backend.clear_session() #Clear any previous model building



epoch = 2 #Number of runs through the data

batchSize = 8 #The number of items in each batch

width = 256 #The width of the images

height = 256 #The height of the images

channels = 3 #The number of channels (RGB)



model = Sequential() #Add a sequential to the model

model.add(L.Lambda(lambda x: x, input_shape = (width, height, channels))) #Put the input into a lambda, because it would not work for some reason in the Conv2D

model.add(Conv2D(32, (3, 3))) #Add a convolutional image layer

model.add(BatchNormalization()) #Normalize the data

model.add(Activation('relu')) #Make the activation relu to discourage negative units

model.add(MaxPooling2D(pool_size=(2, 2))) #Max pool the data to keep the most important characteristics



model.add(Conv2D(32, (3, 3))) #Add a convolutional image layer

model.add(Activation('relu')) #Make the activation relu to discourage negative units

model.add(MaxPooling2D(pool_size=(2, 2))) #Max pool the data to keep the most important characteristics



model.add(Conv2D(64, (3, 3))) #Add a bigger convolutional image layer, layering activations

model.add(Activation('relu')) #Make the activation relu to discourage negative units

model.add(MaxPooling2D(pool_size=(2, 2))) #Max pool the data to keep the most important characteristics



model.add(Flatten()) #Make the layers flat to apply the characteristics into one slot

model.add(Dense(64)) #Add a dense layer to track activation

model.add(Activation('relu')) #Make the activation relu to discourage negative units

model.add(Dropout(0.2)) #Have a 0.2 dropout to prevent overfitting

model.add(Dense(1)) #Add another dense layer to finish the lot

model.add(Activation('sigmoid')) #Make the activation sigmoid 



model.compile(loss = 'binary_crossentropy', #Make the loss binary to fit with the sigmoid endpoint

              optimizer = 'rmsprop', #Use root mean squared prop to optimize (adam came to similar results at much slower speeds)

              metrics = ['accuracy']) #Track the accuracy of the model



history = model.fit_generator(train, epochs = epoch) #Fit the model to the data
loss, accuracy = model.evaluate(test) #Get the loss and Accuracy based on the tests



#Print the loss and accuracy

print("Test Loss: ", loss)

print("Test Accuracy: ", accuracy)