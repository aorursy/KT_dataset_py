from sklearn.datasets import load_files

from keras.utils import np_utils

import numpy as np

from glob import glob
#function to load the dataset

def load_dataset(path):

    data = load_files(path)

    fish_files = np.array(data['filenames'])

    fish_target = np_utils.to_categorical(np.array(data['target']), 8)

    return fish_files,fish_target
#loading the paths of training set

train_files, train_targets = load_dataset('fishImages/train')



#loading the paths of testing set

test_files, _ = load_dataset('fishImages/test')



#printing the number of samples in test and trainig sets.

print ("There are %d images in training dataset"%len(train_files))

print ("There are %d images in the training set"%len(test_files))
import matplotlib.pyplot as plt

import cv2

import seaborn as sns



sns.set(color_codes=True)



#finding the number of samles in each class

[ALB, BET, DOL, LAG, NoF, OTHER, SHARK, YFT] = sum(train_targets)





fish_count =[ALB, BET, DOL, LAG, NoF, OTHER, SHARK, YFT]



x=np.arange(8)



#plotting the barplot between name of classes and number of samples in each class 

fig, ax = plt.subplots()

plt.bar(x, fish_count)

plt.xticks(x, ('ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'))

plt.xlabel("Class Name")

plt.ylabel("Number of samples in the class")

plt.show()
#function for plotting a histogram for the color intensity of an image



def intensity_dist(path):

    #reading the image from its path

    img = cv2.imread(path)

    color = ('b','g','r')

    #calculating the number of pixels of each color

    for i, col in enumerate(color):

        histr = cv2.calcHist([img], [i], None, [256], [0,256])

        plt.plot(histr, color=col)

    print("Histogram for color Internsity of the image below:")

    

    #showing the histogram

    plt.xlabel("value of the pixel for the given channel")

    plt.ylabel("Number of pixels")

    plt.show()

    

    #showing the image

    plt.imshow(img)

    plt.show()

    height, width, channels = img.shape

    print("Size of the image - (%d , %d)"%(height,width)) 

    print("-"*100)

    

intensity_dist(train_files[56])

intensity_dist(train_files[667])

intensity_dist(train_files[660])

intensity_dist(train_files[1547])

#intensity_dist(train_files[1147])

#intensity_dist(test_files[12455])

intensity_dist(test_files[60])
from keras.preprocessing import image

from tqdm import tqdm



#converting image to tensor

def path_to_tensor(img_path):

    # loads RGB image

    img = image.load_img(img_path, target_size=(224,224))

    #convering the image to 3-D tensor with shape (224,224,3)

    x = image.img_to_array(img)

    #convert 3D tensor to 4D tensor

    return np.expand_dims(x, axis=0)



def paths_to_tensor(img_paths):

    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]

    return np.vstack(list_of_tensors)
from PIL import ImageFile                            

ImageFile.LOAD_TRUNCATED_IMAGES = True 



#preprocessing the data

test_tensors = paths_to_tensor(test_files).astype('float32')/255
train_tensors = paths_to_tensor(train_files).astype('float32')/255
#shape of the tensor

print(np.shape(train_tensors))
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout

from keras.models import Sequential



benchmark = Sequential()



# model Convolution layer

benchmark.add(Conv2D(filters=16,kernel_size=2,strides=1,activation='relu',input_shape=(224,224,3)))

# Max Pooling layer to reduce the dimensionality

benchmark.add(MaxPooling2D(pool_size=2,strides=2))

#Dropout layer, for turning off each node with the probability of 0.3

benchmark.add(Dropout(0.3))

benchmark.add(Conv2D(filters=32, kernel_size=2,strides=1,activation='relu'))

benchmark.add(Dropout(0.3))

benchmark.add(GlobalAveragePooling2D())

#A fully connected dense layer with 8 nodes (no of classes of fish)

benchmark.add(Dense(8,activation='softmax'))

benchmark.summary()
benchmark.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping





epochs = 5



#checkpointer saves the best weights.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.benchmark.hdf5', verbose=1, save_best_only=True)



benchmark.fit(train_tensors, train_targets, batch_size=20, epochs=epochs, callbacks=[checkpointer], validation_split=0.2, verbose=1)
benchmark.load_weights('saved_models/weights.best.benchmark.hdf5')
benchmark_model_prediction = [benchmark.predict(np.expand_dims(img_tensor, axis=0)) for img_tensor in test_tensors]
#visaulizing the array

print(benchmark_model_prediction[:][0])
#swapping the axes of the benchmark_model_prediction for easy handling

benchmark_model_prediction = np.swapaxes(benchmark_model_prediction,0,1)



#creating a pandas dataframe for with benchmark model's prediction

df_pred_model1 = pd.DataFrame(benchmark_model_prediction[0][:], columns=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])



#first five rows of df_pred_model1 dataframe

print(df_pred_model1[:5])
#extracting name of the image form its path

image_names = [test_files[i][22:] for i in range(len(test_files))]





#adjusting the filename of the image to match the submission guidelines

for i in range(13153):

    if image_names[i][5]=='_':

        image_names[i] = "test_stg2/" + image_names[i]
#adding image names to our dataframe

df_pred_model1['image'] = pd.DataFrame(image_names)



#reindexing the dataframe

df_pred_model1 = df_pred_model1.reindex_axis(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'], axis=1)



#printing the first five rows of dataframe

print(df_pred_model1[:5])
df_pred_model1.to_csv('submission0.csv',index=False)
from keras.applications.vgg19 import VGG19

from keras.preprocessing import image

from keras.applications.vgg19 import preprocess_input

from keras.models import Model

from keras.layers import Input

import numpy as np



#Extracting the weights of VGG19 model pretrained on Imagenet

#defing the Input shape

input_tensor = Input(shape=(224,224,3))

#extracting the weights wof VGG19, without top layers

#and MaxPooling as pooling layer

base_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling=max)

#removing the last layer

output = base_model.get_layer(index = -1).output

#defining the model

VGG19_model2 = Model(base_model.input, output)

VGG19_model2.summary()
VGG19_features = [VGG19_model2.predict(np.expand_dims(train_tensor, axis=0)) for train_tensor in train_tensors]



VGG19_features_test = [VGG19_model2.predict(np.expand_dims(test_tensor, axis=0)) for test_tensor in test_tensors]
print ("Shape of VGG_19_features: {0}".format(np.shape(VGG19_features)))



print ("Shape of VGG_19_features_test: {0}".format(np.shape(VGG19_features_test)))

#VGG_19_features having 5 dimensions, so we have to squeeze it to a 4 dim array by removing extra dimension

squeezed_VGG19_train = np.squeeze(VGG19_features, axis=1)

#squeezing the test features

squeezed_VGG19_test = np.squeeze(VGG19_features_test, axis=1)



print ("Shape of squeezed_VGG19_train: {0}".format(np.shape(squeezed_VGG19_train)))

print ("Shape of squeezed_VGG_19_test: {0}".format(np.shape(squeezed_VGG19_test)))

from keras.models import Sequential

from keras.layers import MaxPooling2D, GlobalMaxPooling2D, Dense



fish_model = Sequential()

#adding a GlobalMaxPooling2D layer with with input shape same as the shape of Squeezed_VGG19_train.

fish_model.add(GlobalMaxPooling2D(input_shape=squeezed_VGG19_train.shape[1:]))

#adding a fully connected dense layer with relu activation function

fish_model.add(Dense(1024, activation='relu'))

#adding a dense layer with softmax activation function.

#no of nodes are same as the number of classes of fish.

fish_model.add(Dense(8, activation = 'softmax'))

fish_model.summary()
#compiling the model with rmsprop optimizer

fish_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#training fish_model on the trainig dataset

from keras.callbacks import ModelCheckpoint



#checkpointer for saving only best weights

checkpointer_VGG = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', verbose=1, save_best_only=True)



fish_model.fit(squeezed_VGG19_train,train_targets,validation_split=0.3,batch_size=20,

               epochs=5,callbacks=[checkpointer_VGG],verbose=1)
fish_model.load_weights('saved_models/weights.best.VGG19.hdf5')
#making the predictions from fish_model

fish_model_prediction = [fish_model.predict(np.expand_dims(feature, axis=0)) for feature in squeezed_VGG19_test]
print(fish_model_prediction[1])
print(np.shape(fish_model_prediction))
#swapping the axes for better handling

fish_model_prediction = np.swapaxes(fish_model_prediction,0,1)
import pandas as pd



#creating a pandas dataframe for with benchmark model's prediction

df_pred_fish_model = pd.DataFrame(fish_model_prediction[0][:], columns=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
print(df_pred_fish_model[:5])
#extracting name of the image form its path

image_names = [test_files[i][22:] for i in range(len(test_files))]





#adjusting the filename of the image to match the submission guidelines

for i in range(13153):

    if image_names[i][5]=='_':

        image_names[i] = "test_stg2/" + image_names[i]
#adding image names to our dataframe

df_pred_fish_model['image'] = pd.DataFrame(image_names)



#reindexing the dataframe

df_pred_fish_model = df_pred_fish_model.reindex_axis(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'], axis=1)



#printing the first five rows of dataframe

print(df_pred_fish_model[:5])
df_pred_fish_model.to_csv('submission2.csv',index=False)
from keras.applications.vgg19 import VGG19

from keras.preprocessing import image

from keras.applications.vgg19 import preprocess_input

from keras.models import Model

from keras.layers import Input

import numpy as np



#Extracting the weights of VGG19 model pretrained on Imagenet

#defing the Input shape

input_tensor = Input(shape=(224,224,3))

#extracting the weights wof VGG19, without top layers

#and MaxPooling as pooling layer

base_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling=max)

#removing the last 11 layers

output = base_model.get_layer(index = -11).output

#defining the model

VGG19_model3 = Model(base_model.input, output)

VGG19_model3.summary()
VGG_19_features_2 = [VGG19_model3.predict(np.expand_dims(train_tensor, axis=0)) for train_tensor in train_tensors]



VGG_19_features_test_2 = [VGG19_model3.predict(np.expand_dims(test_tensor, axis=0)) for test_tensor in test_tensors]
squeezed_VGG19_train_2 = np.squeeze(VGG_19_features_2, axis=1)

print ("Shape of squeezed_VGG19_train_2: {0}".format(np.shape(squeezed_VGG19_train_2)))
squeezed_VGG19_test_2 = np.squeeze(VGG_19_features_test_2, axis=1)



print ("Shape of squeezed_VGG_19_test_2: {0}".format(np.shape(squeezed_VGG19_test_2)))
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout

from keras.models import Sequential



model4 = Sequential()



# model Convolution layer

model4.add(Conv2D(filters=16,kernel_size=2,strides=1,activation='relu',input_shape=(224,224,3)))

# Max Pooling layer to reduce the dimensionality

model4.add(MaxPooling2D(pool_size=2,strides=2))

#Dropout layer, for turning off each node with the probability of 0.2

model4.add(Dropout(0.2))

model4.add(Conv2D(filters=32, kernel_size=2,strides=1,activation='relu'))

model4.add(MaxPooling2D(pool_size=2,strides=2))

model4.add(Dropout(0.2))

model4.add(Conv2D(filters=64,kernel_size=2,strides=1,activation='relu'))

model4.add(MaxPooling2D(pool_size=2,strides=2))

model4.add(Dropout(0.2))

model4.add(GlobalAveragePooling2D())

#A fully connected dense layer with 8 nodes (no of classes of fish)

model4.add(Dense(8,activation='softmax'))

model4.summary()
model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping





epochs = 10



#checkpointer saves the weight of the best model only

checkpointer_4 = [EarlyStopping(monitor='val_loss',min_delta=0.01, patience=0, verbose=1), ModelCheckpoint(filepath='saved_models/weights.best.from_scratch_6.hdf5',

                                  verbose=1, save_best_only=True)]



model4.fit(train_tensors, train_targets, batch_size=20, epochs=epochs, callbacks=checkpointer_4, validation_split=0.3, verbose=1)
#loading the weights of pretrained model

model4.load_weights('saved_models/weights.best.from_scratch_6.hdf5')
#making predictions

model4_prediction = [model4.predict(np.expand_dims(img_tensor, axis=0)) for img_tensor in test_tensors]
#swapping the axes of the model4_prediction for easy handling

model4_prediction = np.swapaxes(model4_prediction,0,1)
import pandas as pd



#creating a pandas dataframe for with benchmark model's prediction

df_pred_model4 = pd.DataFrame(model4_prediction[0][:], columns=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
#extracting name of the image form its path

image_names = [test_files[i][22:] for i in range(len(test_files))]





#adjusting the filename of the image to match the submission guidelines

for i in range(13153):

    if image_names[i][5]=='_':

        image_names[i] = "test_stg2/" + image_names[i]
#adding image names to our dataframe

df_pred_model4['image'] = pd.DataFrame(image_names)



#reindexing the dataframe

df_pred_model4 = df_pred_model4.reindex_axis(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'], axis=1)
df_pred_model4.to_csv('submission4.csv',index=False)
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout

from keras.models import Sequential



model5 = Sequential()



#Model architecture.

#Convolution layer

model5.add(Conv2D(filters=32,kernel_size=2,strides=1,activation='relu',input_shape=(224,224,3)))

# Max Pooling layer to reduce the dimensionality

model5.add(MaxPooling2D(pool_size=2,strides=2))

#Dropout layer, for turning off each node with the probability of 0.5

model5.add(Dropout(0.5))

model5.add(Conv2D(filters=64, kernel_size=2,strides=1,activation='relu'))

model5.add(MaxPooling2D(pool_size=2,strides=2))

#Dropout layer, for turning off each node with the probability of 0.4

model5.add(Dropout(0.4))

model5.add(Conv2D(filters=128,kernel_size=2,strides=1,activation='relu'))

model5.add(MaxPooling2D(pool_size=2,strides=2))

#Dropout layer, for turning off each node with the probability of 0.2

model5.add(Dropout(0.2))

#Global Average Pooling layer for object localization

model5.add(GlobalAveragePooling2D())

#A fully connected dense layer with 8 nodes (no of classes of fish)

model5.add(Dense(8,activation='softmax'))

#printing the summary of the architecture

model5.summary()
model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping



#number of epochs

epochs = 5

batch_size=20

#split the training data into training and validation datasets (30% for validation and 70 % for training).

validation_split=0.3

# print the progress

verbose=0.1



#checkpointer saves the weight of the best model only

checkpointer_5 = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)



model5.fit(train_tensors, train_targets, batch_size=batch_size, epochs=epochs, 

           callbacks=[checkpointer_5], validation_split=validation_split, verbose=verbose)
model5.load_weights('saved_models/weights.best.from_scratch.hdf5')
model5_prediction = [model5.predict(np.expand_dims(img_tensor, axis=0)) for img_tensor in test_tensors]
#visaulizing the array

print(model5_prediction[:][0])
#swapping the axes of the model4_prediction for easy handling

model5_prediction = np.swapaxes(model5_prediction,0,1)



#creating a pandas dataframe for with benchmark model's prediction

df_pred_model5 = pd.DataFrame(model5_prediction[0][:], columns=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])



#first five rows of df_pred_model1 dataframe

print(df_pred_model5[:5])
#extracting name of the image form its path

image_names = [test_files[i][22:] for i in range(len(test_files))]





#adjusting the filename of the image to match the submission guidelines

for i in range(13153):

    if image_names[i][5]=='_':

        image_names[i] = "test_stg2/" + image_names[i]
#adding image names to our dataframe

df_pred_model5['image'] = pd.DataFrame(image_names)



#reindexing the dataframe

df_pred_model5 = df_pred_model5.reindex_axis(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'], axis=1)



#printing the first five rows of dataframe

print(df_pred_model5[:5])
df_pred_model5.to_csv('submission5.csv',index=False)
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout

from keras.models import Sequential



model6 = Sequential()



# model Convolution layer

model6.add(Conv2D(filters=32,kernel_size=2,strides=1,activation='relu',input_shape=(224,224,3)))

# Max Pooling layer to reduce the dimensionality

model6.add(MaxPooling2D(pool_size=2,strides=2))

#Dropout layer, for turning off each node with the probability of 0.2 

model6.add(Dropout(0.2))

model6.add(Conv2D(filters=64, kernel_size=2,strides=1,activation='relu'))

model6.add(MaxPooling2D(pool_size=2,strides=2))

#Dropout layer, for turning off each node with the probability of 0.2

model6.add(Dropout(0.2))

model6.add(Conv2D(filters=128,kernel_size=2,strides=1,activation='relu'))

model6.add(MaxPooling2D(pool_size=2,strides=2))

#Dropout layer, for turning off each node with the probability of 0.2

model6.add(Dropout(0.2))

model6.add(GlobalAveragePooling2D())

#A fully connected dense layer with 8 nodes (no of classes of fish)

model6.add(Dense(8,activation='softmax'))

model6.summary()
model6.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping





epochs = 10



#checkpointer saves the weight of the best model only

checkpointer_6 = [EarlyStopping(monitor='val_loss',min_delta=0.01, patience=0, verbose=1), ModelCheckpoint(filepath='saved_models/weights.best.from_scratch_5.hdf5',

                                  verbose=1, save_best_only=True)]



model6.fit(train_tensors, train_targets, batch_size=20, epochs=epochs, callbacks=checkpointer_6, validation_split=0.3, verbose=1)
#loading the weights of pretrained model

model6.load_weights('saved_models/weights.best.from_scratch_5.hdf5')
#making predictions

model6_prediction = [model6.predict(np.expand_dims(img_tensor, axis=0)) for img_tensor in test_tensors]
#swapping the axes of the model6_prediction for easy handling

model6_prediction = np.swapaxes(model6_prediction,0,1)
import pandas as pd



#creating a pandas dataframe for with benchmark model's prediction

df_pred_model6 = pd.DataFrame(model6_prediction[0][:], columns=['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'])
#extracting name of the image form its path

image_names = [test_files[i][22:] for i in range(len(test_files))]





#adjusting the filename of the image to match the submission guidelines

for i in range(13153):

    if image_names[i][5]=='_':

        image_names[i] = "test_stg2/" + image_names[i]
#adding image names to our dataframe

df_pred_model6['image'] = pd.DataFrame(image_names)



#reindexing the dataframe

df_pred_model6 = df_pred_model6.reindex_axis(['image','ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT'], axis=1)
df_pred_model6.to_csv('submission6.csv',index=False)