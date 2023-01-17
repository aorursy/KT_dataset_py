import os #to read and write a file

import cv2 #opencv for manipulating images

from glob import glob #finds all the pathnames matching a specified pattern

import h5py #will be helpful for storing huge amt of numerical data

import shutil #offers a high-level operations on files and collection of files

import imgaug as aug #Image augmentation. Helful for creating much more larger dataset from our input.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualising tool

import matplotlib.pyplot as plt #visualising tool

import imgaug.augmenters as iaa #Image augmentation.



from os import listdir, makedirs, getcwd, remove 

#listdr- return list containing names of the entries given in path

#make directory named path with the specified numeric mode.

#getcwd - getting current working directory

#remove- remove/delete a file path



from os.path import isfile, join, abspath, exists, isdir

#isfile - to check specified path is available in that file or not.

#join - to join one or more path.

#abspath - returns a normalised version of the path

#isdir- returns true/false if specified path is there in the directory or not.



from pathlib import Path #object oriented file system path.

from skimage.io import imread #image reading/writing

from skimage.transform import resize #resize

from keras.models import Sequential, Model, load_model #keras NN model

from keras.applications.vgg16 import VGG16, preprocess_input #VGG16

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten #layers to build NN

from keras.optimizers import Adam, SGD, RMSprop #optimizers

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split 

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

from keras import backend as K

import tensorflow as tf





color = sns.color_palette()

%matplotlib inline

%config InlineBackend.figure_format="svg"
columns = ['path', 'label']

no_xray_df = pd.DataFrame(columns = columns)

no_imgs_path = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','brain-tumor-detection' ,'no', '*.jpg'))}



print('Scans found:', len(no_imgs_path))



no_xray_df['path'] = no_imgs_path.values()



no_list = [0]*255 #Labeling 0 for all 255 records

no_xray_df['label'] = no_list
no_xray_df.head() #no_tumor dataset ready
#For tumor dataset

columns1 = ['path', 'label']

yes_xray_df = pd.DataFrame(columns = columns1)

all_imgs_path_yes = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','brain-tumor-detection' ,'yes', '*.jpg'))}



print('Scans found:', len(all_imgs_path_yes))



yes_xray_df['path'] = all_imgs_path_yes.values()

#label them 1 i.e. they have tumor

yes_list = [1]*255

yes_xray_df['label'] = yes_list 
yes_xray_df.head()
#pred data



columns2 = ['path', 'label']

pred_xray_df = pd.DataFrame(columns = columns)

all_imgs_path_pred = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','brain-tumor-detection' ,'pred','*.jpg'))}



print('Scans found:', len(all_imgs_path_pred))



pred_xray_df['path'] = all_imgs_path_pred.values()

pred_xray_df.head() # there are no Labels for this dataset.
frames = [yes_xray_df, no_xray_df]

final_df = pd.concat(frames)



final_df = final_df.sample(frac=1.).reset_index(drop = True) #shuffling the rows

print("Shuffling the dataset")

final_df.head(5)
#train_test split

from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(final_df, test_size = 0.25, random_state = 2020)

print('train', train_df.shape[0], 'validation', valid_df.shape[0])
train_df #Viewing the training dataset
valid_df #Viewing the Valid dataset
print("Number of traininng samples: ", len(train_df))

print("Number of validation samples: ", len(valid_df))
# dimensions to consider for the images

img_rows, img_cols, img_channels = 224,224,3



# batch size for training  

batch_size=8



# total number of classes in the dataset

nb_classes=2
#augmentation

seq = iaa.OneOf([

    iaa.Fliplr(), 

    iaa.Affine(rotate=20), 

    iaa.Multiply((1.2, 1.5))]) 



#Fliplr- Horizontal Flips

#Affine - rotation

#Multiply - Random Brightness
def data_generator(data, batch_size, is_validation_data=False):

    # Get total number of samples in the data

    n = len(data)

    nb_batches = int(np.ceil(n/batch_size))



    # Get a numpy array of all the indices of the input data

    indices = np.arange(n)

    

    # Define two numpy arrays for containing batch data and labels

    batch_data = np.zeros((batch_size, img_rows, img_cols, img_channels), dtype=np.float32)

    batch_labels = np.zeros((batch_size, nb_classes), dtype=np.float32)

    

    while True:

        if not is_validation_data:

            # shuffle indices for the training data

            np.random.shuffle(indices)

            

        for i in range(nb_batches):

            # get the next batch 

            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]

            

            # process the next batch

            for j, idx in enumerate(next_batch_indices):

                img = cv2.imread(data.iloc[idx]["path"])

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                label = data.iloc[idx]["label"]

                

                if not is_validation_data:

                    img = seq.augment_image(img)

                

                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)

                batch_data[j] = img

                batch_labels[j] = to_categorical(label,num_classes=nb_classes)

            

            batch_data = preprocess_input(batch_data)

            yield batch_data, batch_labels
#training data generator 

train_data_gen = data_generator(train_df, batch_size)



# validation data generator 

valid_data_gen = data_generator(valid_df, batch_size, is_validation_data=True)
#Modeling part

#Transfer Learning

#Choosing VGG16



def get_base_model():

    base_model = VGG16(input_shape=(img_rows, img_cols, img_channels), weights='imagenet', include_top=True)

    return base_model
# get the base model

base_model = get_base_model()



#  get the output of the second last dense layer 

base_model_output = base_model.layers[-2].output



# add new layers 

x = Dropout(0.5,name='drop2')(base_model_output)

output = Dense(2, activation='softmax', name='fc3')(x)



# define a new model 

model = Model(base_model.input, output)



# Freeze all the base model layers 

for layer in base_model.layers[:-1]:

    layer.trainable=False



# compile the model and check it 

optimizer = RMSprop(0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()
# the restore_best_weights parameter load the weights of the best iteration once the training finishes

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(patience=10, restore_best_weights=True)



# checkpoint to save model

chkpt = ModelCheckpoint(filepath="model1", save_best_only=True)



# number of training and validation steps for training and validation

nb_train_steps = int(np.ceil(len(train_df)/batch_size))

nb_valid_steps = int(np.ceil(len(valid_df)/batch_size))



# number of epochs 

nb_epochs=30
# train the model 

history1 = model.fit_generator(train_data_gen, 

                              epochs=nb_epochs, 

                              steps_per_epoch=nb_train_steps, 

                              validation_data=valid_data_gen, 

                              validation_steps=nb_valid_steps,

                              callbacks=es)
#get the training and validation accuracy

train_acc = history1.history['accuracy']

valid_acc = history1.history['val_accuracy']



#get the loss

train_loss = history1.history['loss']

valid_loss = history1.history['val_loss']



#get the entries

xvalues = np.arange(len(train_acc))



#visualise

f, ax = plt.subplots(1,2, figsize = (10,5))

ax[0].plot(xvalues, train_loss)

ax[0].plot(xvalues, valid_loss)

ax[0].set_title("Loss curve")

ax[0].set_xlabel("Epoch")

ax[0].set_ylabel("loss")

ax[0].legend(['train', 'validation'])



ax[1].plot(xvalues, train_acc)

ax[1].plot(xvalues, valid_acc)

ax[1].set_title("Accuracy")

ax[1].set_xlabel("Epoch")

ax[1].set_ylabel("accuracy")

ax[1].legend(['train', 'validation'])



plt.show()
valid_loss, valid_acc = model.evaluate_generator(valid_data_gen, steps=nb_valid_steps)

print(f"Final validation accuracy: {valid_acc*100:.2f}%")
#Model Interpretability

#Now, we will create new model which will output all the activations for each convolution created. So, for each sample

#image, we will have outputs of different activations for each convolution in the network
#select the layers for which you want to visulaise the outputs and store that in list.

outputs = [layer.output for layer in model.layers[1:18]]

outputs #outputs from each intermediate layers
#define new model that generates above outputs

vis_model = Model(model.input, outputs)



vis_model.summary()
#now store which all outputs you want in the list

layer_names = []

for layer in outputs:

    layer_names.append(layer.name.split("/")[0])

    

print("Layers which will be used for visualisation: ")

print(layer_names)
def get_CAM(processed_image, predicted_label):

    #will be used to generate heatmap for a sample image

    #processed_image = the image sample which has been preprocessed will come here.

    #predicted_label = label that was predicted by our model

    

    #will return heat map from the last convolution layer output.

    

    #we want the activations for predicted_label

    predicted_output = model.output[:, predicted_label]

    

    #choose the last level of our convolution

    last_conv_layer = model.get_layer('block5_conv3')

    

    #get the gradients for that last layer K = keras

    grads = K.gradients(predicted_output, last_conv_layer.output)[0]

    

    #take the mean grads per feature map

    grads = K.mean(grads, axis = (0,1,2))

    

    #define a function that generates the values for the output and gradients #imp

    evaluation_function = K.function([model.input], [grads, last_conv_layer.output[0]])

    

    #get the values

    grads_values, conv_output_values = evaluation_function([processed_image])

    

    #Now, iterate each feature map in your conv O/P and multiply the gradient values. This can tell

    #how important a feature is..

    for i in range(512): #we have 512 channels (features) in the last layer

        conv_output_values[:,:,i] *= grads_values[i]

        

    #create a heatmap now

    heatmap = np.mean(conv_output_values, axis = -1)

    

    #remove negative values

    heatmap = np.maximum(heatmap, 0)

    

    #normalize now

    heatmap /= heatmap.max()

    

    return heatmap

   
def show_random_sample(idx):

    #This I am creating to select random sample from validation dataframe

    #It also generates the prediction for the same. It also stores the heatmap and intermediate layers 

    #activation maps

    

    #idx: random index to select a sample from validation data

    

    #It will return activation values from intermediate layers.

    

    #select the sample and read the corresponding image and label

    sample_image = cv2.imread(valid_df.iloc[idx]['path'])

    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    sample_image = cv2.resize(sample_image, (img_rows, img_cols))

    sample_label = valid_df.iloc[idx]['label']

    

    #preprocess the image

    sample_image_processed = np.expand_dims(sample_image, axis=0)

    sample_image_processed = preprocess_input(sample_image_processed)

    

    #generate the activation maps from intermediate layers.

    activations = vis_model.predict(sample_image_processed)

    

    #get the label predicted by our original model 

    pred_label = np.argmax(model.predict(sample_image_processed), axis=-1)[0]

    

    #choose any random activation map from the activation maps

    sample_activation = activations[0][0,:,:,32]

    

    #normalize the sample activation map

    sample_activation-=sample_activation.mean()

    sample_activation/=sample_activation.std()

    

    #convert pixel values between 0-255

    sample_activation *=255

    sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)

    

    #get the heatmap now for class activation

    heatmap = get_CAM(sample_image_processed, pred_label)

    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))

    heatmap = heatmap * 255

    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    super_imposed_image = heatmap * 0.5 + sample_image #brightens the image

    super_imposed_image = np.clip(super_imposed_image, 0, 255).astype(np.uint8)

    

    f, ax = plt.subplots(2,2, figsize= (15,8))

    ax[0,0].imshow(sample_image)

    ax[0,0].set_title(f"True Label: {sample_label} \n Predicted Label: {pred_label}")

    ax[0,0].axis('off')

    

    ax[0,1].imshow(sample_activation)

    ax[0,1].set_title("Random Feature Map")

    ax[0,1].axis('off')

    

    ax[1,0].imshow(heatmap)

    ax[1,0].set_title("Class Activation Map")

    ax[1,0].axis('off')

    

    ax[1,1].imshow(super_imposed_image)

    ax[1,1].set_title("Activation map superimposed")

    ax[1,1].axis('off')

    plt.show()

    

    return activations
#Creating same function with different name for pred data set. Previously, it was created for Valid Dataframe.



def show_random_sample1(idx):

    #This I am creating to select random sample from validation dataframe

    #It also generates the prediction for the same. It also stores the heatmap and intermediate layers 

    #activation maps

    

    #idx: random index to select a sample from validation data

    

    #It will return activation values from intermediate layers.

    

    #select the sample and read the corresponding image and label

    sample_image = cv2.imread(pred_xray_df.iloc[idx]['path'])

    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    sample_image = cv2.resize(sample_image, (img_rows, img_cols))

    sample_label = pred_xray_df.iloc[idx]['label']

    

    #preprocess the image

    sample_image_processed = np.expand_dims(sample_image, axis=0)

    sample_image_processed = preprocess_input(sample_image_processed)

    

    #generate the activation maps from intermediate layers.

    activations = vis_model.predict(sample_image_processed)

    

    #get the label predicted by our original model 

    pred_label = np.argmax(model.predict(sample_image_processed), axis=-1)[0]

    

    #choose any random activation map from the activation maps

    sample_activation = activations[0][0,:,:,32]

    

    #normalize the sample activation map

    sample_activation-=sample_activation.mean()

    sample_activation/=sample_activation.std()

    

    #convert pixel values between 0-255

    sample_activation *=255

    sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)

    

    #get the heatmap now for class activation

    heatmap = get_CAM(sample_image_processed, pred_label)

    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))

    heatmap = heatmap * 255

    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    super_imposed_image = heatmap * 0.5 + sample_image #brightens the image

    super_imposed_image = np.clip(super_imposed_image, 0, 255).astype(np.uint8)

    

    f, ax = plt.subplots(2,2, figsize= (15,8))

    ax[0,0].imshow(sample_image)

    ax[0,0].set_title(f"True Label: {sample_label} \n Predicted Label: {pred_label}")

    ax[0,0].axis('off')

    

    ax[0,1].imshow(sample_activation)

    ax[0,1].set_title("Random Feature Map")

    ax[0,1].axis('off')

    

    ax[1,0].imshow(heatmap)

    ax[1,0].set_title("Class Activation Map")

    ax[1,0].axis('off')

    

    ax[1,1].imshow(super_imposed_image)

    ax[1,1].set_title("Activation map superimposed")

    ax[1,1].axis('off')

    plt.show()

    

    return activations
#Now showing the some predictions

#It will also give heatmap and super imposed image to get the details of the X-Ray.

activations = show_random_sample(100) #the output
activations = show_random_sample(123) 
activations = show_random_sample(78) 
activations = show_random_sample(84) 
#This is for predicted dataset

activations = show_random_sample1(15)
activations = show_random_sample1(59) 
activations = show_random_sample1(26)
activations = show_random_sample1(59)
activations = show_random_sample1(24)
activations = show_random_sample1(28)