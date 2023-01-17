import pandas as pd

import numpy as np

import os

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.applications import VGG19

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam

from keras import Sequential

from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense

from keras import optimizers

import os, shutil, cv2, time



"""Initial settings for training/validation, remember to choose!!!"""

EPOCHS = 30

BATCHSIZE = 16

TRAINSET_PERCENT = 0.75 ## percentage of trainingset from totalsamples (from [0, 1.0] )

SAMPLES = 4000  ## take huge sample of images, but only train&validate on subsets of allowed trainable/validatable images

TRAINSTEPS_PERCENT = 1.0  ## percentage of trainingset steps being done (from [0, 1.0] )

VALIDSTEPS_PERCENT = 1.0  ## percentage of validationset steps being done (from [0, 1.0] )

BALANCEDSET = False ## balance the df dataset 50:50 sick/healthy, otherwise simply random sample 

MODELNAME = "borrowedXceptionArchitecture" + ".h5"

trainConvLayers = True  ## train convLayers or freeze them

SEED = 1999-888

ROTANGLE = 10

ZOOM = 0.02

VERBOSE = 1  ## for modeltraining

IMAGESIZE = 299







source_dir = r"../input/preprocessed-diabetic-retinopathy-trainset/300_train/300_train/"

temp_dir = r"./temp/"

#test_dir = r"../input/300_test/300_test/"



# initialize multiple optimizers but you can only choose one at a time of course!

sgd = optimizers.SGD(lr=0.01/2, decay=1e-6, momentum=0.9, nesterov=True)

addyboi = optimizers.Adam(lr=0.01, decay=1e-5)

rmsprop = optimizers.RMSprop(lr=0.01/2)

basic_rms = optimizers.RMSprop()



"""remember to choose optimizer!!!"""

OPTIMIZER = basic_rms
# Convert categorical level to binary and randomly sample

import numpy as np

df = pd.read_csv(r"../input/preprocessed-diabetic-retinopathy-trainset/newTrainLabels.csv")

df['level'] = 1*(df['level'] > 0)



# balance classes

if BALANCEDSET:

    print('balancing dataset...')

    df = pd.concat([

        df[df['level']==0].sample(n=round(SAMPLES/2.0), random_state=SEED),

        df[df['level']==1].sample(n=round(SAMPLES/2.0), random_state=SEED)

    ]).sample(frac=1.0, random_state=SEED) # shuffle

else:

    print('raw dataset(unbalanced)...')

    df= df.sample(n=SAMPLES, random_state=SEED)





df.reset_index(drop=True, inplace=True) # I've had troubles with shuffled dataframes on some earlier Cognitive Mathematics labs,

                                        # and this seemed to prevent those

print(df.head())

print("")




# Create level histogram

df['level'].hist(bins = [0,1, 2], rwidth = 0.5, align = 'left');



# here we can split df into train/validation dataframes, and then we can use

# separated imagedatagenerators, such that we are able to use data augmentation 

# for only the traindata images

# the validdata images should be left raw as they are



setBoundary = round(SAMPLES * TRAINSET_PERCENT)

train_df = df[:setBoundary]

validate_df = df[setBoundary:]

train_df.reset_index(drop=True, inplace=True)

validate_df.reset_index(drop=True, inplace=True)
# Here the issue with original demo4 code from Sakari was that there was the typeError from imageDataGenerators

# at least for myself, so that the error message suggested that I should convert level column to str

df["level"]= df.iloc[:,1].astype(str)

train_df["level"]= train_df.iloc[:,1].astype(str)

validate_df["level"]= validate_df.iloc[:,1].astype(str)
#Test out images can be found

image_path = source_dir + df["image"][0] +".jpeg"

if  os.path.isfile(image_path)==False:

    raise Exception('Unable to find train image file listed on dataframe')

else:

    print('Train data frame and file path ready')

    
# Prepare to crop images to larger size so that the network can work on them...

# apparently too small images are just shit for these convNeuralNetworks

# Create destination directory



try:

    os.mkdir(temp_dir)

    print('Created a directory:', temp_dir)

except:

    # Temp directory already exist, so clear it

    for file in os.listdir(temp_dir):  

        file_path = os.path.join(temp_dir, file)

        try:

            if os.path.isfile(file_path):

                os.unlink(file_path)

        except Exception as e:

            print(e)

    print(temp_dir, ' cleared.')
# Crop the images to larger size from 100x100 upwards to atleast 300x300 

# some contestant winner had images 500x500

# I've tried this vgg19 with 100x100 and it was not good results



# Start timing

start = time.time()



# Crop and resize all images. Store them to dest_dir

print("Cropping and rescaling the images:")

for i, file in enumerate(df["image"]):

    try:

        fname = source_dir + file + ".jpeg"

        img = cv2.imread(fname)

    

        # Crop the image to the height

        h, w, c = img.shape

        if w > h:

            wc = int(w/2)

            w0 = wc - int(h/2)

            w1 = w0 + h

            img = img[:, w0:w1, :]

        # Rescale to N x N

        N = IMAGESIZE

        img = cv2.resize(img, (N, N))

        # Save

        new_fname = temp_dir  + file + ".jpeg"

        cv2.imwrite(new_fname, img)

    except:

        # Display the image name having troubles

        print("problemImagesFound:___ ", fname)

         

    # Print the progress for every N images

    if (i % 500 == 0) & (i > 0):

        print('{:} images resized in {:.2f} seconds.'.format(i, time.time()-start))



# End timing

print('Total elapsed time {:.2f} seconds.'.format(time.time()-start))

print('temp_dir was=' + temp_dir)
path0= temp_dir # specify directory where the f***ing files are

fileList = os.listdir(path0) # get the f***ing file list in the path directory

# list files

print(fileList[0])# print if you even found any f***ing files in this f***ing kaggle environment
for i in range (5):

    print(df['image'][i])
# We must convert the dataframes's image columns into .jpeg extension, so generators can find images into the dataframes

"""wtf is wrong with this code, why kaggle keeps raising warnings

why it is suddenly illegal to MODIFY the goddamn column to have ".jpeg" in the end for all values?!"""



print(df.head(5))

a=df.iloc[:,0] + ".jpeg"

b=train_df.iloc[:,0] + ".jpeg"

c=validate_df.iloc[:,0] + ".jpeg"

df['image']=a

train_df['image']=b

validate_df['image']=c

print(df.head(5))
# Create image data generator

from keras.preprocessing.image import ImageDataGenerator

# I think that it would be possible also to use data augmentation in the training_generator only.

# Also the interview of kaggle contestants showed my own suspicions to be true that one should not use 

# image shear with eye data (the perfectly healthy human eyeball should be round, so you dont get eyeglasses)

# those contestant used rotation and mirrorings of the image as I recall, but small amounts of

#  zoom would not be too bad either, I reckon



# use data augmentation for training

traingen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=ROTANGLE,

    zoom_range=ZOOM,

    horizontal_flip=True,

    vertical_flip=True)



## just take the raw data for validation

validgen = ImageDataGenerator(

    rescale=1./255)



# Data flow for training

trainflow = traingen.flow_from_dataframe(

    dataframe = train_df, 

    directory = temp_dir,

    x_col = "image", 

    y_col = "level", 

    class_mode = "binary", # class_mode binary causes to infer classlabels automatically from y_col, at least according to keras documentation

    target_size = (IMAGESIZE, IMAGESIZE), 

    batch_size = BATCHSIZE,

    shuffle = True,

    seed = SEED)



# Data flow for validation

validflow = validgen.flow_from_dataframe(

    dataframe = validate_df, 

    directory = temp_dir,

    x_col = "image", 

    y_col = "level", 

    class_mode = "binary", 

    target_size = (IMAGESIZE, IMAGESIZE), 

    batch_size = BATCHSIZE,

    shuffle = False, # validgen doesnt need shuffle I think, according to teacher's readymade convnet example (?)

#    seed = SEED

)



"""## We will try Juha Kopus model style for pre-trained VGG16. it was pretty bad in my opinoin, i tested it

x = base_model.output

x = Flatten(name='flatten')(x)

x = Dropout(0.3, seed=SEED)(x) # add dropout just in case

x = Dense(256,activation='relu')(x)

x = Dropout(0.3, seed=SEED)(x) # add dropout just in case

preds = Dense(1,activation='sigmoid')(x) #final layer with sigmoid activation"""

"""## Sakaris Xception model style for Xception convnet ##

## with added Dense model at the end of ConvNet"""

"""x = base_model.output ##this was some old code to make for MobileNet from example tutorial about convnets

x = GlobalAveragePooling2D()(x)

x = Flatten(name='flatten')(x)

x=Dropout(0.2, seed=SEED)(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

x=Dropout(0.3, seed=SEED)(x)

x=Dense(1024,activation='relu')(x) #dense layer 2

x=Dropout(0.3, seed=SEED)(x)

x=Dense(512,activation='relu')(x) #dense layer 3

x=Dropout(0.3, seed=SEED)(x)

preds=Dense(1,activation='sigmoid')(x) #final layer with sigmoid activation

"""





from keras.applications.xception import Xception



base_model = Xception(include_top=False, input_shape=(IMAGESIZE, IMAGESIZE, 3)) # randomized initial weights, Xception_base



if not trainConvLayers:

    for layer in base_model.layers: # make convLayers un-trainable

        layer.trainable=False        

        

base_model.summary()



x = base_model.output



x = GlobalAveragePooling2D()(x)  

#x = Flatten(name='flatten')(x)  

x = Dropout(0.25, seed=SEED)(x) # add dropout just in case      

x = Dense(512, activation='relu')(x)

x = Dropout(0.4, seed=SEED)(x) # add dropout just in case      

x = Dense(512, activation='relu')(x)

x = Dropout(0.4, seed=SEED)(x) # add dropout just in case  

x = Dense(256,activation='relu')(x)

x = Dropout(0.4, seed=SEED)(x) # add dropout just in case

preds = Dense(1,activation='sigmoid')(x) #final layer with sigmoid activation



model=Model(inputs=base_model.input,outputs=preds)

#for i,layer in enumerate(model.layers):

#    print(i,layer.name)



print(model.summary())



model.compile(optimizer = OPTIMIZER,

             loss='binary_crossentropy', 

              metrics = ["accuracy"])


from keras.callbacks import ModelCheckpoint

from time import time, localtime, strftime

# Testing with localtime and strftime

print(localtime())

print(strftime('%Y-%m-%d-%H%M%S', localtime()))





# Calculate how many batches are needed to go through whole train and validation set

STEP_SIZE_TRAIN = round((trainflow.n // trainflow.batch_size) * 1.0 * TRAINSTEPS_PERCENT ) 

STEP_SIZE_VALID = round((validflow.n // validflow.batch_size) * 1.0 * VALIDSTEPS_PERCENT ) 



# Train and count time

model_name = strftime('Case2-%Y-%m-%d-%H%M%S', localtime()) + MODELNAME

print('modelname was=',model_name,'\n')



t1 = time()

h = model.fit_generator(generator = trainflow,

                    steps_per_epoch = STEP_SIZE_TRAIN,

                    validation_data = validflow,

                    validation_steps = STEP_SIZE_VALID,

                    epochs = EPOCHS,

                    verbose = VERBOSE)

t2 = time()

elapsed_time = (t2 - t1)



# Save the model

model.save(model_name)

print('')

print('Model saved to file:', model_name)

print('')



# Print the total elapsed time and average time per epoch in format (hh:mm:ss)

t_total = strftime('%H:%M:%S', localtime(t2 - t1))

t_per_e = strftime('%H:%M:%S', localtime((t2 - t1)/EPOCHS))

print('Total elapsed time for {:d} epochs: {:s}'.format(EPOCHS, t_total))

print('Average time per epoch:             {:s}'.format(t_per_e))



# get the currently trained model, and plot the accuracies and loss for training and validation



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np



epochs = np.arange(EPOCHS) + 1.0



f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,7))



def plotter(ax, epochs, h, variable):

    ax.plot(epochs, h.history[variable], label = variable)

    ax.plot(epochs, h.history['val_' + variable], label = 'val_'+variable)

    ax.set_xlabel('Epochs')

    ax.legend()



plotter(ax1, epochs, h, 'acc')

plotter(ax2, epochs, h, 'loss')

plt.show()





# get true values

y_true = validflow.classes

# note about predict_generator, 

# sometimes it happened that, if you put argument steps = STEP_SIZE_VALID, then

# that throws error because of mismatched steps amount somewhere, I think that the

# np.ceil(validgen.n/validgen.batch_size) seems to fix it

predict = model.predict_generator(validflow, steps= np.ceil(validflow.n / validflow.batch_size))

y_pred = 1*(predict > 0.5)


# Calculate and print the metrics results

from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report



cm = confusion_matrix(y_true, y_pred)

print('Lates Confusion matrix:')

print(cm)

print('')



cr = classification_report(y_true, y_pred)

print('Lates Classification report:')

print(cr)

print('')



from sklearn.metrics import accuracy_score

a = accuracy_score(y_true, (y_pred))

print(a)

print('Lates Accuracy with old decision point {:.4f} ==> {:.4f}'.format(0.5, a))

print('')



# Calculate and plot ROC-curve

# See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_true, predict) 



plt.plot(fpr, tpr, color='darkorange', lw = 2)

plt.plot([0, 1], [0, 1], color='navy', lw = 2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Lates Receiver operating characteristic curve')

plt.show()

from keras.models import load_model



sakarismodel = load_model("../input/sakarisbestmodelxception/best_demo11_SAKARISDEMO11.h5")

predictSakari = sakarismodel.predict_generator(validflow, steps= np.ceil(validflow.n / validflow.batch_size))

y_predSakari = 1*(predictSakari > 0.5)



print('Sakaris best model was as follows: \n')

# Calculate and print the metrics results

from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report



cm = confusion_matrix(y_true, y_predSakari)

print('Sakaris Confusion matrix:')

print(cm)

print('')



cr = classification_report(y_true, y_predSakari)

print('Sakaris Classification report:')

print(cr)

print('')



from sklearn.metrics import accuracy_score

a = accuracy_score(y_true, (y_predSakari))

print(a)

print('Sakaris Accuracy with old decision point {:.4f} ==> {:.4f}'.format(0.5, a))

print('')





fpr0, tpr0, thresholds0 = roc_curve(y_true, y_predSakari) 



plt.plot(fpr0, tpr0, color='darkorange', lw = 2)

plt.plot([0, 1], [0, 1], color='navy', lw = 2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Sakaris Receiver operating characteristic curve')

plt.show()

# Clear the temporary directory

dest_dir = './temp/'

for file in os.listdir(dest_dir):

    file_path = os.path.join(dest_dir, file)

    try:

        if os.path.isfile(file_path):

            os.unlink(file_path)

    except Exception as e:

        print(e)

print(dest_dir, ' cleared.')

os.rmdir(dest_dir)

print(dest_dir,'Removed.')