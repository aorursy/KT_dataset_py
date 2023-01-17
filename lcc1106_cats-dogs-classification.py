# import those packages that I need

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm # to see the running process

from skimage.io import imread, imshow # to read the image from workspace

from skimage.transform import resize # resize those data to a certain size for training

import matplotlib.pyplot as plt # for plotting

%matplotlib inline

import os

import keras 

from keras.utils.np_utils import to_categorical # for One-Hot Encoding

from sklearn.model_selection import train_test_split # for generating validation set (X_val, Y_val)



from keras.applications.resnet50 import ResNet50 # the well-known CNN model imported for transfer learning

from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.core import Dropout

from keras.layers.convolutional import Conv2D

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau # built-in callbacks in keras 

from keras.preprocessing.image import ImageDataGenerator # for data augmentation

from keras import backend as K

from keras.optimizers import Adam # the model optimizer that I choose for this task

from keras.regularizers import l1,l2 # L1, L2 regularization to avoid overfitting

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc

import itertools



# the following remark(code) can be used to see all the files in the 'input' folder in the Workspace



'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''
# function for loading training images and labeling them in the same time according to which folder they come from, 'dogs' or 'cats'



def read_image_train(folder,resize_hight = 224, resize_width = 224):

    

    DATA_ROOT = "../input/"

    dogs = np.ones((1,resize_hight,resize_width,3))

    cats = np.ones((1,resize_hight,resize_width,3))

    

    dogs_label=[]

    cats_label=[]

    

    for image_type in os.listdir(folder):

        # print(image_type)

        if image_type == 'dogs':

            for image_name in tqdm(os.listdir(folder + '/' + image_type)):

                # print(image_name)

                im = imread(os.path.join(DATA_ROOT,folder + '/' + image_type + '/' + image_name))

                # print(im.shape)

                dog_size = im.size

#                 if dog_size >= (resize_hight*resize_width):

                if dog_size >= 0:

                    im_resized = resize(im, (resize_hight, resize_width), anti_aliasing=True)

                    im_resized = im_resized[np.newaxis,:,:,:]

                    # print(im_resized.shape)

                    dogs = np.concatenate((dogs,im_resized),axis=0)

                    dogs_label.append(0)

                    

        elif image_type == 'cats':

            for image_name in tqdm(os.listdir(folder +'/'+ image_type)):

                # print(image_name)

                im = imread(os.path.join(DATA_ROOT,folder + '/' +image_type + '/' +image_name))

                # print(im.shape)

                cat_size = im.size

#                 if cat_size >= (resize_hight*resize_width):

                if cat_size >= 0:

                    im_resized = resize(im, (resize_hight, resize_width), anti_aliasing=True)

                    im_resized = im_resized[np.newaxis,:,:,:]

                    # print(im_resized.shape)

                    cats = np.concatenate((cats,im_resized),axis=0)

                    cats_label.append(1)

                    

    dogs = np.delete(dogs,(0),axis=0)

    cats = np.delete(cats,(0),axis=0)

    image_array = np.concatenate((dogs,cats),axis = 0)

    print(image_array.shape)



    dogs_label = np.asarray(dogs_label)

    cats_label = np.asarray(cats_label)

    label = np.concatenate((dogs_label,cats_label),axis=0)

    print(label.shape)

    

    return image_array, label
# function for loading the testing data and getting their ID at the same time

# the reason that getting the ID is for the submission csv file because the probability of class should match the image name



def read_image_test(folder,resize_hight = 224, resize_width = 224):

    

    DATA_ROOT = "../input/"

    img = np.ones((1,resize_hight,resize_width,3))

    names = []

    

    for image_name in tqdm(os.listdir(folder)):

        # print(image_name)

        im = imread(os.path.join(DATA_ROOT,folder + '/' + image_name))

        # print(im.shape)

        

        im_resized = resize(im, (resize_hight, resize_width), anti_aliasing=True)

        im_resized = im_resized[np.newaxis,:,:,:]

        # print(im_resized.shape)

        img = np.concatenate((img,im_resized),axis=0)

        name_split = image_name.split('.')

        names.append(name_split[0])



                    

    img = np.delete(img,(0),axis=0)

    names = np.asarray(names)

    print("The shape of image",img.shape)

    print("The shape of image name :", names.shape)

    

    return img , names
# Now, We are able to load those data that we need for this task.

# loading images and transform them into numpy array with labels (cats: 1,dogs:0)



x_train, y_train = read_image_train('/kaggle/input/ml-marathon-final/data/kaggle_dogcat/train')

x_test, names = read_image_test('/kaggle/input/ml-marathon-final/data/kaggle_dogcat/test')
# normalization

x_train = x_train.astype('float32')

# x_train /= 255

x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)

x_train -= x_train_mean

x_train /= x_train_std



x_test = x_test.astype('float32')

# x_test /= 255

x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)

x_test -= x_test_mean

x_test /= x_test_std



# label : one-hot encoding 

num_class = 2

y_trainHOT = keras.utils.to_categorical(y_train, num_class)



# extract 1/4 training data as validation data

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_trainHOT, test_size=0.25, random_state=42)



# print out the shape of images and labels

print("The shape of x_train: ",X_train.shape)

print("The shape of y_train: ",Y_train.shape)

print("The shape of x_val: ",X_val.shape)

print("The shape of y_val: ",Y_val.shape)



print("The shape of x_test: ",x_test.shape)
# hyperparameters

resize_hight = 224

resize_width = 224

NUM_CLASSES = 2

batch_size = 4  

epochs = 50

learning_rate = 1e-5

aug = False

initial_train = True
# construct the ResNet50 from keras.applications.resnet50 (the packages that i import above!)

ResNet50_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(resize_hight,resize_width,3))



x = ResNet50_model.output

x = Flatten()(x)

x = Dropout(0.5)(x)

output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

ResNet50_model_final = Model(inputs=ResNet50_model.input, outputs=output_layer)



ResNet50_model_final.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

ResNet50_model_final.summary()
# Data augmentation: Although I may not use it, it is worth trying :)

train_datagen = ImageDataGenerator(shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)



val_datagen = ImageDataGenerator(shear_range=0.2,

                                 zoom_range=0.2,

                                 horizontal_flip=True)



# callbacks

reduce_lr = ReduceLROnPlateau(factor=0.5, 

                              min_lr=1e-12, 

                              monitor='val_loss', 

                              patience=9, 

                              verbose=1)



earlystop = EarlyStopping(monitor="val_acc", 

                          patience=5, 

                          verbose=1)



# save the model for future usage

model_save='model_shuffle_noAug_allimage_bs4.h5' 

model_checkpoint = ModelCheckpoint(filepath=model_save, 

                                   monitor="val_loss", 

                                   save_best_only=True)
# training

if initial_train == True:

    if aug == True:

        train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size,shuffle=False)

        validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size,shuffle=False)

    

        history = ResNet50_model_final.fit_generator(train_generator,

                                steps_per_epoch=len(X_train) // batch_size,

                                epochs=epochs,

                                verbose=1,

                                validation_data=validation_generator,

                                validation_steps = len(X_val) // batch_size,

                                shuffle=True,

                                callbacks = [model_checkpoint,earlystop,reduce_lr])

    else:

        history = ResNet50_model_final.fit(X_train,Y_train,

                                batch_size = batch_size,

                                epochs=epochs,

                                verbose=1,

                                validation_data= (X_val,Y_val),

                                shuffle=True,

                                callbacks = [model_checkpoint,earlystop,reduce_lr])





    # Collect results

    train_loss = ResNet50_model_final.history.history["loss"]

    valid_loss = ResNet50_model_final.history.history["val_loss"]

    train_acc = ResNet50_model_final.history.history["acc"]

    valid_acc = ResNet50_model_final.history.history["val_acc"]



else:

    ResNet50_model_final = keras.models.load_model("/kaggle/input/cats-dogsclassification/model_shuffle_noAug_allimage.h5")

    

score = ResNet50_model_final.evaluate(X_val,Y_val)



print('Test loss:', score[0])

print('Test accuracy:', score[1])
# plot the accuracy and loss during every epoch



plt.plot(range(len(train_loss)), train_loss, label="train loss")

plt.plot(range(len(valid_loss)), valid_loss, label="valid loss")

plt.legend()

plt.title("Loss")

plt.show()



plt.plot(range(len(train_acc)), train_acc, label="train accuracy")

plt.plot(range(len(valid_acc)), valid_acc, label="valid accuracy")

plt.legend()

plt.title("Accuracy")

plt.show()
# functin that plot the confusion matrix, helping understand the performance of the model

def plot_confusion_matrix(cm, classes,

                          normalize=True,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (5,5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cm=(cm*100+.01).astype(int)/100



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# to plot the ROC curve to see the performance of our model



Y_val_roc = np.argmax(Y_val,axis=1) # ground truth

# ResNet50_model_final = keras.models.load_model("/kaggle/input/cats-dogsclassification/model_noshuffle_noAug_allimage.h5")

y_probas = ResNet50_model_final.predict(X_val) # prediction (probability)

Y_pred_classes = np.argmax(y_probas,axis=1) 

fpr, tpr, _ = roc_curve(Y_val_roc,Y_pred_classes)

roc_auc = auc(fpr, tpr)



fig = plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
# plot the confusion matrix

dict_characters = {0:'Dog',1:'Cat'}

confusion_mtx = confusion_matrix(Y_val_roc, Y_pred_classes) 

plot_confusion_matrix(confusion_mtx, classes =list(dict_characters.values()))

plt.show()
# make prediction on testing data and output the csv file for submission

if not initial_train == True: 

    ResNet50_model_final = keras.models.load_model("/kaggle/input/cats-dogsclassification/model_shuffle_noAug_allimage.h5")



y_probas = ResNet50_model_final.predict(x_test)



# read the sample_submission as my csv file format

test = pd.read_csv("/kaggle/input/ml-marathon-final/sample_submission.csv")

test['Predicted'] = y_probas[:,1]

test['ID'] = names

test = test[["ID", "Predicted"]]

test= test.sort_values(by=['ID'])

test.to_csv("20190816ResNet50_bs4.csv", header=["ID", "Predicted"], index=False) # submission format

test.head(10)