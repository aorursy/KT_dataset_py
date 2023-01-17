#only if unable to load vgg16 weights- see comments bellow

#!pip install wget
##imports

#basic

import pandas as pd

import numpy as np 



#sklearn

from sklearn.metrics import confusion_matrix,classification_report

#from sklearn.utils import class_weight, shuffle

from sklearn.model_selection import train_test_split



#keras

from keras.preprocessing.image import ImageDataGenerator#, img_to_array, load_img 

from keras.models import Sequential,  Model

from keras import optimizers,applications 

from keras.layers import Dropout, Flatten, Dense

from keras.utils.np_utils import to_categorical 



#matplotlib

import matplotlib.pyplot as plt 

import matplotlib.image as mpimg

%matplotlib inline



#misc

import math 

import datetime

import time

import os

import cv2

#import wget



#to show predictions

#from keras.applications.vgg16 import decode_predictions

#from keras.applications.vgg16 import preprocess_input

#from keras.preprocessing import image

#how to migrate fro tf1 to tf2

#https://www.tensorflow.org/guide/migrate



tic = time.process_time()
# set random seeds for more reproducible results

from numpy.random import seed

seed(42)

#from tensorflow import set_random_seed #dosen't work on kaggle 

#set_random_seed(43)
#global variables

img_width,img_height=224,224

batch_size=16

epochs=4

#filepaths



#input datasets 

base_in='../input/dogs-vs-cats-redux-kernels-edition'

train_dir = "train"

test_dir="test"



#output data- will be saved in output/kaggle/working

bottleneck_features_train_path="bottleneck_features_train.npy"

bottleneck_features_validation_path="bottleneck_features_validation.npy"

top_model_weights_path = "bottleneck_fc_model.h5"

final_model_weights_path="final_model_weights.h5"
times=[]
#view image

path = os.path.join(base_in,train_dir)

i=0

for p in os.listdir(path):

    category = p.split(".")[0]

    img_array = cv2.imread(os.path.join(path,p))

    new_img_array = cv2.resize(img_array, dsize=(img_width,img_height))

    plt.imshow(new_img_array,cmap="gray")

    i+=1

    if i==3:

      break
#train data prep 

filenames = os.listdir(base_in+ "/" +train_dir)

files = []

labels = []

convert = lambda category : int(category == 'dog')

for file in filenames:

    if file.split(".")[-1]=="jpg":

        files.append(base_in+"/" +train_dir + "/" + file)

        category = file.split(".")[0]

        category = convert(category)

        labels.append(category)



df = pd.DataFrame({

    'filename': files,

    'label': labels

})
df.shape
df.head()
df.iloc[0,0]
#test data prep 

filenames = os.listdir(base_in+ "/" +test_dir)

files = []

#labels = []

#convert = lambda category : int(category == 'dog')

for file in filenames:

    if file.split(".")[-1]=="jpg":

        files.append(base_in+"/" +test_dir + "/" + file)

        #category = file.split(".")[0]

        #category = convert(category)

        #labels.append(category)



df_test = pd.DataFrame({

    'filename': files,

  #  'label': labels

})
df_test.shape
df_test.iloc[1,0]
#split training set

X_train, X_val = train_test_split(df.iloc[:,:], test_size=0.1, random_state=42)

print("y_train includes:",np.unique(X_train.iloc[:,1],return_counts=True),"total iages:",X_train.shape[0])

print("y_val includes:",np.unique(X_val.iloc[:,1],return_counts=True),"total iages:",X_val.shape[0])
#splt val and test sets

X_test = df_test

print("y_test total images:", X_test.shape[0])
# the weights for the following vgg16 model can be loaded useing:

#option 1:

#weights="imagenet" #easiest

#option 2:

#if that dosen't work you can download them manualy like I did from : 

#https://github.com/fchollet/deep-learning-models/releases/  

#make sure to use vgg16 weights NO TOP

# or https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

#OPTION 3 use wget

#import wget

#url= https://github.com/fchollet/deep-learning-models/releases/

#weights= wget.download(url)
#Loading vgc16 model

#vgg16_weights = "../input/vgg16-w-notop/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

vgg16 = applications.VGG16(include_top=False, weights='imagenet',input_shape=(img_width,img_height,3))
#initialize data generator instance

augment= False

if augment:

    datagen = ImageDataGenerator(

        rescale=1. / 255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)

else:

    datagen=ImageDataGenerator(rescale=1. / 255)
# datagenerator from df

#this predicts all train data vgg16 model final conv layer output using generator which passes data batch by batch 

#"Using the bottleneck features of a pre-trained network"

#training data

start = datetime.datetime.now()



generator = datagen.flow_from_dataframe(

        dataframe=X_train,

        x_col="filename",

        y_col="label",

        target_size=(img_width,img_height),

        batch_size=batch_size,

        class_mode=None,

        shuffle=False)

nb_train_samples = len(generator.filenames) 

num_classes = 2

predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 



bottleneck_features_train = vgg16.predict_generator(generator,predict_size_train)

np.save(bottleneck_features_train_path, bottleneck_features_train)

end= datetime.datetime.now()

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)
bottleneck_features_train.shape
#this predicts all validation data vgg16 model final conv layer output using generator which passes data batch by batch 

#validation data

start = datetime.datetime.now()

generator = datagen.flow_from_dataframe(

        dataframe=X_val,

        x_col='filename',

        y_col="label",

        target_size=(img_width,img_height),

        batch_size=batch_size,

        class_mode=None,

        shuffle=False)

nb_val_samples = len(generator.filenames) 

num_classes = 2

predict_size_val = int(math.ceil(nb_val_samples / batch_size)) 

bottleneck_features_validation = vgg16.predict_generator(generator, predict_size_val)

np.save(bottleneck_features_validation_path, bottleneck_features_validation)

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)
bottleneck_features_validation.shape
#We can then load our saved data (vgg16 final conv output) and train a small fully-connected model:

start = datetime.datetime.now()



train_data = np.load(bottleneck_features_train_path)

train_labels = X_train.iloc[:,1]



validation_data = np.load(bottleneck_features_validation_path)

validation_labels = X_val.iloc[:,1]



top_model = Sequential()

top_model.add(Flatten(input_shape=train_data.shape[1:],name="flat1"))

top_model.add(Dense(256, activation='relu',name="class1"))

top_model.add(Dropout(0.5,name="drop1"))

top_model.add(Dense(1, activation='sigmoid',name="output"))





top_model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])



history1=top_model.fit(train_data, train_labels,

          epochs=epochs,

          batch_size=batch_size,

          validation_data=(validation_data, validation_labels))

top_model.save_weights(top_model_weights_path)



(eval_loss, eval_accuracy) = top_model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 

print("[INFO] Loss: {}".format(eval_loss)) 



end= datetime.datetime.now()

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)
#After instantiating the VGG base and loading its weights, we add our previously trained fully-connected classifier on top:

start = datetime.datetime.now()



# build a classifier model to put on top of the convolutional model

top_model = Sequential()

top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))

top_model.add(Dense(256, activation='relu'))

top_model.add(Dropout(0.5))

top_model.add(Dense(1, activation='sigmoid'))



# note that it is necessary to start with a fully-trained

# classifier (vgg16 w/ imagenet weights and "top classfier"- just trained above)

# in order to successfully do fine-tuning

top_model.load_weights(top_model_weights_path)



# add the model on top of the convolutional base

full_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))





#We then proceed to freeze all convolutional layers up to the last convolutional block:

# set the first 15 layers (up to the last conv block)

# to non-trainable (weights will not be updated)

for layer in full_model.layers[:15]:

    layer.trainable = False



# compile the model with a SGD/momentum optimizer

# and a very slow learning rate.

full_model.compile(loss='binary_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])

#lets see the full model

full_model.summary()

#full_model.load_weights(final_model_weights_path)# to load weights of the model I previously trained

end= datetime.datetime.now()

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)
###in stage 2 we will use binary data generators, all y and y_pred need to be strings

#conversion to strings for binary class

X_train.label=X_train.label.astype("str")

print("X_train.dtypes:","\n",X_train.dtypes)

X_val.label=X_val.label.astype("str")

print("X_val.dtypes:","\n",X_val.dtypes)

#X_test.label=X_test.label.astype("str")

#print("X_test.dtypes:","\n",X_test.dtypes)
#Finally, we start training the whole thing, with a very slow learning rate:

start = datetime.datetime.now()



epochs=5



# prepare data and optional augmentation configuration

augment= False

if augment:

    train_datagen = ImageDataGenerator(

        rescale=1. / 255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)

else:

    train_datagen=ImageDataGenerator(rescale=1. / 255)



test_datagen = ImageDataGenerator(rescale=1. / 255)



#will augment and throw the train data into full model

train_generator = train_datagen.flow_from_dataframe(

        dataframe=X_train,

        x_col='filename',

        y_col="label",

        target_size=(img_height,img_width),

        batch_size=batch_size,

        class_mode='binary')

#will a throw the val data into full model

validation_generator = test_datagen.flow_from_dataframe(

        dataframe=X_val,

        x_col='filename',

        y_col="label",

        target_size=(img_height, img_width),

        batch_size=batch_size,

        class_mode='binary')



nb_train_samples = len(train_generator.filenames) 

nb_val_samples = len(validation_generator.filenames) 





# fine-tune the model

history2=full_model.fit_generator(

        train_generator,

        steps_per_epoch=int(math.ceil(nb_train_samples / batch_size)),

        epochs=epochs,

        validation_data=validation_generator,

        validation_steps=int(math.ceil(nb_val_samples / batch_size)))

#steps_per_epoch= how many batches to pass before epoch is finished. In current setting whole dataset passed. 

full_model.save_weights(final_model_weights_path)



end= datetime.datetime.now()

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)
#test model on val data

start = datetime.datetime.now()

test_datagen = ImageDataGenerator(rescale=1. / 255)

#will a throw the val data into full model

val_generator = test_datagen.flow_from_dataframe(

        dataframe=X_val,

        x_col='filename',

        y_col="label",

        target_size=(img_height, img_width),

        batch_size=batch_size,

        class_mode='binary')

nb_val_samples = len(val_generator.filenames) 

end= datetime.datetime.now()

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)
start = datetime.datetime.now()

(eval_loss, eval_accuracy) =full_model.evaluate_generator(val_generator,steps=int(math.ceil(nb_val_samples / batch_size)))

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 

print("[INFO] Loss: {}".format(eval_loss)) 



end= datetime.datetime.now()

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)
Y_pred[:5]
#test generator

start = datetime.datetime.now()

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_dataframe(

        dataframe=X_test,

        x_col='filename',

        target_size=(img_height, img_width),

        batch_size=batch_size,

        class_mode=None)

nb_test_samples = len(test_generator.filenames) 



#Y_test_pred = full_model.predict_generator(test_generator, steps=int(math.ceil(nb_test_samples / batch_size)))

#y_prob=Y_test_pred 



end= datetime.datetime.now()

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)
#make predictions

start = datetime.datetime.now()

Y_test_pred = full_model.predict_generator(test_generator, steps=int(math.ceil(nb_test_samples / batch_size)))

end= datetime.datetime.now()

elapsed= end-start

print ("Time: ", elapsed)

times.append(elapsed)


#test data prep  to df and csv

filenames =X_test.filename

ids = []

labels = Y_test_pred.flatten()

#convert = lambda category : int(category == 'dog')

for file in filenames:

    #if file.split(".")[-1]=="jpg":

    # files.append(base_in+"/" +train_dir + "/" + file)

    new_id = file.split(".")[0]

    #category = convert(category)

    ids.append(new_id)



df_sub = pd.DataFrame({

    'id': ids,

    'label': labels

})

sub1=df_sub.to_csv('submission.csv',index=True)

#kaggle competitions submit -c dogs-vs-cats-redux-kernels-edition -f sub1
print("total net runtime:",np.sum(times))

toc= time.process_time()

print("start to end time:",(toc-tic)/60,"min")
from IPython.display import FileLink

FileLink('submission.csv')
FileLink("bottleneck_features_train.npy")
FileLink("bottleneck_features_validation.npy")
FileLink("bottleneck_fc_model.h5")