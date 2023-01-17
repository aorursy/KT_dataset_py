import numpy as np

import pandas as pd 

import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D

from keras import applications

from pathlib import Path

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint, History

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

import random

import os
filenames = os.listdir("../input/dogs-vs-cats/train/train")

sample = random.choice(filenames)

print(sample)

image = load_img("../input/dogs-vs-cats/train/train/"+sample)

plt.imshow(image)
filenames = os.listdir("../input/dogs-vs-cats/train/train")

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})

df.head()
df.tail()
#See Total In count

df['category'].value_counts().plot.bar()
from sklearn.model_selection import train_test_split

#dimensions of our images.

img_width, img_height = 150, 150

IMG_SHAPE = (img_width, img_height, 3)



#IMAGE_FOLDER_PATH="../input/dogs-vs-cats/train"



train_data_dir = '../input/dogs-vs-cats/train/train'

validation_data_dir = '../input/dogs-vs-cats/train/train'

dataset = df

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

train_df, validate_df=train_test_split(df,

                                       test_size=0.2,

                                       random_state=42)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)



nb_train_samples = train_df.shape[0]

nb_validation_samples = validate_df.shape[0]



#nb_train_samples = 4000

#nb_validation_samples = 800

epochs = 50

batch_size = 40
datagen = ImageDataGenerator(rescale=1. / 255)



# build the VGG16 network

model = applications.VGG16(include_top=False, weights='imagenet')



#Traning Generator

train_generator = datagen.flow_from_dataframe(

    train_df,

    "../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(img_width, img_height),

    batch_size=batch_size,

    shuffle=True)

bottleneck_features_train = model.predict_generator(

    train_generator, nb_train_samples // batch_size)

#np.save('bottleneck_features_train.npy', bottleneck_features_train)
#Validation Generator

validation_generator = datagen.flow_from_dataframe(

    validate_df,

    "../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(img_width, img_height),

    batch_size=batch_size,

    shuffle=False)

bottleneck_features_validation = model.predict_generator(

    validation_generator, nb_validation_samples // batch_size)

#np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
os.makedirs("model")

#train_data = np.load('bottleneck_features_train.npy')

train_data =bottleneck_features_train

train_labels = np.array( 

    [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))



#validation_data = np.load('bottleneck_features_validation.npy')

validation_data = bottleneck_features_validation



validation_labels = np.array(

    [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))



model = Sequential()

model.add(Flatten(input_shape=train_data.shape[1:])) #shape (4, 4, 512) 

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(#optimizer='rmsprop',

              optimizer=optimizers.RMSprop(lr=1e-4),

              loss='binary_crossentropy', metrics=['accuracy'])



checkpointer = ModelCheckpoint(filepath = "model/model.weights.best.hdf5", 

            verbose=1,  save_best_only=True)

hist=model.fit(train_data, train_labels,

          epochs=epochs,

          batch_size=batch_size,

          validation_data=(validation_data, validation_labels),

          callbacks=[checkpointer] )
# Save neural network structure

model_structure = model.to_json()

f = Path("model/model_structure_AM.json")

f.write_text(model_structure)



# Save neural network's trained weights

model.save_weights("model/bottleneck_model_AM.h5")
def Polt_history(hist):

    acc = hist.history['accuracy']

    val_acc = hist.history['val_accuracy']



    loss = hist.history['loss']

    val_loss = hist.history['val_loss']



    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)

    plt.plot(acc, label='Training Accuracy')

    plt.plot(val_acc, label='Validation Accuracy')

    plt.legend(loc='lower right')

    plt.ylabel('Accuracy')

    plt.ylim([min(plt.ylim()),1])

    plt.title('Training and Validation Accuracy')



    plt.subplot(2, 1, 2)

    plt.plot(loss, label='Training Loss')

    plt.plot(val_loss, label='Validation Loss')

    plt.legend(loc='upper right')

    plt.ylabel('Cross Entropy')

    plt.ylim([0,1.0])

    plt.title('Training and Validation Loss')

    plt.xlabel('epoch')

    plt.show()



Polt_history(hist)

plt.savefig('model/hist.png')
# build the VGG16 network

base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=IMG_SHAPE )

print("base_model.layers", len(base_model.layers)) #19



#Feature extraction

#Freeze the convolutional base

for layer in base_model.layers[:15]:

    layer.trainable = False

    

# build a top model to put on top of the convolutional model

top_model = Sequential()

#top_model.add(GlobalAveragePooling2D())

top_model.add(Flatten(input_shape=base_model.output_shape[1:]))

top_model.add(Dense(256, activation='relu'))

top_model.add(Dropout(0.5))

top_model.add(Dense(1, activation='sigmoid'))



top_model_weights_path='model/bottleneck_model_AM.h5'

top_model.load_weights(top_model_weights_path)



model = Sequential()

model.add(base_model)

model.add(top_model)

model.summary()
epochs = 5

batch_size = 16



# prepare data augmentation configuration

train_datagen = ImageDataGenerator(

    rescale=1./ 255,

    rotation_range=15,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)

train_generator = train_datagen.flow_from_dataframe(

    train_df,

    "../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(img_width, img_height),

    batch_size=batch_size,

    shuffle=False)



#Validation Generator

test_datagen = ImageDataGenerator(rescale=1./ 255)

validation_generator = test_datagen.flow_from_dataframe(

    validate_df,

    "../input/dogs-vs-cats/train/train/", 

    x_col='filename',

    y_col='category',

    class_mode='binary',

    target_size=(img_width, img_height),

    batch_size=batch_size,

    shuffle=False)





#Early Stop

#To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased

#earlystop = EarlyStopping(patience=10)





checkpointer = ModelCheckpoint(filepath='model/model.weights.best_2.hdf5', 

                               verbose=1, save_best_only=True)



# compile the model with a SGD/momentum optimizer and a very slow learning rate.

model.compile(loss='binary_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),

              metrics=['accuracy'])



# fine-tune the model

hist=model.fit_generator(

    train_generator,

    samples_per_epoch=nb_train_samples,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size,

    #nb_val_samples=nb_validation_samples)

    callbacks=[checkpointer])
# Save neural network structure and weights

''''''

model_structure = model.to_json()

f = Path("model/model_structure2_AM.json")

f.write_text(model_structure)



model.save_weights("model/model_weights2_AM.h5")
Polt_history(hist)

plt.savefig('model/hist_AM.png')
import cv2

import glob

from tqdm import tqdm

import pandas as pd

from PIL import Image as img



x_test = []

def model_predict(itr,start,end,x_test):

	result = []

	#image_path = 'test1/'+"*["+rang+"].jpg"

	#image_path  ='/home/abdallah/datasets/dogs-vs-cats/train/cat*.jpg'

	#print (image_path)

	image_path  ='../input/dogs-vs-cats/test1/test1/*.jpg'

	test_list = glob.glob(image_path)[start:end]

	print("test_list",test_list[1])

	for i in tqdm(test_list):

	    temp = cv2.imread(i)

	    temp = cv2.resize(temp, (150, 150))

	    temp = temp.reshape(1, 150, 150, 3)

	    #out= model.predict_classes(temp)

	    #result.append(out[0][0])

	    result.append(model.predict_classes(temp)[0][0])

	    #if out == 1:    errors +=1



	#print("errors= ",errors, "from total of", i)

	idx = []

	for i in test_list:

	    name = Path(i).stem  #get the filename without the extension 

	    idx.append(name)

	#print(result)

	data = {"id": idx, "label": result}

	submission = pd.DataFrame(data)

	#print(submission)

	#submission.index += 1 

	#submission.to_csv('submission_AAA22.csv')  #, index_label='Event_id'



	plt.figure(figsize=(12, 24))

	count=0

	for row in test_list:

	    img = load_img(row, target_size=(150,150))

	    plt.subplot(6, 3, count+1)

	    plt.imshow(img)

	    plt.xlabel(row + '(' + "{}".format(result[count]) + ')' )

	    count +=1

	plt.tight_layout()

	plt.show()

	plt.savefig('model/predicted.png')



model_predict(1, 20,    38 , x_test)