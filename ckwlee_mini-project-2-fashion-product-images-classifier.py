

#setting up all the essentials

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import matplotlib.image as mpimg





#machine learning libraries 

from keras.models import Sequential, Model

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras import regularizers, optimizers

from keras.applications.mobilenet_v2 import MobileNetV2

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler



import os # accessing directory structure
#import and setup training dataset

TRAINING_DATASET_PATH = "/kaggle/input/myntradataset/"

training_data = pd.read_csv(TRAINING_DATASET_PATH + "styles.csv", error_bad_lines=False)

training_data['image'] = training_data.apply(lambda row: str(row['id']) + ".jpg", axis=1)

training_data = training_data.sample(frac=1).reset_index(drop=True)

training_data.head(10)
#import and setup test dataset

TEST_DATASET_PATH = "/kaggle/input/"

test_data = pd.read_csv(TEST_DATASET_PATH + "styles.csv", nrows=128, error_bad_lines=False)

test_data['image'] = test_data.apply(lambda row: str(row['id']) + ".jpg", axis=1)

test_data = test_data.sample(frac=1).reset_index(drop=True)

test_data.head(10)
#create additional images for training, while splitting dataset into two datasets for training and validation data

train_datagen = ImageDataGenerator(

        rescale=1./255,

        validation_split=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



#process training dataset images

training_generator = train_datagen.flow_from_dataframe(

    dataframe=training_data,

    directory=TRAINING_DATASET_PATH + "images",

    x_col="image",

    y_col="masterCategory",

    target_size=(96,96),

    batch_size=32,

    subset="training"

)



#process validation dataset images

valid_generator = train_datagen.flow_from_dataframe(

    dataframe=training_data,

    directory=TRAINING_DATASET_PATH + "images",

    x_col="image",

    y_col="masterCategory",

    target_size=(96,96),

    batch_size=32,

    subset="validation"

)



#process testing dataset images

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(

    dataframe=test_data,

    directory=TEST_DATASET_PATH + "images",

    x_col="image",

    y_col="masterCategory",

    target_size=(96,96),

    batch_size=32

)



#create classes (categories to be sorted into)

classes = len(training_generator.class_indices)
# # create the base pre-trained model

base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(classes, activation='softmax')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
#training the model

model.fit_generator(

    generator=training_generator,

    steps_per_epoch=training_generator.n/training_generator.batch_size,



    validation_data=valid_generator,

    validation_steps=valid_generator.n/valid_generator.batch_size,



    epochs=3 #the number of times to repeat training for higher accuracy

)



#save model to file

model.save('/kaggle/working/model.h5')
# Evaluate model with validation data set

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



evaluation = model.evaluate_generator(generator=valid_generator,

steps=STEP_SIZE_TEST)



# Print out final values of all metrics

key2name = {'acc':'Accuracy', 'loss':'Loss', 

    'val_acc':'Validation Accuracy', 'val_loss':'Validation Loss'}

results = []

for i,key in enumerate(model.metrics_names):

    results.append('%s = %.2f' % (key2name[key], evaluation[i]))

print(", ".join(results))



fig = plt.figure(figsize=(10,5))



# Plot loss function

plt.subplot(222)

plt.plot(history.history['loss'],'bo--', label = "loss")

plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")

plt.title("train_loss vs val_loss")

plt.ylabel("loss")

plt.xlabel("epochs")



# Plot accuracy

plt.subplot(221)

plt.plot(history.history['acc'],'bo--', label = "acc")

plt.plot(history.history['val_acc'], 'ro--', label = "val_acc")

plt.title("train_acc vs val_acc")

plt.ylabel("accuracy")

plt.xlabel("epochs")

plt.legend()





plt.legend()

plt.show()
#reset the test generator to make sure the order is correct 

test_generator.reset()



#make predictions

pred=model.predict_generator(test_generator,

steps=STEP_SIZE_TEST,

verbose=1)
#map out the indices to the category names to make more legible 

predicted_class_indices=np.argmax(pred,axis=1)



labels = (training_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

labels



#compare the prediction to the original test dataset

filenames=test_generator.filenames

actual = test_data['masterCategory']



match = []

count = 0

correct = 0

for x in predictions: 

    if predictions[count] == actual[count]:

        match.append("True")

        correct = correct + 1

    else: 

        match.append("False")

    count = count + 1

        
results=pd.DataFrame({"Filename":filenames,

                      "Predictions":predictions,

                      "Actual":actual,

                      "Correct":match})



#expore results to csv and print accuracy

results.to_csv("results.csv",index=False)

print("Number Predictions Correct: " + str(correct) + " out of " + str(count) + " Percent Correct: " + str(correct/count))