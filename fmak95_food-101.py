import numpy as np

import pandas as pd

import os

from collections import defaultdict

from shutil import copy

from shutil import copytree, rmtree



import tensorflow as tf

import matplotlib.pyplot as plt
# Check if GPU is enabled

print(tf.__version__)

print(tf.test.gpu_device_name())
# Take a look at the dataset file structure

%cd /kaggle/input/food-101/food-101/

os.listdir('./food-101/images')
os.listdir('./food-101/meta')
# Using helper function from @Avinash!

# Helper method to split dataset into train and test folders

def prepare_data(filepath, src,dest):

    classes_images = defaultdict(list)

    with open(filepath, 'r') as txt:

        paths = [read.strip() for read in txt.readlines()]

        for p in paths:

            food = p.split('/')

            classes_images[food[0]].append(food[1] + '.jpg')



    for food in classes_images.keys():

        print("\nCopying images into ",food)

        if not os.path.exists(os.path.join(dest,food)):

            os.makedirs(os.path.join(dest,food))

        for i in classes_images[food]:

            copy(os.path.join(src,food,i), os.path.join(dest,food,i))

    print("Copying Done!")
# Prepare train dataset by copying images from food-101/images to food-101/train using the file train.txt

%cd /

print("Creating train data...")

prepare_data('/kaggle/input/food-101/food-101/food-101/meta/train.txt', '/kaggle/input/food-101/food-101/food-101/images', 'train')
# Prepare test data by copying images from food-101/images to food-101/test using the file test.txt

print("Creating test data...")

prepare_data('/kaggle/input/food-101/food-101/food-101/meta/test.txt', '/kaggle/input/food-101/food-101/food-101/images', 'test')
!ls /train
!ls /test
# Helper method to create train_mini and test_mini data samples

def dataset_mini(food_list, src, dest):

    if os.path.exists(dest):

        rmtree(dest) # removing dataset_mini(if it already exists) folders so that we will have only the classes that we want

        os.makedirs(dest)

    for food_item in food_list :

        print("Copying images into",food_item)

        copytree(os.path.join(src,food_item), os.path.join(dest,food_item))
# picking 10 of my favorite foods

food_list = ['nachos','ice_cream','sushi','french_fries','bibimbap','cheesecake','donuts','dumplings','waffles','omelette']

src_train = 'train'

dest_train = 'train_mini'

src_test = 'test'

dest_test = 'test_mini'
print("Creating train data folder with new classes")

dataset_mini(food_list, src_train, dest_train)
print("Creating test data folder with new classes")

dataset_mini(food_list, src_test, dest_test)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet_v2 import ResNet50V2

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D

import tensorflow.keras.backend as K



tf.compat.v1.disable_eager_execution()

K.clear_session()



train_dir = 'train_mini'

test_dir = 'test_mini'

img_height = 224

img_width = 224

batch_size = 16

num_classes = 10

num_train_samples = 7500

num_test_samples = 2500



# Training datagen and generator

train_datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.1,

    fill_mode='nearest')



train_gen = train_datagen.flow_from_directory(

    train_dir,

    target_size = (224,224),

    batch_size = batch_size,

    class_mode = 'categorical',

    subset = 'training')



# Validation datagen and generator

val_datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    rescale=1./255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.1,

    fill_mode='nearest')



val_gen = train_datagen.flow_from_directory(

    train_dir,

    target_size = (224,224),

    batch_size = 1,

    class_mode = 'categorical',

    subset = 'validation',

    shuffle = False)



# Testing datagen and generator

test_datagen = ImageDataGenerator()



test_gen = test_datagen.flow_from_directory(

    test_dir,

    target_size = (224,224),

    batch_size = 1,

    class_mode = 'categorical',

    shuffle = False)



# Building Model Architecture

resnet50 = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

resnet50.trainable = False

x = resnet50.output

x = GlobalAveragePooling2D()(x)

prediction = Dense(num_classes, activation='softmax')(x)



model = Model(inputs = resnet50.input, outputs = prediction)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit_generator(train_gen,

                    steps_per_epoch = train_gen.samples // batch_size,

                    epochs=5,

                    validation_data = val_gen,

                    validation_steps = val_gen.samples)
# Save model weights

model.save('transfer_learning_10class.hdf5') #Model with presaved ImageNet weights and only training the fully connected layers
plt.figure(figsize = (10,5))

plt.plot(history.history['val_accuracy'])

plt.plot(history.history['accuracy'])

plt.legend(['val_accuracy','train_accuracy'])

plt.show()
# Load the pre-trained model if it is not already

# from tensorflow.keras.models import load_model

# model = load_model(transfer_learning_10class.hdf5)
from tensorflow.keras import optimizers



#The last 17 layers as trainable (which is the last convolutional block + the fully connected layers)

for layer in model.layers:

    layer.trainable = False



for layer in model.layers[-17:]:

    layer.trainable = True



optim = optimizers.SGD(learning_rate=1e-4, momentum=0.9)

model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])

model.summary()
history2 = model.fit_generator(train_gen,

                    steps_per_epoch = train_gen.samples // batch_size,

                    validation_data=val_gen,

                    validation_steps=val_gen.samples,

                    initial_epoch =  history.epoch[-1]+1,

                    epochs=10)
#Save model after fine-tuning

model.save('fine_tuning_10class.hd5') #Model weights after fine-tuning
plt.figure(figsize=(10,5))

plt.plot(history.history['val_accuracy'] + history2.history['val_accuracy'])

plt.plot(history.history['accuracy'] + history2.history['accuracy'])

plt.plot([history.epoch[-1],history.epoch[-1]],

         plt.ylim(), label='Start Fine Tuning')

plt.legend(['val_accuracy','train_accuracy'])

plt.show()
# Test the model with images scraped from the internet

!wget -O sushi.jpg https://i.pinimg.com/originals/df/2a/90/df2a90188e22f3505691cb0fe84022c2.jpg

!wget -O french_fries.jpg https://bigoven-res.cloudinary.com/image/upload/d_recipe-no-image.jpg,t_recipe-256/french-fries-nacho-style-7e28eb.jpg

!wget -O ice_cream.jpg https://www.aspicyperspective.com/wp-content/uploads/2012/06/Chocolate-Dipped-Cones1-256x256.jpg

!wget -O nacho.jpg https://bigoven-res.cloudinary.com/image/upload/d_recipe-no-image.jpg,t_recipe-256/cheesy-nacho-dinner-ea1c9010003d2562400dc132.jpg

!wget -O omelette.jpg https://www.readyseteat.com/sites/g/files/qyyrlu501/files/uploadedImages/img_4043_1542.JPEG

!wget -O waffles.jpg https://www.aspicyperspective.com/wp-content/uploads/2014/03/best-waffle-recipe-9-256x256.jpg

!wget -O cheesecake.jpg https://bigoven-res.cloudinary.com/image/upload/d_recipe-no-image.jpg,t_recipe-256/chantals-new-york-cheesecake-21.jpg

!wget -O bibimbap.jpg https://mealplannerpro.com/images/recipes/4/731974_256x256.jpg

!wget -O donuts.jpg https://bigoven-res.cloudinary.com/image/upload/t_recipe-256/raised-doughnuts-5.jpg

!wget -O dumplings.jpg https://i.pinimg.com/474x/bb/f8/e9/bbf8e98f99e4bea3a488c9b077cb244c--chinese-dumplings-dips.jpg
!ls
import cv2

sushi = cv2.imread('./sushi.jpg')[:,:,::-1]

french_fries = cv2.imread('./french_fries.jpg')[:,:,::-1]

ice_cream = cv2.imread('./ice_cream.jpg')[:,:,::-1]

waffles = cv2.imread('./waffles.jpg')[:,:,::-1]

nacho = cv2.imread('./nacho.jpg')[:,:,::-1]

omelette = cv2.imread('./omelette.jpg')[:,:,::-1]

bibimbap = cv2.imread('./bibimbap.jpg')[:,:,::-1]

donuts = cv2.imread('./donuts.jpg')[:,:,::-1]

dumplings = cv2.imread('./dumplings.jpg')[:,:,::-1]

cheesecake = cv2.imread('./cheesecake.jpg')[:,:,::-1]



X_test = np.array([sushi/255,french_fries/255,ice_cream/255,waffles/255,nacho/255,

                   omelette/255,bibimbap/255,donuts/255,dumplings/255,cheesecake/255])

X_test.shape
#Plot image with its prediction

predictions = model.predict(X_test)

predictions = np.argmax(predictions,axis=1)



ind_2_labels = {v: k for k, v in train_gen.class_indices.items()}



plt.figure(figsize=(15,10))

plt.subplot(2,5,1)

plt.imshow(sushi)

plt.title(ind_2_labels[predictions[0]])



plt.subplot(2,5,2)

plt.imshow(french_fries)

plt.title(ind_2_labels[predictions[1]])



plt.subplot(2,5,3)

plt.imshow(ice_cream)

plt.title(ind_2_labels[predictions[2]])



plt.subplot(2,5,4)

plt.imshow(waffles)

plt.title(ind_2_labels[predictions[3]])



plt.subplot(2,5,5)

plt.imshow(nacho)

plt.title(ind_2_labels[predictions[4]])



plt.subplot(2,5,6)

plt.imshow(omelette)

plt.title(ind_2_labels[predictions[5]])



plt.subplot(2,5,7)

plt.imshow(bibimbap)

plt.title(ind_2_labels[predictions[6]])



plt.subplot(2,5,8)

plt.imshow(donuts)

plt.title(ind_2_labels[predictions[7]])



plt.subplot(2,5,9)

plt.imshow(dumplings)

plt.title(ind_2_labels[predictions[8]])



plt.subplot(2,5,10)

plt.imshow(cheesecake)

plt.title(ind_2_labels[predictions[9]])
def get_activation_map(model,image,label):

    

    image = image/255

    image = np.expand_dims(image,axis=0)

    

    

    # Run the model to get predicted label as well as convolutional layer output

    prediction = model.predict(image)

    class_id = np.argmax(prediction[0])

    last_conv_layer = model.get_layer('post_relu')

    get_output = K.function([model.input],[last_conv_layer.output])

    conv_layer_output = get_output([image])

    conv_layer_output = conv_layer_output[0][0]

    

    # Get weights of the predicted label

    predicted_weights = model.layers[-1].get_weights()[0][:,class_id]

    

    # Calculate heatmaps

    heatmap_predicted = np.zeros(dtype = np.float32, shape = (8,8))

    for i in range(2048):

        heatmap_predicted += predicted_weights[i] * conv_layer_output[:,:,i]

    print(class_id)

    # Get weights of the actual label (if prediction is wrong)

    if class_id != label:

        actual_weights = model.layers[-1].get_weights()[0][:,label]

        heatmap_actual = np.zeros(dtype = np.float32, shape = (8,8))

        for i in range(2048):

            heatmap_actual += actual_weights[i] * conv_layer_output[:,:,i]

        heatmap_actual = np.maximum(heatmap_actual, 0)

        return heatmap_predicted, heatmap_actual

    

    return heatmap_predicted, None
heatmap_predicted, heatmap_actual = get_activation_map(model, dumplings, train_gen.class_indices['dumplings'])



heatmap_predicted = cv2.resize(heatmap_predicted, (dumplings.shape[0],dumplings.shape[1]))

heatmap_actual = cv2.resize(heatmap_actual, (dumplings.shape[0],dumplings.shape[1]))



plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.imshow(dumplings, alpha=0.65)

plt.imshow(heatmap_predicted, alpha=0.5, cmap='jet')

plt.title("Regions of interest that make model think waffles")



plt.subplot(1,2,2)

plt.imshow(dumplings, alpha=0.65)

plt.imshow(heatmap_actual, alpha=0.5, cmap='jet')

plt.title("Regions of interest that make model think dumplings")
heatmap_predicted, heatmap_actual = get_activation_map(model, sushi, train_gen.class_indices['sushi'])



heatmap_predicted = cv2.resize(heatmap_predicted, (sushi.shape[0],sushi.shape[1]))

heatmap_actual = cv2.resize(heatmap_actual, (sushi.shape[0],sushi.shape[1]))



plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.imshow(sushi, alpha=0.65)

plt.imshow(heatmap_predicted, alpha=0.5, cmap='jet')

plt.title("Regions of interest that make model think waffles")



plt.subplot(1,2,2)

plt.imshow(sushi, alpha=0.65)

plt.imshow(heatmap_actual, alpha=0.5, cmap='jet')

plt.title("Regions of interest that make model think sushi")
heatmap_predicted, heatmap_actual = get_activation_map(model, waffles, train_gen.class_indices['waffles'])



heatmap_predicted = cv2.resize(heatmap_predicted, (waffles.shape[0],waffles.shape[1]))



plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.imshow(waffles, alpha=0.65)

plt.imshow(heatmap_predicted, alpha=0.5, cmap='jet')

plt.title("Regions of interest that make model think waffles")
predictions = []

labels = []



val_gen.reset()

for _ in range(val_gen.samples):

    X_val, y_val = next(val_gen)

    pred = model.predict(X_val)

    predictions.append(pred)

    labels.append(y_val)

predictions = np.array(predictions)

predictions = np.argmax(predictions,axis=2)

labels = np.array(labels)

labels = np.argmax(labels, axis=2)

print(predictions.shape)

print(labels.shape)
from sklearn.metrics import accuracy_score



accuracy = accuracy_score(predictions,labels)

print("Accuracy without TTA: {}".format(accuracy))
tta_steps = 5

predictions = []

for _ in range(tta_steps):

    val_gen.reset()

    preds = []

    for _ in range(val_gen.samples):

        X_val, y_val = next(val_gen)

        pred = model.predict(X_val)

        preds.append(pred)

    predictions.append(preds)

predictions = np.array(predictions)

predictions = np.mean(predictions,axis=0)

predictions = np.argmax(predictions,axis=2)
accuracy = accuracy_score(predictions,labels)

print("Accuracy with TTA: {}".format(accuracy))