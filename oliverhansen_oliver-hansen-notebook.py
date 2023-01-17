import tensorflow as tf

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import random

import os

import cv2

import multiprocessing

import tensorflow.keras.layers as L

from PIL import Image

from collections import Counter

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.utils import to_categorical

from keras import Model

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

from tensorflow.keras.applications.xception import Xception

from keras.optimizers import Adam 

from keras.optimizers import Adagrad











#Finding available CPU threads just in case i have to use multiprocessing in the training. Just in case i run out of GPU time. 

from tensorflow.python.client import device_lib

threads = multiprocessing.cpu_count()

print("There are% 2d threads available " %(threads))  





print(device_lib.list_local_devices())



tf.debugging.set_log_device_placement(False)



# Create some tensors to test that the GPU is turned on and available. 

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)



print(c)

# Read data from train and test csv files. 

train_data = pd.read_csv('../input/landmark-recognition-2020/train.csv')

test_data = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')



# See shape of our data to determine size of dataset 

print("The size of the dataset is% 2d images" %(train_data.shape[0]))  

train_data.head(5)

# Get amount of classes in our dataset. 

class_count = train_data.landmark_id.value_counts()

print(class_count.head(20))

print("There are% 2d classes " %(class_count.shape[0]))  

# Histogram showing instances per class

hist_data = pd.DataFrame(train_data['landmark_id'].value_counts()) 

hist_data.reset_index(inplace=True) 

hist_data.columns=['landmark_id','count']



figure = plt.figure(figsize = (14, 14))



plt.hist(hist_data['count'],500,range = (0,300))#Histogram of the distribution

plt.xlabel("Training Samples")

plt.ylabel("Occurences")



#plt.hist(train_data["landmark_id"],bins=train_data["landmark_id"].unique())

#freq_info = n[0]



#plt.xlim(0,hist_data['landmark_id'].max())

#plt.ylim(0,2000)

#plt.xlabel('Landmark ID')

#plt.ylabel('Number of images')

# Classes with less than 5 training samples

belowFive = class_count[class_count < 5].index.shape[0]

print("There are% 2d classes with less than 5 training samples" %(belowFive))  

# Classes with between 5 and 10 training samples

filtered_classes = class_count[class_count >= 5]

betweenFiveTen = filtered_classes[filtered_classes <= 10].index.shape[0]

print("There are% 2d classes with between 5 and 10 training samples" %(betweenFiveTen))  

# Select 4 random images from the dataset.

def get_image_path(img_id):

    image_path = f"../input/landmark-recognition-2020/train/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"

    img = np.array(Image.open(image_path).resize((224, 224), Image.LANCZOS))

    return img



classes = train_data.landmark_id.unique()

random_classes = random.choices(classes, k=4)

    

figure = plt.figure(figsize = (14, 14))



for i in range(len(random_classes)):

    random_image = train_data.loc[train_data['landmark_id'] == random_classes[i]]

    random_path = random.choice(random_image.id.values)

    # Display the randomly selected images.

    image = get_image_path(random_path)

    figure.add_subplot(2, 2, i+1)

    plt.title(random_path)

    plt.imshow(image)





#Hyperparameters 

val_rate = 0.2

batch_size = 32

min_samples = 20

img_width = img_height = 192



selected_classes = class_count[class_count >= min_samples].index

train_data = train_data.loc[train_data.landmark_id.isin(selected_classes)]

print(train_data.shape)



keep_classes = 1000 # Since many classes have a low sample count we only keep the 1000 most frequent classes in the dataset. 

#Only keep the 1000 most common classes in the dataset. 

c = train_data.landmark_id.values

count = Counter(c).most_common(keep_classes)

keep_labels = [i[0] for i in count]

train_data = train_data[train_data.landmark_id.isin(keep_labels)]



train_data['landmark_id'] = train_data.landmark_id.astype(str)

train_data["id"] = train_data.id.str[0]+"/"+train_data.id.str[1]+"/"+train_data.id.str[2]+"/"+train_data.id+".jpg"

print(train_data.shape)

num_classes = len(count)

print(num_classes)



train_datagen_augmented = ImageDataGenerator(

        validation_split=val_rate,

        rotation_range=10,

        rescale=1. / 255,      

        shear_range=0.2,       

        zoom_range=0.2,        

        horizontal_flip=True)  



datagen = ImageDataGenerator(rescale=1. / 255)





train_generator = train_datagen_augmented.flow_from_dataframe(

    train_data,

    directory="/kaggle/input/landmark-recognition-2020/train/",

    x_col="id",

    y_col="landmark_id",

    weight_col=None,

    target_size=(img_width, img_height),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,

    subset="training",

    interpolation="nearest",

    validate_filenames=False)



validation_generator = train_datagen_augmented.flow_from_dataframe(

    train_data,

    directory="/kaggle/input/landmark-recognition-2020/train/",

    x_col="id",

    y_col="landmark_id",

    weight_col=None,

    target_size=(img_width, img_height),

    color_mode="rgb",

    classes=None,

    class_mode="categorical",

    batch_size=batch_size,

    shuffle=True,

    subset="validation",

    interpolation="nearest",

    validate_filenames=False)

def my_model(input_shape, num_classes, dropout, learning_rate = 0.0002):

    '''

    base_model = Xception(weights=None, include_top=False, input_shape=input_shape)

    base_model.load_weights("../input/keraspretrainedmodel/xception_weights_tf_dim_ordering_tf_kernels_notop.h5")

    x = base_model.output

    x = L.GlobalAveragePooling2D()(x)

    x = L.Dropout(dropout)(x)

    predictions = L.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    '''



    base_model = Xception(input_shape=input_shape, 

                           weights=None, include_top=False)

    base_model.load_weights("../input/keraspretrainedmodel/xception_weights_tf_dim_ordering_tf_kernels_notop.h5")





    x = base_model.output

    x = L.Dropout(dropout)(x)

    x = L.SeparableConv2D(256, kernel_size=(3, 3), activation='relu',kernel_initializer = tf.keras.initializers.he_uniform(seed=1))(x)

    x = L.BatchNormalization()(x)

    x = L.SeparableConv2D(128, kernel_size=(3, 3), activation='relu',kernel_initializer = tf.keras.initializers.he_uniform(seed=3))(x)

    x = L.BatchNormalization()(x)

    x = L.SeparableConv2D(num_classes,kernel_size = (1,1), depth_multiplier=1, activation = 'relu',

                kernel_initializer = tf.keras.initializers.he_uniform(seed=0),

                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)

                )(x)

    x = L.GlobalMaxPooling2D()(x)

    x = L.BatchNormalization()(x)

    x = L.Flatten()(x)



    pred = L.Dense(num_classes, activation = 'softmax')(x)



    for layer in base_model.layers:

        layer.trainable = False



    model = Model(inputs = base_model.input,outputs = pred,name='model')



    model.compile(loss='categorical_crossentropy',experimental_steps_per_execution=8, optimizer = Adagrad(learning_rate=0.01), metrics='categorical_accuracy')



    model.summary()

    return model

model = my_model(input_shape = (img_width, img_height, 3), num_classes = num_classes, dropout = 0.5)
epochs = 30 # maximum number of epochs

train_samples  = int(len(train_data)*(1-val_rate))//batch_size

validation_samples  = int(len(train_data)*val_rate)//batch_size



print(train_samples)

print(validation_samples)


# Model saving callback

checkpointer = ModelCheckpoint('basic_cnn.h5', monitor='val_loss', verbose=1, save_best_only=True)



# Early stopping

early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=10)



history = model.fit_generator(

        train_generator,

        steps_per_epoch=train_samples // batch_size,

        epochs=epochs,

        callbacks=[checkpointer, early_stopping],

        use_multiprocessing=True,

        workers=threads,

        verbose=1,

        validation_data=validation_generator,

        validation_steps=validation_samples // batch_size,)



model.save("basic_cnn.h5")
#for layer in model.layers:

#    layer.trainable = True

    

#model.compile(loss='categorical_crossentropy', experimental_steps_per_execution=8, optimizer = Adam(learning_rate=0.0001), metrics='categorical_accuracy')

#model.summary()
#history = model.fit_generator(

#        train_generator,

#        steps_per_epoch=train_samples // batch_size,

#        epochs=epochs,

#        callbacks=[checkpointer, early_stopping],

#        use_multiprocessing=True,

#        workers=threads,

#        verbose=1,

#        validation_data=validation_generator,

#        validation_steps=validation_samples // batch_size)


#model.evaluate_generator(validation_generator, validation_samples, use_multiprocessing=True, workers=threads, verbose=1)
submission = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")

submission["id"] = submission.id.str[0]+"/"+submission.id.str[1]+"/"+submission.id.str[2]+"/"+submission.id+".jpg"

best_model = load_model("basic_cnn.h5")



test_gen = ImageDataGenerator().flow_from_dataframe(

    submission,

    directory="/kaggle/input/landmark-recognition-2020/test/",

    x_col="id",

    y_col=None,

    weight_col=None,

    target_size=(img_width, img_height),

    color_mode="rgb",

    classes=None,

    class_mode=None,

    batch_size=1,

    shuffle=True,

    subset=None,

    interpolation="nearest",

    validate_filenames=False)
y_pred_one_hot = best_model.predict_generator(test_gen, verbose=1, steps=len(submission))
y_pred = np.argmax(y_pred_one_hot, axis=-1)

y_prob = np.max(y_pred_one_hot, axis=-1)

print(y_pred.shape, y_prob.shape)
def get_test_image_path(img_id):

    #image_path = f"../input/landmark-recognition-2020/test/{img_id[0]}{img_id[1]}{img_id[2]}{img_id}"

    image_path = f"../input/landmark-recognition-2020/test/{img_id}"



    img = np.array(Image.open(image_path).resize((224, 224), Image.LANCZOS))

    return img
y_uniq = np.unique(train_data.landmark_id.values)

y_pred = [y_uniq[Y] for Y in y_pred]
temp_sub = submission



for i in range(len(temp_sub)):

    temp_sub.loc[i, "landmarks"] = str(y_pred[i])



temp_sub.insert(2, "pred", y_prob)    



worst_preds = temp_sub.sort_values(by=['pred'])

worst_preds = worst_preds[0:5]

best_preds = temp_sub.sort_values(by=['pred'], ascending=False)

best_preds = best_preds[0:5]
figure = plt.figure(figsize = (14, 14))

worst_images = worst_preds.id.values



for i in range(len(worst_images)):

    path = worst_images[i]

    # Display the randomly selected images.

    image = get_test_image_path(path)

    figure.add_subplot(3, 3, i+1)

    plt.title(worst_preds.pred.values[i])

    plt.imshow(image)
figure = plt.figure(figsize = (14, 14))

best_images = best_preds.id.values



for i in range(len(best_images)):

    path = best_images[i]

    image = get_test_image_path(path)

    figure.add_subplot(3, 3, i+1)

    plt.title(best_preds.pred.values[i])

    plt.imshow(image)
for i in range(len(submission)):

    submission.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])

    splitText1 = submission.loc[i, "id"].split("/")

    splitText2 = splitText1[3].split(".")

    submission.loc[i, "id"] = splitText2[0]





submission = submission.drop(columns="pred")

#submission = submission.drop(columns="id")

submission.to_csv("submission.csv", index=False)

submission
#model.evaluate_generator(validation_generator, 

#                         validation_samples, 

#                         verbose=1,

#                         use_multiprocessing=True,

#                         workers=threads)
#fig, ax = plt.subplots()

#plt.plot(history.history['accuracy'])

#plt.plot(history.history['val_accuracy'])

#plt.title('Model accuracy')

#plt.ylabel('Accuracy')

#plt.xlabel('Epoch')

#plt.legend(['Train', 'Test'], loc='upper left')

#plt.show()
