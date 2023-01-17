import os
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw
import glob
import cv2
import random
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Sequential
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import VGG19
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import seaborn as sns
from sklearn import preprocessing
from collections import Counter
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder

train_images_count = sum([len(files) for r, d, files in os.walk('../input/landmark-recognition-2020/train')])
print('The number of train images is :', train_images_count)
test_images_count = sum([len(files) for r, d, files in os.walk('../input/landmark-recognition-2020/test')])
print('The number of test images is :', test_images_count)
print('The total number of images is :', train_images_count+test_images_count)

Base_path = '../input/landmark-recognition-2020/'
Train_DIR = f'{Base_path}/train'
Test_DIR = f'{Base_path}/test'
train = pd.read_csv(f'{Base_path}/train.csv')
submission = pd.read_csv(f'{Base_path}/sample_submission.csv')
print('Reading data completed')
my_train_data = train
my_test_data = submission
#We will use something called label encoding/decoding.
#The way this work is assigning each class a known label that can be used later,during the prediction, to know which class is being predicted. \n we will use the number of classes for indexing
print("This is how the raw data looks like.\n", my_train_data)
#Encoding
# le = preprocessing.LabelEncoder()
# le.fit(my_train_data.landmark_id.values)
# new_df = le.transform(my_train_data["landmark_id"])
# my_train_data.landmark_id = new_df
print("This is how the data looks like after the encoding.\n", my_train_data)
#adding a filename column which will contain the full path to the sample, which will later be used to access the data.
my_train_data["filename"] = my_train_data.id.str[0]+"/"+my_train_data.id.str[1]+"/"+my_train_data.id.str[2]+"/"+my_train_data.id+".jpg"
my_test_data["filename"] = my_test_data.id.str[0]+"/"+my_test_data.id.str[1]+"/"+my_test_data.id.str[2]+"/"+my_test_data.id+".jpg"
#adding a "label" column which is basically the same as "landmark_id" but as string. This will be needed later for the data generator.
my_train_data["label"] = my_train_data.landmark_id.astype(str)
print("This is how the data looks like after the encoding and adding the needed columns. \n",my_train_data)
landmark_count=pd.value_counts(my_train_data["landmark_id"])
landmark_count=landmark_count.reset_index()
landmark_count.rename(columns={"index":'landmark_ids','landmark_id':'count'},inplace=True)
print(landmark_count)
# sample = landmark_count[0:50]
# sample.rename(columns={"index":'landmark_ids','landmark_id':'count'},inplace=True)
# sample.sort_values(by=['count'],ascending=False,inplace=True)
# sample['landmark_ids']=sample['landmark_ids'].map(str)
# sample.info()
# print(sample)
number_of_classes = len(my_train_data['landmark_id'].unique())
print('Number of unique classes in training images:',number_of_classes)
nb_images_pr_class= pd.DataFrame(my_train_data.landmark_id.value_counts())
nb_images_pr_class.reset_index(inplace=True)
nb_images_pr_class.columns = ['landmark_id','count']
print(nb_images_pr_class)
                

fig=plt.figure(figsize=(18, 3))
n = plt.hist(my_train_data["landmark_id"],bins=my_train_data["landmark_id"].unique())
plt.title("Distribution of labels")
plt.xlabel("Landmark_id")
plt.ylabel("Number of images")
plt.show()
less_than_five = 0
between_five_and_ten = 0
for x in n[0]:
    if(x<5):
        less_than_five+=1
    elif(x<10):
        between_five_and_ten+=1
    
print('Number of classes that have less than 5 training samples :',less_than_five)
print('Number of classes that have between 5 and 10 training samples :',between_five_and_ten)
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')
plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(2, 2, figsize=(10, 8))

curr_row = 0
for i in range(4):
    example = cv2.imread(train_list[random.randint(0,len(train_list)-1)])
    example = example[:,:,::-1]
    
    col = i%2
    axarr[col, curr_row].imshow(example)
    if col == 1:
        curr_row += 1
plt.figure(figsize = (10, 8))
plt.title('Landmark ID Distribuition')
sns.distplot(my_train_data['landmark_id'])
plt.show()

print("The data distribution will affect the training process negatively, since some classes have a very large number of samples when compared with other classes. \n For example the largest class contains 6272 images where there are 4749 classes which contain only 2 images, meaning that the largest class will have higher impact \n when trying to do some predictions after training the model. \n in other words, when trying to predict a sample from the small classes, 99% of the times, the classifier will predict it as it belongs to the large class, which is refered to as generalization. ") 
c = my_train_data.landmark_id.values
count = Counter(c).most_common(100)
print(len(count), count[-1])
# only keep 100 classes
keep_labels = [i[0] for i in count]
train_keep = my_train_data[my_train_data.landmark_id.isin(keep_labels)]
print(train_keep)
plt.figure(figsize = (10, 8))
plt.title('Landmark ID Distribuition')
sns.distplot(train_keep['landmark_id'])

plt.show()
val_rate = 0.2# The percentage of the validation data
epochs = 5 # The maximum number of epochs
batch_size = 10 # The batch size
#opt = RMSprop(learning_rate=0.01, momentum = 0.9) # The used optimizer  
#loss_function = 'categorical_crossentropy' # The loss function


#First we start by creating the generator object, which will work as the container of our data.
#This generator object will be split into two, validation and training, where the size of the validating will be specificed using the "validation_split" parameter.
#and the rest will belong to the training.
gen = ImageDataGenerator(validation_split=val_rate, rescale=1.0/255.0)

train_gen = gen.flow_from_dataframe(
    my_train_data,
    directory="/kaggle/input/landmark-recognition-2020/train/",
    x_col="filename",
    y_col="label", # The argument to this parameter has to be string, and that is why we created the "label" column at the beginning.
    target_size=(256, 256),# Since the images in the dataset have different sizes, they will get resized into a unified size-256,256 to each color channel-
    color_mode="rgb",
    class_mode="categorical",#We have to use the "categorical" argument since we multiple classes
    batch_size=batch_size,
    shuffle=True, # This parameter will shuffle the data while being passed to the model
    subset="training",# The name of the subset
    interpolation="nearest", # This parameter is used to interpolate the pixel values when images get scaled to the target size.
    validate_filenames=False)
    
val_gen = gen.flow_from_dataframe(
    my_train_data,
    directory="/kaggle/input/landmark-recognition-2020/train/",
    x_col="filename",
    y_col="label",
    target_size=(256, 256),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    subset="validation",
    interpolation="nearest",
    validate_filenames=False)
samples = 20000

data = my_train_data.loc[:samples,:]
classes = len(data['landmark_id'].unique())

lencoder = LabelEncoder()
lencoder.fit(data["landmark_id"])

print(classes)
model = Sequential()
model.add(Input(shape=(256,256,3)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(64, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(128, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(Conv2D(256, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(4096, activation = "relu"))
model.add(Dense(4096, activation = "relu"))
model.add(Dense(classes, activation="softmax"))
print(model.summary())
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(data.landmark_id),
                                                 data.landmark_id)
class_weights = dict(enumerate(class_weights))
class_weights

opt = Adagrad(learning_rate = 0.001, initial_accumulator_value=0.1, epsilon=1e-07)
model.compile(optimizer=opt,
             loss="categorical_crossentropy",
             metrics=["accuracy"])
train_steps = int(len(data)*(1-val_rate))//batch_size
val_steps = int(len(data)*val_rate)//batch_size

#model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)
history = model.fit(train_gen, steps_per_epoch = train_steps, epochs = epochs,
                    validation_data = val_gen, validation_steps=val_steps, 
                    class_weight=class_weights
                   )

model.save("weightedClasses.h5")

print(history.history.keys())
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
best_model = load_model("best_model.h5")
test_gen = ImageDataGenerator().flow_from_dataframe(
    my_test_data,
    directory="/kaggle/input/landmark-recognition-2020/test/",
    x_col="filename",
    y_col=None,
    weight_col=None,
    target_size=(256, 256),
    color_mode="rgb",
    classes=None,
    class_mode=None,
    batch_size=1,
    shuffle=True,
    subset=None,
    interpolation="nearest",
    validate_filenames=False)
predictions_list = best_model.predict_generator(test_gen, verbose=1, steps=len(my_test_data))
y_pred = np.argmax(predictions_list, axis=-1)
y_prob = np.max(predictions_list, axis=-1)
print(y_pred.shape, y_prob.shape)
y_uniq = np.unique(train_keep.landmark_id.values)

y_pred = [y_uniq[Y] for Y in y_pred]


print(y_pred)
for i in range(len(my_test_data)):
    my_test_data.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])
my_test_data = my_test_data.drop(columns="filename")
my_test_data.to_csv("submission.csv", index=False)
my_test_data