import sys
import os
import csv
import shutil
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


csv_path = "../input/face-mask-detection-dataset/train.csv"
img_dir = "../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images"

def cleaning_csv(filepath):
    '''
    Inputs a csv file and removes all the unecessary bounding box entries and
    retains only the required classes for training mask_no_mask classifirer
    --filename : path of the csv file    
    '''
    csv_path = filepath
    data = pd.read_csv(csv_path)
    #print(data.info())
    #print(data['classname'].unique())

    array = ['face_with_mask', 'face_no_mask', 'face_with_mask_incorrect']
    data_reduce = pd.DataFrame(data.loc[(data['classname'].isin(array))])

    data_reduce['class'] = np.where((data_reduce['classname']=='face_with_mask') |
                                    (data_reduce['classname']=='face_with_mask_incorrect'),
                                    'with_mask','without_mask')
    del data_reduce['classname']
    data_reduce.columns = ['filename','xmin','ymin','xmax','ymax','classname']

    #print(data_reduce.info())
    print("[INFO] Total images :",len(data_reduce['filename'].unique()))
    return data_reduce
def visualize_images(data,num):
    random_subset = data.sample(n=num)
    i = 1
    fig = plt.gcf()
    fig.set_size_inches(10,10)
      
    for index, row in random_subset.iterrows():
        filename = row[0]
        x,y,w,h = row[1],row[2],row[3],row[4]
        classname = row[5]
        
        img_path = os.path.join(img_dir,filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.rectangle(image,(x, y), (w, h),(0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, classname, (int(x),int(y+10)), font, 1, (255,0,0), 2)

        sp = plt.subplot(num/2, num/2, i)
        sp.axis('Off')
        plt.imshow(image)
        i+=1
    plt.show()


data = cleaning_csv(csv_path)
visualize_images(data,6)
data['classname'].value_counts()
base_dir = '/dataset'
pos_img_dir = '/dataset/with_mask'
neg_img_dir = '/dataset/without_mask'
os.mkdir(base_dir)
os.mkdir(pos_img_dir)
os.mkdir(neg_img_dir)

def crop_face(data):
    print("[INFO]Creating positive and negative images ...")
    counter1 = 0
    counter2 = 0
    for index,row in data.iterrows():
        img_name = row[0]
        x,y,w,h = row[1],row[2],row[3],row[4]
        img_label = row[5]
        
        img_path =os.path.join(img_dir,img_name)
        image = cv2.imread(img_path)
        cropped_img = image[y:y+h, x:x+w]
        
        if img_label == 'with_mask':
            filename = str(counter1) + ".jpg"
            cv2.imwrite(os.path.join(pos_img_dir,filename), cropped_img)
            counter1+=1
        elif img_label == 'without_mask':
            filename = str(counter2) + ".jpg"
            cv2.imwrite(os.path.join(neg_img_dir,filename), cropped_img)
            counter2+=1
        else:
            print("[Warning]Label Missing")
            continue
    print("[INFO]Dataset creation completed")
        
        
print(data.head())
crop_face(data)
img_name,x,y,w,h,img_label = data.loc[15396]

img_path =os.path.join(img_dir,img_name)
print(img_label)
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cropped_img = image[y:y+h, x:x+w]

fig = plt.gcf()
fig.set_size_inches(10,10)
sp = plt.subplot(1,2,1)
plt.imshow(image)
sp = plt.subplot(1,2,2)
plt.imshow(cropped_img)

plt.show()

print("With_mask images :",len(os.listdir(pos_img_dir)))
print("Without_mask images :",len(os.listdir(neg_img_dir)))
train_dir = os.path.join( base_dir, 'train')
test_dir = os.path.join( base_dir, 'test')

train_mask_dir = os.path.join(train_dir, 'with_mask')
train_nomask_dir = os.path.join(train_dir, 'without_mask')
test_mask_dir = os.path.join(test_dir, 'with_mask') 
test_nomask_dir = os.path.join(test_dir, 'without_mask')

to_create = [
    train_dir,
    test_dir,
    train_mask_dir,
    train_nomask_dir,
    test_mask_dir,
    test_nomask_dir,
]

for directory in to_create:
    try:
        os.mkdir(directory)
        print(directory, 'Created')
    except:
        print(directory, 'Already Exists')



def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    print("[INFO]Spliting images into test and train directories...")
    
    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name
        if os.path.getsize(file_path):
            files.append(file_name)
        else:
            print('{} is empty, so ignoring'.format(file_name))
    
    n_files = len(files)
    split_point = int(n_files * SPLIT_SIZE)
    
    shuffled = random.sample(files, n_files)
    
    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]
    
    for file_name in train_set:
        shutil.copy(SOURCE + file_name, TRAINING + file_name)
        
    for file_name in test_set:
        shutil.copy(SOURCE + file_name, TESTING + file_name)
        
    print("[INFO]Split completed.")

WITH_MASK_SOURCE_DIR = "/dataset/with_mask/"
TRAINING_MASK_DIR = "/dataset/train/with_mask/"
TESTING_MASK_DIR = "/dataset/test/with_mask/"
WITHOUT_MASK_SOURCE_DIR = "/dataset/without_mask/"
TRAINING_NOMASK_DIR = "/dataset/train/without_mask/"
TESTING_NOMASK_DIR = "/dataset/test/without_mask/"


split_size = .9
split_data(WITH_MASK_SOURCE_DIR, TRAINING_MASK_DIR, TESTING_MASK_DIR, split_size)
split_data(WITHOUT_MASK_SOURCE_DIR, TRAINING_NOMASK_DIR, TESTING_NOMASK_DIR, split_size)

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import RMSprop


pre_trained_model = MobileNetV2(weights="imagenet",
                                include_top=False,
                                input_shape=(150, 150, 3))

for layer in pre_trained_model.layers:
  layer.trainable = False


last_output = pre_trained_model.output


# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(128, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.5)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (2, activation='softmax')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()


TRAINING_DIR = train_dir
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=64,
    class_mode='binary',
    target_size=(150, 150)
)

VALIDATION_DIR = test_dir
validation_datagen = ImageDataGenerator(rescale = 1.0/255,)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=64,
    class_mode='binary',
    target_size=(150, 150)
)


history = model.fit_generator(train_generator,
                              epochs=20,
                              verbose=1,
                              validation_data=validation_generator)
model.save("mask_detector.model", save_format="h5")
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

import numpy as np
from keras.preprocessing import image

print("[INFO] loading face mask detector model...")
model = load_model("mask_detector.model")

path = img_dir + "/0001.jpg"

org_img = cv2.imread(path)
org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
img = cv2.resize(org_img, (150, 150))

image = img_to_array(img)
image = preprocess_input(img)
image = np.expand_dims(img, axis=0)

(mask, withoutMask) = model.predict(image)[0]
label = "Mask" if mask > withoutMask else "No Mask"
color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

# include the probability in the label
label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
cv2.putText(org_img, label, (0,20),cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 2)
# display the label and bounding box rectangle on the output
# frame
print(label)
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.imshow(org_img)
plt.show()
