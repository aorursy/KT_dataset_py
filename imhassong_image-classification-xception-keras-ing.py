import pandas as pd
import numpy as np

import datetime as dt

import os
from os import listdir, makedirs
from os.path import join, exists,expanduser
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)

import tensorflow as tf 
from sklearn.utils import shuffle     
import tensorflow.keras.utils as Utils

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.models import Sequential, Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras import backend as K

from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications import xception
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from sklearn.linear_model import LogisticRegression
          
import cv2                               
               
from tqdm import tqdm
from random import randint

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline

import gc
print (gc.get_threshold())
start = dt.datetime.now()
!ls ../input/keras-pretrained-models/
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)
    
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/

print("Available Pretrained Models:\n")
!ls ~/.keras/models
!ls ../input/intel-image-classification
class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)


data_dir = '../input/intel-image-classification'
train_dir = os.path.join(data_dir, 'seg_train/seg_train')
test_dir = os.path.join(data_dir, 'seg_test/seg_test')

# pred_dir = os.path.join(data_dir, 'seg_pred/seg_pred')

# cat = os.listdir(train_dir)
'''
augs = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,  
    zoom_range=0.2,        
    horizontal_flip=True,
    validation_split=0.3)  

train_gen = augs.flow_from_directory(
    train_dir,
    target_size = IMAGE_SIZE,
    batch_size=8,
    class_mode = 'categorical'
)

test_gen = augs.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=8,
    class_mode='categorical'
)
'''

def load_data():
    """
        Load the data:
            - 14,034 images to train the network.
            - 3,000 images to evaluate how accurately the network learned to classify images.
    """
    
    datasets = ['../input/intel-image-classification/seg_train/seg_train', 
                '../input/intel-image-classification/seg_test/seg_test']
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output
(train_images, train_labels), (test_images, test_labels) = load_data()
print("Shape of Train Images:",train_images.shape)
print("Shape of Train Labels:",train_labels.shape)
print("Shape of Test Images:",test_images.shape)
print("Shape of Test Labels:",test_labels.shape)
print ("Each image is of size: ", IMAGE_SIZE)
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)
pd.DataFrame({'train': train_counts, 
              'test': test_counts}, 
             index=class_names
            ).plot.bar()

plt.show()
plt.pie(train_counts,
        explode=(0, 0, 0, 0, 0, 0) , 
        labels=class_names,
        autopct='%1.1f%%', 
        startangle=90, 
        shadow = True)
plt.axis('equal')
plt.title('Proportion of each observed category')
plt.show()
train_images = train_images / 255.0 
test_images = test_images / 255.0
def display_random_image(class_names, images, labels):
    """
        Display a random image from the images array and its correspond label from the labels array.
    """
    
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image #{} : '.format(index) + class_names[labels[index]])
    plt.show()
display_random_image(class_names, train_images, train_labels)
'''
def display_random_images(class_names, images, labels):

    fig = plt.figure(figsize=(15,15))
    index = np.random.randint(images.shape[0])

    for i in range(25):
 
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.title('Image #{} : '.format(index) + class_names[labels[index]], fontsize=11)
        
    plt.show()
    
'''
# display_random_images(class_names, train_images, train_labels)
def display_examples(class_names, images, labels):
    """
        Display 25 images from the images array with its corresponding labels
    """
    
    fig = plt.figure(figsize=(16,16))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
        

    plt.show()
display_examples(class_names, train_images, train_labels)
def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(16,16))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'], 'bo--', label = "accuracy")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_accuracy")
    plt.title("train_accuracy vs val_accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
def print_mislabeled_images(class_names, test_images, test_labels, pred_labels):
    """
        Print 25 examples of mislabeled images by the classifier, e.g when test_labels != pred_labels
    """
    BOO = (test_labels == pred_labels)
    mislabeled_indices = np.where(BOO == 0)
    mislabeled_images = test_images[mislabeled_indices]
    mislabeled_labels = pred_labels[mislabeled_indices]

    title = "Some examples of mislabeled images by the classifier:"
    display_examples(class_names,  mislabeled_images, mislabeled_labels)
from keras.applications.xception import Xception
X_model = Xception(weights='imagenet',
                    include_top=False,
                    input_shape=(150, 150, 3)
                  )



X_model.summary()
# Utils.plot_model(X_model,to_file='xception.png', show_shapes=True, show_layer_names=True, expand_nested=True)


train_features = X_model.predict(train_images)
test_features = X_model.predict(test_images)

train_features.shape

model = Sequential()
# model.add(X_model)
model.add(Flatten(input_shape = (5, 5, 2048)))
model.add(Dense(50,activation='relu'))
model.add(Dense(6,activation='softmax'))

model.summary()
Utils.plot_model(model,to_file='model.png', show_shapes=True, show_layer_names=True, expand_nested=True)

'''
# data_gen
#model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('../working/best_model.hdf5', verbose = 1, monitor = 'val_accuracy', save_best_only = True)

history = model.fit_generator(
    train_gen, 
    steps_per_epoch  = 500, 
    validation_data  = test_gen,
    validation_steps = 500,
    epochs = 20,
    callbacks = [checkpoint]
)
'''



model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint('../working/best_model.hdf5', verbose = 1, monitor = 'val_accuracy', save_best_only = True)

# history = model.fit(train_images, train_labels, batch_size=128, epochs=15, validation_split = 0.2)
history = model.fit(train_features, train_labels, batch_size=128, epochs=15, validation_split = 0.2, callbacks = [checkpoint])

plot_accuracy_loss(history)
test_loss = model.evaluate(test_features, test_labels)
predictions = model.predict(test_features)     # Vector of probabilities
pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability

display_random_image(class_names, test_images, pred_labels)
print_mislabeled_images(class_names, test_images, test_labels, pred_labels)
from sklearn.metrics import accuracy_score

predictions = model.predict(test_features)    
pred_labels = np.argmax(predictions, axis = 1)
print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))
CM = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           fmt='d',
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, 
           ax = ax)
ax.set_title('Confusion matrix')
plt.show()
n_train, x, y, z = train_features.shape
n_test, x, y, z = test_features.shape
numFeatures = x * y * z
from sklearn import decomposition

pca = decomposition.PCA(n_components = 2)

X = train_features.reshape((n_train, x*y*z))
pca.fit(X)

C = pca.transform(X) # Représentation des individus dans les nouveaux axe
C1 = C[:,0]
C2 = C[:,1]
### Figures

plt.subplots(figsize=(10,10))

for i, class_name in enumerate(class_names):
    plt.scatter(C1[train_labels == i][:1000], C2[train_labels == i][:1000], label = class_name, alpha=0.4)
plt.legend()
plt.title("PCA Projection")
plt.show()
def get_images(directory):
    Images = []
    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street
    label = 0
    
    for labels in os.listdir(directory): #Main Directory where each class label is present as folder name.
        if labels == 'glacier': #Folder contain Glacier Images get the '2' class label.
            label = 2
        elif labels == 'sea':
            label = 4
        elif labels == 'buildings':
            label = 0
        elif labels == 'forest':
            label = 1
        elif labels == 'street':
            label = 5
        elif labels == 'mountain':
            label = 3
        
        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)
            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(label)
    
    return shuffle(Images,Labels,random_state=817328462) #Shuffle the dataset you just prepared.
os.listdir('../input/intel-image-classification/seg_pred/seg_pred/')
pred_images,no_labels = get_images('../input/intel-image-classification/seg_pred/seg_pred/')
pred_images = np.array(pred_images)
pred_images.shape
fig = plot.figure(figsize=(30, 30))
outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

for i in range(25):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    rnd_number = randint(0,len(pred_images))
    pred_image = np.array([pred_images[rnd_number]])
    pred_class = get_classlabel(model.predict_classes(pred_image)[0])
    pred_prob = model.predict(pred_image).reshape(6)
    for j in range(2):
        if (j%2) == 0:
            ax = plot.Subplot(fig, inner[j])
            ax.imshow(pred_image[0])
            ax.set_title(pred_class)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        else:
            ax = plot.Subplot(fig, inner[j])
            ax.bar([0,1,2,3,4,5],pred_prob)
            fig.add_subplot(ax)


fig.show()