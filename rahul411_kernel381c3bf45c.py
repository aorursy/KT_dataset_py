# Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
import seaborn as sns
# Reading the input images and putting them into a numpy array
data=[]
labels=[]

height = 32
width = 32
channels = 3
classes = 43

for i in range(classes) :
    path = "../input/gtsrb-german-traffic-sign/train/{0}/".format(i)
    
    Class=os.listdir(path)
    
    for a in Class:
        
        image=cv2.imread(path + a)
        image_from_array = Image.fromarray(image, 'RGB')  
        size_image = image_from_array.resize((height, width))

        data.append(np.array(size_image))
        labels.append(i)
        
        
data = np.array(data)
labels = np.array(labels)

#normalising data to values between 0 to 1.

data = data.astype('float32')/255

#loading test data
y_test = pd.read_csv("../input/gtsrb-german-traffic-sign/Test.csv")
paths =  y_test['Path'].as_matrix()
y_test = y_test['ClassId'].values

test_data=[]

for p in paths:
    image = cv2.imread('../input/gtsrb-german-traffic-sign/test/'+p.replace('Test/', ''))
    image_from_array = Image.fromarray(image, 'RGB')
    
    size_image = image_from_array.resize((height, width))
    test_data.append(np.array(size_image))

X_test=np.array(test_data)
#normalising test data
X_test = X_test.astype('float32')/255 

#Plotting the train and test classes distribution
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(30, 10))
axs[0].set_title('Train classes distribution')
axs[0].set_xlabel('Class')
axs[0].set_ylabel('Count')
axs[1].set_title('Test classes distribution')
axs[1].set_xlabel('Class')
axs[1].set_ylabel('Count')

sns.countplot(labels, ax=axs[0], color='blue')
sns.countplot(y_test, ax=axs[1], color='red')
axs[0].set_xlabel('Class ID');
axs[1].set_xlabel('Class ID');
#Spliting the images into train and validation sets
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data,
                                                    labels,
                                                    test_size=0.25,
                                                    random_state=42)


#Showing sample images from the dataset

def show_images(images, labels, pred_labels=[], amount=16):
    ids = np.random.randint(len(images), size=amount)
    
    plt.figure(figsize=(24,10))
    for i, id in enumerate(ids):
        plt.subplot(int(amount/8)+1, 8,1+i)
        plt.axis('off')
        if(len(pred_labels)):
            plt.title("true- " + str(labels[id]) + "  pred- " + str(pred_labels[id]))
        else:
            plt.title(str(labels[id]))
        plt.imshow(images[id])
    
print("Train images")
show_images(X_train, y_train, amount=24)
#Using one hote encoding for the train and validation labels
from keras.utils import to_categorical

y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
#building the CNN model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(4, 4), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)
model.summary()
#using ten epochs for the training and saving the accuracy for each epoch
epochs = 10
history = model.fit(X_train, y_train, batch_size=24, epochs=epochs, validation_data=(X_val, y_val))

#Ploting accuracy and loss values vs epochs for train and val data

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
#Accuracy with the test data
test_pred = model.predict_classes(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, test_pred)
# Confussion matrix 
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=75) 
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
class_names = range(43)
# evaluate the output in a confusion matrix. 
plt.figure(figsize=(15,15))
cm = confusion_matrix(y_test, test_pred)

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
#showing few predicted images with their true values

show_images(X_test, y_test, test_pred, 32)