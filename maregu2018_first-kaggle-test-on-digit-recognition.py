# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sbn

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import tensorflow as tf
from keras import backend as K

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Loading Dataset

train = pd.read_csv("../input/sign_mnist_train.csv")
#test = pd.read_csv("../input/sign_mnist_test.csv")

#print(train.shape)
#print(test.shape)
#train.head()

# Create Labels
labels = train.pop('label')
labels = to_categorical(labels)

#print(labels.shape)

# Create Train data
train = train.values
train = np.array([np.reshape(i, (28,28)) for i in train])
train = train/255

train.shape

plt.imshow(train[0])


# Split train images and test images
train_images, test_images, train_labels, test_labels = train_test_split(train, labels, test_size=0.2, random_state=2)
'''print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# Reshaping image data
if K.image_data_format() == "channels_first":
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
else:
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
'''
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
print(train_images.shape, test_images.shape)

model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='softmax'))

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

#print(model.summary())

#plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)

#model.load_weights("mnist-model.h5")

history = model.fit(train_images, train_labels, batch_size=512, epochs=32, verbose = 0, validation_data=(test_images, test_labels))

#model.save("mnist-model.h5")
accuracy = model.evaluate(test_images, test_labels, batch_size=512, verbose=0)



history.history['acc']
print('Accuracy on Test set: ', accuracy[1])


#%% Accuracy plot

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss Plot

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


Y_pred = model.predict(test_images)
y_pred = np.argmax(Y_pred, axis=1)

print("Confusion Matrix")
print(confusion_matrix(np.argmax(test_labels, axis=1), y_pred))

print("Classification Report")
#target_names = ['ሀ','ለ','ሐ','መ','ሠ','ረ','ሰ','ሸ','ቀ','በ','ተ','ቸ','ኀ','ነ','ኝ','አ','ከ','ኸ','ወ','ዐ','ዘ','ዠ','የ','ደ','ጀ','ገ','ጠ','ጨ','ጰ','ጸ','ፀ','ፈ','ፐ']
target_name1 = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']
print(classification_report(np.argmax(test_labels, axis=1), y_pred, target_names=target_name1))


#%%
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
   
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel("Predicted Label\nAccuracy={:0.4f};Misclass={:0.4f}".format(accuracy, misclass))
    plt.tight_layout()
    plt.show()
#%%
# Compute confusion matrix
cnf_matrix=(confusion_matrix(np.argmax(test_labels, axis=1), y_pred))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix,classes=target_name1,title='Confusion Matrix')



