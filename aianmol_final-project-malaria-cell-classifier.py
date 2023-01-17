import numpy as np

import pandas as pd



import cv2

import matplotlib.pyplot as plt 

import seaborn as sns

import os

from PIL import Image

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import load_img

from keras.utils import np_utils
import os

print(os.listdir("../input"))
infected_data = os.listdir('../input/cell_images/cell_images/Parasitized/')

print(infected_data[:10]) #the output we get are the .png files



uninfected_data = os.listdir('../input/cell_images/cell_images/Uninfected/')

print('\n')

print(uninfected_data[:10])
plt.figure(figsize = (10,10))

for i in range(4):

    plt.subplot(1, 4, i+1)

    img = cv2.imread('../input/cell_images/cell_images/Parasitized' + "/" + infected_data[i])

    plt.imshow(img)

    plt.title('INFECTED : 1')

    plt.tight_layout()

plt.show()
plt.figure(figsize = (10,10))

for i in range(4):

    plt.subplot(1, 4, i+1)

    img = cv2.imread('../input/cell_images/cell_images/Uninfected' + "/" + uninfected_data[i])

    plt.imshow(img)

    plt.title('UNINFECTED : 1')

    plt.tight_layout()

plt.show()
## as per the knowledge base the blood smear image will be stained and parasite will be visible.
data = []

labels = []

for img in infected_data:

    try:

        img_read = plt.imread('../input/cell_images/cell_images/Parasitized/' + "/" + img)

        img_resize = cv2.resize(img_read, (50, 50))

        img_array = img_to_array(img_resize)

        img_aray=img_array/255

        data.append(img_array)

        labels.append(1)

    except:

        None

        

for img in uninfected_data:

    try:

        img_read = plt.imread('../input/cell_images/cell_images/Uninfected' + "/" + img)

        img_resize = cv2.resize(img_read, (50, 50))

        img_array = img_to_array(img_resize)

        img_array= img_array/255

        data.append(img_array)

        labels.append(0)

    except:

        None
plt.imshow(data[1])

plt.show()
type(data)
image_data = np.array(data)

labels = np.array(labels)

idx = np.arange(image_data.shape[0])

np.random.shuffle(idx)

image_data = image_data[idx]

labels = labels[idx]
type(image_data)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(image_data, labels, test_size = 0.2, random_state = 42)
y_train = np_utils.to_categorical(y_train,  2)

y_test = np_utils.to_categorical(y_test,  2)
print(f'Shape of training image : {x_train.shape}')

print(f'Shape of testing image : {x_test.shape}')

print(f'Shape of training labels : {y_train.shape}')

print(f'Shape of testing labels : {y_test.shape}')
import keras

from keras.layers import Dense, Conv2D

from keras.layers import Flatten

from keras.layers import MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Dropout

from keras.models import Sequential

from keras import backend as K



from keras import optimizers
inputShape= (50,50,3)

model=Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = inputShape))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis =-1))

model.add(Dropout(0.2))



model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(MaxPooling2D(2,2))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.2))



model.add(Flatten())



#ANN architecture

model.add(Dense(512, activation = 'relu'))

model.add(BatchNormalization(axis = -1))

model.add(Dropout(0.5))

model.add(Dense(2, activation = 'softmax'))
model.summary()
#compile the model

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=25)
print(H.history.keys())
# summarize history for accuracy

plt.plot(H.history['acc'])

plt.plot(H.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(H.history['loss'])

plt.plot(H.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','test'], loc='upper right')

plt.show()
# make predictions on the test set

preds = model.predict(x_test)
from sklearn.metrics import accuracy_score



print(accuracy_score(y_test.argmax(axis=1), preds.argmax(axis=1)))


from sklearn.metrics import classification_report

print(classification_report(y_test.argmax(axis=1), preds.argmax(axis=1)))
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    from sklearn.metrics import confusion_matrix

    from sklearn.utils.multiclass import unique_labels

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
class_names=np.array((0,1))

plot_confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1), classes=class_names, title='Confusion Matrix')
model.save("malaria_model.h5")
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import load_model


# load the model

model = load_model("malaria_model.h5")



y=x_test[0]

y=np.expand_dims(y,axis=0)





for i in range(5,65,5):

    y=x_test[i]

    y=np.expand_dims(y,axis=0)

    pred=model.predict(y)

    print("image ",i )

    if(pred[0][0]>pred[0][1]):

        print("no malaria")

    else:

        print("malaria")

    