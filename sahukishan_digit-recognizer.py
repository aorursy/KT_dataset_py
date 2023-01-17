import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from PIL import Image

import cv2

import keras

from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dropout

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

data=pd.read_csv("../input/digit-recognizer/train.csv")

test_data=pd.read_csv("../input/digit-recognizer/test.csv")
sns.set_style('whitegrid')

sns.countplot(x='label',data=data,palette='RdBu_r')
X_train=data.iloc[:,1:].values

Y_train=data.iloc[:,0].values

X_test=test_data.values
from sklearn.preprocessing import StandardScaler

X_train = StandardScaler().fit_transform(X_train)

X_test  = StandardScaler().fit_transform(X_test)
X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_test = X_test.reshape(X_test.shape[0],28,28,1)

num_classes = 10

Y_train = keras.utils.to_categorical(Y_train, num_classes)

print(Y_train[:5])
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
plt.imshow(X_train[0][:,:,0])
model=Sequential()

model.add(Convolution2D(32,3,3,input_shape=(28,28,1),activation="relu",kernel_initializer='he_normal'))

model.add(Convolution2D(32,3, 3, activation = 'relu', kernel_initializer='he_normal'))

model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.25))

model.add(Convolution2D(64,3, 3, activation = 'relu', kernel_initializer='he_normal'))

model.add(Convolution2D(64,3, 3, activation = 'relu', kernel_initializer='he_normal'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.25))



model.add(Convolution2D(128,3, 3, activation = 'relu',kernel_initializer='he_normal'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(output_dim=264,init='uniform',activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(output_dim=10,init='uniform',activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
datagen = ImageDataGenerator(

    featurewise_center=False,

    featurewise_std_normalization=False,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=False,

    vertical_flip=False)

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),validation_data = (X_val,Y_val),

                    steps_per_epoch=len(X_train) // 32, epochs=40)

#Without data augmentation

#model.fit(X_train, Y_train, batch_size = 32, epochs = 30,validation_data = (X_val, Y_val), verbose = 2)
Y_pred=model.predict(X_val)
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

    plt.xticks(tick_marks, classes, rotation=45)

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



# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# from sklearn.linear_model import LogisticRegression

# from sklearn import metrics

# logreg=LogisticRegression()

# logreg.fit(X_train,Y_train)

# predict=logreg.predict(X_test)

# metrics.accuracy_score(Y_test,predict)
sample=pd.read_csv("../input/digit-recognizer/sample_submission.csv")

X_test.shape
predict=model.predict(X_test) 

pred=np.argmax(predict,axis=1)

pred[:5]
d={"ImageId":sample["ImageId"],"Label":pred.astype(np.int32)}

check=pd.DataFrame(d)

check.to_csv("result.csv",index=False)
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")