from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense

from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix

from glob import glob

from keras.utils.np_utils import to_categorical #Converting to one hot encoding 

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings 

warnings.filterwarnings("ignore")
train = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")

test = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")
print("Train shape : %s  \nTest: Shape : %s"%(train.shape,test.shape))
train.head(10)
train.label.nunique() # We got 10 Classes(Labels)
test.label.nunique()
print(train.label.value_counts())

plt.figure(figsize = (12,10))

sns.countplot(train.label, palette ="cubehelix")
#Visualizing with PieChart 

labelsx = train.label.value_counts().index

colors = ["grey","red","blue","yellow","brown","orange","pink","green","purple","indigo"]

explode = [0,0,0,0,0,0,0,0,0,0]

sizes = train.label.value_counts().values



plt.figure(figsize = (9,9))

plt.pie(sizes,explode = explode, labels = labelsx, colors = colors, autopct ="%1.1f%%")

plt.title("Label Counting by using PieChart (Seaborn)",color = "violet",fontsize = 15, fontstyle ="oblique")

plt.show()
print(test.label.value_counts())

plt.figure(figsize = (12,10))

sns.countplot(test.label, palette = "icefire")
#Train Dataset

Y_train = train.label

X_train = train.drop(labels = ["label"], axis = 1)
#Test Dataset

Y_test = test.label

X_test = test.drop(labels = ["label"],axis = 1)
plt.subplot(3,2,1)

img1 = X_train.iloc[0].to_numpy().reshape((28,28))

plt.imshow(img1,cmap='gray')

plt.subplot(3,2,2)

img2 = X_train.iloc[10].to_numpy().reshape((28,28))

plt.imshow(img2,cmap='gray')

plt.subplot(3,2,3)

img3 = X_train.iloc[98].to_numpy().reshape((28,28))

plt.imshow(img3,cmap='gray')

plt.subplot(3,2,4)

img4 = X_train.iloc[25].to_numpy().reshape((28,28))

plt.imshow(img4,cmap='gray')

plt.subplot(3,2,5)

img5 = X_train.iloc[120].to_numpy().reshape((28,28))

plt.imshow(img5,cmap='gray')

plt.subplot(3,2,6)

img6 = X_train.iloc[264].to_numpy().reshape((28,28))

plt.imshow(img6,cmap='gray')



plt.show()
#Normalization

X_train = X_train.astype("float32")

X_test = X_test.astype("float32")

X_train = X_train / 255.0

X_test = X_test / 255.0

print("X_train Shape : %s \nX_Test Shape :%s"%(X_train.shape,X_test.shape))
#Reshaping 

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

print("X_train shape : ",X_train.shape)

print("X_Test shape : ",X_test.shape)
#Label Encoding - IF there more Labels we could use Glob Function

from keras.utils.np_utils import to_categorical 

Y_train = to_categorical(Y_train,num_classes = 10)

Y_test = to_categorical(Y_test,num_classes = 10)
from sklearn.model_selection import train_test_split 

X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.15,random_state = 42)

print("X_train shape",X_train.shape)

print("X_val shape",X_val.shape)

print("Y_train shape",Y_train.shape)

print("Y_val shape",Y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools 



from keras.utils.np_utils import to_categorical #Converting to one hot encoding 

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization



epochs = 75

batch_size = 240

num_of_classes = 10



model = Sequential()



model.add(Conv2D(filters = 128, kernel_size = (3,3), padding ="same",

                 activation ="relu", input_shape =(28,28,1)))

model.add(MaxPooling2D(pool_size =(3,3)))



model.add(Conv2D(64,3,3))

model.add(Activation("relu"))

model.add(MaxPooling2D(3,3))



model.add(Flatten())

model.add(Dense(1024))  #Hidden layer1

model.add(Activation("relu"))

model.add(Dropout(0.25))



model.add(Dense(num_of_classes)) # Output layer size must equal to number of classes (labels)

model.add(Activation("softmax"))
learning_rate_optimizer = ReduceLROnPlateau(monitor = "val_accuracy",

                                           patience = 2, verbose = 1,

                                           factor = 0.5, min_lr = 0.000001)
optimizer = RMSprop()

model.compile(optimizer = optimizer, loss  ="categorical_crossentropy", metrics =["accuracy"])
model.summary()
datagen = ImageDataGenerator( 

        shear_range = 0.2,

        zoom_range = 0.1,

        width_shift_range=0.1,

        height_shift_range=0.1,

        horizontal_flip = True,

        vertical_flip = True)



datagen.fit(X_train)
history = model.fit(datagen.flow(X_train,Y_train, 

                                batch_size = batch_size), 

                                epochs = epochs,

                                validation_data = (X_val,Y_val),

                                steps_per_epoch = X_train.shape[0]//batch_size,

                                callbacks = [learning_rate_optimizer])
score = model.evaluate(X_test, Y_test, verbose = 0)

print("Test Loss : %f \nTest Accuracy : %f "%(score[0],score[1]))
print(history.history.keys())

plt.plot(history.history["loss"], label ="Train Loss")

plt.plot(history.history["val_loss"], label ="Test Loss")

plt.legend()

plt.show()



#-----------------------------------------------------------------------



print(history.history.keys())

plt.plot(history.history["accuracy"], label ="Train Accuracy")

plt.plot(history.history["val_accuracy"], label ="Test Accuracy")

plt.legend()

plt.show()
import seaborn as sns

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(confusion_mtx, annot=True, cmap="cubehelix", linewidths=0.01,linecolor="green", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()