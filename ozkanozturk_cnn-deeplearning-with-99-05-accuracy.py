# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings 

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
print(train.shape)
train.head()
sns.countplot(train.label)

plt.title("Counts of pictures in dataset")

plt.show()



train.label.value_counts()
train[train.label == 1].head()
train[train.label == 1].iloc[0]
train[train.label == 1].iloc[0][1:]
image = train[train.label == 1].iloc[0][1:]  # filtering data as explained above

plt.imshow(image.values.reshape(28,28))

plt.axis("off")

plt.show()
fig, ax = plt.subplots(1,6, figsize = (36,6))

for i in range(0,6):

    ax[i].imshow(train[train.label == 1].iloc[i][1:].values.reshape(28,28))

    ax[i].axis("off")

plt.show()


fig, ax = plt.subplots(10,15, figsize = (20,10))

j = 0   # flag for rows of fig

while j < 10:

    for i in range(0,15):  # i for columns of fig

        ax[j,i].imshow(train[train.label == j].iloc[i][1:].values.reshape(28,28))

        ax[j,i].axis("off")

    j = j + 1

plt.show()
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(test.shape)
test.head()
y_train = train["label"]
X_train = train.drop(columns = ["label"], axis = 0)
print("Shape of X_train :", X_train.shape)

print("Shape of y_train :", y_train.shape)
X_train.iloc[0,:].values.reshape(28,28)
# Normalization

# As i previously showed, our data have 255 as maximum pixel values.

# In order to normalize, i will divide X_train and test data to 255.

# Thus all values to be between 0 and 1.



X_train = X_train / 255

test = test / 255
# Reshape

print("Shape of X_train before reshape :", X_train.shape )

print("Shape of test before reshape :", test.shape )



X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)



print("Shape of X_train after reshape :", X_train.shape )

print("Shape of test after reshape :", test.shape )
X_train[0][0]
a = np.array([list(range(16))])

a.shape
a = a.reshape(-1,2)

a.shape
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 10)
y_train.shape
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(

    X_train, 

    y_train, 

    test_size = 0.1, 

    random_state = 2 )
# print shape of split datas

print("Shape of X_train :", X_train.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of X_val :", X_val.shape)

print("Shape of y_val :", y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical   # for converting to one hot encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
# Creating model

model = Sequential()   



#----------------

# Convolution_1

#----------------

# Adding convolutin layer to model

# Convolution layer will be created by 8 filters(feature detectors), 

# feature detector size will be 5x5,

# padding type will be same padding,

# and activation function will be relu.

model.add(

    Conv2D(

        filters = 8, 

        kernel_size = (5,5),

        padding = "Same",

        activation = "relu",

        input_shape = (28,28,1)))



#----------------

# Maxpooling_1

#----------------



# Adding maxpooling to model, poolsize will be 2x2.

model.add(

    MaxPool2D(pool_size = (2,2)))



#----------------

# Dropout_1

#----------------



# Adding drop out, drop out ratio will be %25.

model.add(Dropout(0.25))





#----------------

# Convolution_2

#----------------

model.add(

    Conv2D(

        filters = 16, 

        kernel_size = (3,3),

        padding = "Same",

        activation = "relu"))



#----------------

# Maxpooling_2

#----------------

model.add(

    MaxPool2D(

        pool_size = (2,2),

        strides = (2,2)))



#----------------

# Dropout_2

#----------------

model.add(Dropout(0.25))



#----------------

# Fully Connected layers

#----------------

model.add(Flatten()) # flattening

model.add(Dense(256, activation = "relu")) # adding 1st hidden layer with 256 neurons

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax")) # adding 2nd hidden layer with 10 neurons

# softmax function is used for multiple classification and sigmoid is used for binary classification

model.summary()
optimizer = Adam(lr = 0.001,

                 beta_1 = 0.9,

                 beta_2 = 0.999)
model.compile(optimizer = optimizer,

              loss = "categorical_crossentropy",

              metrics = ["accuracy"])
epochs = 5

batch_size = 250
datagen = ImageDataGenerator(

    featurewise_center = False,# sets input mean to 0

    samplewise_center = False, # sets each sample mean to 0

    featurewise_std_normalization = False, # divide inputs by std of dataset

    samplewise_std_normalization = False, # divides each input by its std

    zca_whitening = False, # dimension reduction

    rotation_range = 0.5, # randomly rotate images in 5 degree ranges

    zoom_range = 0.5, # randomly zoom image

    width_shift_range = 0.5, # randomly shift image horizantally

    height_shift_range = 0.5, # randomly shift image vertically

    horizontal_flip = False, # randomly flip image

    vertical_flip = False # randomly flip image

    )





datagen.fit(X_train)  # apply data augmentation to our X_train data
batch_size

X_train.shape[0]  # samples
X_train.shape[0] // batch_size
result = model.fit_generator(

    datagen.flow(X_train,y_train,batch_size = batch_size),

    epochs = epochs,

    validation_data = (X_val, y_val),

    steps_per_epoch = X_train.shape[0] // batch_size

)
result.history
# Losses 

plt.subplots(figsize=(8,6))

plt.plot(result.history["loss"], color = "b", label = "train loss")

plt.plot(result.history["val_loss"], color = "r", label = "test loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Losses of fit with Augmented Data")

plt.legend()

plt.show()
# Accuracies

plt.subplots(figsize=(8,6))

plt.plot(result.history["accuracy"], color = "b", label = "train accuracy")

plt.plot(result.history["val_accuracy"], color = "r", label = "test accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.title("Accuracies of fit with Augmented Data")

plt.legend()

plt.show()
print("Shape of X_val :", X_val.shape)

print("Shape of y_val :", y_val.shape)
# checking 1st row of y_val



y_val[0]
# Predicting labels of X_val data



y_pred = model.predict(X_val)
print("Shape of y_pred :", y_pred.shape)
y_pred[0]
# predicted class of first row of X_val data

np.argmax(y_pred[0])    
# true label of first row of X_val data

np.argmax(y_val[0])
# getting predicted classes (convert ot one-hot-vectors)

 

y_pred_classes = np.argmax(y_pred, axis = 1)



# validation observations to one-hot-vectors



y_true = np.argmax(y_val, axis = 1)



print("Shape of y_pred_classes :", y_pred_classes.shape)

print("Shape of y_true_classes :", y_true.shape)
# confusion matrix

import seaborn as sns



cm = confusion_matrix(y_true, y_pred_classes)



f,ax = plt.subplots(figsize = (8,5)) # creating f,ax object

sns.heatmap(

    cm, 

    annot = True, 

    linewidths = 0.01, 

    cmap = "Greens", 

    linecolor = "gray", 

    fmt = ".1f", 

    ax = ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix of fit with Augmented Data")

plt.show()
result_1 = model.fit(

    X_train,

    y_train,

    epochs = epochs,

    validation_data = (X_val, y_val),

    steps_per_epoch =  X_train.shape[0] // batch_size

)
# Losses

plt.subplots(figsize=(8,6))

plt.plot(result_1.history["loss"], label = "train loss")

plt.plot(result_1.history["val_loss"], label = "test loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Losses of fit w/o data augmentation")

plt.legend()

plt.show()
# Accuracies

plt.subplots(figsize=(8,6))

plt.plot(result_1.history["accuracy"], label = "train accuracy")

plt.plot(result_1.history["val_accuracy"], label = "test accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracies")

plt.title("Accuracies of fit w/o data augmentation")

plt.legend()

plt.show()
# Predicting labels of X_val data



y_pred = model.predict(X_val)



# getting predicted classes (convert ot one-hot-vectors)

 

y_pred_classes = np.argmax(y_pred, axis = 1)



# validation observations to one-hot-vectors



y_true = np.argmax(y_val, axis = 1)



# confusion matrix



cm = confusion_matrix(y_true, y_pred_classes)



f,ax = plt.subplots(figsize = (8,5)) # creating f,ax object

sns.heatmap(

    cm, 

    annot = True, 

    linewidths = 0.01, 

    cmap = "Greens", 

    linecolor = "gray", 

    fmt = ".1f", 

    ax = ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix of fit without Augmented Data")

plt.savefig("Confusion Matrix of fit without Augmented Data")

plt.show()

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV



# Preparing cnn model



def create_model(optimizer , activation  , drop_out_rate, filters, kernel_size, padding ):

    model = Sequential()  

    

    model.add(

    Conv2D(

        filters = filters, 

        kernel_size = kernel_size,

        padding = padding,

        activation = activation,

        input_shape = (28,28,1)))

    

    model.add(

    MaxPool2D(pool_size = (2,2)))

    

    model.add(Dropout(drop_out_rate))

    

    model.add(

    Conv2D(

        filters = filters, 

        kernel_size = kernel_size,

        padding = padding,

        activation = activation))

    

    model.add(

    MaxPool2D(

        pool_size = (2,2),

        strides = (2,2)))

    

    

    model.add(Dropout(drop_out_rate))

    

    model.add(Flatten()) # flattening

    model.add(Dense(256, activation = activation)) # adding 1st hidden layer with 256 neurons

    model.add(Dropout(drop_out_rate))

    model.add(Dense(10, activation = "softmax")) # adding 2nd hidden layer with 10 neurons

    



    

    model.compile(optimizer = optimizer,

              loss = "categorical_crossentropy",

              metrics = ["accuracy"])

    

    

    return model

    
new_model = KerasClassifier(build_fn = create_model)



param_grid = {

              'epochs':[3],

              'batch_size':[250],

              'optimizer' : ["Adam"],

              'drop_out_rate' : [0.2],

              'activation' : ['relu'],

              "filters":[15],

              "kernel_size":[(3,3)],

              "padding":["same"]

             }





grid = GridSearchCV(

    estimator = new_model, 

    param_grid = param_grid,

    cv =2)



result_2 = grid.fit(X_train, y_train)
result_2.best_params_
result_2.best_score_
from keras.callbacks import ReduceLROnPlateau

learning_rate_red = ReduceLROnPlateau(monitor='val_accuracy', 

                                            mode = "auto",

                                            patience=1, 

                                            cooldown=2,

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0)
model_lr_reduction = create_model(

    optimizer = result_2.best_params_["optimizer"],

    activation = result_2.best_params_["activation"],

    drop_out_rate = result_2.best_params_["drop_out_rate"],

    filters = result_2.best_params_["filters"],

    kernel_size = (3,3),

    padding = result_2.best_params_["padding"]

)
result_3 = model_lr_reduction.fit(

    X_train,

    y_train,

    epochs = 30,

    validation_data = (X_val, y_val),

    steps_per_epoch =  X_train.shape[0] // batch_size,

    callbacks = [learning_rate_red]

)
# Losses

plt.subplots(figsize=(8,6))

plt.plot(result_3.history["loss"], label = "train loss")

plt.plot(result_3.history["val_loss"], label = "test loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Losses of fit w LR reduction")

plt.legend()

plt.savefig("Losses of fit w LR reduction")

plt.show()
# Losses

plt.subplots(figsize=(8,6))

plt.plot(result_3.history["accuracy"], label = "train accuracy")

plt.plot(result_3.history["val_accuracy"], label = "test accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.title("Accuracies of fit w LR reduction")

plt.legend()

plt.savefig("Accuracies of fit w LR reduction")

plt.show()
# Final results for submission



final_result = model_lr_reduction.predict(test)



# Selecting the classes which have the maximum probability



final_result = np.argmax(final_result, axis = 1)



final_result = pd.Series(final_result,name="Label")
submission = pd.concat(

    [pd.Series(range(1,28001),name = "ImageId"),final_result],

    axis = 1

)



submission.to_csv("cnn_mnist_datagen.csv",index=False)