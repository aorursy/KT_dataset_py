# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from random import randint
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/digit-rec-model"))
print(os.listdir("../input/d-r-model"))
# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('../input/digit-recognizer/train.csv')
test_data=pd.read_csv('../input/digit-recognizer/test.csv')
sub_sample=pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train_data.head(10)
test_data.head()
test_data.shape
sub_sample.head()
Y_train = train_data["label"]
Y_train1 = np.array(Y_train, np.uint8)
train_images = train_data.drop(labels = ["label"],axis = 1) 
train_images = np.array(train_images)
test_images=np.array(test_data)
print(train_images.shape)
print(Y_train1.shape)
print(test_images.shape)
Y_train1
#Convert train datset to (num_images, img_rows, img_cols) format 
X_train1 = train_images.reshape(train_images.shape[0], 28, 28,1)
#Convert test datset to (num_images, img_rows, img_cols) format 
X_test = test_images.reshape(test_images.shape[0], 28, 28,1)
print(X_train1.shape)
print(X_test.shape)
def plot_images(images, classes):
    assert len(images) == len(classes) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3,figsize=(28,28),sharex=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
   
    for i, ax in enumerate(axes.flat):
        # Plot image.
        
        ax.imshow(images[i][:,:,0], cmap=plt.get_cmap('gray'))    
        xlabel = "the number is: {0}".format(classes[i])
    
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        ax.xaxis.label.set_size(28)
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    
    plt.show()
random_numbers = [randint(0, len(X_train1)) for p in range(0,9)]
images_to_show = [X_train1[i] for i in random_numbers]
classes_to_show = [Y_train[i] for i in random_numbers]
print("Images to show: {0}".format(len(images_to_show)))
print("Classes to show: {0}".format(len(classes_to_show)))
#plot the images
plot_images(images_to_show, classes_to_show)

from keras.utils.np_utils import to_categorical

Y_train1= to_categorical(Y_train1)
Y_train1.shape
#Splitting the train_images into the Training set and validation set
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val= train_test_split(X_train1, Y_train1,
               test_size=0.1, random_state=42,stratify=Y_train1)

print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)
print(X_test.shape)
X_train1 = X_train1.astype('float32')/255
X_val=X_val.astype('float32')/255
X_test = X_test.astype('float32')/255
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras import regularizers
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 10:
        lrate = 0.0003
    if epoch > 20:
        lrate = 0.00003
    elif epoch > 30:
        lrate = 0.000003       
    return lrate
lr_scheduler=LearningRateScheduler(lr_schedule)
#we can reduce the LR by half if the accuracy is not improved after 3 epochs.using the following code
reduceOnPlateau = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001, mode='auto')

#Save the model after every decrease in val_loss 
checkpoint = ModelCheckpoint(filepath='bestmodel.hdf5', verbose=0,monitor='val_loss',save_best_only=True,save_weights_only=False)

#Stop training when a monitored quantity has stopped improving.
earlyStopping=EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
model=load_model("../input/d-r-model/bestmodel (1).hdf5")
datagen = ImageDataGenerator(
        rotation_range=3,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.01, # Randomly zoom image 
        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train1)
callbacks_list = [reduceOnPlateau,checkpoint]
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)
rmsprp_opt = keras.optimizers.rmsprop(lr=0.00003 ,decay=1e-4)
adam=keras.optimizers.adam(lr=0.00003)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
H1=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=250),
                    steps_per_epoch=len(X_train)//225, epochs=30,
                    verbose=1,callbacks=callbacks_list,
                    validation_data=(X_val, Y_val))
plt.figure(0)
plt.plot(H1.history['acc'],'r')
plt.plot(H1.history['val_acc'],'g')
plt.xticks(np.arange(0, 51, 1.0))
plt.rcParams['figure.figsize'] = (14, 8)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
plt.figure(1)
plt.plot(H1.history['loss'],'r')
plt.plot(H1.history['val_loss'],'g')
plt.xticks(np.arange(0, 51, 1.0))
plt.rcParams['figure.figsize'] = (14, 8)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
score = model.evaluate(X_val, Y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
from sklearn.metrics import classification_report

preds = model.predict_classes(X_val)
y_lable = [y.argmax() for y in Y_val]
print(classification_report(y_lable,preds))
preds1 = model.predict_classes(X_train)
ytr_lable = [y.argmax() for y in Y_train]
print(classification_report(ytr_lable,preds1))
# predict results
Test_perdect = model.predict(X_test)

# select the indix with the maximum probability
Test_perdect = np.argmax(Test_perdect,axis = 1)

Test_perdect = pd.Series(Test_perdect,name="Label")

submission1 = pd.concat([pd.Series(range(1,28001),name = "ImageId"),Test_perdect],axis = 1)

submission1.to_csv("submission1.csv",index=False)
model.save("cifar-10_model.h5")
