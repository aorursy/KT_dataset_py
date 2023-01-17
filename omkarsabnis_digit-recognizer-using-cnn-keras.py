# IMPORTING NECESSARY PACKAGES AND MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from collections import Counter
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#IMPORTING KERAS AND RELATED MODULES
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#SETTING GLOBAL VARIABLES:
batch = 64
classes = 10
epochs = 20
input_shape = (28,28,1)
#LOADING THE TRAINING DATASET
trainingset = pd.read_csv("../input/train.csv")
print(trainingset.shape)
trainingset.head()
#LOADING THE TESTING DATASET
testingset = pd.read_csv("../input/test.csv")
print(testingset.shape)
testingset.head()
# CHECKING THE TRAINING DATASET FOR THE DISTRIBUTION OF THE NUMBERS
print(trainingset['label'].value_counts())
x_train = trainingset.ix[:,1:].values.astype('float32')
y_train = trainingset.ix[:,0].values.astype('int32')
x_test = testingset.values.astype('float32')
plt.figure(figsize=(12,5))
x,y = 10,2
for i in range(20):
    plt.subplot(y,x,i+1)
    plt.imshow(x_train[i].reshape(28,28),interpolation='nearest')
plt.show()
X_Tr = x_train.reshape(x_train.shape[0],28,28,1)
X_Te = x_test.reshape(x_test.shape[0],28,28,1)
y_train = keras.utils.to_categorical(y_train,classes)
X_Tr,X_Va,Y_tr,Y_Va = train_test_split(X_Tr,y_train,test_size=0.1,random_state=42)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

lrr = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

data = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

model.summary()
data.fit(X_Tr)
runs =  model.fit_generator(data.flow(X_Tr,Y_tr, batch_size=batch),
                              epochs = epochs, validation_data = (X_Va,Y_Va),
                              verbose = 1, steps_per_epoch=X_Tr.shape[0] // batch
                              , callbacks=[lrr],)
fl,fac = model.evaluate(X_Va,Y_Va,verbose=0)
print("Final Loss =",fl)
print("Final Accuracy =",fac)
y_pred = model.predict(X_Va)
y_classes = np.argmax(y_pred,axis=1)
y_ohv = np.argmax(Y_Va,axis=1)
cm = confusion_matrix(y_ohv,y_classes)
print("Confusion Matrix:")
print(cm)
errors = (y_classes - y_ohv != 0)

y_classes_errors = y_classes[errors]
y_pred_errors = y_pred[errors]
y_ohv_errors = y_ohv[errors]
X_Va_errors = X_Va[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    n = 0
    nrows = 3
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

#PROBABILITY OF WRONG PREDICTION
y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

# PROBABILITY OF TRUE VALUES IN ERROR SET
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_ohv_errors, axis=1))

# DIFFERENCE BETWEEN TRUE AND ERROR SET
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 9 errors 
most_important_errors = sorted_dela_errors[-9:]

# Show the top 9 errors
display_errors(most_important_errors, X_Va_errors, y_classes_errors, y_ohv_errors)
pred = model.predict_classes(X_Te)
y_real = testingset.iloc[:,0]
true = np.nonzero(pred==y_real)[0]
false = np.nonzero(pred!=y_real)[0]

submit = pd.DataFrame({'ImageId': list(range(1,len(pred)+1)),'Label':pred})
submit.to_csv("final.csv",index=False,header=True)