import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau



sns.set(style='white', context='notebook', palette='bright')
#load training data

train_orig = pd.read_csv("../input/digit-recognizer/train.csv")

train_orig.describe()
#checking distribution of target variable

def check_label_distribution(target):

    print(target.value_counts())

    sns.countplot(target)
check_label_distribution(train_orig["label"])
#separate out target variable

def separate_x_y(data):

    return (data.drop("label",axis=1), data["label"])
x_train,y_train = separate_x_y(train_orig)
#normalize pixel values

def normalize_pixels(train):

    return train/255.
x_train = normalize_pixels(x_train)

x_train.describe()
#reshaping for feeding into CNN

def reshaping_images(images,height, width ,canal):

    return images.values.reshape(-1, height, width, canal)
#reshaping 

x_train = reshaping_images(x_train, 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)
X_train, X_dev, Y_train, Y_dev = train_test_split(x_train, y_train, test_size=0.2, random_state = 7)
print(X_train.shape)

print(Y_train.shape)
plt.imshow(X_train[4][:,:,0])
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=5, 

                                            verbose=1, 

                                            factor=0.2, 

                                            min_lr=0.0001)
epoch= 31

batch_size= 64
model_fit = model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epoch, verbose=1, callbacks=[learning_rate_reduction],validation_data=(X_dev, Y_dev), shuffle=True)

Y_pred = model.predict(X_dev)

confusion_matrix(np.argmax(Y_dev,axis = 1) , np.argmax(Y_pred,axis = 1) )
arr=np.argmax(Y_dev,axis = 1)

pred_arr = np.argmax(Y_pred,axis = 1)

ind=np.where(arr==1)

ind2=np.where(pred_arr==7)

match_ind = np.intersect1d(ind,ind2)

if len(match_ind)>0:

    match_ind=match_ind[0]

    print("Actual: ",arr[match_ind])

    print("Predicted: ",pred_arr[match_ind])

    plt.imshow(X_dev[match_ind][:,:,0])
arr=np.argmax(Y_dev,axis = 1)

pred_arr = np.argmax(Y_pred,axis = 1)

ind=np.where(arr==4)

ind2=np.where(pred_arr==9)

match_ind = np.intersect1d(ind,ind2)

if len(match_ind)>0:

    match_ind=match_ind[0]

    print("Actual: ",arr[match_ind])

    print("Predicted: ",pred_arr[match_ind])

    plt.imshow(X_dev[match_ind][:,:,0])
arr=np.argmax(Y_dev,axis = 1)

pred_arr = np.argmax(Y_pred,axis = 1)

ind=np.where(arr==8)

ind2=np.where(pred_arr==9)

match_ind = np.intersect1d(ind,ind2)

if len(match_ind)>0:

    match_ind=match_ind[0]

    print("Actual: ",arr[match_ind])

    print("Predicted: ",pred_arr[match_ind])

    plt.imshow(X_dev[match_ind][:,:,0])
arr=np.argmax(Y_dev,axis = 1)

pred_arr = np.argmax(Y_pred,axis = 1)

ind=np.where(arr==7)

ind2=np.where(pred_arr==2)

match_ind = np.intersect1d(ind,ind2)

if len(match_ind)>0:

    match_ind=match_ind[0]

    print("Actual: ",arr[match_ind])

    print("Predicted: ",pred_arr[match_ind])

    plt.imshow(X_dev[match_ind][:,:,0])
test_orig = pd.read_csv("../input/digit-recognizer/test.csv")

test = normalize_pixels(test_orig)

test = reshaping_images(test, 28, 28, 1)

results = model.predict(test)

results = np.argmax(results,axis = 1)
results.shape

results = pd.Series(results,name="Label")

results.describe()
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission
submission.to_csv("cnn_adam.csv",index=False)