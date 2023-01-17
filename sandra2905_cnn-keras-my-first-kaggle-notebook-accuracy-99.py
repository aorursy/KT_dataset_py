import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



np.random.seed(2905)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





import os

os.getcwd()
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print(train.shape)

print(test.shape)

print(train.columns)

print(test.columns)
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

Y_train.value_counts()


Y_train = to_categorical(Y_train, num_classes = 10)

print(Y_train.shape)


null=X_train.isnull().any()

null.describe()
test.isnull().any().describe()
# Normalize the data so that the model will converge fast if the values are between o and 1

X_train = X_train / 255.0

test = test / 255.0
X_train.shape
#Visualizing

plt.imshow(X_train.values.reshape(-1,28,28)[126])

print(Y_train[126])
# Reshape image in 3 dimensions ..n*784 to n*(28*28*1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train[0].shape
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)


model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model.summary()



model.compile(RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) , loss = "categorical_crossentropy", metrics=["accuracy"])
cnn = model.fit(X_train, Y_train, batch_size = 32, epochs = 5, validation_data = (X_val, Y_val))




datagen = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False, 

        featurewise_std_normalization=False, 

        samplewise_std_normalization=False, 

        rotation_range=10,

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally 

        height_shift_range=0.1,  # randomly shift images vertically 

        horizontal_flip=False,  # randomly flip images not applied because of numbers such as 6 and 9

        vertical_flip=False)  # randomly flip images not applied because of numbers such as 6 and 9





datagen.fit(X_train)
X_train.shape[0] //32
# Fit the model

cnn_augmented = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=32),epochs = 5, validation_data = (X_val,Y_val),steps_per_epoch=X_train.shape[0] // 32)
Y_pred = model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 



Y_true = np.argmax(Y_val,axis = 1) 

cm =confusion_matrix(Y_true, Y_pred_classes)

print(cm)

accuracy=sum(np.diag(cm))/np.sum(cm)

print('Accuracy: ',accuracy)


errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 9 images with their predicted and real labels"""

    n = 0

    nrows = 3

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(10, 10),constrained_layout=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)),cmap='gray')

            ax[row,col].set_title("Predicted label :{}\n\n True label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 9 errors 

most_important_errors = sorted_dela_errors[-9:]



# Show the top 9 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
results.value_counts()

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("pradeep_cnn_practise.csv",index=False)
submission_view=pd.read_csv('/kaggle/working/pradeep_cnn_practise.csv')
submission_view.tail(20)