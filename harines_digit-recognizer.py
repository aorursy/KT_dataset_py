#Import the necessary Python packages
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,MaxPool2D,Convolution2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Drop 'label' column and use pixel data alone
X_train = train.drop(labels = ["label"],axis = 1) 
Y_train = train["label"]
len(Y_train)
X_train = X_train / 255.0
# test consists only image pixel values and no labels
test = test / 255.0
img_width = 28
img_height = 28
n_channels = 1 #grayscale

# Reshape image
X_train = X_train.values.reshape(-1,img_height,img_width,n_channels)
test = test.values.reshape(-1,img_height,img_width,n_channels)
# Encode labels to one hot vectors (ex : 1 -> [0,1,0,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("Total Images:",len(Y_train)+len(Y_val))
print("Training Images:",len(Y_train))
print("Validation Images:",len(Y_val))
# Define the input shape
input_shape = (img_height,img_width,n_channels)
# Build a sequential model
model = Sequential()

model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = input_shape))
model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Convolution2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.summary()
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Data Augmentation to prevent overfitting
datagen = ImageDataGenerator(
        featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
        samplewise_std_normalization=False, zca_whitening=False, rotation_range=10,
        zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip=False, vertical_flip=False)

datagen.fit(X_train)
Model = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=200),epochs=30,verbose=1,validation_data=(X_val, Y_val))
model.save("cnn_digit_recognizer.h5")
# Compute Train Loss and Accuracy
score = model.evaluate(X_train, Y_train, verbose=1)
print('Train Loss:', score[0])
print('Train Accuracy:', score[1])
# Compute Validation Loss and Accuracy
score = model.X_valuate(X_val, Y_val, verbose=1)
print('Validation Loss:', score[0])
print('Validation Accuracy:', score[1])
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_Matrix = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
print(confusion_Matrix)
#Plot Error Images with labels
def plot_error_images_with_labels(images,pred,obs):
    fig=plt.figure(figsize=(10, 10))
    columns = 5
    rows = 4
    for i in range(1, columns*rows +1):
        image_index = i
        sub_plot = fig.add_subplot(rows, columns, i)
        sub_plot.axis('off')
        if pred is not None:
            sub_plot.set_title("Predicted label :{}\nTrue label :{}".format(pred[i],obs[i]),fontsize = 12)
            plt.imshow(images[image_index].reshape(28, 28),cmap='Greys')
    plt.show()
# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

plot_error_images_with_labels(X_val_errors, Y_pred_classes_errors, Y_true_errors)
# predict results
results = model.predict(test)
# Convert result to test predictions classes
results = np.argmax(results,axis = 1)
# Convert results as a series
results = pd.Series(results,name="Label")
# Convert final Test Results(Labels) to CSV
final_Result = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
final_Result.to_csv("cnn_mnist_datagen.csv",index=False)