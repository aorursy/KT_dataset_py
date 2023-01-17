#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import random
from scipy import stats

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
#passaging data
path = "../input/digit-recognizer"
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

#understanding data
print(train.head())
print(test.head())

print(train.describe())
print(test.describe())

print(train.columns)

#Splitting training sets
Y_train = train.loc[:, "label"].to_frame()
X_train = train.drop("label", axis = 1)

print(Y_train.head)
print(X_train.head)

sns.countplot(Y_train["label"])
print(Y_train.shape)
print(type(Y_train))
print(type(X_train))

#Noted that there were more 1s, more 7s, and less 5s than average

#Visualisations
sns.set(color_codes=True)
current_palette = sns.color_palette()
pixel_not_zero = train.values.flatten()[train.values.flatten() > 0]
sns.distplot(pixel_not_zero, color = "r")
#sns.distplot(test)
#Processing (Normalisation) and reshaping of data to fit into neural network 
print(type(X_train))
print(type(test))
X_train = X_train/255
test = test/255 #Possible to divide values of whole df like that  
print(type(X_train)) #Dividing pandas df will not change dataset to become numpy array
print(type(test))

X_train = X_train.to_numpy().reshape(-1, 28,28,1) #But using to_numpy will, and reshape requires numpy arrays
test = test.to_numpy().reshape(-1, 28,28,1)

print(len(X_train))
print(X_train.shape)
print(len(test))
print(test.shape)

print(X_train[0][:,:,0]) #necessary to include in the second set of [], because otherwise you won't see the actual structure
print(type(X_train))

#checking for coherency of transformation
image = X_train[1][:,:,0]
plt.imshow(image, cmap = 'gray', interpolation='nearest')
plt.show()

#for i in range(5): #Checking first 5 images in training data
#    plt.imshow(train_image_df[i][:,:,0], cmap = 'gray')
#    plt.show()

#print(Y_train_OHE[0:5]) #Checking if Y_train first 5 rows corresponds to first 5 images of X_train
# What the difference between one hot encoded prediction and non OHE prediction?
colT = ColumnTransformer(transformers = [('onehot', OneHotEncoder(), ["label"])], remainder = "passthrough")
Y_train_OHE = colT.fit_transform(Y_train).toarray()#? or can use sparse matrix
#Y_test_OHE = colT.fit_transform(Y_test)

print(Y_train_OHE.shape)
#print(Y_test_OHE.shape)
type(Y_train_OHE)
type(Y_train_OHE[0])
#splitting into training and validation set. Might not even be necessary because keras' fitting function
#already allows you to create a validation set instantly
from sklearn.model_selection import train_test_split
random.seed(42)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, 
                                                  Y_train_OHE, 
                                                  test_size = 0.1, 
                                                  random_state= 256)

print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))
#g = plt.imshow(X_train[0][:,:,0], c_map = 'gray')
# Neural network building time
classifier = None
classifier = Sequential()

classifier.add(Conv2D(filters = 32,kernel_size = (3,3), input_shape = (28,28,1), 
                      activation = 'relu', padding = 'valid'))
#classifier.add(BatchNormalization()) #No need for this layer because normalised before this already
classifier.add(Conv2D(filters = 32,kernel_size = (3,3), input_shape = (28,28,1), 
                      activation = 'relu', padding = 'valid'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4)) #Dropout is added to the previous subsampling layer

classifier.add(Conv2D(filters = 64,kernel_size = (3,3), input_shape = (28,28,1), 
                      activation = 'relu', padding = 'valid'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(filters = 64,kernel_size = (3,3), input_shape = (28,28,1), 
                      activation = 'relu', padding = 'valid'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4)) #Dropout then flatten
classifier.add(Flatten())

#classifier.add(Dropout(0.25))
classifier.add(Dense(units = 128, activation = 'relu')) #128 selected based on 
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#if one-hot encoded, use categorical_crossentropy for loss function. Otherwise, use sparse_cross entropy.
#Note, both are for multi-class classification problems

# Suggested CNN architecture: conv(32 filters), conv(64 filters), Maxpooling, dropout(0.25), Flatter(), 
# Dense(128), dropout(0.5), Dense(softmax)

#Output size = ((Width-filter size + 2*padding)/Stride) + 1
(28-3+2)/2 + 1

classifier.summary()
# Initial run of the baseline model
random.seed(42)
model_baseline = classifier.fit(X_train, 
                           Y_train,verbose = 2, 
                           validation_data = (X_test,Y_test), 
                           epochs = 30, 
                           batch_size = 32)
#This produces History-type output 
#Diagnostic plots
accuracy = model_baseline.history['accuracy']
val_accuracy = model_baseline.history['val_accuracy']
val_loss = model_baseline.history['val_loss']
loss = model_baseline.history['loss']

#Loss Curve
plt.plot(range(1,len(val_loss)+1), val_loss, "r-", label = "val_loss")
plt.plot(range(1,len(loss)+1), loss, "y-", label = "loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Values")

plt.legend()
plt.grid(True)
plt.show()


#Accuracy Curve (Training accuracy and validation accuracy)
plt.plot(range(1,len(accuracy)+1), accuracy, "b-", label = "accuracy")
plt.plot(range(1,len(val_accuracy)+1), val_accuracy, "g-", label = "val_accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Values")

plt.legend()
plt.grid(True)
plt.show()
#Best number of epoch shown to be 4

#Uncertainty Curve

#
#image data generator to create more variation of images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2)

# fits the model on batches with real-time data augmentation:
epoch = 24
classifier.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    steps_per_epoch=len(X_train)/32, epochs=epoch)
#CM of y_pred vs y_actual. ALways remember to check prediction against actual results. Otherwise, what's the
# point of a validation set?
y_pred = classifier.predict(X_test, batch_size = 32, verbose = 1)
y_pred_num = np.argmax(y_pred, axis = 1)
y_test_num = np.argmax(Y_test, axis = 1)
cm = confusion_matrix(y_test_num, y_pred_num)
print(cm)

sns.set(rc={'figure.figsize':(5,5)})
sns.heatmap(cm, cmap=plt.cm.Blues, annot = True, xticklabels = True, yticklabels = True)

#Fitting model to test dataset
predictions = classifier.predict(test, batch_size = 32, verbose = 1)

type(predictions)
predictions.shape

results = np.argmax(predictions, axis = 1) #Returned the indices of the maximum value found in specified axis.
# Only works because column values range from 0-9 as well. Also because it's one-hotencoded data. 
type(results)
results.shape
results[:5]

pd_results = pd.Series(results, name = "Label") #Conversion to pd.Series automatically turns np.array into a column. 
type(pd_results)
pd_results.shape
pd_results[:5] #Series automatically 

#Just to check if results align
image = test[0][:,:,0]
plt.rcParams['figure.figsize'] = (5,5)
plt.imshow(image, cmap = 'gray', interpolation='nearest')
plt.show()
#Prepare submission file aaaand submit!
index = pd.Series(range(1,len(pd_results)+1), name = "ImageId")
submission = pd.concat([index, pd_results], axis = 1)

submission.shape
submission.head
                       
submission.to_csv("Digits_submission_19112019_8.csv", index = False)
