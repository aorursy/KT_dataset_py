import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set() # setting seaborn default for plots



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



from keras.utils import np_utils

from keras.datasets import mnist



# for Multi-layer Perceptron (MLP) model

from keras.models import Sequential

from keras.layers import Dense



# for Convolutional Neural Network (CNN) model

from keras.layers import Dropout, Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D



# fix for issue: https://github.com/fchollet/keras/issues/2681

from keras import backend as K

K.set_image_dim_ordering('th')
train = pd.read_csv('../input/train.csv')

print (train.shape)

train.head()
test = pd.read_csv('../input/test.csv')

print (test.shape)

test.head()
y_train = train['label']

X_train = train.drop(labels=['label'], axis=1)

X_test = test



print (y_train.value_counts())
sns.countplot(y_train)
X_train.head()
# check for corrupted images in the datasets

# i.e. check if there are any empty pixel values

print (X_train.isnull().any().sum())

print (X_test.isnull().any().sum())
X_train = X_train.values.astype('float32') # pixel values of all images in train set

y_train = y_train.values.astype('int32') # labels of all images

X_test = test.values.astype('float32') # pixel values of all images in test set
print (X_train.shape)

print (y_train.shape)
print (y_train[0])

print (X_train[0])
plt.figure(figsize=[20,8])

for i in range(6):

    plt.subplot(1,6,i+1)

    # Here, we reshape the 784 pixels vector values into 28x28 pixels image

    plt.imshow(X_train[i].reshape(28, 28), cmap='gray', interpolation='none')

    plt.title("Class {}".format(y_train[i]))
# fix random seed for reproducibility

random_seed = 7

np.random.seed(random_seed)
# pixel values are gray scale between 0 and 255

# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255

print (X_train[1])
print (y_train.shape)

print (y_train[0])
# one hot encode outputs

# note that we have new variables with capital Y

# Y_train is different than y_train

Y_train = np_utils.to_categorical(y_train)

num_classes = Y_train.shape[1]
print (y_train.shape, Y_train.shape)

print (y_train[0], Y_train[0])
# Split the entire training set into two separate sets: Training set and Validation set

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.10, random_state=random_seed)
print (X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

num_pixels = X_train.shape[1]
print (Y_val)

# converting one-hot format of digits to normal values/labels

y_val = np.argmax(Y_val, 1) # reverse of to_categorical

print (y_val)

# Note that: capital Y_val contains values in one-hot format and small y_val contains normal digit values
def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model = baseline_model()

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=200, verbose=1)
model.summary()
scores = model.evaluate(X_val, Y_val, verbose=0)

print (scores)

print ('Score: {}'.format(scores[0]))

print ('Accuracy: {}'.format(scores[1]))
# get predicted values

predicted_classes = model.predict_classes(X_val)
# get index list of all correctly predicted values

correct_indices = np.nonzero(np.equal(predicted_classes, y_val))[0]



# get index list of all incorrectly predicted values

incorrect_indices = np.nonzero(np.not_equal(predicted_classes, y_val))[0]
print ('Correctly predicted: %i' % np.size(correct_indices))

print ('Incorrectly predicted: %i' % np.size(incorrect_indices))
plt.figure(figsize=[20,8])

for i, correct in enumerate(correct_indices[:6]):

    plt.subplot(1,6,i+1)

    plt.imshow(X_val[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_val[correct]))

    

plt.figure(figsize=[20,8])

for i, incorrect in enumerate(incorrect_indices[:6]):

    plt.subplot(1,6,i+1)

    plt.imshow(X_val[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_val[incorrect]))
# we have digit labels from 0 to 9

# we can either manually create a class variable with those labels

# class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



# or, we can take unique values from train dataset's labels

class_names = np.unique(y_train)



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_val, predicted_classes)

np.set_printoptions(precision=2)



print ('Confusion Matrix in Numbers')

print (cnf_matrix)

print ('')



cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]



print ('Confusion Matrix in Percentage')

print (cnf_matrix_percent)

print ('')



true_class_names = class_names

predicted_class_names = class_names



df_cnf_matrix = pd.DataFrame(cnf_matrix, 

                             index = true_class_names,

                             columns = predicted_class_names)



df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 

                                     index = true_class_names,

                                     columns = predicted_class_names)



plt.figure(figsize = (8,6))



#plt.subplot(121)

ax = sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

ax.set_ylabel('True values')

ax.set_xlabel('Predicted values')

ax.set_title('Confusion Matrix in Numbers')



'''

plt.subplot(122)

ax = sns.heatmap(df_cnf_matrix_percent, annot=True)

ax.set_ylabel('True values')

ax.set_xlabel('Predicted values')

'''
train = pd.read_csv('../input/train.csv')

print (train.shape)

train.head()
test = pd.read_csv('../input/test.csv')

print (test.shape)

test.head()
y_train = train['label']

X_train = train.drop(labels=['label'], axis=1)

X_test = test
X_train = X_train.values.astype('float32') # pixel values of all images in train set

y_train = y_train.values.astype('int32') # labels of all images

X_test = test.values.astype('float32') # pixel values of all images in test set
print (X_train.shape)

print (y_train.shape)

print (X_train[1])
# pixel values are gray scale between 0 and 255

# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255

print (X_train[1])
print (y_train.shape)

print (y_train[0])
# one hot encode outputs

# note that we have new variables with capital Y

# Y_train is different than y_train

Y_train = np_utils.to_categorical(y_train)

num_classes = Y_train.shape[1]
print (y_train.shape, Y_train.shape)

print (y_train[0], Y_train[0])
# Split the entire training set into two separate sets: Training set and Validation set

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.10, random_state=random_seed)
print (X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)

num_pixels = X_train.shape[1]
print (Y_val)

# converting one-hot format of digits to normal values/labels

y_val = np.argmax(Y_val, 1) # reverse of to_categorical

print (y_val)

# Note that: capital Y_val contains values in one-hot format and small y_val contains normal digit values
# reshape to be [samples][pixels][width][height]

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_val = X_val.reshape(X_val.shape[0], 1, 28, 28).astype('float32')



print (num_pixels, X_train.shape, X_test.shape, X_val.shape)
print (X_train[1])
# baseline model for CNN

def baseline_model():

    # create model    

    model = Sequential()    

    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))    

    model.add(MaxPooling2D(pool_size=(2, 2)))    

    model.add(Dropout(0.2))

    model.add(Flatten())    

    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))    

    # compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    

    return model
# Example of using RMSprop optimizer

#from keras.optimizers import RMSprop, SGD

#model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001), metrics=['accuracy'])
model = baseline_model()

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, batch_size=200, verbose=1)
history_dict = history.history

history_dict.keys()
plt.figure(figsize=[10,4])



plt.subplot(121)

plt.plot(range(1, len(history_dict['val_acc'])+1), history_dict['val_acc'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.subplot(122)

plt.plot(range(1, len(history_dict['val_loss'])+1), history_dict['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')
model.summary()
scores = model.evaluate(X_val, Y_val, verbose=0)

print (scores)

print ('Score: {}'.format(scores[0]))

print ('Accuracy: {}'.format(scores[1]))
# get predicted values

predicted_classes = model.predict_classes(X_val)
# get index list of all correctly predicted values

correct_indices = np.nonzero(np.equal(predicted_classes, y_val))[0]



# get index list of all incorrectly predicted values

incorrect_indices = np.nonzero(np.not_equal(predicted_classes, y_val))[0]
print ('Correctly predicted: %i' % np.size(correct_indices))

print ('Incorrectly predicted: %i' % np.size(incorrect_indices))
plt.figure(figsize=[20,8])

for i, correct in enumerate(correct_indices[:6]):

    plt.subplot(1,6,i+1)

    plt.imshow(X_val[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_val[correct]))

    

plt.figure(figsize=[20,8])

for i, incorrect in enumerate(incorrect_indices[:6]):

    plt.subplot(1,6,i+1)

    plt.imshow(X_val[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_val[incorrect]))
# we have digit labels from 0 to 9

# we can either manually create a class variable with those labels

# class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



# or, we can take unique values from train dataset's labels

class_names = np.unique(y_train)



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_val, predicted_classes)

np.set_printoptions(precision=2)



print ('Confusion Matrix in Numbers')

print (cnf_matrix)

print ('')



cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]



print ('Confusion Matrix in Percentage')

print (cnf_matrix_percent)

print ('')



true_class_names = class_names

predicted_class_names = class_names



df_cnf_matrix = pd.DataFrame(cnf_matrix, 

                             index = true_class_names,

                             columns = predicted_class_names)



df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 

                                     index = true_class_names,

                                     columns = predicted_class_names)



plt.figure(figsize = (8,6))



#plt.subplot(121)

ax = sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

ax.set_ylabel('True values')

ax.set_xlabel('Predicted values')

ax.set_title('Confusion Matrix in Numbers')



'''

plt.subplot(122)

ax = sns.heatmap(df_cnf_matrix_percent, annot=True)

ax.set_ylabel('True values')

ax.set_xlabel('Predicted values')

'''
def baseline_model():

    # create model

    model = Sequential()

    

    model.add(Conv2D(filters=30, kernel_size=(5, 5), input_shape=(1, 28, 28), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    

    model.add(Conv2D(filters=15, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    

    model.add(Flatten())

    

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.25))

    

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(num_classes, activation='softmax'))

    

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model
# build the model

model = baseline_model()



# fit the model

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=200)
history_dict = history.history

history_dict.keys()
plt.figure(figsize=[10,4])



plt.subplot(121)

plt.plot(range(1, len(history_dict['val_acc'])+1), history_dict['val_acc'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')



plt.subplot(122)

plt.plot(range(1, len(history_dict['val_loss'])+1), history_dict['val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')
model.summary()
scores = model.evaluate(X_val, Y_val, verbose=0)

print (scores)

print ('Score: {}'.format(scores[0]))

print ('Accuracy: {}'.format(scores[1]))
# get predicted values for test dataset

predicted_classes = model.predict_classes(X_test)



submissions = pd.DataFrame({'ImageId': list(range(1, len(predicted_classes) + 1)), 

                            "Label": predicted_classes})



#submissions.to_csv("submission.csv", index=False, header=True)