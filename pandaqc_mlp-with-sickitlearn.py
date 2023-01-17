# imports



# data analysis

import pandas as pd

import numpy as np

from scipy import stats, integrate



# machine learning

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder



# data visualization

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
# load training data in memory

# the training set is quite large (75 MB), we read only the first 10 records for now

train = pd.read_csv('../input/train.csv', nrows=10)
train.info()
train
# define function that displays the n-th first images of a data set

def display_image(n, dataset, cmap):

   

    # create figure

    fig = plt.figure(figsize=(10,4))



    # loop over the first images of the training set

    for i in range(n):

        ax = plt.subplot(int(n/5), 5, i+1)

        ax.axis('off')

        image = dataset.iloc[i, :].values.reshape(28, 28)    

        imgplot = plt.imshow(image, cmap=cmap)

    

    plt.show()





# we omit the first column of the training set since it contains the image label

display_image(10, train.drop('label', axis=1), 'gray')
# turning 

train[train > 0] = 1



display_image(10, train.drop('label', axis=1), 'binary')
# define number of images to load from the training set

TRAINING_SET_SIZE = 10000

VAL_SET_SIZE_PCT = 0.1 # percentage of images of the training set put aside for validation purpose



# load images from the training set in memory

train = pd.read_csv('../input/train.csv', nrows=TRAINING_SET_SIZE)

X_train = train.iloc[:, 1:] # omit the first column since it contains the label

y_train = train.iloc[:, 0] # extract the labels from the dataset (1st column)



# convert images to black and white

X_train[X_train > 0] = 1



# we retain 10% of the dataset for our validation set

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SET_SIZE_PCT, random_state=1)



print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape)

print('X_val shape:', X_val.shape, 'y_val shape:', y_val.shape)
# instanciate the MLP classifier

clf = MLPClassifier(solver='lbfgs', # we use the L-BFGS optimizer (Limited-memory BFGS)

                    activation='relu', # activation function is the rectified linear unit function, returns f(x) = max(0, x)

                    hidden_layer_sizes=(500,) # one hidden layer of 500 nodes 

                     )



# fit the model with the training set

clf.fit(X_train, y_train)
# run prediction on the validation set 

y_pred = clf.predict(X_val)



print('Classification report:\n\n', classification_report(y_val, y_pred), '\n')

print('------------------------------\n')

print('Confusion matrix:\n\n', confusion_matrix(y_val, y_pred))
# load ALL images from the training set in memory

train = pd.read_csv('../input/train.csv')

X_train = train.iloc[:, 1:] # omit the first column since it contains the label

y_train = train.iloc[:, 0] # extract the labels from the dataset (1st column)



# convert images to black and white

X_train[X_train > 0] = 1



print('X_train shape:', X_train.shape, 'y_train shape:', y_train.shape)



# instanciate the MLP classifier

clf = MLPClassifier(solver='lbfgs', # we use the L-BFGS optimizer (Limited-memory BFGS)

                    activation='relu', # activation function is the rectified linear unit function, returns f(x) = max(0, x)

                    hidden_layer_sizes=(500,) # one hidden layer of 500 nodes 

                     )



# fit the model with the training set

clf.fit(X_train, y_train)
# load ALL images from the testing set

X_test = pd.read_csv('../input/test.csv')



# convert images to black and white

X_test[X_test > 0] = 1



print('Loaded', X_test.shape[0], 'images from the test set')



# run prediction on the entire testing set

y_pred = clf.predict(X_test)
# insert our prediction in a DataFrame with the ImageId

submit = pd.DataFrame({'ImageId': range(1, y_pred.size+1),

                       'Label': y_pred.astype(int)}

                     )

print('Completed prediction of {} images'.format(submit.shape[0]))
# create figure

fig = plt.figure(figsize=(10,4))



# loop over the first 10 images of the test set

for i in range(10):

    ax = plt.subplot(2, 5, i+1)

    ax.set_title('Prediction: {}'.format(submit.iloc[i].Label))

    ax.axis('off')

    image = X_test.iloc[i, :].values.reshape(28, 28)

    imgplot = plt.imshow(image, cmap='binary')



plt.show()
submit.to_csv("prediction-20170627-A.csv", index=False)