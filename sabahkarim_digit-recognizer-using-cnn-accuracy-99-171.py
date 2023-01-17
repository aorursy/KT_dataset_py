# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

# convert to one-hot-encoding
from keras.utils.np_utils import to_categorical 

# for Multi-layer Perceptron (MLP) model
from keras.models import Sequential
from keras.layers import Dense

# for Convolutional Neural Network (CNN) model
from keras.layers import Dropout, Flatten, Convolution2D, MaxPool2D

# for optimization
from keras.optimizers import RMSprop

train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")

# check attributes of loaded test and train sets
print('Training data is (%d, %d).'% train.shape)
print('Test data is (%d, %d).'% test.shape)
train.head()
Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
# del train 

g = sns.countplot(Y_train)

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# one hot encode outputs
# note that we have new variables with capital Y
Y_train = to_categorical(Y_train, num_classes = 10) # 10 here indicates 10 digits : 0-9
# fix random seed for reproducibility
random_seed = 7

# Split the train and the validation set for the fitting
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# Some examples
g = plt.imshow(X_train[0][:,:,0])
X_train
Y_train
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Convolution2D(64, (3, 3), input_shape = (28,28,1), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPool2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))

# apply maxpooling on 2nd layer
classifier.add(MaxPool2D(pool_size = (2, 2)))
# Adding a third convolutional layer
# classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))

# apply maxpooling on 3rd layer
# classifier.add(MaxPool2D(pool_size = (2, 2)))
# randomly exclude 20% of neurons in the layer in order to reduce overfitting.
classifier.add(Dropout(0.2))
# Step 3 - Flattening
# allows the output to be processed by standard fully connected layers.
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ----- OR TRY THIS -------
# classifier.compile(optimizer= RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, 
               Y_train, 
               validation_data=(X_validation, Y_validation), 
               epochs=100, 
               batch_size=200, 
               verbose=2)
classifier.summary()
scores = classifier.evaluate(X_validation, Y_validation, verbose=0)
print (scores)
print ('Score: {}'.format(scores[0]))
print ('Accuracy: {}'.format(scores[1]))
pred_classes = classifier.predict_classes(X_validation,verbose=0)
i=10
plt.imshow(X_validation[i,:,:,0],cmap='gray')
plt.title('prediction:%d'%pred_classes[i])
plt.show()
def plot_difficult_samples(classifier,x,y, verbose=True):
    """
    model: trained model from keras
    x: size(n,h,w,c)
    y: is categorical, i.e. onehot, size(n,10)
    """ 
    
    pred_classes = classifier.predict_classes(x,verbose= 0)
    y_val_classes = np.argmax(y, axis=1)
    er_id = np.nonzero(pred_classes!=y_val_classes)[0]
    
    K = np.ceil(np.sqrt(len(er_id)))
    fig = plt.figure()
    print('There are %d wrongly predicted images out of %d validation samples'%(len(er_id),x.shape[0]))
    for i in range(len(er_id)):
        ax = fig.add_subplot(K,K,i+1)
        k = er_id[i]
        ax.imshow(x[er_id[i],:,:,0])
        ax.axis('off')
        if verbose:
            ax.set_title('%d as %d'%(y_val_classes[k],pred_classes[k]))
plot_difficult_samples(classifier, X_validation,Y_validation,verbose=False)
plt.show()
# predict results
results = classifier.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("sub_Sabah1.csv",index=False)