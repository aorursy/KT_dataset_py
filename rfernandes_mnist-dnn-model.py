# Importing Libraries

#Dataframe
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
# Plot the Figures Inline
%matplotlib inline

# Machine learning 
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
import tensorflow as tf
import random
import scipy.stats as st

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# Importimg Datasets
train_set = pd.read_csv('../input/train.csv')
test_set= pd.read_csv('../input/test.csv')
# Separating features from labels 
X_train = (train_set.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train_set.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test_set.values.astype('float32')
#Printing the data
X_train_fig = X_train.reshape(X_train.shape[0], 28, 28)

fig = plt.figure()
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.tight_layout()
  plt.imshow(X_train_fig[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
X_train.shape
X_test.shape
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)
#One Hot encoding of labels (y_train)
from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes
# Now let's make the Deep Neural Network with Keras-Tensorflow!


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer (input nodes = 784, 1st hidden layer nodes = 512 )
classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu', input_dim = 784))

# Adding the second hidden layer (2nd hidden layer nodes = 512 )
classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer (output layer nodes = 10 (predicted digits))
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 128, epochs = 20)

# Part 3 - Making the predictions 

# Predicting the Test set results
y_pred = classifier.predict_classes(X_test, verbose=0)

y_pred.shape

# Metricss available from the model training
history_dict = history.history
history_dict.keys()
# plotting the metrics for the Training set
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

# preparing results dataframe with image index and predictions for testing dataset (X_test)
df = pd.DataFrame(y_pred)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.head()
# CSV file with image predictions
df.to_csv('submission.csv', header=True)
