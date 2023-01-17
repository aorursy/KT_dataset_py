# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the basic libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import tensorflow as tf
# checking the version of the tensorflow 



print(tf.__version__)
(x_train,y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# Checking the shape of the dataset



x_train.shape
y_train.shape
x_test.shape
y_test.shape
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
# Getting the Model as well



from tensorflow.keras.models import Model
# kind of normalization



x_train, x_test =  x_train/255.0, x_test/255.0
print("Shape of x_train is:-",x_train.shape)
print("Shape of x_test is:-", x_test.shape)
x_train = np.expand_dims(x_train, -1)
x_train.shape
# same for x_test



x_test = np.expand_dims(x_test, -1)
x_test.shape
# Getting the number of classes 



k_classes = len(set(y_train))
print("Total number of classes are:-",k_classes)
# Giving the shape of Input on the basis of first data of input data.



i = Input(shape = x_train[0].shape)
x = Conv2D(32, (3,3), strides=2, activation='relu')(i)
x = Conv2D(64, (3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
# To convert the image into the feature vector



x = Flatten()(x)
# Dropout is for regularization



x = Dropout(0.2)(x)
# Applying the Dense layer

x = Dense(512, activation = 'relu',)(x)
x = Dropout(0.2)(x)
x = Dense(k_classes, activation = 'softmax')(x)
# passing inside the model constructor



# First parameter can be considered as input and second is considered as output

cnn_model_1 = Model(i, x)
cnn_model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy',

                   metrics=['accuracy'])
my_result_1 = cnn_model_1.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 20)
plt.plot(my_result_1.history['loss'],label = 'loss line')

plt.plot(my_result_1.history['val_loss'],label = 'validation loss line')



plt.legend()
### Plotting the accuracy per Iteration



plt.plot(my_result_1.history['accuracy'], label = 'Accuracy line')

plt.plot(my_result_1.history['val_accuracy'], label = 'Validation Accuracy line')



plt.legend()
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize = False,

                         title = 'Confusion Matrix',

                         cmap = plt.cm.Blues):

    

    """

    This function prints and plots the confusion matrix. 

    Normalization can be applied by setting 'normalize=True'.

    """

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized Confusion Matrix")

        

    else:

        print("Confusion Matrix, without Normalization")

        

    print(cm)

    

    

    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)

    plt.title(title)

    

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    

    plt.xticks(tick_marks, classes, rotation = 45)

    plt.yticks(tick_marks, classes)

    

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                horizontalalignment='center',

                color="white" if cm[i, j] > thresh else 'black')

    

    

    plt.tight_layout()

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    

    plt.show()

    



p_test = cnn_model_1.predict(x_test).argmax(axis =1) 

cm = confusion_matrix(y_test, p_test)

plot_confusion_matrix(cm, list(range(10)))
# Now, performing the label mapping 



my_labels = '''T-shirt/Top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle boot'''.split()
# We will get a list of these dresses



my_labels
misclassified_idx = np.where(p_test!=y_test)[0]
misclassified_idx
# randomly selecting one data from all those misclassified data



i = np.random.choice(misclassified_idx)
i
plt.imshow(x_test[i].reshape(28,28), cmap = 'gray')



plt.title("True Label: %s Predicted %s" %(my_labels[y_test[i]], my_labels[p_test[i]]))