import pandas as pd

import numpy as np



import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras.models import Model

from tensorflow.keras import metrics

from tensorflow.keras import backend as K
# Reading the folder architecture of Kaggle to get the dataset path.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading the Train and Test Datasets.

mnist_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

mnist_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# Let's see the shape of the train and test data

print(mnist_train.shape, mnist_test.shape)
# Looking at a few rows from the data isn't a bad idea.

mnist_train.head()
# and yeah, here you will see the basic statistical insights of the numerical features of train data.

mnist_train.describe()
mnist_train.isna().any().any()
# dividing the data into the input and output features to train make the model learn based on what to take in and what to throw out.

mnist_train_data = mnist_train.loc[:, "pixel0":]

mnist_train_label = mnist_train.loc[:, "label"]



# Notmailzing the images array to be in the range of 0-1 by dividing them by the max possible value. 

# Here is it 255 as we have 255 value range for pixels of an image. 

mnist_train_data = mnist_train_data/255.0

mnist_test = mnist_test/255.0
# Let's make some beautiful plots.

digit_array = mnist_train.loc[3, "pixel0":]

arr = np.array(digit_array) 



#.reshape(a, (28,28))

image_array = np.reshape(arr, (28,28))



digit_img = plt.imshow(image_array, cmap=plt.cm.binary)

plt.colorbar(digit_img)

print("IMAGE LABEL: {}".format(mnist_train.loc[3, "label"]))
from sklearn.preprocessing import StandardScaler



standardized_scalar = StandardScaler()

standardized_data = standardized_scalar.fit_transform(mnist_train_data)

standardized_data.shape
cov_matrix = np.matmul(standardized_data.T, standardized_data)

cov_matrix.shape
from scipy.linalg import eigh



lambdas, vectors = eigh(cov_matrix, eigvals=(782, 783))

vectors.shape
vectors = vectors.T

vectors.shape
new_coordinates = np.matmul(vectors, standardized_data.T)

print(new_coordinates.shape)

new_coordinates = np.vstack((new_coordinates, mnist_train_label)).T
df_new = pd.DataFrame(new_coordinates, columns=["f1", "f2", "labels"])

df_new.head()
sns.FacetGrid(df_new, hue="labels", size=6).map(plt.scatter, "f1", "f2").add_legend()

plt.show()
from sklearn import decomposition



pca = decomposition.PCA()

pca.n_components = 2

pca_data = pca.fit_transform(standardized_data)

pca_data.shape
pca_data = np.vstack((pca_data.T, mnist_train_label)).T
df_PCA = pd.DataFrame(new_coordinates, columns=["f1", "f2", "labels"])

df_PCA.head()
sns.FacetGrid(df_new, hue="labels", size=12).map(plt.scatter, "f1", "f2").add_legend()

plt.savefig("PCA_FacetGrid.png")

plt.show()
pca.n_components = 784

pca_data = pca.fit_transform(standardized_data)

percent_variance_retained = pca.explained_variance_ / np.sum(pca.explained_variance_)



cum_variance_retained = np.cumsum(percent_variance_retained)
plt.figure(1, figsize=(10, 6))

plt.clf()

plt.plot(cum_variance_retained, linewidth=2)

plt.axis("tight")

plt.grid()

plt.xlabel("number of compoments")

plt.ylabel("cumulative variance retained")

plt.savefig("pca_cumulative_variance.png")

plt.show()

# Let's build a count plot to see the count of all the labels.

sns.countplot(mnist_train.label)

print(list(mnist_train.label.value_counts().sort_index()))
# Converting dataframe into arrays

mnist_train_data = np.array(mnist_train_data)

mnist_train_label = np.array(mnist_train_label)
# Reshaping the input shapes to get it in the shape which the model expects to recieve later.

mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)

print(mnist_train_data.shape, mnist_train_label.shape)
# But first import some cool libraries before getting our hands dirty!! 

# TensorFlow is Google's open source AI framework and we are using is here to build model.

# Keras is built on top of Tensorflow and gives us

# NO MORE GEEKY STUFF, Know more about them here:  https://www.tensorflow.org     https://keras.io



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D

from tensorflow.keras.optimizers import Adadelta

from keras.utils.np_utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import LearningRateScheduler
# Encoding the labels and making them as the class value and finally converting them as categorical values.

nclasses = mnist_train_label.max() - mnist_train_label.min() + 1

mnist_train_label = to_categorical(mnist_train_label, num_classes = nclasses)

print("Shape of ytrain after encoding: ", mnist_train_label.shape)
# Warning!!! Here comes the beast!!!



def build_model(input_shape=(28, 28, 1)):

    model = Sequential()

    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.4))



    model.add(Conv2D(128, kernel_size = 4, activation='relu'))

    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))

    return model



    

def compile_model(model, optimizer='adam', loss='categorical_crossentropy'):

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    

    

def train_model(model, train, test, epochs, split):

    history = model.fit(train, test, shuffle=True, epochs=epochs, validation_split=split)

    return history
# Training the model using the above function built to build, compile and train the model

cnn_model = build_model((28, 28, 1))

compile_model(cnn_model, 'adam', 'categorical_crossentropy')



# train the model for as many epochs as you want but I found training it above 80 will not help us and eventually increase overfitting.

model_history = train_model(cnn_model, mnist_train_data, mnist_train_label, 80, 0.2)
def plot_model_performance(metric, validations_metric):

    plt.plot(model_history.history[metric],label = str('Training ' + metric))

    plt.plot(model_history.history[validations_metric],label = str('Validation ' + metric))

    plt.legend()

    plt.savefig(str(metric + '_plot.png'))
plot_model_performance('accuracy', 'val_accuracy')
plot_model_performance('loss', 'val_loss')
# reshaping the test arrays as we did to train images above somewhere.

mnist_test_arr = np.array(mnist_test)

mnist_test_arr = mnist_test_arr.reshape(mnist_test_arr.shape[0], 28, 28, 1)

print(mnist_test_arr.shape)
# Now, since the model is trained, it's time to find the results for the unseen test images.

predictions = cnn_model.predict(mnist_test_arr)
# Finally, making the final submissions assuming that we have to submit it in any comptition. P)

predictions_test = []



for i in predictions:

    predictions_test.append(np.argmax(i))
submission =  pd.DataFrame({

        "ImageId": mnist_test.index+1,

        "Label": predictions_test

    })



submission.to_csv('my_submission.csv', index=False)