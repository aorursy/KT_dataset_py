# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
data1 = cv2.imread("../input/cat-and-dog/training_set/training_set/cats/cat.1004.jpg") #cat image

data1 = cv2.resize(data1,(320,320)) 

plt.subplot(1,2,1)

plt.imshow(data1)

plt.axis("off")

plt.title("Cat")

data2 = cv2.imread("../input/cat-and-dog/training_set/training_set/dogs/dog.1010.jpg") #dog image

data2 = cv2.resize(data2,(320,320))

plt.subplot(1,2,2)

plt.imshow(data2)

plt.axis("off")

plt.title("Dog")
TRAIN_DIR = ('../input/cat-and-dog/training_set/training_set')

TEST_DIR = ('../input/cat-and-dog/test_set/test_set')

img_size = 64,64
image_names = []

data_labels = []

data_images = []
def create_data(DIR):    

    for folder in os.listdir(DIR):    # ../training_set    

        for file in os.listdir(os.path.join(DIR,folder)):    # ../training_set/cats

            if file.endswith("jpg"):

                image_names.append(os.path.join(DIR,folder,file))    # ../training_set/cats/cat.1004.jpg

                data_labels.append(folder)    # cat

                img = cv2.imread(os.path.join(DIR,folder,file))

                img = cv2.resize(img,img_size)

                data_images.append(img)

            else:

                continue
create_data(TRAIN_DIR)

create_data(TEST_DIR)
data = np.array(data_images)

data.shape
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

label = le.fit_transform(data_labels)

len(label)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)



print("x_train shape:", x_train.shape)

print("x_test shape:", x_test.shape)

print("y_train shape:", y_train.shape)

print("y_test shape:", y_test.shape)

X_train_flatten = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])

X_test_flatten = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

Y_train = y_train.reshape(y_train.shape[0],1)

Y_test = y_test.reshape(y_test.shape[0],1)

print("X train flatten",X_train_flatten.shape)

print("X test flatten",X_test_flatten.shape)

print("Y train flatten",Y_train.shape)

print("Y test flatten",Y_test.shape)
x_train = X_train_flatten

x_test = X_test_flatten

y_train = Y_train

y_test = Y_test
# Normalization - Reducing to grayscale

x_train = x_train / 255.0

x_test = x_test / 255.0
print("x_train shape",x_train.shape)

print("x_test shape",x_test.shape)

print("y_train shape",y_train.shape)

print("y_test shape",y_test.shape)


from sklearn import linear_model

lr = linear_model.LogisticRegression(random_state=42, max_iter = 100, solver='lbfgs')

print("Test accuracy: {}".format(lr.fit(x_train,y_train).score(x_test,y_test)))



from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense



def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units= 32, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units= 16, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units= 16, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units= 4, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units= 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 2)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
