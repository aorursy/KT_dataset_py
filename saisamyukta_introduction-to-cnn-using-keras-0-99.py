import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

import itertools



from keras.utils.np_utils import to_categorical 

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')
# Load the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 



g = sns.countplot(Y_train)
# Check the data

X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train / 255.0

test = test / 255.0
# Observe the data

X_train.head
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1 (as it is gray scale))

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
#observe the labels

Y_train.head(10)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)
# Set the random seed

random_seed = 2
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# Some examples

g = plt.imshow(X_train[0][:,:,0])
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu] -> MaxPool2D -> Dropout]*3 -> Flatten -> Dense -> Dropout -> Out

# If you want to increase accuracy then you can increase the number of CNNs



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
# Try experimenting different Epoch and Batch_size values and fix it to the one which gives maximum accuracy

epochs = 6

batch_size = 128
# Without data augmentation i obtained an accuracy of 0.98114

history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 

          validation_data = (X_val, Y_val), verbose = 2)
from pandas         import DataFrame

predicted_value = model.predict(test)

classes=[0,1,2,3,4,5,6,7,8,9]

list1=[]

for index in range(0, len(predicted_value)):

    list1.append(classes[np.argmax(predicted_value[index])])

results= DataFrame(list1, columns=['Label'])
#change the index value and check the results

index=25

g = plt.imshow(test[index][:,:,0])

print(list1[index])