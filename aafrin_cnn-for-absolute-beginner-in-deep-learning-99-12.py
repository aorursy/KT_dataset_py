# importing required libraries. 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
# Load Train and Test data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
#checking the shape of train and test dataset
train.shape
test.shape
train_label = train.label.values

# Dropping 'abel' column from train dataset
train = train.drop("label", axis = 1).values

train.shape
# train dataset null value check
np.isnan(train).sum()
# test dataset null value check
test.isnull().any().describe()
train = train/255.0
test= test/255.0
fig1, ax1 = plt.subplots(1,15, figsize=(15,10))
for i in range(15):
    # reshaping the images into 28*28 shape
    ax1[i].imshow(train[i].reshape((28,28)))
    ax1[i].axis('off')
    ax1[i].set_title(train_label[i]) 
train_image =np.array(train).reshape(-1,28,28,1)
test_image =np.array(test).reshape(-1,28,28,1)
train_image.shape
test_image.shape
# first checkin the shape of train_label
train_label.shape
# Encoding labels to one hot encoder
train_label = to_categorical(train_label)
# again checking the shape of train_label
train_label.shape
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), padding = 'Same', activation="relu", input_shape=(28, 28, 1)))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#adding another convulationary layer. Since we are adding another convolution layer, we are not required 
#- to pass input shape parameter
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
#Step 4 - Full connection. 
#Here output_dim is the number of neurons per layer
classifier.add(Dense(output_dim = 256, activation = 'relu'))
#output layer
# here we are using 10 output_dim (neurons) because there are 10 classes
classifier.add(Dense(output_dim = 10, activation = 'softmax'))
# Compilint the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
epochs=30
batch_size=90

# fitting the CNN model 
classifier.fit(train_image, train_label, batch_size=batch_size, epochs=epochs)
#Prediction
results = classifier.predict(test_image)
# Submission
pred = []
numTest = results.shape[0]
for i in range(numTest):
    pred.append(np.argmax(results[i])) 
predictions = np.array(pred) 

sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

result=pd.DataFrame({'ImageId':sample_submission.ImageId, 'Label':predictions})
result.to_csv("submission.csv",index=False)
print(result)