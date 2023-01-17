import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#exploring the data
train.head()
train.shape
test.head()
test.shape
train_images=train.iloc[:,1:].values 
#.values is used to convert dataframe to numpy for better accesibility 
train_labels=train.iloc[:,:1].values
test_images=test.values
np.max(train_images)
train_images=train_images/255.
test_images=test_images/255.
test_images.shape
train_images.shape
train_images=train_images.reshape(train_images.shape[0],28,28)
train_images.shape
test_images=test_images.reshape(test_images.shape[0],28,28)
test_images.shape
plt.figure(figsize=(10,10))
plt.imshow(train_images[1])
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
    plt.xticks([])
    plt.yticks([])
train_images=train_images.reshape(train_images.shape[0],28,28,1)
test_images=test_images.reshape(test_images.shape[0],28,28,1)
classifier=Sequential()
classifier.add(Conv2D(32,(3,3), input_shape=(28, 28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(activation='relu',units=128))
classifier.add(Dense(units=10,activation='softmax'))
classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
from sklearn.model_selection import train_test_split
X = train_images
y = train_labels
train_images1, X_val, train_labels1, y_val = train_test_split(train_images, train_labels, test_size=0.10, random_state=42)
batches = datagen.flow(train_images, train_labels, batch_size=64)
val_batches=datagen.flow(X_val, y_val, batch_size=64)
history=classifier.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=2, 
                    validation_data=val_batches, validation_steps=val_batches.n)
predictions = classifier.predict_classes(test_images)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv('digit_submission3.csv', index=False, header=True)