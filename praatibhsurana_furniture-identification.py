import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras import layers
import pandas as pd
# Preprocessing function
def preprocess(cat, split, label):
    train_images = []
    train_labels = []
    for i in os.listdir(cat):
            image = cv2.imread(cat + '/' + i)
            res = cv2.resize(image, dsize=(128,128), interpolation=cv2.INTER_CUBIC) # Resize
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) # Converting to grayscale
            train_images.append(gray)
            train_labels.append(label)
            
    size = len(train_images)
    return train_images[int(split*size):], train_images[:int(split*size)], train_labels[int(split*size):], train_labels[:int(split*size)] 
        
# Preparing train and test sets
train = []
test = []
labeltrain = []
labeltest = []
# Bed
train_images, test_images, train_labels, test_labels = preprocess('../input/day-3-kaggle-competition/data_comp/data_comp/train/bed', 0.2, 0)
train.extend(train_images)
test.extend(test_images)
labeltrain.extend(train_labels)
labeltest.extend(test_labels)

# Chair
train_images, test_images, train_labels, test_labels = preprocess('../input/day-3-kaggle-competition/data_comp/data_comp/train/chair', 0.15, 1)
train.extend(train_images)
test.extend(test_images)
labeltrain.extend(train_labels)
labeltest.extend(test_labels)

# Sofa
train_images, test_images, train_labels, test_labels = preprocess('../input/day-3-kaggle-competition/data_comp/data_comp/train/sofa', 0.15, 2)
train.extend(train_images)
test.extend(test_images)
labeltrain.extend(train_labels)
labeltest.extend(test_labels)

# Swivelchair
train_images, test_images, train_labels, test_labels = preprocess('../input/day-3-kaggle-competition/data_comp/data_comp/train/swivelchair', 0.1, 3)
train.extend(train_images)
test.extend(test_images)
labeltrain.extend(train_labels)
labeltest.extend(test_labels)

# Table
train_images, test_images, train_labels, test_labels = preprocess('../input/day-3-kaggle-competition/data_comp/data_comp/train/table', 0.15, 4)
train.extend(train_images)
test.extend(test_images)
labeltrain.extend(train_labels)
labeltest.extend(test_labels)

train = np.array(train)
test = np.array(test)
labeltrain = np.array(labeltrain)
labeltest = np.array(labeltest)

train = train.reshape(train.shape[0], 128, 128, 1).astype('float32')
train = (train - 127.5) / 127.5 # Normalize the images to [-1, 1]

test = test.reshape(test.shape[0], 128, 128, 1).astype('float32')
test = (test - 127.5) / 127.5 # Normalize the images to [-1, 1]

print (train.shape)
print (test.shape)
# Model
model = tf.keras.models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0028)),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.75),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    #layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    #layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    #layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')])

model.summary()
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


history = model.fit(train, labeltrain, epochs = 30,
                    validation_data=(test, labeltest), verbose=2)
val = []
fol = []
for i in range(2164):
    image = cv2.imread('../input/day-3-kaggle-competition/data_comp/data_comp/test/img' + str(i) + '.jpg')
    res = cv2.resize(image, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) # Converting to grayscale
    gray = np.array(gray)
    gray = gray.reshape(1, 128, 128, 1).astype('float32')
    gray = (gray - 127.5) / 127.5
    pred = np.argmax(model.predict(gray)) # Predicting class
    val.append(pred)
    fol.append('img' + str(i))

#print (val[0:5])
#print (fol[0:5])
output = pd.DataFrame({'image':fol ,'target':val}) # Creating dataframe
print (output)
output.to_csv('submission3.csv')





