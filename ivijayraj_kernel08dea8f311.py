# Importing the Required Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
training_data = pd.read_csv('../input/train.csv')

testing_data = pd.read_csv('../input/test.csv')
x=training_data.iloc[:,1:]
y=training_data.iloc[:,0]
def show_image(image_no,r):

    plt.figure(figsize=(8,8))

    if image_no >= 0:

        for i in range(r):

            image_pixel = x.iloc[image_no].values.reshape(28,28)

            image_no = image_no+1

            plt.subplot(4, 4, i+1) #devide figure into 4x4 and choose i+1 to draw

            plt.imshow(image_pixel)

        plt.show();
x=x/255.0
x_testing_data = testing_data
x_testing_data.shape
x=x.values.reshape(-1,28,28,1)

testing_data = testing_data.values.reshape(-1,28,28,1)
y = tf.keras.utils.to_categorical(y,num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
classifier = tf.keras.models.Sequential()



# Adding the first 2 convolution layer



classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))

classifier.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))



# Pooling layer



classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))



classifier.add(tf.keras.layers.Dropout(.1))



# Adding the second 2 convolution layer



classifier.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

classifier.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))



# 2nd Pooling layer



classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))



classifier.add(tf.keras.layers.Dropout(.1))



# Adding the flattening layer



classifier.add(tf.keras.layers.Flatten())



# Adding the ANN



classifier.add(tf.keras.layers.Dense(units=300, activation='relu'))

classifier.add(tf.keras.layers.Dropout(rate=0.1))

classifier.add(tf.keras.layers.Dense(units=150, activation='relu'))

classifier.add(tf.keras.layers.Dropout(rate=0.1))

classifier.add(tf.keras.layers.Dense(units=10, activation='softmax'))
classifier.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train,y_train, batch_size=40,epochs=30)
testing_data.shape
# Prediction



submission = classifier.predict(testing_data)
submission = np.argmax(submission,axis=1)
len(testing_data)
df =pd.DataFrame({'ImageId':range(1,len(testing_data)+1),'Label':pd.Series(submission)})
df.head()
df.to_csv("cnn_mnist.csv",index=False)