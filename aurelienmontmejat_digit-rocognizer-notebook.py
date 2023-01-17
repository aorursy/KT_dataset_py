import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


training_images = pd.read_csv('../input/train.csv')

test_images = pd.read_csv('../input/test.csv')

training_labels = training_images['label']



training_images = training_images.drop(labels = ['label'], axis = 1)



training_images = training_images.values.reshape(42000, 28, 28, 1)

test_images = test_images.values.reshape(28000, 28, 28, 1)



training_images = training_images / 255.0

test_images = test_images / 255.0
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(training_images, training_labels, epochs=15)
predict = model.predict(test_images)

predict_array = np.argmax(predict, axis=1)

predict_array = predict_array.tolist()



#print(predict_array)
image_set = pd.read_csv('../input/test.csv')



for i in np.random.randint(28001, size=10):

  print("Predicted : " + str(predict_array[i]))

  image = image_set.iloc[i].values.reshape(28, 28)

  plt.imshow(image, cmap='gray')

  plt.show()
data_to_submit = pd.DataFrame({

    'ImageId': range(1, 28001),

    'Label': predict_array

})



data_to_submit.to_csv('csv_to_submit3.csv', index = False)