import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
training_data1 = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

training_labels1 = training_data1.label

training_data1 = training_data1.drop(['label'],axis=1)



training_data1 = training_data1.to_numpy()

training_data1 = training_data1.reshape(training_data1.shape[0],28,28,1)

training_data1 = training_data1.astype('float32')

training_data1 /= 255

training_data1.shape
training_labels1 = tf.keras.utils.to_categorical(training_labels1,num_classes=10)

training_labels1.shape
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu',padding='same'),

    

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),

  

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),

    

    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(10,activation='softmax')

])



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
from sklearn.model_selection import train_test_split



x_train,x_val,y_train,y_val = train_test_split(training_data1,training_labels1,test_size=0.2)



history = model.fit(training_data1,training_labels1,

          epochs=10,

         validation_data=(x_val,y_val)

         )
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'])

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'])

plt.show()
testing_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

testing_data = testing_data.to_numpy()

testing_data = testing_data.reshape(-1,28,28,1)

testing_data = testing_data.astype('float32')

testing_data /= 255



test_predictions = model.predict_classes(testing_data)

image_ids =  np.arange(1,28001)

submission = pd.DataFrame({"ImageId":image_ids,"Label":test_predictions})

submission.to_csv('submission_digit_recignizer.csv',index=False)