import os

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



print(os.listdir('/kaggle/input/digit-recognizer'))
# Import 'train.csv' using pandas and see how many rows and columns are there.

all_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0)

print(all_data.shape)

all_data.head()
# Assign 'label' column to target variable (y_all_data)

y_all_data = all_data['label']



# Remove 'label' column from rest of the data 

X_all_data = all_data.drop('label', axis=1)



# Convert dataframe into numpy array 

X_all_data = np.asarray(X_all_data)

y_all_data = np.asarray(y_all_data)



#Normalize Data

X_all_data = X_all_data/255



# No. of rows and columns per image

in_shape = X_all_data[0].shape



print(in_shape)
# Re-shape image into 28x28 pixels

disp_image = X_all_data[1].reshape(28,28)



# Display image using matplotlib

plt.imshow(disp_image)

print("Label: ",y_all_data[1])
# Function to stop training once desired accuracy is acheived

class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('acc')>0.999):

            print('99.9% Accuracy reached')

            self.model.stop_training = False

            

mycallback = myCallback()  



# Start building model

model = tf.keras.models.Sequential([

                                    tf.keras.layers.Reshape((28,28,1), input_shape=in_shape),

                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

                                    tf.keras.layers.MaxPooling2D(2,2),

                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

                                    tf.keras.layers.MaxPooling2D(2,2),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512, activation='relu'),

                                    tf.keras.layers.Dense(10, activation='softmax')

]

)



# Compiling the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])



#model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=10, callbacks=[mycallback])



# Start Training Process

model.fit(X_all_data,y_all_data, epochs=50, callbacks=[mycallback])

# Making Prediction

all_test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', header=0)

final_test = np.asarray(all_test_data)

count=1

label = []

id = []

for test in final_test:

    #print(test.reshape(1,784))

    label.append(np.argmax(model.predict(test.reshape(1,784), batch_size=1000)))

    id.append(count)

    count += 1

#Save to csv



submission = pd.DataFrame()

submission['ImageId'] = id

submission['Label'] = label



submission.to_csv('submission.csv', index=False)