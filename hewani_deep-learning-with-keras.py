import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

from subprocess import check_output


np.random.seed(2)

img_rows, img_cols = 28, 28
num_classes = 10

train_file = "../input/train.csv"
test_file = "../input/test.csv"
output_file = "submission.csv"

raw_data = pd.read_csv(train_file)
test = pd.read_csv(test_file)

def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

x, y = data_prep(raw_data)

#Specify Model
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),strides=2, activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


#Compile Model

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#Fit Model

hist = model.fit(x, y,
          batch_size=128,
          epochs=20,
          validation_split = 0.2)

#Evaluate the model
#Training and validation curves
# Plot the loss and accuracy curves for training and validation 

plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()

test = pd.read_csv('../input/test.csv') # Read csv file in pandas dataframe
test_data = StandardScaler().fit_transform(np.float32(test.values)) # Convert the dataframe to a numpy array
test_data = test_data.reshape(-1, 28, 28, 1) # Reshape the data into 42000 2d images

# Prediction:
mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
x_test = mnist_testset.astype("float32")
x_test = x_test.reshape(-1, 28, 28, 1)/255

# predict results
y_hat = model.predict(x_test, batch_size=64)

#select the class with highest probability
y_pred = np.argmax(y_hat,axis=1)


#Submission
with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(len(y_pred)) :
        f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))  

print("Done")
print(check_output(["ls", "../input"]).decode("utf8"))

