import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python import keras

# Each image is a 28*28 pixcel grey-scale image
img_rows, img_cols = 28, 28
# Fashion MNIST has only 10 classes i.e 10 type of  objects (shoe, bag, shirt etc)
num_classes = 10

def prep_data(raw):
    #, train_size, val_size):
    # ALl rows, but first column. That is the Label of each row
    y = raw[:, 0]
    # Create one-hot encoding matrix. So each label in y will be a vector of size 10
    out_y = keras.utils.to_categorical(y, num_classes)
    
    # All rows, but columns from 1 to 256
    x = raw[:,1:]
    
    #shape[0] provides number of rows, which is nothing but number of images
    num_images = raw.shape[0]
    # creating a new multi dimension array. 60k * 28 * 28 * 1 . Here 1 means each pixel has a single numerical value telling how dark it is. (this is grey-scale).
    # If this was a  color image, then this is a vecor of size 3. (RGB)
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    
    #Each value in this matrix is going to be a value between 0 and 1. (after dividng by 255)
    out_x = out_x / 255
    return out_x, out_y

# Location of the file
fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
# Get raw data, not sure why skipping first row. Each column separated by comma
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')

#Prepare data as per above function. what are these 50000 and 5000? why is this required?
#x, y = prep_data(fashion_data, train_size=50000, val_size=5000)
x, y = prep_data(fashion_data)
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

#New Sequential model
fashion_model = Sequential()
#INput shape is required only for the first Conv layer. It is the input for the first layer. 
#For the subsequent layers, shape of output of previous layer is the input shape.
#Increasing teh Kernel size is reducing the accuracy.
#Increasing the neurons here is increasing the trainsing time, but slightly increases the accuracy (10%)
fashion_model.add(Conv2D(24, kernel_size=(3,3), strides=2, activation='relu', input_shape=(img_rows, img_cols, 1)))
fashion_model.add(Dropout(0.5))
fashion_model.add(Conv2D(24, kernel_size=(3,3), strides=2, activation='relu'))
fashion_model.add(Dropout(0.5))
fashion_model.add(Conv2D(24, kernel_size=(3,3), strides=2, activation='relu'))
fashion_model.add(Dropout(0.5))
fashion_model.add(Conv2D(24, kernel_size=(3,3), strides=2, activation='relu'))

#what is this Flatten layer?
fashion_model.add(Flatten())

#Dense model means it has links ocming from each of the previous layer's neurons.
#Why this Dense model.?
fashion_model.add(Dense(128, activation='relu'))
#Last model is the output layer. Number of neurons in this layer should be same as number of output classes. Why is the activation 'softmax'?
fashion_model.add(Dense(num_classes, activation='softmax'))

# Your code to compile the model in this cell
fashion_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer='adam',
                     metrics=['accuracy'])
# Your code to fit the model here
#Batch size is number of examples after which weights are updated.
fashion_model.fit(x,y,batch_size = 50, epochs = 4, validation_split=0.2)