import pandas as pd                                  # for data manipulation
import matplotlib.pyplot as plt                      # for data representation
import numpy as np                                   # linear algebra
from keras.utils.np_utils import to_categorical      # one hot enconding
from sklearn.model_selection import train_test_split # selecting train and test samples
from keras.models import Sequential                  # to make our model customizable
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPool2D, Input
from tensorflow.python import keras                  # I'll import loss from here

# Load data:

train_data = pd.read_csv('../input/train.csv')
test_data  = pd.read_csv('../input/test.csv')
print(train_data.head())
print(test_data.head())
print('train.csv dataset contains %d different images.' %(len(train_data)))
print('test.csv dataset contains %d different images.' %(len(test_data)))
# defining x and y:

x = train_data.drop(['label'], axis = 'columns')  # we quit y column from the dataset
y = train_data['label']
'''
plt.figure(1)
plt.title('Y_train')
plt.hist(Y_train, rwidth = 0.9)
plt.xlabel('numbers')
plt.ylabel('count')
plt.show()
'''
# Models usually works better when values are normalized:

x         = x / 255.0
test_data = test_data / 255.0

# We also need Y_train to be categorical in order to train our model

y_categorical = to_categorical(y, num_classes = 10)
x = x.values.reshape(-1, 28, 28, 1)  # (28, 28) is image length, 1 is to build a 3D matrix which contains
                                       #  all images and -1 is because of Keras channel dimension.
    
test_data = test_data.values.reshape(-1, 28, 28, 1)

for i in range(0, 9):
    plt.subplot(3,3,i+1)
    plt.imshow(x[i][:,:,0])
    plt.title(y[i])

# Define the model:

'''
model = Sequential()
model.add(Conv2D(12, input_shape = (28, 28, 1), kernel_size = 3, activation = 'relu'))
model.add(Conv2D(2, kernel_size = 3, activation = "relu"))
model.add(Conv2D(2, kernel_size = 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(100, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

'''
from keras.models import Model
def buildnetwork():
    
    
    inputs = Input(shape=(28, 28, 1))
    model = Sequential()
    model = Convolution2D(filters=32, kernel_size=(3,3),activation='relu')(inputs)
    
    model = Convolution2D(filters=32, kernel_size=(3,3), activation='relu',)(model)
    
    model = MaxPool2D(pool_size=(2,2))(model)
    model= Dropout(0.25)(model)
    
    
    model=Convolution2D(filters = 64, kernel_size = (3,3), activation ='relu')(model)
    model=Convolution2D(filters = 64, kernel_size = (3,3), activation ='relu')(model)
    
    model=MaxPool2D(pool_size=(2,2), strides=(2,2))(model)
    model=Dropout(0.25)(model)


    model=Flatten()(model)
    model=Dense(256, activation = "relu")(model)
    model=Dropout(0.5)(model)
    
    q_values=Dense(10, activation = "softmax")(model)
    
    m = Model(input=inputs, output=q_values)
    return m
d_model = buildnetwork()
'''model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
'''
# Compile the model

d_model.compile(loss = keras.losses.categorical_crossentropy,
                     optimizer = "adam",
                     metrics = ['accuracy'])
# Fit the model

history = d_model.fit(x, y_categorical, batch_size = 100, epochs = 20, validation_split = 0.2)
print(history.history.keys())
answer = d_model.predict(test_data)
print(answer) # array of probabilities. We'll take the most probable.
answer = np.argmax(answer, axis = 1)
answer = pd.Series(answer, name = "Label")
print(answer)
#Special THanks to Juan Diaz for this