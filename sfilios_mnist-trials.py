import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout

path = '../input/train.csv'
train = pd.read_csv(path)
test = pd.read_csv('../input/test.csv')

#Because the images will are 28x28, and we want to convert them back to this format before we analyze them
#we add variables representing the length and width
img_rows, img_cols = 28, 28
#because we are predicting numbers 0-9, there are 10 final nodes for our prediction layer
num_classes = 10

#print(train.label)
#because all of the labels are ints and all of the features are floats, we can use this to select each for th X and y
# X_train = train.iloc[:,1:].values.astype('float64')
# y_train = train.iloc[:,0].values.astype('int32')
#define a function to extract the labels and reshape the pixel intensity data (not as a column for each, but 
#as a 28X28 grid again) before applying model
def prep_data(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)
    
    num_images = raw.shape[0] # the number of images is equal to the size of one of the columns
    x_as_array = raw.values[:,1:] #take all the rows, and take all but the first column (which is label column)
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1) #reshape the X values into a 4d array.
    out_x = x_shaped_array / 255 #divide by a large number to make the outputs between 0 and 1. Helps with optimizing.
    return out_x, out_y
X, y = prep_data(train) #pass my raw data - from the csv - to the prep data function defined above

#define the model
model = Sequential()
#add the first layer to the Neural network
model.add(Conv2D(20, kernel_size=(3,3), #specify number of nodes (20), size of the convolutions (3 pixels)
                 activation='relu', #include the activation function - what says whether or not a neuron "fired"
                 input_shape=(img_rows, img_cols, 1))) #for the first layer, we specify dimensions. rows, cols, and layers (grayscale = 1 layer)
model.add(Dropout(0.5))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu')) #add a second layer
model.add(Flatten()) #this layer turns the data back into a long row with lots of columns instead of 28x28 format
model.add(Dense(128, activation='relu')) #models perform best with a dense (all connected) layer before final predictions layer
model.add(Dense(num_classes, activation='softmax')) #final prediction layer for the numbers, using softmax to make predictions as percentages

#compile the model
model.compile(loss=keras.losses.categorical_crossentropy, #state which loss function model to use
              optimizer='adam', #using adam optimizer automatically to make gradient descent optimal
              metrics=['accuracy']) #measure things by accuracy of the model

#fit the model
model.fit(X, y,#pass the prepared data into the model to fit it
          batch_size=128, #larger batch sizes go faster
          epochs=2, #number of times you iterate through this process
          validation_split = 0.2) #use this argument to say we want to preserve 20% of the data for validation
#make a new function to work on the test data, because the test data has no labels
def prep_data_test(raw):
    num_images = raw.shape[0] # the number of images is equal to the size of one of the columns
    x_as_array = raw.values[:,:] #take all the rows, and take all but the first column (which is label column)
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1) #reshape the X values into a 4d array.
    out_x = x_shaped_array / 255 #divide by a large number to make the outputs between 0 and 1. Helps with optimizing.
    return out_x

# #prep the test data
X2 = prep_data_test(test)

#re-train using the full training dataset
model.fit(X, y, batch_size=128, epochs=2)




#make predictions
predictions = model.predict_classes(X2)

#write the predictions file to a dataframe
#make a column called 'imageid' with a list starting from one and going to the length of the predictions file
#you have to +1 on the previous line because of 0-indexing
predictions_file = pd.DataFrame({
    'ImageId': list(range(1,len(predictions)+1)), 
    'Label': predictions 
    })

predictions_file.to_csv('submission_mnist_nn.csv', index=False)

