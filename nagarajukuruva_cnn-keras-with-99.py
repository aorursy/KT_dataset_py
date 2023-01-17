# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # For Visualization
import matplotlib.image as mpimg # To display image
import seaborn as sns # For Visualization
# Magic Command
%matplotlib inline 

np.random.seed(2)

from sklearn.model_selection import train_test_split # Splitting train data into train and validation
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential # Sequeencial model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D # Convolution Network layers
from keras.optimizers import RMSprop # Optimizer
from keras.preprocessing.image import ImageDataGenerator # Image Augmentation
from keras.callbacks import ReduceLROnPlateau # Call backs/Early stopping


sns.set(style='white', context='notebook', palette='deep') # Set aesthetic parameters in one step.
## set max how many rows and columns you want to display in jupyter notebook
pd.options.display.max_columns = 200 
pd.get_option('display.max_rows') 
pd.set_option('display.max_rows',None) 
## Get the file path and file name from kaggel 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load the train and test data
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv') # you can use '../input/digit-recognizer/train.csv'
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv') # you can use '../input/digit-recognizer/test.csv'
# Check dimensions of train
train.shape
# Check dimensions of test
test.shape
# Get first 5 rows from train data
train.head()
# Get first 5 rows from test data
test.head()
# Get last 5 rows from train data
train.tail()
# Get last 5 rows from teste data
test.tail()
# Check data types for train
train.dtypes
# Check data types for test
test.dtypes
# Check column name for train
train.columns
# Check column names for test
test.columns
# Check statistics for Train
train.describe(include='all')
# Check statistics for Test
test.describe(include='all')
# Check the index value for Train
train.index
# Check the index value for Test
test.index
# Seperate target from train data
Y_train = train["label"]

# Drop 'label' column (or) store features/independent columns 
X_train = train.drop(labels = ["label"],axis = 1) 

# Get each label count
Y_train.value_counts()
# Plot graph for each label Vs Count
g = sns.countplot(Y_train)
# Check null values for train data
X_train.isnull().any().describe()
X_train.isna().sum()
# Check null values for test
test.isnull().any().describe()
test.isna().sum()
### this method will return number of levels,null values,unique values,data types

def Observations(df):
    return(pd.DataFrame({'dtypes' : df.dtypes,
                         'levels' : [df[x].unique() for x in df.columns],
                         'null_values' : df.isna().sum(),
                         'Unique Values': df.nunique()
                        }))
# Get number of levels,null values,unique values,data types for train
Observations(X_train)
# Get number of levels,null values,unique values,data types for test
Observations(test)
## below logic is used for checking special charcter in numeric columns

def specialCharcterVerification(data):
    for col in data.columns: 
        print('\n',col,'----->')
        for index in range(1,len(data)):
            try:
                #skip=float(data.loc[index,col])
                skip=int(data.loc[index,col])
            except ValueError :
                print(index,data.loc[index,col])
            
# Check special charcters in train data
specialCharcterVerification(train)
# Check special charcters in test data
specialCharcterVerification(test)
# Normalize the train and test data
X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , chanel = 1)
X_train = X_train.values.reshape(-1,28,28,1) # The -1 can be thought of as a flexible value for the library to fill in for you. The restriction here would be that the inner-most shape of the Tensor should be (28, 28, 1). Beyond that, the library can adjust things as needed. In this case, that would be the # of examples in a batch.
test = test.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
Y_train[1]
Y_train[9]
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
# Some examples
g = plt.imshow(X_train[0][:,:,0])
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(Y_train[i])
    plt.axis('off')
plt.show()
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

# Sequential model
model = Sequential()

# Convolution layer with feature map size 5X5,32 filters,input shape 28X28X1,Relu Activation function
model.add(Conv2D(filters = 32, #  Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
                 kernel_size = (5,5), # An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.Can be a single integer to specify the same value for 1all spatial dimensions.
                 padding = 'Same', # one of `"valid"` or `"same"`  
                 activation ='relu',# Activation function.If you don't specify anything, no activation is applied
                 input_shape = (28,28,1))) # input shapes(28X28X1)

# Convolution layer with feature map size 5X5,32 filters,Relu Activation function
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))

# Maxpooling layer with kernal size 2X2,default stride (pool_size)
model.add(MaxPool2D(pool_size=(2,2)))

# Droput 25% Nodes
model.add(Dropout(0.25))

# Convolution layer with feature map size 3X3,64 filters,Relu Activation function
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))

# Convolution layer with feature map size 3X3,64 filters,Relu Activation function
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))

# Maxpooling layer with kernal size 2X2,Stride 2X2
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Droput 25% Nodes
model.add(Dropout(0.25))

# Convert 2D data into 1D data
model.add(Flatten())

# Fully connected layer with 256 output shape,Relu Activation function
model.add(Dense(256, activation = "relu"))

# Dropouts 50% Nodes
model.add(Dropout(0.5))

# Fully connnected layer with 10 output shape,Softmax activation function
model.add(Dense(10, activation = "softmax"))
# Define the RMSprop optimizer with leaning rate 0.001,
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , # String (name of optimizer) or optimizer instance.
              loss = "categorical_crossentropy", # String (name of objective function) or objective function or`Loss` instance. 
              metrics=["accuracy"]) # List of metrics to be evaluated by the model during training and testing.
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', # quantity to be monitored.
                                            patience=3, # number of epochs that produced the monitored quantity with no improvement after which training will be stopped. 
                                            verbose=1, # int. 0: quiet, 1: update messages.
                                            factor=0.5, # factor by which the learning rate will be reduced. new_lr = lr * factor
                                            min_lr=0.00001) # lower bound on the learning rate.
epochs = 30 # number of epochs to train a model
batch_size = 86 # number of sample to process at a time
# Without data augmentation i obtained an accuracy of 0.98
#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
#          validation_data = (X_val, Y_val), verbose = 2)
# With data augmentation to prevent overfitting (accuracy 0.99)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

# Fit data augmentation model
datagen.fit(X_train)
# Fit the model
history = model.fit_generator(datagen.flow(X_train, # Input data
                                           Y_train, # Labels/ Target
                                           batch_size=batch_size), # batch size (default: 32)
                              epochs = epochs, # Number of epochs to train the model.
                              validation_data = (X_val,Y_val), # on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
                              verbose = 2, # 0, 1, or 2. Verbosity mode 0 = silent, 1 = progress bar, 2 = one line per epoch.
                              steps_per_epoch=X_train.shape[0] // batch_size, # Total number of steps (batches of samples) to yield from `generator` before declaring one epoch finished and starting the next epoch. It should typically be equal to `ceil(num_samples / batch_size)`
                              callbacks=[learning_rate_reduction]) # List of callbacks to apply during training.
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# Get predictions on test data
results = model.predict(test)
results

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results
# create pandas array series to store result
results = pd.Series(results,name="Label")
results.name
# create a pandas data frame to append Image id and Label 
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
# store results in csv file in current directory
submission.to_csv("cnn_submission.csv",index=False)