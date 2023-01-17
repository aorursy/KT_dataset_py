# Import numpy and pandas
import numpy as np 
import pandas as pd

# We'll use this later for splitting our data up into training and validation sets
from sklearn.model_selection import train_test_split

# Import matplotlib and sns 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Get everything we need from Keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(5)
# Get number of training examples
print('Number of training examples: ' + str(train.shape[0]))

# Plot the number of examples per class in our training set
sns.countplot(train['label'])
# Get a list of 25 random examples
example_ids = np.random.randint(28000, size = 25)

# Set up your figure
cnt = 1
fig = plt.figure()
title = fig.suptitle('MNIST Handwritten Digit Examples', fontsize = 20)
title.set_position([.5, .92])
fig.set_figheight(12)
fig.set_figwidth(10)
for id in example_ids:
    # Get the data associated with each example
    example_data = train.iloc[[id]]
    
    # Get the label associated with each example
    example_truth = example_data['label'].unique()[0]
    
    # Drop the label so we are only left with the image data
    example_data = example_data.drop(['label'], axis = 1)
    
    # Convert the data to a numpy array so that it's compatible with imshow
    example_data = np.array(example_data)
    
    # Reshape the example to a 28 x 28 array
    example_data = np.reshape(example_data, [28, 28])
    
    # Plot that array in a subplot
    plt.subplot(5, 5, cnt)
    plt.title('Label = ' + str(np.array(example_truth)))
    plt.axis('off')
    plt.imshow(example_data)
    cnt += 1
def digit_conv_net(input_shape, num_classes):
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv0', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv1', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D(pool_size = (2, 2), name = 'max_pool0')(X)
    
    # Add dropout to regularize
    X = Dropout(0.25)(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(64, (5, 5), strides = (1, 1), name = 'conv2', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(64, (5, 5), strides = (1, 1), name = 'conv3', padding = 'same', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D(pool_size = (2, 2), name = 'max_pool1')(X)
    
    # Add dropout to regularize
    X = Dropout(0.25)(X)
    
    # FLATTEN X -> FULLY CONNECTED
    X = Flatten()(X)
    X = Dense(128, activation = 'relu', name = 'fc0')(X)
    X = Dropout(0.25)(X)
    X = Dense(128, activation = 'relu', name = 'fc1')(X)
    X = Dropout(0.25)(X)
    X = Dense(num_classes, activation = 'softmax', name = 'fc2')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name = 'LeNet5')

    return model
# Initialize model with input size and number of classes
DigitModel = digit_conv_net([28, 28, 1], 10)

# Complie the model. Use Adam optimizer.
DigitModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
DigitModel.summary()
train_labels = train['label']
train_labels = pd.get_dummies(train_labels)
train_features = train.drop(['label'], axis = 1)
X_train, X_val, y_train, y_val = train_test_split(train_features,
                                                   train_labels,
                                                   train_size = 0.98,
                                                   test_size = 0.02,
                                                   random_state = 0)
X_train = np.array(X_train)
X_train = np.reshape(X_train, [X_train.shape[0], 28, 28, 1]) / 255.

X_val = np.array(X_val)
X_val = np.reshape(X_val, [X_val.shape[0], 28, 28, 1]) / 255.

test = np.array(test)
test = np.reshape(test, [test.shape[0], 28, 28, 1]) / 255.
ReduceLearningRate = ReduceLROnPlateau(monitor = 'val_acc', 
                                       factor = 0.5, 
                                       patience = 3, 
                                       min_lr = 0.00001)
DataGenerator = ImageDataGenerator(rotation_range = 10,
                                   zoom_range = 0.1,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1)

# Fits the model on batches with real-time data augmentation
DataGenerator.fit(X_train)
# Store this in a variable called mdl which we will use later to visualize the model performance
mdl = DigitModel.fit_generator(DataGenerator.flow(X_train, y_train, batch_size = 32), 
                               epochs = 15,
                               validation_data = (X_val, y_val),
                               callbacks = [ReduceLearningRate])
fig = plt.figure()
title = fig.suptitle('Model Accuracy/Loss Summary', fontsize = 16)
title.set_position([.5, 1.0])
fig.set_figheight(5)
fig.set_figwidth(10)

# Summarize history for accuracy
plt.subplot(1, 2, 1)
plt.plot(mdl.history['acc'])
plt.plot(mdl.history['val_acc'])
plt.title('Model Accuracy', fontsize = 10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(mdl.history['loss'])
plt.plot(mdl.history['val_loss'])
plt.title('Model Loss', fontsize = 10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
# predict results
results = DigitModel.predict(test)

# select the index with the maximum probability
results = np.argmax(results, axis = 1)
results = pd.Series(results, name = "Label")
my_submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results], axis = 1)
my_submission.to_csv("my_submission.csv", index = False)
