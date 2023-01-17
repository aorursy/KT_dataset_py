# Import libraries and tools
# Data preprocessing and linear algebra
import pandas as pd
import numpy as np
np.random.seed(2)

# Visualisation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

# Tools for cross-validation, error calculation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical

# Machine Learning
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
train.info()
test.info()
# train.head()
# As we can see our dataset consists of label (meaning 1-9 digit) and pixels of handwritten digits.
# So we can go next to form X_train and Y_train datasets which gonna be used in ML algorhytm later.
# Form X_train, Y_train
# Put digits aka true answer in Y_train
Y_train = train['label']
# Drop it as Target variable from X_train 
X_train = train.drop(['label'], axis = 1)
# By the way we can drop train dataset in order to save some disk space since we will use only X_train further.
del train
# Count how many digits we have in Y_train set
Y_train.value_counts(ascending=False)
X_train.isnull().any().count()
test.isnull().any().count()
# Lets normalize the image pixel values from [0, 255] to [-0.5, 0.5] 
# to make our network easier to train (using smaller, centered values leads to better results).
X_train = (X_train / 255) - 0.5
test = (test / 255) - 0.5
# Reshape each image from (28, 28) to (28, 28, 1) because Keras requires the third dimension.
# MNIST images are gray scaled - only one channel. For RGB images, there is 3 channels, 
# so we will reshape 784px vectors to 28x28x3 3D matrices.
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print(X_train.shape)
print(test.shape)
Y_train = to_categorical(Y_train, num_classes = 10)
# Split X_train to train and validation datasets
# Set random seed
random_seed = 2
# Split data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
# # Before class initiation we need to define models hypermarameters, which we will use in our class
# num_filters = 8 #lets use 8 filters
# filter_size = 3 #filter is matrix 3x3
# pool_size = 2 #traverse the input image in 2x2 blocks
# # Initiate class
# model = Sequential([
#     Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)), #input layer
#     MaxPooling2D(pool_size=pool_size),
#     Flatten(),
#     Dense(10, activation='softmax'), #output softmax layer has 10 nodes
# ])
# # Compile the model
# # We decide 3 factors: the optimizer, the loss function, a list of metrics
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy'],
# )
# # # Train the model
# # # We decide 3 parameters: training data, number of epochs, batch size
# # model.fit(
# #     X_train,
# #     Y_train,
# #     epochs=3,
# #     #batch_size=32,
# # )
# Epoch 1/3
# 37800/37800 [==============================] - 24s 630us/step - loss: 0.4036 - accuracy: 0.8837
# Epoch 2/3
# 37800/37800 [==============================] - 15s 394us/step - loss: 0.2115 - accuracy: 0.9386
# Epoch 3/3
# 37800/37800 [==============================] - 15s 404us/step - loss: 0.1537 - accuracy: 0.9562
# # Evaluate the model
# model.evaluate(
#     X_val,
#     Y_val,
# )
# 4200/4200 [==============================] - 1s 243us/step
# [0.1418064293833006, 0.9576190710067749]
# Predict
# predictions = model.predict(X_train)
# # print(np.argmax(predictions, axis=1))
# [8 7 9 ... 2 9 4]
# Initialize model
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
# Define the optimizer
# In our previous model we used Adam optimizer. Now lets try another one - RMSprop, which is enough
# powerfull but can save comp resource. We will use default params.
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(
    optimizer = optimizer , 
    loss = "categorical_crossentropy", 
    metrics=["accuracy"]
)
# Define an annealing method of the learning rate (LR)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001
                                           )
# A. Fit model without data augmentation
history = model.fit(X_train, Y_train, batch_size = 128, epochs = 10, 
validation_data = (X_val, Y_val), verbose = 2)
# Make some data augmentation. Used [2] approach, but it can easily be modified. It is a very intuitive work.
augment = ImageDataGenerator(
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
        vertical_flip=False
        )
# Re-fit using augmentation
augment.fit(X_train)
# B. Fit the model using our augmentaton
history = model.fit_generator(augment.flow(X_train,Y_train, batch_size=128),
                              epochs = 10, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch = X_train.shape[0] // 128,
                              callbacks=[learning_rate_reduction]
                             )
predictions_complex_model = model.predict(X_train)
print(np.argmax(predictions_complex_model, axis=1))
# Loss and accuracy curves for training and validation 
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# Calculate the confusion matrix
conf_mat = confusion_matrix(Y_true, Y_pred_classes)
# PLot confusion matrix
sns.set(font_scale=1.2) # for label size
sns.heatmap(conf_mat, annot=True, annot_kws={"size": 10}) # font size
plt.figure(figsize=(16,10))
plt.show()
# Make final prediction showing our model a real-test data for the first time
results = model.predict(test)
# Select the indix with the maximum probability
results = np.argmax(results,axis = 1)
# Save result as pandas series
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_result.csv",index=False)
# Literature
# [1] https://victorzhou.com/blog/keras-cnn-tutorial/#the-full-code
# [2] https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook
# [3] https://en.wikipedia.org/wiki/Convolutional_neural_network
# [4] https://keras.io/
# [5] https://www.tensorflow.org/
# [6] https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist/notebook
