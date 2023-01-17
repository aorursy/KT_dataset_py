import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns 
from sklearn import preprocessing
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.initializers import he_normal, glorot_normal 
from keras import metrics
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

#importing additional packages for Stage 2: CNN
from keras.layers import Flatten, Conv2D,MaxPooling2D
from keras.optimizers import SGD
import matplotlib.image as mpimg  #For pixel plots

#additional imports for SqueezeNet
from keras.layers import Input, Concatenate, Convolution2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
#READ THE DATA INTO DATAFRAME    
#print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
labels = pd.get_dummies(df_train["label"])

df_train = df_train.drop(["label"], axis = 1)
std_scale = preprocessing.StandardScaler().fit(df_train)
x_train = std_scale.transform(df_train) #Delivers a (42,000 x 784 array)

std_scale = preprocessing.StandardScaler().fit(df_test)
x_test = std_scale.transform(df_test) #Delivers a (42,000 x 784 array)

y_train = labels
print(x_train.shape,x_test.shape,y_train.shape)
# Get training and test loss histories
#This code block is used with thanks to https://chrisalbon.com/deep_learning/keras/visualize_loss_history/
def plot_loss(h):
    m = h.history.keys()
    #print(m)
    training_loss = h.history['loss']
    test_loss = h.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.figure(figsize=[10,10])
    plt.plot(epoch_count, training_loss,'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'CrossVal Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();
    
def plot_accuracy(h):
    training_acc = h.history['acc']
    test_acc = h.history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_acc) + 1)

    # Visualize loss history
    plt.figure(figsize=[10,10])
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, test_acc, 'b-')
    plt.legend(['Training Accuracy', 'CrossVal Accuracy',])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show();
#Stage 1 - without CNN. But full use of L2 regularization, ReLu activation and Batch Normalization
#I have not used Dropouts here as BatchNorm and L2 regularization were already present. Thought Dropouts might be an overkill.
#No improvement in accuracy due to deeper network or due to running more epochs. I got same results with single layer NN itself. 
init = glorot_normal(seed=1)
model = Sequential()
model.add(Dense(128, kernel_initializer = init,kernel_regularizer=regularizers.l2(0.01), input_dim=784))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64, kernel_initializer = init,kernel_regularizer=regularizers.l2(0.01), input_dim=784))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(32, kernel_initializer = init,kernel_regularizer=regularizers.l2(0.01), input_dim=784))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adamop = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)
model.compile(optimizer=adamop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
h=model.fit(x_train, y_train, epochs=100, validation_split=0.3,batch_size = 1000, shuffle=True, verbose=0)
plot_loss(h)
plot_accuracy(h)
predictions = model.predict(x_test)
print(predictions.shape)
pred_labels = K.argmax(predictions, axis= -1)
print(pred_labels.shape)

sess = tf.Session()
with sess.as_default():
    pred_array = pred_labels.eval()

pred_array = np.reshape(pred_array,(28000,1))
print(pred_array.shape)
df = pd.DataFrame(pred_array)
df.describe()
#print(df)
df.to_csv("results.csv",header=["Prediction"],index='False')
#Stage 2 - implementation with CNN 
#Not following any specific CNN architecture - just brute force Design Space Exploration(DSE)

#Reshape the training data and test into 28x28x1 (channel_last form) shape
#No reshaping required for y_train
x_train_cnn = x_train.reshape(42000,28,28,1)
x_train_cnn[1,:].shape
x_test_cnn = x_test.reshape(28000,28,28,1)
x_test_cnn.shape

#Visualize the pixel data  
img = x_train_cnn[1,:,:,0]
plt.imshow(img,cmap='nipy_spectral')

#Build the CNN layers
init = glorot_normal(seed=1) #proper initialization is very important to avoid exploding/vanishing gradients 
model = Sequential()

#First Hidden Layer 
#No padding is done since all images are essentially in greyscale and borders are simply white. 
#Stride is zero?
model.add(Conv2D(16,(8,8),init = init,kernel_regularizer=regularizers.l2(0.01), input_shape=(28,28,1)))
model.add(BatchNormalization()) #Add BatchNormalization prior to Activation
model.add(Activation('relu'))
model.add(Dropout(0.50)) 

#Second Hidden Layer 
model.add(Conv2D(32,(4,4),kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#Not including pooling at this stage - this is to be done
model.add(Dropout(0.50)) 

#Third Hidden Layer 
model.add(Conv2D(64,(2,2),kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#Not including pooling at this stage - this is to be done
model.add(Dropout(0.50)) 

#Fourth Hidden Layer - Flatten
model.add(Flatten())
model.add(Dense(256,kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

#Output layer for classification
model.add(Dense(10, activation='softmax'))
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

h=model.fit(x_train_cnn, y_train, batch_size=100,validation_split=0.3,shuffle=True, epochs=20, verbose=1)
plot_loss(h)
plot_accuracy(h)
predictions = model.predict(x_test_cnn)
print(predictions.shape)
pred_labels = K.argmax(predictions, axis= -1)
print(pred_labels.shape)

sess = tf.Session()
with sess.as_default():
    pred_array = pred_labels.eval()

pred_array = np.reshape(pred_array,(28000,1))
print(pred_array.shape)
df = pd.DataFrame(pred_array)
df.describe()
#print(df)
df.to_csv("results_cnn.csv",header=["Label"],index='False')
#STAGE 3: Using the SqueezeNet approach - anything faster and using less infra is sweet ! 
#Reference: https://arxiv.org/abs/1602.07360 

#Starting with this code base to understand this architecture - https://github.com/DT42/squeezenet_demo/blob/master/model.py
#Will fine tune my own way once I understand this better
def SqueezeNet(nb_classes, inputs=(1, 28, 28)):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """
    input_img = Input(shape=inputs)
    #Modified the stride part here as the default (2,2) stride was causing error
    #Need to investigate further
    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(1, 1), padding='same', name='conv1',
        data_format="channels_first")(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_first")(conv1)
    fire2_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_first")(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_first")(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_first")(fire2_squeeze)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format="channels_first")(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format="channels_first")(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format="channels_first")(fire3_squeeze)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_first")(merge3)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_first")(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_first")(fire4_squeeze)
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_first")(merge4)

    fire5_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format="channels_first")(maxpool4)
    fire5_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format="channels_first")(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format="channels_first")(fire5_squeeze)
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_first")(merge5)
    fire6_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_first")(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_first")(fire6_squeeze)
    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format="channels_first")(merge6)
    fire7_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format="channels_first")(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format="channels_first")(fire7_squeeze)
    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format="channels_first")(merge7)
    fire8_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format="channels_first")(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format="channels_first")(fire8_squeeze)
    merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format="channels_first")(merge8)
    fire9_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze',
        data_format="channels_first")(maxpool8)
    fire9_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1',
        data_format="channels_first")(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2',
        data_format="channels_first")(fire9_squeeze)
    merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        nb_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_first")(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format='channels_first')(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)
    
    return Model(inputs=input_img, outputs=softmax)
#Reshape the training data and test into 28x28x1 (channel_last form) shape
#No reshaping required for y_train
x_train_cnn = x_train.reshape(42000,1,28,28)
x_train_cnn[1,:].shape
x_test_cnn = x_test.reshape(28000,1,28,28)
x_test_cnn.shape

model = SqueezeNet(10)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adamop = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
#model.summary()

#Since Kaggle was timing out, I have reduced epoch count and increased batch size. 
h=model.fit(x_train_cnn, y_train, batch_size=100,validation_split=0.3,shuffle=True, epochs=10, verbose=1)
plot_loss(h)
plot_accuracy(h)
predictions = model.predict(x_test_cnn)
print(predictions.shape)
pred_labels = K.argmax(predictions, axis= -1)
print(pred_labels.shape)

sess = tf.Session()
with sess.as_default():
    pred_array = pred_labels.eval()

pred_array = np.reshape(pred_array,(28000,1))
print(pred_array.shape)
df = pd.DataFrame(pred_array)
df.describe()
#print(df)
df.to_csv("results_cnn_sqn.csv",header=["Label"],index='False')