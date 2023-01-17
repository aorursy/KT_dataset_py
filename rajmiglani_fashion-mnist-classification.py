import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Load the fashion-mnist data
data_train = pd.read_csv('../input/fashion-mnist_train.csv');
data_test = pd.read_csv('../input/fashion-mnist_test.csv');

#separate features and target
train_x = data_train[list(data_train.columns)[1:]].values;
train_y = data_train['label'].values;
#normalize
train_x = train_x / 255;
train_x = np.reshape(train_x,(-1,28,28));
#training and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_x,train_y,test_size = 0.2);
print("train_x shape:", train_x.shape, "train_y shape:", train_y.shape)
#one hot encoding of targets
from keras.utils import to_categorical
train_y = to_categorical(train_y);
val_y = to_categorical(val_y);

print(train_y.shape);
print(val_y.shape);
# Show one of the images from the training dataset
plt.imshow(train_x[0])
#reshape input into 28*28 matrix for the conv layers
train_x = train_x.reshape(-1,28,28,1);
val_x = val_x.reshape(-1,28,28,1);
#Test data

#separate features and target
test_x = data_test[list(data_train.columns)[1:]].values;
test_y = data_test['label'].values;

#normalize
test_x = test_x / 255;
test_x = np.reshape(test_x,(-1,28,28));
# Show one of the images from the training dataset
plt.imshow(test_x[0])

#one hot encoding
test_y = to_categorical(test_y);

#reshape input into 28*28 matrix for the conv layers
test_x = test_x.reshape(-1,28,28,1);
#import layers
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Dropout, Flatten
from keras.models import Model

#define the model

#input layer
input_layer = Input(shape=(28,28,1));

#conv layer 1
l_1_conv = Conv2D(64,(3,3),activation = 'elu',padding = 'same')(input_layer);
l_1_max = MaxPool2D((2,2),padding = 'same')(l_1_conv);

#conv layer 2
l_2_conv = Conv2D(32,(3,3),activation = 'elu',padding = 'same')(l_1_max);
l_2_max = MaxPool2D((2,2),padding = 'same')(l_2_conv);

#dropout layer
l_drop_1 = Dropout(0.2)(l_2_max);

#flatten the output
l_flat = Flatten()(l_drop_1);

#dense layer
l_dense = Dense(256, activation = 'relu')(l_flat);

#droupout layer
l_drop_2 = Dropout(0.5)(l_dense);

#output layer
output_layer =  Dense(10,activation = 'softmax')(l_drop_2);

#create the model
model = Model(input_layer,output_layer);

#summary
model.summary();
#compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']);
from keras.callbacks import EarlyStopping
#define early stopping
early_stopping = EarlyStopping(monitor = 'val_loss',min_delta = 0, patience = 10,verbose = 1,mode = 'auto');

#train the model
model.fit(train_x,
         train_y,
         batch_size=64,
         epochs=10,
         validation_data=(val_x, val_y),
         callbacks = [early_stopping]);
# Evaluate the model on test set
score = model.evaluate(test_x, test_y, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])
#predictions
print(test_x.shape)
pred = model.predict(test_x)
# check with the original data
print('pred',np.argmax(pred[8]))
print('orig',np.argmax(test_y[8]))
test_x = test_x.reshape((-1,28,28));
plt.imshow(test_x[8])