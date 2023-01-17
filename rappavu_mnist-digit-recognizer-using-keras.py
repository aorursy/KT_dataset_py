import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K

%matplotlib inline
K.set_image_data_format('channels_last')
def load_data():
    # Read the MNIST training data
    df = pd.read_csv('../input/train.csv')
    X = df.drop(columns=['label'])
    X = X.values.reshape((-1, 28, 28, 1)) / 255.0 
    Y = df['label'].values
    Y = to_categorical(Y, num_classes=10)
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)

    # shuffle training data
    indices = np.arange(len(Y))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    # Keep part of data for validation
    index = 40000
    X_train = X_shuffled[:index]
    Y_train = Y_shuffled[:index]
    X_val = X_shuffled[index+1:]
    Y_val = Y_shuffled[index+1:]
    
    return X_train, Y_train, X_val, Y_val
X_train, Y_train, X_val, Y_val = load_data()
plt.imshow(X_train[0,:,:,0])  # view sample data
def DigitModel():
    # Input placeholder
    X_input = Input((28, 28, 1), name='input')

    # Layer-0 : Conv -> Relu -> Max Pooling
    X = Conv2D(16, (5,5), strides=(1,1), padding='same', name='conv0')(X_input)
    X = Activation('relu', name='a0')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    
    # Layer-1 : conv -> Relu -> Max Pooling
    X = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv1')(X)
    X = Activation('relu', name='a1')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    # Layer-2 : Flatten -> Fully Connected -> Softmax
    X = Flatten(name='flatten2')(X)
    X = Dense(10, name='dense2')(X)
    Y = Activation('softmax', name='a2')(X)

    # Create keras model
    model = Model(inputs=X_input, outputs=Y, name='DigitModel')
    
    return model
digit_model = DigitModel()
digit_model.summary()
digit_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
digit_model.fit(X_train, Y_train, epochs=30, batch_size=128)
digit_model.evaluate(X_val, Y_val, batch_size=64)
# Read test data
df = pd.read_csv('../input/test.csv')
X_test = df.values.reshape((-1, 28, 28, 1)) / 255.0
# Predict results for the test data
results = digit_model.predict(X_test)
digits = [np.argmax(r) for r in results]
# Write the predictions
file = open('predictions.csv', 'w')
file.write('ImageId,Label\n')
for i, d in zip(range(len(digits)), digits):
    file.write(str(i+1)+','+str(d)+'\n')
file.close()
