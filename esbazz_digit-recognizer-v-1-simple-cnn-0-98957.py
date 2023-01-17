def get_data(im_shape):    
    
    train_set = np.genfromtxt("../input/train.csv", delimiter=',', dtype = 'int32')
    test_set = np.genfromtxt("../input/test.csv", delimiter=',', dtype = 'int32')
    
    train_set = train_set[1:, :]
    test_set = test_set[1:, :]

    rand_per = np.random.permutation(train_set.shape[0])

    train = train_set[rand_per[0:int(train_set.shape[0]*0.9)], :]
    dev = train_set[rand_per[int(train_set.shape[0]*0.9): \
                                  int(train_set.shape[0]*0.9) + int(train_set.shape[0]*0.05)], :]
    test = train_set[rand_per[int(train_set.shape[0]*0.9) + int(train_set.shape[0]*0.05):], :]

    train_x = train[:, 1:]
    train_y = train[:, 0]
    dev_x = dev[:, 1:]
    dev_y = dev[:, 0]
    test_x = test[:, 1:]
    test_y = test[:, 0]

    train_x = train_x/255.
    dev_x  = dev_x/255.
    test_x = test_x/255.
    test_set = test_set/255.
    
    train_y = to_categorical(train_y, num_classes = 10)
    dev_y = to_categorical(dev_y, num_classes = 10)
    test_y = to_categorical(test_y, num_classes = 10)
    
    train_x = np.reshape(train_x, (train_x.shape[0], im_shape[0], im_shape[1], 1))
    dev_x = np.reshape(dev_x, (dev_x.shape[0], im_shape[0], im_shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], im_shape[0], im_shape[1], 1))
    test_set = np.reshape(test_set, (test_set.shape[0], im_shape[0], im_shape[1], 1))

    return train_x, train_y, dev_x, dev_y, test_x, test_y, test_set

def digit_model(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(10, (5, 5), strides = (1, 1), name = 'conv1')(X_input)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    
    X = Conv2D(20, (5, 5), strides = (1, 1), name = 'conv2')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    
    X = Dropout(0.25)(X)
    
    X = Flatten()(X)
    X = Dense(100, activation='relu', name='fc1')(X)
    X = Dense(10, activation='softmax', name='fc2')(X)
    
    model = Model(inputs = X_input, outputs = X, name='MNIST_model')
    
    return model
    
    
    
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
K.set_image_data_format('channels_last')

def digit_classifier():
    
    im_shape = (28, 28, 1)
    
    train_x, train_y, dev_x, dev_y, test_x, test_y, test_set = get_data(im_shape)
    
    model = digit_model(im_shape)
    
    model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    
    model.fit(x = train_x, y = train_y, epochs = 100, batch_size = 32, validation_data = (dev_x, dev_y))
    
    preds = model.evaluate(x = test_x, y = test_y)
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    
    imgIDs = np.arange(test_set.shape[0]) 
    pred = model.predict(test_set)
    results = np.argmax(pred, axis=1)
    
    submission = pd.DataFrame({
        "ImageId": imgIDs,
        "Label": results})
    submission.to_csv('submission.csv', index = False)
    
digit_classifier()
