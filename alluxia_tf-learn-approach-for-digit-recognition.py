import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf

import tflearn

%config IPCompleter.greedy=True
# Load the input data

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print('Number of records in training file:'+str(len(train)))

print('Number of records in testing file:'+str(len(test)))

print(test.head(5))
# Generate trainX,trainY

trainX=train.drop('label',axis=1)

trainX=trainX.values

print(trainX[:5])

print()

print('Shape of trainX:'+str(trainX.shape))

print()

#Convert trainY to one_hot encoding values

trainY=train['label']

trainY=np.eye(max(trainY)+1)[trainY]

print(trainY[:5])
# Visualizing the data



def show_digit(index):

    label=trainY[index].argmax(axis=0)

    #Reshape 784 array into 28x28 image

    image=trainX[index].reshape([28,28])

    plt.title('Training data,index:%d,label:%d' %(index,label))

    plt.imshow(image,cmap='gray_r')

    plt.show

 

 # Display the first training image

show_digit(9997)

 

    
# Building the network



def build_model():

    # Input Layer

    net=tflearn.input_data([None,trainX.shape[1]])

    # Hidden Layer

    net=tflearn.fully_connected(net,64,activation='ReLU',regularizer='L2',weight_decay=0.001)

    net=tflearn.dropout(net,0.75)

    net=tflearn.fully_connected(net,64,activation='ReLU',regularizer='L2',weight_decay=0.001)

    net=tflearn.dropout(net,0.75)

    # Output Layer

    net=tflearn.fully_connected(net,10,activation='softmax')

    

    # Set up how to train the network

    net=tflearn.regression(net,optimizer='sgd',learning_rate=0.001,loss='categorical_crossentropy')

    

    model=tflearn.DNN(net)

    return model

    

    
# Build the model

model=build_model()

# Train the model



model.fit(trainX,trainY,validation_set=0.25,show_metric=True,batch_size=128,n_epoch=200)
# Run test records and generate predictions

testX=test.values

prediction=np.array(model.predict(testX)).argmax(axis=1)

prediction=pd.Series(prediction,name='label')

print(prediction[:10])

# Generate the submission file



submission=pd.concat([pd.Series(range(1,len(test)+1),name='ImageID'),prediction],axis=1)

submission.to_csv('cnn_mnist_datagen.csv',index=False)