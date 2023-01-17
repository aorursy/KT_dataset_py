from keras.models import Sequential
from keras.layers import Dense
#from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.models import Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
Xtrain = np.loadtxt("../input/train_x.csv", dtype = int, delimiter=",");
Ytrain = np.loadtxt("../input/train_y.csv", dtype = int, delimiter=",");
Xtest = np.loadtxt("../input/test_x.csv", dtype = int, delimiter=",");
Ytest = np.loadtxt("../input/test_y.csv", dtype = int, delimiter=",");

print(Xtrain.shape)
print(Ytrain.shape)
print(Xtest.shape)
print(Ytest.shape)


print("NOT Already normalised look at first image " + str(Xtrain[0,:].max()))
print("Check left upper pixel is always 0  " + str(Xtrain[:,0].max()))

# make a version for cnn
input_shape = (28,28,1)
# note these reshaped inputs will be used in CNN 
trainX = np.array([i for i in Xtrain]).reshape(-1,28,28,1)
testX = np.array([i for i in Xtest]).reshape(-1,28,28,1)
trainY = Ytrain
testY = Ytest

for i in range(10):
     plt.imshow(trainX[i,:,:,0],cmap = 'gray')
     str_label = 'Label is :' + str(trainY[i,:].argmax())
     plt.title(str_label)
     plt.show()

# Conventional wisdom seems to be to standardise the inputs for fully connected 
# networks although now that relu is commonly used as an internal layer activation
# rather than sigmoid it may not be as necessary?
Xtrain = Xtrain/255
Xtest = Xtest/255     
trainX = trainX/255
testX = testX/255


model = Sequential()
model.add(Dense(12, input_dim=784, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='softmax'))
# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Fit the model
model.fit(Xtrain, Ytrain, epochs=50, batch_size=10,  verbose=0,
          validation_data=(Xtest, Ytest))

train_performance = model.evaluate(Xtrain, Ytrain,batch_size=10, verbose=0)
test_performance =  model.evaluate(Xtest, Ytest,batch_size=10, verbose=0)

print('Train loss fully connected:', train_performance[0])
print('Train accuracy fully connected:', train_performance[1])

print('Test loss fully connected:', test_performance[0])
print('Test accuracy fully connected:', test_performance[1])

model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform', activation='softmax'))
# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Fit the model
model.fit(Xtrain, Ytrain, epochs=50, batch_size=10,  verbose=0,
          validation_data=(Xtest, Ytest))

glm_train_performance = model.evaluate(Xtrain, Ytrain,batch_size=10, verbose=0)
glm_test_performance =  model.evaluate(Xtest, Ytest,batch_size=10, verbose=0)

print('Train loss fully connected:', glm_train_performance[0])
print('Train accuracy fully connected:', glm_train_performance[1])

print('Test loss fully connected:', glm_test_performance[0])
print('Test accuracy fully connected:', glm_test_performance[1])

model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
# this final layer will compromise a 1x1 feature map only - 32 of them
model1.add(MaxPooling2D(pool_size=(2, 2)))
# need to connect it to the flatten and fully connected network so it is 
# motivated to learn the right features 
model1.add(Flatten())
# this part is really just a 1 hidden layer MLP from here
model1.add(Dense(12, activation='relu'))
model1.add(Dense(10, activation='softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model1.fit(trainX, trainY,
          batch_size=128,
          epochs=20,
          verbose=0,
          validation_data=(testX, testY))
train_performance_cnn = model1.evaluate(trainX, trainY,batch_size=10, verbose=1)
test_performance_cnn =  model1.evaluate(testX, testY,batch_size=10, verbose=1)

#compare to the fully connected model
print('Test loss cnn:', test_performance_cnn[0])
print('Test accuracy cnn:', test_performance_cnn[1])
print('Test loss fully connected:', test_performance[0])
print('Test accuracy fully connected:', test_performance[1])

model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))

model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(10, activation='softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model1.fit(trainX, trainY,
          batch_size=128,
          epochs=20,
          verbose=0,
          validation_data=(testX, testY))
train_performance_cnn_no_hid = model1.evaluate(trainX, trainY,batch_size=10, verbose=1)
test_performance_cnn_no_hid =  model1.evaluate(testX, testY,batch_size=10, verbose=1)


print('Test loss cnn no hidden layer in mlp fc part:', test_performance_cnn_no_hid[0])
print('Test accuracy cnn no hidden layer in mlp fc part::', test_performance_cnn_no_hid[1])
print('Test loss cnn:', test_performance_cnn[0])
print('Test accuracy cnn:', test_performance_cnn[1])
# layer_name = 'max_pooling2d_16'
# model.get_layer(index=0) specify the index
# Indices are based on order of horizontal graph traversal (bottom-up).
# change get_layer(layer_name) to get_layer(index=5) as it is the 6th layer
# just before flatten

intermediate_layer_model = Model(inputs=model1.input,
                                 outputs=model1.get_layer(index=5).output)
intermediate_output = intermediate_layer_model.predict(trainX)
# as the size of the output is 60000*1*1*32 (32 feature maps of 1*1 for each image)
# we can put each image as a row in a matrix - essentially flattenning it ourselves
# this is the part learnt by the fully connected part
intermediate_output_reshape = intermediate_output.reshape(60000,32)


for i in range(20):
    plt.imshow(trainX[i,:,:,0],cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
    plt.imshow(intermediate_output_reshape[i,:].reshape(1,32),cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
    
for i in range(20):
    #plt.imshow(trainX[i,:,:,0],cmap = 'gray')
    #str_label = 'Label is :' + str(trainY[i,:].argmax())
    #plt.title(str_label)
    #plt.show()
    plt.imshow(intermediate_output_reshape[i,:].reshape(1,32),cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
# may want to use the imtermediate layer to generate features that are scored by any ML model
intermediate_output_test = intermediate_layer_model.predict(testX)
print(intermediate_output_test.shape)
intermediate_output_test_reshape = intermediate_output_test.reshape(10000,32)
print(intermediate_output_test_reshape.shape)
# glm equivalent
model = Sequential()
model.add(Dense(10, input_dim=32, init='uniform', activation='softmax'))
# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Fit the model
model.fit(intermediate_output_reshape, Ytrain, epochs=50, batch_size=10,  verbose=0,
          validation_data=(intermediate_output_test_reshape, Ytest))

glm_train_performance = model.evaluate(intermediate_output_reshape, Ytrain,batch_size=10, verbose=0)
glm_test_performance =  model.evaluate(intermediate_output_test_reshape, Ytest,batch_size=10, verbose=0)

print('Train loss fully connected:', glm_train_performance[0])
print('Train accuracy fully connected:', glm_train_performance[1])

print('Test loss fully connected:', glm_test_performance[0])
print('Test accuracy fully connected:', glm_test_performance[1])
# some hidden layers
model = Sequential()
model.add(Dense(12, input_dim=32, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='softmax'))
# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# Fit the model
model.fit(intermediate_output_reshape, Ytrain, epochs=50, batch_size=10,  verbose=0,
          validation_data=(intermediate_output_test_reshape, Ytest))

train_performance = model.evaluate(intermediate_output_reshape, Ytrain,batch_size=10, verbose=0)
test_performance =  model.evaluate(intermediate_output_test_reshape, Ytest,batch_size=10, verbose=0)

print('Train loss fully connected:', train_performance[0])
print('Train accuracy fully connected:', train_performance[1])

print('Test loss fully connected:', test_performance[0])
print('Test accuracy fully connected:', test_performance[1])
from sklearn.ensemble import RandomForestClassifier
# becareful for random forest not to use the image data but the flattened data
rfc = RandomForestClassifier(40)
rfc.fit(intermediate_output_reshape, trainY.argmax(axis=1))
print('\n RF train_performance ' + str(rfc.score(intermediate_output_reshape, trainY.argmax(axis=1))))
print('\n RF test_performance ' + str(rfc.score(intermediate_output_test_reshape, testY.argmax(axis=1))))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs',)
logreg.fit(intermediate_output_reshape, trainY.argmax(axis=1))
print('\n Logistic Regression train_performance ' + str(logreg.score(intermediate_output_reshape, trainY.argmax(axis=1))))
print('\n Logistic Regression test_performance ' + str(logreg.score(intermediate_output_test_reshape, testY.argmax(axis=1))))

model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model1.add(MaxPooling2D(pool_size=(2, 2)))
# add some more if you want
model1.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

# at this point only make a set of 3 1d feature maps
model1.add(Conv2D(3, kernel_size=(3, 3),
                 activation='relu'))

model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
model1.add(Dense(10, activation='softmax'))

model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model1.fit(trainX, trainY,
          batch_size=128,
          epochs=20,
          verbose=0,
          validation_data=(testX, testY))
train_performance_cnn = model1.evaluate(trainX, trainY,batch_size=10, verbose=1)
test_performance_cnn =  model1.evaluate(testX, testY,batch_size=10, verbose=1)


print('Train loss cnn:', train_performance_cnn[0])
print('Train accuracy cnn:', train_performance_cnn[1])
print('Test loss cnn:', test_performance_cnn[0])
print('Test accuracy cnn:', test_performance_cnn[1])
print('Test loss fully connected:', test_performance[0])
print('Test accuracy fully connected:', test_performance[1])
intermediate_layer_model = Model(inputs=model1.input,
                                 outputs=model1.get_layer(index=5).output)
intermediate_output = intermediate_layer_model.predict(trainX)
# as the size of the output is 60000*1*1*3 (3 feature maps of 1*1 for each image)
# we can put each image as a row in a matrix - essentially flattenning it ourselves
# this is the part learnt by the fully connected part
intermediate_output_reshape = intermediate_output.reshape(60000,3)
for i in range(20):
    plt.imshow(trainX[i,:,:,0],cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
    plt.imshow(intermediate_output_reshape[i,:].reshape(1,3),cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
 
for i in range(20):
    #plt.imshow(trainX[i,:,:,0],cmap = 'gray')
    #str_label = 'Label is :' + str(trainY[i,:].argmax())
    #plt.title(str_label)
    #plt.show()
    plt.imshow(intermediate_output_reshape[i,:].reshape(1,3),cmap = 'gray')
    str_label = 'Label is :' + str(trainY[i,:].argmax())
    plt.title(str_label)
    plt.show()
   
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(intermediate_output_reshape[:,0], intermediate_output_reshape[:,1], intermediate_output_reshape[:,2],c=trainY.argmax(axis=1))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


plotly.offline.init_notebook_mode(connected=True)
plotly.__version__
type(intermediate_output_reshape)
intermediate_output_reshape.shape
target = trainY.argmax(axis=1)
print(target.shape)
fours = intermediate_output_reshape[target == 4,:]
zeros = intermediate_output_reshape[target == 0,:]
fives = intermediate_output_reshape[target == 5,:]
sixs = intermediate_output_reshape[target == 6,:]
print(fours.shape)
print(zeros.shape)
trace1 = go.Scatter3d(
    x=fives[:,0],
    y=fives[:,1],
    z=fives[:,2],
    mode='markers',
    marker=dict(
        size=5,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

trace2 = go.Scatter3d(
    x=sixs[:,0],
    y=sixs[:,1],
    z=sixs[:,2],
    mode='markers',
    marker=dict(
        size=5,
        symbol= "x",
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.9
    )
)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='mnist-3d-scatter')
trace1 = go.Scatter3d(
    x=fours[:,0],
    y=fours[:,1],
    z=fours[:,2],
    mode='markers',
    marker=dict(
        size=5,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

trace2 = go.Scatter3d(
    x=zeros[:,0],
    y=zeros[:,1],
    z=zeros[:,2],
    mode='markers',
    marker=dict(
        size=5,
        symbol= "x",
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.9
    )
)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='mnist-3d-scatter1')