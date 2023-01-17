import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras import backend as K


random_seed=21
np.random.seed(random_seed)

# load Kaggle train
data = pd.read_csv("../input/train.csv")
X = data.iloc[:,1:].values
X = X.astype(np.float)
X = np.multiply(X, 1.0 / 255.0)
Ylabel = np.array(data["label"])
del data

Y = to_categorical(Ylabel, num_classes = (np.max(Ylabel)+1))

Nfig=28
NNfig=Nfig*Nfig
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=random_seed)
    
Ytrain_label=np.argmax(Ytrain,axis=0)
Ytest_label=np.argmax(Ytest,axis=0)
    
modelSNN = Sequential()
modelSNN.add(Dense(12, activation = "relu", input_shape=(NNfig,)))
modelSNN.add(Dense((np.max(Ylabel)+1), activation = "softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
modelSNN.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

modelSNN.summary()
epochs = 10
batch_size = 512

historySNN = modelSNN.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size,
             validation_data = (Xtest, Ytest), verbose = 1)
    
Ytrain_pred = modelSNN.predict(Xtrain)
Ytest_pred = modelSNN.predict(Xtest)

plt.figure(figsize=[10,7])
plt.plot(range(1,epochs+1),historySNN.history['acc'],range(1,epochs+1),historySNN.history['val_acc'])
plt.legend(['Training','Validation'])
plt.ylabel('Accuracy [%]')
plt.xlabel('Epoch')
plt.xlim(1,epochs)
plt.grid(True)
plt.title('Learning curve SNN')
plt.show()
W=modelSNN.get_weights()
W1k=W[0].T
vmin, vmax = W1k.min(), W1k.max()
fig, axes = plt.subplots(2, 6, figsize=[12,4])
for coef, ax in zip(W1k, axes.ravel()):
    ax.imshow(coef.reshape(Nfig, Nfig), cmap=plt.cm.binary, vmin=vmin/2, vmax=vmax/2)
    ax.set_xticks(())
    ax.set_yticks(())
plt.suptitle('Weights of hidden layer units', fontsize=16)
plt.show()
X=np.reshape(X,(-1,Nfig,Nfig,1))

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

modelCNN = Sequential()

modelCNN.add(Conv2D(filters = 6, kernel_size = (5,5),padding = 'same', 
                 activation ='relu', input_shape = (Nfig,Nfig,1)))
modelCNN.add(MaxPool2D(pool_size=(2,2)))
modelCNN.add(Flatten())
modelCNN.add(Dense((np.max(Ylabel)+1), activation = "softmax"))
    
modelCNN.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

modelCNN.summary()
epochs = 5
batch_size = 512

historyCNN = modelCNN.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size,
             validation_data = (Xtest, Ytest), verbose = 1)
    
Ytrain_pred = modelCNN.predict(Xtrain)
Ytest_pred = modelCNN.predict(Xtest)

plt.figure(figsize=[10,7])
plt.plot(range(1,epochs+1),historyCNN.history['acc'],range(1,epochs+1),historyCNN.history['val_acc'])
plt.legend(['Training','Validation'])
plt.ylabel('Accuracy [%]')
plt.xlabel('Epoch')
plt.xlim(1,epochs)
plt.grid(True)
plt.title('Learning Curve CNN')
plt.show()
W=modelCNN.get_weights()
W1k=W[0].T
vmin, vmax = W1k.min(), W1k.max()

fig, axes = plt.subplots(2, 3, figsize=[6,4])
for coef, ax in zip(W1k, axes.ravel()):
    ax.imshow(np.squeeze(coef), cmap=plt.cm.binary, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())
plt.suptitle('Weights of filters in the convolution', fontsize=16)
plt.show()


i_plot=np.array([ 1,  3,  6, 10])

get_layer_0_output = K.function([modelCNN.layers[0].input],
                                  [modelCNN.layers[0].output])
layer_0_output = get_layer_0_output([X[i_plot]])[0]

for i in range(len(i_plot)):
    plt.imshow(np.squeeze(X[i_plot[i]]), cmap=plt.cm.binary)
    plt.gca().set_xticks(())
    plt.gca().set_yticks(())
    plt.title('CNN layer input', fontsize=16)
    plt.show()
    
    fig, axes = plt.subplots(2, 3, figsize=[6,4])
    for coef, ax in zip(np.moveaxis(layer_0_output[i,:,:,:],-1,0), axes.ravel()):
        ax.imshow(np.squeeze(coef), cmap=plt.cm.binary)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.suptitle('Output of CNN layer per filter', fontsize=16)
    plt.show()
