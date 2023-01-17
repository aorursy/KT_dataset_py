import pandas as pd
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from ipywidgets import interact
from keras.layers import Input, Dense, Dropout,Conv2D,MaxPooling2D,Flatten
from keras.layers import GlobalMaxPooling2D,UpSampling2D,GlobalMaxPooling1D
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Print out the folders where our datasets live. 
print("Datasets: {0}".format(os.listdir("../input/danbury-ai-june-2018")))
# The training images. 
X = np.load("../input/danbury-ai-june-2018/train_x.npy")
y = np.load("../input/danbury-ai-june-2018/train_y.npy")

# We subtract 1 from the labels in order to scale the the labels between 0,9. 
y = y - 1

# These are the images we will need to predict lables for. 
test  = np.load("../input/danbury-ai-june-2018/test.npy")
X_train, X_validation, y_train , y_validation = train_test_split(X,to_categorical(y,10), test_size=0.2)
def makeModel(inputSize):
    inputs = Input(shape=inputSize,name="input")
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = GlobalMaxPooling2D()(x)
    out = Dense(10,activation='softmax', name="output")(x)

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])
    
    return model
model2 = makeModel((32,32,3,))
model2.summary()
hist2 = model2.fit(X_train, y_train, batch_size=100,epochs=10, validation_data=(X_validation,y_validation))
def learningCurves(hist):
    histAcc_train = hist.history['acc']
    histLoss_train = hist.history['loss']
    histAcc_validation = hist.history['val_acc']
    histLoss_validation = hist.history['val_loss']
    maxValAcc = np.max(histAcc_validation)
    minValLoss = np.min(histLoss_validation)

    plt.figure(figsize=(12,12))
    epochs = len(histAcc_train)

    plt.plot(range(epochs),histLoss_train, label="Training Loss", color="#acc6ef")
    plt.plot(range(epochs),histLoss_validation, label="Validation Loss", color="#a7e295")

    plt.scatter(np.argmin(histLoss_validation),minValLoss,zorder=10,color="green")

    plt.xlabel('Epochs',fontsize=14)
    plt.title("Learning Curves",fontsize=20)

    plt.legend()
    plt.show()

    print("Max validation accuracy: {0}".format(maxValAcc))
    print("Minimum validation loss: {0}".format(minValLoss))
    
learningCurves(hist2)
pred = model2.predict(test)
pred = np.argmax(pred,1)
pred = pred + 1
submission = pd.DataFrame.from_items([
    ('id',list(range(pred.shape[0]))),
    ('label', pred)])

submission.to_csv('submission.csv', index = False)