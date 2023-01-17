import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from ipywidgets import interact
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input/digit-recognizer"))
mnistTrainingData = pd.read_csv("../input/digit-recognizer/train.csv")
mnistTrainingData.head()
X = mnistTrainingData.values[:,1:]
y = mnistTrainingData.values[:,0]
def disp(imSelIdx=0):
    plt.title(y[imSelIdx])
    plt.imshow(X[imSelIdx].reshape(28,28), cmap="gray")

interact(disp,imSelIdx=(0,X.shape[0]))
def oneHotEncoder(integerVal,maxClasses):
    out = np.zeros(maxClasses)
    out[integerVal] = 1
    return out

y_onehot = []

for i in y:
    y_onehot.append(oneHotEncoder(i,y.max()+1))

y_onehot = np.stack(y_onehot)

print(y_onehot)
print("Shape of y vector: {0}".format(y.shape))
print("Shape of y one-hot matrix: {0}".format(y_onehot.shape))
X_train, X_validation, y_train , y_validation = train_test_split(X,y_onehot, test_size=0.2)
inputs = Input(shape=(X.shape[1],))

x = Dense(100, activation='sigmoid')(inputs)
x = Dense(y_onehot.shape[1], activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
hist = model.fit(X_train,y_train,epochs=50, batch_size=100, validation_data=(X_validation,y_validation)) 
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

learningCurves(hist)
inputs = Input(shape=(X.shape[1],))

x = Dense(100, activation='sigmoid')(inputs)
x = Dropout(0.5)(x)
x = Dense(y_onehot.shape[1], activation='softmax')(x)

model3 = Model(inputs=inputs, outputs=x)
model3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model3.summary()
hist3 = model3.fit(X_train,y_train,epochs=50, batch_size=100, validation_data=(X_validation,y_validation)) 
learningCurves(hist3)