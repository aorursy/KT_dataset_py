import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras import layers
import matplotlib.pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

np.set_printoptions(linewidth=200)
(xTrain,yTrain),(xTest,yTest) = tf.keras.datasets.mnist.load_data()

xTrainNorm = xTrain/255.0
xTestNorm = xTest/255.0
def plot_curve(epochs,hist,metricList):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    
    for i in metricList:
        x = hist[i]
        plt.plot(epochs[1:],x[1:],label=i)
    plt.legend()
def create_model(learningRate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(units=128,activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=256,activation='relu'))
    model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learningRate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
    return model

def train(model,features,labels,epochs,batchSize=None,validation=0.1):
    history = model.fit(x=features,y=labels,batch_size=batchSize,epochs=epochs,
                       shuffle=True,validation_split=validation)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    
    return epochs,hist
lr=0.01
epochs=100
batch = 2000
valid = 0.2

model = create_model(lr)
epochs , hist = train(model,xTrainNorm,yTrain,epochs,batch,valid)

metricList=['accuracy']
plot_curve(epochs,hist,metricList)

model.evaluate(x=xTest,y=yTest,batch_size=batch)