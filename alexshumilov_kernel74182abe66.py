from google.colab import files

uploaded = files.upload()
import pandas as pd
import tensorflow as tf
trainData = pd.read_csv('fashion-mnist_train.csv')
testData = pd.read_csv('fashion-mnist_test.csv')

testLabel = testData['label'].to_numpy()
testImg = testData.iloc[:,1:].to_numpy()
testLabel = tf.keras.utils.to_categorical(testLabel)

trainLabel = trainData['label'].to_numpy()
trainImg = trainData.iloc[:,1:].to_numpy()
trainLabel = tf.keras.utils.to_categorical(trainLabel)

testImg /= 250
trainImg = trainImg/250

from sklearn.linear_model import LogisticRegression
modelLogReg = LogisticRegression()
modelLogReg.fit(trainImg[:100],trainData['label'].to_numpy()[:100])
modelLogReg.score(testImg,testData['label'].to_numpy())


model = tf.keras.models.Sequential(name='NorsOlFasol')
model.add(tf.keras.layers.Dense(512,activation='relu',input_shape =(784,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adadelta(),
    metrics = ['accuracy']
)
model.fit(trainImg,trainLabel,epochs=10)
loss, accuracy  = model.evaluate(testImg,testLabel)
  
xtrain = trainImg.reshape(60000,28,28,1)
xtest = testImg.reshape(10000,28,28,1)

model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Convolution2D( filters=32, kernel_size= (3,3), activation='relu',input_shape =(28,28,1,) ))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.Convolution2D( filters=64, kernel_size= (3,3), activation='tanh'))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(64,activation='relu'))
model2.add(tf.keras.layers.Dense(10,activation='softmax'))


model2.compile(
    loss = 'mean_squared_error',
    optimizer = tf.keras.optimizers.Adadelta(),
    metrics = ['accuracy'])
model2.fit(xtrain,trainLabel,epochs=10)
loss, accuracy  = model2.evaluate(xtest,testLabel)

model3 = tf.keras.models.Sequential()
model3.add(tf.keras.layers.Convolution2D( filters=32, kernel_size= (3,3), activation='relu',input_shape =(28,28,1,) ))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.Convolution2D( filters=64, kernel_size= (3,3), activation='tanh'))
model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model3.add(tf.keras.layers.Dense(128,activation='selu'))
model3.add(tf.keras.layers.Dense(128,activation='relu'))
model3.add(tf.keras.layers.Flatten())
model3.add(tf.keras.layers.Dense(64,activation='relu'))
model3.add(tf.keras.layers.Dense(10,activation='softmax'))


model3.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adadelta(),
    metrics = ['accuracy'])
model3.fit(xtrain,trainLabel,epochs=10)
loss, accuracy  = model3.evaluate(xtest,testLabel)
model4 = tf.keras.models.Sequential()
model4.add(tf.keras.layers.Convolution2D( filters=32, kernel_size= (3,3), activation='relu',input_shape =(28,28,1,) ))
model4.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model4.add(tf.keras.layers.Convolution2D( filters=64, kernel_size= (3,3), activation='tanh'))
model4.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model4.add(tf.keras.layers.Dense(128,activation='selu'))
model4.add(tf.keras.layers.BatchNormalization())
model4.add(tf.keras.layers.Dense(128,activation='relu'))
model4.add(tf.keras.layers.BatchNormalization())
model4.add(tf.keras.layers.Flatten())
model4.add(tf.keras.layers.Dense(64,activation='relu'))
model4.add(tf.keras.layers.Dense(10,activation='softmax'))


model4.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adadelta(),
    metrics = ['accuracy'])
model4.fit(xtrain,trainLabel,epochs=10)
loss, accuracy  = model4.evaluate(xtest,testLabel)
SimpleModelAcc = [0.5191, 0.5933, 0.6355, 0.6595, 0.6794,0.6931,0.7053, 0.7125,0.7188, 0.7251, 0.7914]
SimpleConv = [ 0.2596,0.4656,0.5494,0.6348,0.6918,0.7183,0.7333,0.7429,0.7510,0.7578,0.7660]
ConvModified = [ 0.4164,0.6556,0.7054,0.7280,0.7425,0.7539,0.7629,0.7703,0.7764,0.7822,0.7878]
ConvBatch = [0.4127,0.6491,0.7248,0.7545,0.7717,0.7863,0.7950,0.8030,0.8086,0.8145,0.8154]
import matplotlib.pyplot as plt
#plt.legend(["Полносвязная модель","Свёрточная нейронная сеть","Свёрточная нейронная сеть с новыми слоями", "Batch_Normalization"])
pl = plt.plot(SimpleModelAcc)
plt.plot(SimpleConv)
plt.plot(ConvModified)
plt.plot(ConvBatch)
plt.legend(["Полносвязная модель","Свёрточная нейронная сеть","Свёрточная нейронная сеть с новыми слоями", "Batch_Normalization"])
