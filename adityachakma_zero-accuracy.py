import numpy as np

from tensorflow import keras

import pickle

import matplotlib.pyplot as plt
f = open('../input/zero-accuracy-character-recognition-dataset/TrainX.m','rb')

DataX = pickle.load(f)

f.close()



f = open('../input/zero-accuracy-character-recognition-dataset/TrainY.m','rb')

DataY = pickle.load(f)

f.close()
print(np.unique(DataY))

print(len(np.unique(DataY)))
plt.hist(DataY,bins=29)

plt.show()
plt.imshow(DataX[100])

plt.show()

print(DataY[100])
plt.imshow(DataX[1100])

plt.show()

print(DataY[1100])
mnv2 = keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=True,classes=29,weights=None)
opt = keras.optimizers.Adam(learning_rate=0.001)

mnv2.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
mnv2.fit(DataX,DataY,epochs=2,validation_split=0.15)
mnv2.evaluate(DataX,DataY)
mnv2 = keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights=None)
X = keras.layers.Flatten()(mnv2.output)

X = keras.layers.Dense(1024,activation='relu')(X)

X = keras.layers.BatchNormalization()(X)

X = keras.layers.Dense(1024,activation='relu')(X)

X = keras.layers.BatchNormalization()(X)

X = keras.layers.Dense(30, activation = 'softmax')(X)



model = keras.models.Model(inputs=mnv2.input,outputs=X)



opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.train(DataX,DataY)
# model.evaluate(DataX,DataY)
model.load_weights('../input/zero-accuracy-character-recognition-dataset/weights.h5')
print('Accuracy and loss with pre-trained weight')

model.evaluate(DataX,DataY)