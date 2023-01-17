import pandas as pd

import numpy as np

import keras

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Activation

from keras import optimizers

from keras import regularizers



import matplotlib.pyplot as plt

size = 10000

index = pd.Series(range(0,size))

#print(index)

matrix = np.zeros(shape=(size,2))

for i in index:

    rand_value = np.random.random_sample() *10-5 #intervall [0..1] gives a better result

    matrix[i]=[rand_value,rand_value*rand_value]

    

#print(matrix)             



df = pd.DataFrame(matrix,index=index, columns=['x','x^2'])
df.head()
# shuffel the dataframe and reset index

df = df.sample(frac=1).reset_index(drop=True)
df.head()
training_data, test_data = train_test_split(df, test_size=0.2)

print(len(training_data),len(test_data))
Xtraining_data = training_data['x']# inputdata

#print(Xtraining_data)

Ytraining_data = training_data['x^2']# output data

#print(Ytraining_data)

Xtest_data = test_data['x']

#print(Xtraining_data)

Ytest_data = test_data['x^2']

#print(Ytraining_data)

NN = Sequential()

NN.add(Dense(32, input_dim=1,activation='linear', kernel_regularizer=regularizers.l2(0.001) ))

NN.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001) ))

NN.add(Dense(1))  
#sgd = optimizers.SGD(lr=0.01, decay=1e-6)
#NN.compile(optimizer='adam', loss='mse')

# rmsprop: unpublished optimization algorithm designed for neural networks, first proposed by Geoff Hinton

NN.compile(optimizer='rmsprop', loss='mse') # best result with rmsprop 
history=NN.fit(Xtraining_data,Ytraining_data,epochs=50,batch_size=256, validation_split=0.2)
#history.history
#visualisation off loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper right')

plt.show()
print(NN.predict([0,1,2,3,4,5,6,7]))