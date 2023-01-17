#loading the necessary libaries
from keras.datasets import mnist
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#spliting the data in training and testing
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#changing data types from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#using keras inbuild categorical function
Y_train = keras.utils.to_categorical(y_train, num_classes=10)
Y_test = keras.utils.to_categorical(y_test, num_classes=10)
#Reshaping the images into 1D/keras.layers.Flatten() can also be used alternatively
X_train_new = X_train.reshape(60000, 28*28)
X_test_new = X_test.reshape(10000, 28*28)
#normalizing the data (z=(x-µ)/σ), where µ=mean and σ=standard deviation
X_train_normalize = (X_train_new-np.mean(X_train_new))/np.std(X_train_new)
X_test_normalize = (X_test_new-np.mean(X_test_new))/np.std(X_test_new)
#Building the keras training model
model = keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10,activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Finally fitting and training the model with 30 epochs and batch_size-128
model.fit(X_train_normalize,Y_train, epochs=30, batch_size=128, validation_data=(X_test_normalize, Y_test))
#Evaluating the model
model.evaluate(X_test_normalize,Y_test)
y_predct = model.predict_classes(X_test_normalize)
#Taking a random image
print('Actual number is {}'.format(np.argmax(Y_test[6669])))
print('Predicted number is {}'.format(y_predct[6669]))
plt.imshow(X_test[6669])
plt.show()
#Taking a random image
print('Actual number is {}'.format(np.argmax(Y_test[5483])))
print('Predicted number is {}'.format(y_predct[5483]))
plt.imshow(X_test[5483])
plt.show()
random_num = np.random.randint(10000)
print('random checking for {}'.format(random_num))
print('***********************************')
print('Actual number is {}'.format(np.argmax(Y_test[random_num])))
print('Predicted number is {}'.format(y_predct[random_num]))
plt.imshow(X_test[random_num])
plt.show()

