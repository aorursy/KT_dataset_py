import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
(train_x , train_y),(test_x , test_y) = mnist.load_data()
train_x.shape
batch_size = 128
num_class = 10
epochs = 20
train_x = train_x.reshape(60000 , 784)
test_x = test_x.reshape(10000 , 784)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_y = keras.utils.to_categorical(train_y , num_class)
test_y = keras.utils.to_categorical(test_y , num_class)
model = Sequential()
model.add(Dense(512 , activation = 'relu' , input_shape = (784,)))
model.add(Dropout(0.2))
model.add(Dense(512 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(num_class , activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy' , 
             optimizer = RMSprop(),
             metrics = ['accuracy'])
history = model.fit(train_x , train_y , 
                   batch_size = batch_size , 
                   epochs = epochs , 
                   validation_data = (test_x , test_y))
score = model.evaluate(test_x , test_y)
print(score)
plt.plot(history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()