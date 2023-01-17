import keras
from keras.layers import Dense,Conv2D,MaxPool2D,Activation, Dropout, Flatten, Dense,BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
import pandas as pd
train = pd.read_csv("../input/train.csv")
Y_train=train["label"]
X_train=train.drop(labels = ["label"],axis = 1)
X_train /= 255
Y_train = keras.utils.to_categorical(Y_train, 10)
print(X_train.shape)
print(Y_train.shape)
X_train = X_train.values.reshape(42000, 28,28, 1)
print(X_train.shape)

model = Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters = 32, kernel_size = (5,5),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',
optimizer=RMSprop(0.001),
metrics=['accuracy'])  
model.fit(X_train,Y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.2)
test=pd.read_csv('../input/test.csv')

test=test.values.reshape(28000,28,28,1)
predict=model.predict(test)
print(predict)
accuracy = model.evaluate(X_train,Y_train,batch_size=32)
print("Accuracy: ",accuracy[1])



predict = model.predict_classes(test, verbose=1)
data_predictions = pd.DataFrame({"ImageId": list(range(1,len(predict)+1)),"Label": predict})
data_predictions.to_csv('predictions.csv', index=False, header=True)
print(data_predictions)
