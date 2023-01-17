from tensorflow import keras
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
data=keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=data.load_data()
x_train=x_train/255
x_test=x_test/255
x_train,y_train=x_train[5000:],y_train[5000:]
x_valid,y_valid=x_train[:5000],y_train[:5000]
#Note: I have commented those lines that are used for regularizing the model. Don't touch it unless you want to play around with it!
model=keras.models.Sequential()
model.add(keras.layers.Input(shape=[28,28]))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300,activation='relu'))
#model.add(keras.layers.Dense(300,activation='relu'),kernel_regularizer=regularizers.l2(0.02))
model.add(keras.layers.Dense(200,activation='relu'))
#model.add(keras.layers.Dense(300,activation='relu'),kernel_regularizer=regularizers.l2(0.02))
model.add(keras.layers.Dense(10,activation='softmax'))
model.summary()
keras.utils.plot_model(model,show_shapes=True)
model.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')
history=model.fit(x_train,y_train,batch_size=32,epochs=20,validation_data=(x_valid,y_valid))
plt.plot(history.history['loss'],'r--')
plt.plot(history.history['val_loss'],'b--')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('losses')
plt.legend(['training','validating'])
plt.title('Loss analysis')
plt.plot(history.history['accuracy'],'r--')
plt.plot(history.history['val_accuracy'],'b--')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('accuracies')
plt.legend(['training','validating'])
plt.title('Accuracy analysis')
mse=model.evaluate(x_test,y_test)

x_test[:10]
y_test[:10]
y_pred=model.predict(x_test[:10])
y_pred
y_pred=np.round(y_pred)
print(y_pred)
#here y_pred gave me 10 values. So the value which is the highest one is the corresponding number's location.