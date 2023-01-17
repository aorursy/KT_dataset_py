#Load needed libraries
import numpy as np
import matplotlib.pyplot as plt
#load data 
X = np.load('../input/Sign-language-digits-dataset/X.npy') #feature dataset
Y = np.load('../input/Sign-language-digits-dataset/Y.npy') #target dataset
f, ax = plt.subplots(2, 2, figsize=(15, 10))
sample = [290, 1000, 1800, 650]
for i in range(0, 4):
    ax[i//2, i%2].imshow(X[sample[i]].reshape(64, 64))
    ax[i//2, i%2].axis('off')
plt.show()
#shape of feature
X.shape
#reshape the feature
X = X.reshape(X.shape[0], 64, 64, 1)
X.shape
#split data for train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
model_input = Input(shape=(64, 64,1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)
x = MaxPooling2D(pool_size=(2, 2))(x)    
x = Dropout(0.25)(x)
    
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)    
x = Dropout(0.25)(x)
    
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)       
x = MaxPooling2D(pool_size=(2, 2))(x)    
x = Dropout(0.25)(x)
    
x = Flatten()(x)
x = Dense(512, activation='relu')(x)    
x = Dropout(0.25)(x)
    
y1 = Dense(10, activation='softmax')(x)
    
model = Model(inputs=model_input, outputs= y1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#Fiting 
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)
#About accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'])
plt.show()
#About loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training', 'validation'])
plt.show()
#score
score = model.evaluate(X_test, y_test)
print("Accuracy : ", score[1])