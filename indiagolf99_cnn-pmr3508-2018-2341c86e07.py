import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
K.set_image_dim_ordering('th')
# fixando a random seed:
seed = 0
np.random.seed(seed)
train_data_pure = np.load('../input/train_images_pure.npy')

plt.subplot(241)
plt.imshow(train_data_pure[0], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(train_data_pure[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(train_data_pure[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(train_data_pure[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(train_data_pure[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(train_data_pure[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(train_data_pure[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(train_data_pure[7], cmap=plt.get_cmap('gray'))
plt.show()
train_data_noisy = np.load('../input/train_images_noisy.npy')

plt.subplot(241)
plt.imshow(train_data_noisy[0], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(train_data_noisy[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(train_data_noisy[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(train_data_noisy[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(train_data_noisy[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(train_data_noisy[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(train_data_noisy[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(train_data_noisy[7], cmap=plt.get_cmap('gray'))
plt.show()
train_data_rotated = np.load('../input/train_images_rotated.npy')

plt.subplot(241)
plt.imshow(train_data_rotated[0], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(train_data_rotated[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(train_data_rotated[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(train_data_rotated[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(train_data_rotated[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(train_data_rotated[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(train_data_rotated[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(train_data_rotated[7], cmap=plt.get_cmap('gray'))
plt.show()
train_data_both = np.load('../input/train_images_both.npy')

plt.subplot(241)
plt.imshow(train_data_both[0], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(train_data_both[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(train_data_both[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(train_data_both[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(train_data_both[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(train_data_both[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(train_data_both[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(train_data_both[7], cmap=plt.get_cmap('gray'))
plt.show()
test_data = np.load('../input/Test_images.npy')

plt.subplot(241)
plt.imshow(test_data[0], cmap=plt.get_cmap('gray'))
plt.subplot(242)
plt.imshow(test_data[1], cmap=plt.get_cmap('gray'))
plt.subplot(243)
plt.imshow(test_data[2], cmap=plt.get_cmap('gray'))
plt.subplot(244)
plt.imshow(test_data[3], cmap=plt.get_cmap('gray'))
plt.subplot(245)
plt.imshow(test_data[4], cmap=plt.get_cmap('gray'))
plt.subplot(246)
plt.imshow(test_data[5], cmap=plt.get_cmap('gray'))
plt.subplot(247)
plt.imshow(test_data[6], cmap=plt.get_cmap('gray'))
plt.subplot(248)
plt.imshow(test_data[7], cmap=plt.get_cmap('gray'))
plt.show()
train_labels = pd.read_csv('../input/train_labels.csv', index_col=0)
X_fit1 = train_data_pure.reshape(train_data_pure.shape[0], 1, 28, 28).astype('float32')
X_fit2 = train_data_both.reshape(train_data_both.shape[0], 1, 28, 28).astype('float32')
X_fit3 = train_data_rotated.reshape(train_data_rotated.shape[0], 1, 28, 28).astype('float32')
X_Test = test_data.reshape(test_data.shape[0], 1, 28, 28).astype('float32')
# normalizando os inputs 0-255 to 0-1
X_fit1 = X_fit1 / 255
X_fit2 = X_fit2 / 255
X_fit3 = X_fit3 / 255
X_Test = X_Test / 255
# one hot encode outputs
y_fit = np_utils.to_categorical(train_labels)
num_classes = y_fit.shape[1]
# separando as bases de treino e validação: 
X_train1,X_validation1,y_train1,y_validation1 = train_test_split(X_fit1,y_fit, test_size = 0.2)
X_train2,X_validation2,y_train2,y_validation2 = train_test_split(X_fit2,y_fit, test_size = 0.2)
X_train3,X_validation3,y_train3,y_validation3 = train_test_split(X_fit3,y_fit, test_size = 0.2)
# utilizando a mesma estrutura do tutorial para baseline
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model_pure = baseline_model()
model_pure.summary()
#fitando e determinando a baseline para dados limpos
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fit the model
model_pure.fit(X_train1,y_train1, validation_data=(X_validation1,y_validation1), epochs=20, 
          batch_size=200, verbose=2, callbacks = callbacks)
#Acurácia do modelo
print('Acurácia da CNN treinada com dados puros:')
scores = model_pure.evaluate(X_fit1, y_fit, verbose=0)
print("CNN Accuracy (Pure data): %.2f%%" % (scores[1]*100))
scores = model_pure.evaluate(X_fit3, y_fit, verbose=0)
print("CNN Accuracy (Rotated data): %.2f%%" % (scores[1]*100))
scores = model_pure.evaluate(X_fit2, y_fit, verbose=0)
print("CNN Accuracy (Currupted data): %.2f%%" % (scores[1]*100))
model_rotated = baseline_model()
model_rotated.summary()
#fitando e determinando a baseline para dados limpos
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fit the model
model_rotated.fit(X_train3,y_train3, validation_data=(X_validation3,y_validation3), epochs=20, 
          batch_size=200, verbose=2, callbacks = callbacks)
#Acurácia do modelo
print('Acurácia da CNN treinada com dados rotacionados:')
scores = model_rotated.evaluate(X_fit1, y_fit, verbose=0)
print("CNN Accuracy (Pure data): %.2f%%" % (scores[1]*100))
scores = model_rotated.evaluate(X_fit3, y_fit, verbose=0)
print("CNN Accuracy (Rotated data): %.2f%%" % (scores[1]*100))
scores = model_rotated.evaluate(X_fit2, y_fit, verbose=0)
print("CNN Accuracy (Currupted data): %.2f%%" % (scores[1]*100))
model_both = baseline_model()
model_both.summary()
#fitando e determinando a baseline para dados sujos
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
# Fit the model
model_both.fit(X_train2,y_train2, validation_data=(X_validation2,y_validation2), epochs=20, 
          batch_size=200, verbose=2, callbacks = callbacks)
#Acurácia do modelo
print('Acurácia da CNN treinada com dados sujos:')
scores = model_both.evaluate(X_fit1, y_fit, verbose=0)
print("CNN Accuracy (Pure data): %.2f%%" % (scores[1]*100))
scores = model_both.evaluate(X_fit3, y_fit, verbose=0)
print("CNN Accuracy (Rotated data): %.2f%%" % (scores[1]*100))
scores = model_both.evaluate(X_fit2, y_fit, verbose=0)
print("CNN Accuracy (Currupted data): %.2f%%" % (scores[1]*100))
def encode(p):
    resp=[]
    for linha in p:
        resp.append(np.argmax(linha))
    return resp
Pred1 = model_pure.predict(X_Test)

Pred2 = model_rotated.predict(X_Test)

Pred = model_both.predict(X_Test)


P1 = pd.DataFrame(columns = ['Id','label'])
P1.label = encode(Pred1)
P1.Id = range(len(X_Test))
P1.to_csv("Pred1.csv",index=False)

P2 = pd.DataFrame(columns = ['Id','label'])
P2.label = encode(Pred2)
P2.Id = range(len(X_Test))
P2.to_csv("Pred2.csv",index=False)

P = pd.DataFrame(columns = ['Id','label'])
P.label = encode(Pred)
P.Id = range(len(X_Test))
P.to_csv("Pred.csv",index=False)