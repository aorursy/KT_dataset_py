import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras as ke
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
train_l    = pd.read_csv("../input/train_labels.csv", header=0, index_col=0, na_values="?")
train_p    = np.load("../input/train_images_pure.npy")
train_r    = np.load("../input/train_images_rotated.npy")
train_n    = np.load("../input/train_images_noisy.npy")
train_b    = np.load("../input/train_images_both.npy")
test       = np.load("../input/Test_images.npy")
train_l['label'].value_counts()
train_l['label'][:1800].value_counts()
plt.subplot(141)
plt.imshow(train_p[1000], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(train_p[1001], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(train_p[1004], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(train_p[1005], cmap=plt.get_cmap('gray'))
plt.show()
plt.subplot(141)
plt.imshow(train_r[1000], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(train_r[1001], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(train_r[1004], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(train_r[1005], cmap=plt.get_cmap('gray'))
plt.show()
plt.subplot(141)
plt.imshow(train_n[1000], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(train_n[1001], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(train_n[1004], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(train_n[1005], cmap=plt.get_cmap('gray'))
plt.show()
plt.subplot(141)
plt.imshow(train_b[1000], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(train_b[1001], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(train_b[1004], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(train_b[1005], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
np.random.seed(720)
num_pixels = train_b.shape[1] * train_b.shape[2]
train_p    = train_p.reshape(train_b.shape[0], num_pixels).astype('float32')
train_r    = train_r.reshape(train_r.shape[0], num_pixels).astype('float32')
train_n    = train_n.reshape(train_n.shape[0], num_pixels).astype('float32')
train_b    = train_b.reshape(train_b.shape[0], num_pixels).astype('float32')
test       = test.reshape(test.shape[0],       num_pixels).astype('float32')
train_p = train_p/255
train_r = train_r/255
train_n = train_n/255
train_b = train_b/255
test    = test/255
train_l = np_utils.to_categorical(train_l)
num_classes = train_l.shape[1]
def baseline_model():
    # criamos o modelo
    # começamos por instanciar o esqueleto do modelo, neste caso, o sequencial
    model = Sequential()
    # em seguida, adicionamos as camadas uma a uma. Adicionamos uma camada de entrada com ativação relu
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # e a camada de saída, com ativação softmax para transformar as saídas numéricas em probabilidades
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # compilamos o modelo escolhendo a função objetiva, o otimizador (cuja escolha é empírica) e a métrica 
    #de performance mais conveniente.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# construimos o modelo com nossa classe
from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=baseline_model, verbose=0)
#from sklearn.model_selection import GridSearchCV
#from keras.callbacks import EarlyStopping
#callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
#epochs = [100]
#batches = [50, 100, 200, 300, 350]
#param_grid = dict(epochs=epochs, batch_size=batches)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
#grid_result = grid.fit(train_r[:40000], train_l[:40000],
#                      validation_data=(train_r[40000:50000], train_l[40000:50000]),
#                      verbose=0, callbacks=callbacks)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
from keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
model_keras = baseline_model()
model_keras.fit(train_r[:40000, :], train_l[:40000, :],
                validation_data=(train_r[40000:50000], train_l[40000:50000]),
                epochs=30, batch_size=3000, verbose=1, callbacks = callbacks)
scores = model_keras.evaluate(train_r[50000:], train_l[50000:], verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
scores = model_keras.evaluate(train_b[50000:], train_l[50000:], verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
y_pred = model_keras.predict_classes(test)
resultado = pd.DataFrame(columns = ['Id','label'])
resultado.label = y_pred
resultado.Id = range(len(test))
resultado.to_csv("pred_basico.csv",index=False)
train_l    = pd.read_csv("../input/train_labels.csv", header=0, index_col=0, na_values="?")
train_p    = np.load("../input/train_images_pure.npy")
train_r    = np.load("../input/train_images_rotated.npy")
train_n    = np.load("../input/train_images_noisy.npy")
train_b    = np.load("../input/train_images_both.npy")
test       = np.load("../input/Test_images.npy")
train_p    = train_p.reshape(train_b.shape[0], 1, 28, 28).astype('float32')
train_r    = train_r.reshape(train_r.shape[0], 1, 28, 28).astype('float32')
train_n    = train_n.reshape(train_n.shape[0], 1, 28, 28).astype('float32')
train_b    = train_b.reshape(train_b.shape[0], 1, 28, 28).astype('float32')
test       = test.reshape(test.shape[0],       1, 28, 28).astype('float32')
train_p    = train_p/255
train_r    = train_r/255
train_n    = train_n/255
train_b    = train_b/255
test       = test/255
train_l    = np_utils.to_categorical(train_l)
num_classes = train_l.shape[1]

def baseline_model_nop():
    model = Sequential()
    model.add(Conv2D(100, (10, 10), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(50, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(30, (5, 5), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = baseline_model_nop()
model.summary()
from keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
model.fit(train_r[:40000], train_l[:40000],
         validation_data=(train_r[40000:50000], train_l[40000:50000]),
         epochs=10, batch_size=3500, verbose=1, callbacks=callbacks)
scores = model.evaluate(train_r[50000:], train_l[50000:], verbose=0)
print("Large CNN Error: %.2f%%"% (100-scores[1]*100))
scores = model.evaluate(train_b[50000:], train_l[50000:], verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
y_pred = model.predict_classes(test)
resultado = pd.DataFrame(columns = ['Id','label'])
resultado.label = y_pred
resultado.Id = range(len(test))
resultado.to_csv("pred_camadas.csv",index=False)
train_l    = pd.read_csv("../input/train_labels.csv", header=0, index_col=0, na_values="?")
train_p    = np.load("../input/train_images_pure.npy")
train_r    = np.load("../input/train_images_rotated.npy")
train_n    = np.load("../input/train_images_noisy.npy")
train_b    = np.load("../input/train_images_both.npy")
test       = np.load("../input/Test_images.npy")
train_p    = train_p.reshape(train_b.shape[0], 1, 28, 28).astype('float32')
train_r    = train_r.reshape(train_r.shape[0], 1, 28, 28).astype('float32')
train_n    = train_n.reshape(train_n.shape[0], 1, 28, 28).astype('float32')
train_b    = train_b.reshape(train_b.shape[0], 1, 28, 28).astype('float32')
test       = test.reshape(test.shape[0],       1, 28, 28).astype('float32')
train_p    = train_p/255
train_r    = train_r/255
train_n    = train_n/255
train_b    = train_b/255
test       = test/255
train_l    = np_utils.to_categorical(train_l)
num_classes = train_l.shape[1]
def baseline_model_pooling():
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = baseline_model_pooling()
model.summary()
from keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
model.fit(train_r[:40000], train_l[:40000],
         validation_data=(train_r[40000:50000], train_l[40000:50000]),
         epochs=10, batch_size=3500, verbose=1, callbacks=callbacks)
scores = model.evaluate(train_r[50000:], train_l[50000:], verbose=0)
print("Large CNN Error: %.2f%%"% (100-scores[1]*100))
scores = model.evaluate(train_b[50000:], train_l[50000:], verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
y_pred = model.predict_classes(test)
resultado = pd.DataFrame(columns = ['Id','label'])
resultado.label = y_pred
resultado.Id = range(len(test))
resultado.to_csv("pred_camadas_pooling.csv",index=False)
import cv2
def noiseless(img):
    no_noise = []
    for i in range(len(img)):
        blur = cv2.GaussianBlur(img[i], (3, 3), 0)
        no_noise.append(blur)
    image = no_noise[1]
    return image
train_b = noiseless(train_b)
test = noiseless(test)
from keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
model.fit(train_b[:40000], train_l[:40000],
         validation_data=(train_b[40000:50000], train_l[40000:50000]),
         epochs=10, batch_size=3500, verbose=1, callbacks=callbacks)
scores = model.evaluate(train_b[50000:], train_l[50000:], verbose=0)
print("Large CNN Error: %.2f%%"% (100-scores[1]*100))
y_pred = model.predict_classes(test)
resultado = pd.DataFrame(columns = ['Id','label'])
resultado.label = y_pred
resultado.Id = range(len(test))
resultado.to_csv("pred_camadas_pooling_noiseless.csv",index=False)