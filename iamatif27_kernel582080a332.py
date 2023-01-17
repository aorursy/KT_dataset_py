#Cleaned FER
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred)
    conf_Mat=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat

#K.set_image_dim_ordering('th')
#import tensorflow as tf
#import numpy as np
X=np.genfromtxt('/kaggle/input/datanfer/CleanedFERNFinal/FERFinalN.csv', delimiter=' ', dtype='float32')
#np.random.shuffle(X)
X_test=X[X.shape[0]-2000:, :]
X=X[:X.shape[0]-2000, :]
#X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ', dtype='float32')
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=5)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=5)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
model = Sequential()
p=0.2
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(Dropout(p))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(p))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(p))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(p))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(p))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(128, activation='relu'))
model.add(Dropout(p))
model.add(Dense(64, activation='relu'))
model.add(Dropout(p))
model.add(Dense(5 , activation='softmax'))
print(model.summary())
#adam=tf.keras.optimizers.Adam(learning_rate=0.1)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=15,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=1,
    min_lr=0.00001
)
opt = SGD(lr=0.001, momentum=0.9, decay=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
try:
    model.fit(X, Y, validation_data=(X_test, y_test), epochs=1000, batch_size=500, shuffle=True)
    conf=ConfusionMatrix(X_test, y_test, model)
    print(conf)
    print(np.sum(conf))
    print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 5, axis=1)) 
    model.save('/kaggle/working/model1.h5')
except:
    conf=ConfusionMatrix(X_test, y_test, model)
    print(conf)
    print(np.sum(conf))
    print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 5, axis=1)) 
    model.save('/kaggle/working/model1.h5')
#For FER
from keras.models import Sequential
from imblearn.over_sampling import RandomOverSampler 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import numpy as np
ros = RandomOverSampler(random_state=42)
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred.shape)
    conf_Mat=np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat
#K.set_image_dim_ordering('th')
#import tensorflow as tf
#import numpy as np
X=np.concatenate((np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data34/data3N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data34/data4N.csv', delimiter=' ', dtype='float32')), axis=0)
#Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
Y=X[:, X.shape[1]-1]
#X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
X=X[:, :X.shape[1]-1]
X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ', dtype='float32')
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
hist=np.zeros(7)
#Yor=np.argmax(Y, axis=1)
for i in range(7):
    #hist[i]=np.sum(Yor==i)
    hist[i]=np.sum(Y==i)
X=X.reshape(X.shape[0], 48, 48, 1)
Y=tf.keras.utils.to_categorical(Y)
model = Sequential()
p=0.2
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(Dropout(p))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(p))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(p))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(p))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(p))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(p))
model.add(Dense(256, activation='relu'))
model.add(Dropout(p))
model.add(Dense(128, activation='relu'))
model.add(Dropout(p))
model.add(Dense(64, activation='relu'))
model.add(Dropout(p))
model.add(Dense(7 , activation='softmax'))
print(model.summary())
#adam=tf.keras.optimizers.Adam(learning_rate=0.1)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=15,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=1,
    min_lr=0.00001
)
opt = SGD(lr=0.001, momentum=0.9, decay=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
try:
    model.fit(X, Y, validation_data=(X_test, y_test), epochs=1000, batch_size=500, shuffle=True)
except:
    conf=ConfusionMatrix(X_test, y_test, model)
    conf=conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 5, axis=1)
    print(conf)
    print(np.sum(conf, axis=1))
    model.save('/kaggle/working/model1.h5')
    plt.bar(range(7), hist)
    plt.show()
#For CKPlus
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred.shape)
    conf_Mat=np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat
def toEmotions(x):
    if x==0:
        return 'Angry'
    elif x==1:
        return 'Disgust'
    elif x==2:
        return 'Fear'
    elif x==3:
        return 'Happy'
    elif x==4:
        return 'Sad'
    elif x==5:
        return 'Surprise'
    else:
        return 'Contempt'

#K.set_image_dim_ordering('th')
#import tensorflow as tf
#import numpy as np
X=np.genfromtxt('/kaggle/input/ckplusfinal/CKPlusFinalN.csv', delimiter=' ', dtype='float32')
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
#X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ', dtype='float32')
#y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
#X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
def genModel():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.3))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7 , activation='softmax'))
    #print(model.summary())
    return model
history=[]
es=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=40,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
for k in range(5):
    dbyf=X.shape[0]/5.0
    X_test=X[int(k*dbyf):int((k+1)*dbyf), :, :, :]
    Y_test=Y[int(k*dbyf):int((k+1)*dbyf), :]
    X_train=np.concatenate((X[0:int(k*dbyf), :, :, :], X[int((k+1)*dbyf):, :, :, :]), axis=0)
    Y_train=np.concatenate((Y[0:int(k*dbyf), :], Y[int((k+1)*dbyf):, :]), axis=0)
    model=genModel()
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    history.append(model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=220, batch_size=500, shuffle=True, callbacks=[es]))
    if k==0:
        conf=ConfusionMatrix(X_test, Y_test, model)
    else:
        conf=conf+ConfusionMatrix(X_test, Y_test, model)
    model.save('/kaggle/working/model1.h5')
print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 7, axis=1))
y=model.predict(X_test[:15, : ,:, :])
imgs=X_test[:15, :, : ,:].reshape(15, 48, 48)
for i in range(15):
    plt.title('Prediction: '+toEmotions(np.argmax(y[i]))+'; Actual: '+toEmotions(np.argmax(Y_test[i])))
    plt.imshow(imgs[i, :, :], cmap='gray')
    plt.show()
#print(history.history.keys())
#print(max(history[5].history['val_accuracy']))
#adam=tf.keras.optimizers.Adam(learning_rate=0.1)
#For CKPlus Validation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred.shape)
    conf_Mat=np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat
def toEmotions(x):
    if x==0:
        return 'Angry'
    elif x==1:
        return 'Disgust'
    elif x==2:
        return 'Fear'
    elif x==3:
        return 'Happy'
    elif x==4:
        return 'Sad'
    elif x==5:
        return 'Surprise'
    else:
        return 'Contempt'
X=np.genfromtxt('../input/ckplusfinal/CKPlusFinalN.csv', delimiter=' ', dtype='float32')
np.random.shuffle(X)
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
model=keras.models.load_model('../input/models/CKPlusModel.h5')
#model.evalaute(X, Y)
imgs=X[:15, :, :, :].reshape(15, 48, 48)
y=model.predict(X[:15, :, :, :])
for i in range(15):
    plt.title('Prediction: '+toEmotions(np.argmax(y[i]))+'; Actual: '+toEmotions(np.argmax(Y[i])))
    plt.imshow(imgs[i, :, :], cmap='gray')
    plt.show()
#model.evaluate(X_test, y_test)
conf=ConfusionMatrix(X, Y, model)
print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 7, axis=1))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import numpy as np
def toEmotions(x):
    if x==0:
        return 'Angry'
    elif x==1:
        return 'Disgust'
    elif x==2:
        return 'Fear'
    elif x==3:
        return 'Happy'
    else:
        return 'Neutral'
earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred.shape)
    conf_Mat=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat
X=np.genfromtxt('/kaggle/input/datanfer/CleanedFERNFinal/FERFinalN.csv', delimiter=' ', dtype='float32')
X_test=X[X.shape[0]-2000:, :]
X=X[:X.shape[0]-2000, :]
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=5)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=5)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
model=keras.models.load_model('../input/models/CleanedFER.h5')
imgs=X_test[:10, :, :, :].reshape(10, 48, 48)
y=model.predict(X_test[:10, :, :, :])
for i in range(10):
    plt.title('Prediction: '+toEmotions(np.argmax(y[i]))+'; Actual: '+toEmotions(np.argmax(y_test[i])))
    plt.imshow(imgs[i, :, :], cmap='gray')
    plt.show()
model.evaluate(X_test, y_test)
conf=ConfusionMatrix(X_test, y_test, model)
print(conf/np.repeat(np.sum(conf, axis=1)[:, np.newaxis], 5, axis=1))
