#Prepared training data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
data=pd.read_csv('/kaggle/input/fergit/FERGIT.csv')
print(data)
imgs=data['pixels'].to_numpy()
usage=data['Usage'].to_numpy()
Y=np.array(data['emotion'].to_numpy(), dtype='int32')
mask=usage=='Training'
#print(np.sum(mask))
imgs=imgs[mask]
print(imgs.shape)
Y=Y[mask]
img2=str(imgs[0])
x=img2.split(' ')
x=np.array(x, dtype='int32')
x=x.reshape(48, 48)
X=x[np.newaxis, :, :]
print(X.shape)
count=0
import glob
files=glob.glob('/kaggle/input/ckplus/CK+48/anger/*.png')
Y=np.concatenate((Y, np.repeat(0, len(files))))
for img in imgs[1:]:
    img2=str(img)
    x=img2.split(' ')
    x=np.array(x, dtype='int32')
    x=x.reshape(48, 48)
    X=np.concatenate((X, x[np.newaxis, :, :]), axis=0)
    print(count)
    count=count+1
for name in files:
    img=plt.imread(name)
    X=np.concatenate((X, img[np.newaxis, :, :]), axis=0)
files=glob.glob('/kaggle/input/ckplus/CK+48/disgust/*.png')
Y=np.concatenate((Y, np.repeat(1, len(files))))
for name in files:
    img=plt.imread(name)
    X=np.concatenate((X, img[np.newaxis, :, :]), axis=0)
files=glob.glob('/kaggle/input/ckplus/CK+48/fear/*.png')
Y=np.concatenate((Y, np.repeat(2, len(files))))
for name in files:
    img=plt.imread(name)
    X=np.concatenate((X, img[np.newaxis, :, :]), axis=0)
files=glob.glob('/kaggle/input/ckplus/CK+48/sadness/*.png')
Y=np.concatenate((Y, np.repeat(4, len(files))))
for name in files:
    img=plt.imread(name)
    X=np.concatenate((X, img[np.newaxis, :, :]), axis=0)
files=glob.glob('/kaggle/input/ckplus/CK+48/surprise/*.png')
Y=np.concatenate((Y, np.repeat(5, len(files))))
for name in files:
    img=plt.imread(name)
    X=np.concatenate((X, img[np.newaxis, :, :]), axis=0)
X=X.reshape(X.shape[0], 48*48)
np.savetxt("/kaggle/working/imgs.csv", X, delimiter=",")
np.savetxt("/kaggle/working/lab.csv", Y, delimiter=",")
#Test  data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
data=pd.read_csv('/kaggle/input/fergit/FERGIT.csv')
print(data)
imgs=data['pixels'].to_numpy()
usage=data['Usage'].to_numpy()
Y=np.array(data['emotion'].to_numpy(), dtype='int32')
mask=np.logical_not(usage=='Training')
#print(np.sum(mask))
imgs=imgs[mask]
print(imgs.shape)
Y=Y[mask]
img2=str(imgs[0])
x=img2.split(' ')
x=np.array(x, dtype='int32')
x=x.reshape(48, 48)
X=x[np.newaxis, :, :]
count=0
for img in imgs[1:]:
    img2=str(img)
    x=img2.split(' ')
    x=np.array(x, dtype='int32')
    x=x.reshape(48, 48)
    X=np.concatenate((X, x[np.newaxis, :, :]), axis=0)
    print(count)
    count=count+1
X=X.reshape(X.shape[0], 48*48)
X=np.concatenate((X, Y[:, np.newaxis]), axis=1)
np.savetxt("/kaggle/working/test.csv", X, delimiter=",")
import numpy as np
from matplotlib import pyplot as plt
hist=np.zeros(7, dtype='int32')
Y=np.genfromtxt('/kaggle/input/facialexpression/lab.csv', delimiter=',')
for i in range(7):
    hist[i]=np.sum(Y==i)
print(hist)
plt.bar(range(7), hist, color='green')
plt.show()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 2D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
for ctount in range(5):
    X=np.genfromtxt('/kaggle/input/facialexpression/data1.csv', delimiter=',')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    #model.fit(X, Y, batch_size=100, epochs=1)
    X=np.genfromtxt('/kaggle/input/facialexpression/data2.csv', delimiter=',')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    for i in range (50):
        plt.imshow(X[i, :, :, 0])
        plt.show()
    model.fit(X, Y, batch_size=100, epochs=1)
    X=np.genfromtxt('/kaggle/input/facialexpression/data3.csv', delimiter=',')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1)
    X=np.genfromtxt('/kaggle/input/facialexpression/data4.csv', delimiter=',')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1)
model.save('/kaggle/working/model1.h5')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
#K.set_image_dim_ordering('th')
import tensorflow as tf
import numpy as np
def swish_activation(x):
    return (K.sigmoid(x) * x)

model = Sequential()
model.add(Conv2D(48, (7, 7), activation='relu', input_shape=(48,48, 1)))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(GlobalAveragePooling2D())
#model.add(Flatten())
model.add(Dense(256, activation=swish_activation))
model.add(Dropout(0.2))
model.add(Dense(128, activation=swish_activation))
model.add(Dropout(0.2))
model.add(Dense(64, activation=swish_activation))
model.add(Dropout(0.2))
model.add(Dense(7 , activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
X_test=np.genfromtxt('/kaggle/input/testfacialexpression/test.csv', delimiter=',')
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)/255.0
for ctount in range(10):
    X=np.genfromtxt('/kaggle/input/facialexpression/data1.csv', delimiter=',')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)/255.0
    model.fit(X, Y,validation_data=(X_test, y_test), batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/facialexpression/data2.csv', delimiter=',')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)/255.0
    model.fit(X, Y, validation_data=(X_test, y_test), batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/facialexpression/data3.csv', delimiter=',')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)/255.0
    model.fit(X, Y, validation_data=(X_test, y_test), batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/facialexpression/data4.csv', delimiter=',')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)/255.0
    model.fit(X, Y, validation_data=(X_test, y_test), batch_size=100, epochs=1, shuffle=True)
    print(model.predict(X_test[0:10, :, :, :]), y_test[0:10])
model.save('/kaggle/working/model1.h5')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf
import numpy as np
X=np.genfromtxt('/kaggle/input/facialexpression/imgs.csv', delimiter=',')
Y=np.genfromtxt('/kaggle/input/facialexpression/lab.csv', delimiter=',')
X=np.concatenate((X, Y[:, np.newaxis]), axis=1)
print("not_shuffled1")
np.random.shuffle(X)
np.random.shuffle(X)
np.random.shuffle(X)
np.savetxt("/kaggle/working/data.csv", X, delimiter=",")

import numpy as np
X=np.genfromtxt('/kaggle/input/facialexpression/data.csv', delimiter=',')
np.savetxt("/kaggle/working/data1.csv", X[:int(X.shape[0]/4), :], delimiter=",")
np.savetxt("/kaggle/working/data2.csv", X[int(X.shape[0]/4):int(X.shape[0]/2), :], delimiter=",")
np.savetxt("/kaggle/working/data3.csv", X[int(X.shape[0]/2):int(3*X.shape[0]/4), :], delimiter=",")
np.savetxt("/kaggle/working/data4.csv", X[int(3*X.shape[0]/4):, :], delimiter=",")
#Mix CKPlus48 and FER2018+MUX dataset and train, oversample or undersample when required
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from matplotlib import pyplot as plt
from keras import backend as K
#K.set_image_dim_ordering('th')
import tensorflow as tf
import numpy as np
def swish_activation(x):
    return (K.sigmoid(x) * x)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7 , activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#X=np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ')
#print(X.shape)
#print(X[0, :])
#X=np.concatenate((np.genfromtxt('/kaggle/input/datanfer/data1N/data1N.csv', delimiter=','), np.genfromtxt('/kaggle/input/datanfer/data2N/data2N.csv', delimiter=',')), axis=0)
#Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
#X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ')
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
for ctount in range(30):
    X=np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data34/data3N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data34/data4N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, validation_data=(X_test, y_test), batch_size=100, epochs=1, shuffle=True)
    model.save('/kaggle/working/model1.h5')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
#K.set_image_dim_ordering('th')
import tensorflow as tf
import numpy as np
def swish_activation(x):
    return (K.sigmoid(x) * x)

model = Sequential()
model.add(Conv2D(48, (7, 7), activation='relu', input_shape=(48,48, 1)))
model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation=swish_activation))
model.add(Dropout(0.5))
model.add(Dense(512, activation=swish_activation))
model.add(Dropout(0.5))
model.add(Dense(256, activation=swish_activation))
model.add(Dropout(0.5))
model.add(Dense(128, activation=swish_activation))
model.add(Dropout(0.5))
model.add(Dense(64, activation=swish_activation))
model.add(Dropout(0.5))
model.add(Dense(32, activation=swish_activation))
model.add(Dropout(0.5))
model.add(Dense(7 , activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ')
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
for ctount in range(30):
    X=np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data34/data3N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data34/data4N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, validation_data=(X_test, y_test), batch_size=100, epochs=1, shuffle=True)
    model.save('/kaggle/working/model1.h5')
#Saved version
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from matplotlib import pyplot as plt
from keras import backend as K
#K.set_image_dim_ordering('th')
import tensorflow as tf
import numpy as np
def swish_activation(x):
    return (K.sigmoid(x) * x)

model = Sequential()
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48, 1)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7 , activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#X=np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ')
#print(X.shape)
#print(X[0, :])
#X=np.concatenate((np.genfromtxt('/kaggle/input/datanfer/data1N/data1N.csv', delimiter=','), np.genfromtxt('/kaggle/input/datanfer/data2N/data2N.csv', delimiter=',')), axis=0)
#Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
#X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ')
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
while True:
    X=np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data34/data3N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, batch_size=100, epochs=1, shuffle=True)
    X=np.genfromtxt('/kaggle/input/datanfer/data34/data4N.csv', delimiter=' ')
    Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
    X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
    model.fit(X, Y, validation_data=(X_test, y_test), batch_size=100, epochs=1, shuffle=True)
    model.save('/kaggle/working/model1.h5')
from skimage.filters import gabor
from matplotlib import pyplot as plt
import numpy as np
X=np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ')
Y=X[:, X.shape[1]-1]
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48)
filt_real, filt_imag = gabor(X[1, :, :], frequency=0.25, sigma_x=np.sqrt(2), sigma_y=np.sqrt(2), theta=0)
res=np.sqrt(filt_real*filt_real+filt_imag*filt_imag)
plt.imshow(X[1, :, :], cmap='gray')
plt.show()
plt.imshow(res, cmap='gray')
plt.show()
filt_real, filt_imag = gabor(X[1, :, :], frequency=0.25, sigma_x=np.sqrt(2), sigma_y=np.sqrt(2), theta=3.141592653/4)
res=np.sqrt(filt_real*filt_real+filt_imag*filt_imag)
plt.imshow(res, cmap='gray')
plt.show()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from matplotlib import pyplot as plt
from keras import backend as K
#K.set_image_dim_ordering('th')
import tensorflow as tf
import numpy as np
def swish_activation(x):
    return (K.sigmoid(x) * x)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7 , activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
X=np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ')
print(X.shape)
print(X[0, :])
#X=np.concatenate((np.genfromtxt('/kaggle/input/datanfer/data1N/data1N.csv', delimiter=','), np.genfromtxt('/kaggle/input/datanfer/data2N/data2N.csv', delimiter=',')), axis=0)
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ')
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
model.fit(X, Y, validation_data=(X_test, y_test), batch_size=100, epochs=40, shuffle=True)
import numpy as np
from matplotlib import pyplot as plt
X=np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ')
Y=X[:, X.shape[1]-1]
hist=np.zeros(7)
for i in range(7):
    hist[i]=np.sum(Y==i)
plt.bar(range(7), hist)
plt.show()
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import gabor
X=np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ')
Y=X[:int(X.shape[0]/5), X.shape[1]-1]
X=X[:int(X.shape[0]/5), :X.shape[1]-1]
X=X.reshape(X.shape[0], 48, 48)
Z=np.zeros((X.shape[0], 40, 48, 48))
for i in range(X.shape[0]):
    for u in range(5):
        for v in range(8):
            filt_real, filt_imag = gabor(X[1, :, :], frequency=0.2/pow(2, u/2.0), sigma_x=np.sqrt(2), sigma_y=np.sqrt(2), theta=v*3.141592653/8)
            res=np.sqrt(filt_real*filt_real+filt_imag*filt_imag)
            Z[i, u*5+v, :, :]=res
    print(i)
np.savetxt("/kaggle/working/data1G1.csv", np.concatenate((Z.reshape(Z.shape[0], 40*48*48), Y[:, np.newaxis]), axis=1), delimiter=",")
    
import pickle
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel="linear",coef0=0, degree=1)

import numpy as np
X=np.genfromtxt('/kaggle/working/data1G1.csv', delimiter=',')
print(X.shape)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import numpy as np
#K.set_image_dim_ordering('th')
#import tensorflow as tf
#import numpy as np
X=np.concatenate((np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data34/data3N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data34/data4N.csv', delimiter=' ', dtype='float32')), axis=0)
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ', dtype='float32')
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
model = Sequential()
model.add(Conv2D(256, (5, 5), activation='relu', input_shape=(48,48, 1)))
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.4))
model.add(Conv2D(1024, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
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
model.fit(X, Y, validation_data=(X_test, y_test), epochs=1000, batch_size=500, shuffle=True, callbacks=[reduce_lr])
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
X, Y = ros.fit_resample(X, Y)
X=X.reshape(X.shape[0], 48, 48, 1)
Y=tf.keras.utils.to_categorical(Y)
model = Sequential()
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
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
    model.fit(X, Y, validation_data=(X_test, y_test), epochs=1000, batch_size=500, shuffle=True, callbacks=[reduce_lr])
except:
    conf=ConfusionMatrix(X_test, y_test, model)
    conf=conf/np.sum(conf, axis=1)
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
es=EarlyStopping(
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
print(conf/np.sum(conf, axis=1))
#print(history.history.keys())
for  i in range(5):
    print(i)
    plt.plot(history[i].history['val_accuracy'])
    plt.plot(history[i].history['accuracy'])
    plt.show()
    plt.plot(history[i].history['val_loss'])
    plt.plot(history[i].history['loss'])
    plt.show()
print(max(history[0].history['val_accuracy']))
print(max(history[1].history['val_accuracy']))
print(max(history[2].history['val_accuracy']))
print(max(history[3].history['val_accuracy']))
print(max(history[4].history['val_accuracy']))
#print(max(history[5].history['val_accuracy']))
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
#ct_keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])
#Cleaned FER
#For FER
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
    model.fit(X, Y, validation_data=(X_test, y_test), epochs=1000, batch_size=500, shuffle=True, callbacks=[reduce_lr])
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
#Cleaned CKPLUS extra
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
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
#K.set_image_dim_ordering('th')
#import tensorflow as tf
#import numpy as np
X=np.genfromtxt('/kaggle/input/ckplusfinal/CKPlusFinalN.csv', delimiter=' ', dtype='float32')
np.random.shuffle(X)
X_test=X[X.shape[0]-150:, :]
X=X[:X.shape[0]-150, :]
#X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ', dtype='float32')
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
model = Sequential()
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
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
    model.save('/kaggle/working/modelCK.h5')
    print(ConfusionMatrix(X_test, y_test, model))
    
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
    p=0.3
    model = Sequential()
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
    #print(model.summary())
    return model
history=[]
es=EarlyStopping(
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
print(conf/np.sum(conf, axis=1))
#print(history.history.keys())
for  i in range(5):
    print(i)
    plt.plot(history[i].history['val_accuracy'])
    plt.plot(history[i].history['accuracy'])
    plt.show()
    plt.plot(history[i].history['val_loss'])
    plt.plot(history[i].history['loss'])
    plt.show()
print(max(history[0].history['val_accuracy']))
print(max(history[1].history['val_accuracy']))
print(max(history[2].history['val_accuracy']))
print(max(history[3].history['val_accuracy']))
print(max(history[4].history['val_accuracy']))
#print(max(history[5].history['val_accuracy']))
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

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from matplotlib import pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import numpy as np
#K.set_image_dim_ordering('th')
#import tensorflow as tf
#import numpy as np
def ConfusionMatrix(X_test, Y_test, model):
    Y_pred=np.argmax(model.predict(X_test), axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    print(Y_pred.shape)
    conf_Mat=np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            conf_Mat[i, j]=np.sum(np.logical_and(Y_test==i, Y_pred==j))
    return conf_Mat

X=np.concatenate((np.genfromtxt('/kaggle/input/datanfer/data12/data1N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data12/data2N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data34/data3N.csv', delimiter=' ', dtype='float32'), np.genfromtxt('/kaggle/input/datanfer/data34/data4N.csv', delimiter=' ', dtype='float32')), axis=0)
Y=tf.keras.utils.to_categorical(X[:, X.shape[1]-1], num_classes=7)
X=X[:, :X.shape[1]-1].reshape(X.shape[0], 48, 48, 1)
X_test=np.genfromtxt('/kaggle/input/datanfer/testN2/testN.csv', delimiter=' ', dtype='float32')
y_test=tf.keras.utils.to_categorical(X_test[:, X_test.shape[1]-1], num_classes=7)
X_test=X_test[:, :X_test.shape[1]-1].reshape(X_test.shape[0], 48, 48, 1)
model = Sequential()
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48, 1)))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.4))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.4))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
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
    model.fit(X, Y, validation_data=(X_test, y_test), epochs=1000, batch_size=500, shuffle=True, callbacks=[reduce_lr])
except:
    conf=ConfusionMatrix(X_test, y_test, model)
    print(conf/np.sum(conf, axis=1))
model.save('/kaggle/working/model1.h5')