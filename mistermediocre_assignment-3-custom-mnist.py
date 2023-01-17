
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

import skimage
from skimage.morphology import square, diamond, disk
from keras.datasets import mnist

def prepareData():
    train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv').values
    X_train = np.array(train[:,1:])/255  
    y_train =  np.array(train[:,0])

    val = pd.read_csv('../input/mnist-in-csv/mnist_test.csv').values
    X_val = np.array(val[:,1:])/255
    y_val =  np.array(val[:,0])
    
    X_train2 = np.array(train[:,1:])/255
    for i in range(X_train.shape[0]):
        if(i%4 == 0): shape = square(1)
        if(i%4 == 1): shape = diamond(1)
        if(i%4 == 2): shape = disk(1)
        if(i%4 == 3): shape = None
        X_train2[i,:] = skimage.morphology.erosion(X_train[i,:].reshape(28,28), shape).reshape(1,784)
    X_train = np.concatenate((X_train, X_train2))
    y_train = np.concatenate((y_train, y_train))
    
    X_train2 = np.array(X_train)
    for i in range(X_train.shape[0]):
        X_train2[i,:] = X_train[i,:] * (1 + np.random.normal(0, 0.25))
    X_train = np.concatenate((X_train, X_train2))
    y_train = np.concatenate((y_train, y_train))

    y_test = np.ravel(pd.read_csv("../input/x-mnist/Y_MNIST.csv").values)
    return (X_train, y_train, X_val, y_val)

test = pd.read_csv('../input/x-test-nearlymnist/x_test.csv')
Index = test['Index']
X_test = test.values[:, 1:]/255




from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, Conv2D, MaxPooling2D, MaxPool2D
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

num_classes = 10
batch_size = 64
epochs = 20
input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(32, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))


model.compile(optimizer=RMSprop(),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)


#history=model.fit(X_train, train_labels, validation_split=0.05,
#                 epochs=10, batch_size=200)
model.save_weights('initial')

%%time
from keras.utils.np_utils import to_categorical #One hot 
(X_train, y_train, X_val, y_val) = prepareData()

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_val = X_val.reshape(X_val.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print("Data ready")

#model.load_weights('initial')
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.20,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.20,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
        
datagen.fit(X_train)

h = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 2048),
                              epochs = 100, validation_data = (X_val , y_val),
                              steps_per_epoch=10,  # // batch_size
                              callbacks=[learning_rate_reduction]), 

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train.argmax(axis =1 ), model.predict(X_train).argmax(axis = 1))

from sklearn.metrics import accuracy_score
accuracy_score(y_train.argmax(axis =1 ), model.predict(X_train).argmax(axis = 1))
confusion_matrix(y_train.argmax(axis =1 ), model.predict(X_train).argmax(axis = 1))
confusion_matrix(y_test2.argmax(axis =1 ), model2.predict(X_test2).argmax(axis = 1))
[doubts[x] for x in model2.predict(X_test2).argmax(axis=1).tolist()]

def classifier(doubts, epochs = 3):
    indices_train = y_train.argmax(axis = 1) == -1
    indices_val = y_val.argmax(axis = 1) == -1

    for doubt in doubts:
        indices_train = np.logical_or(indices_train, y_train.argmax(axis = 1) == doubt)
        indices_val = np.logical_or(indices_val, y_val.argmax(axis = 1) == doubt)

    X_train2 = np.array(X_train[indices_train])
    y_train2 = np.array(y_train[indices_train])[:,doubts]
    X_val2 = np.array(X_val[indices_val])
    y_val2 = np.array(y_val[indices_val])[:,doubts]
    
    num_classes = len(doubts)
    input_shape = (28, 28, 1)

    model2 = Sequential()
    model2.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
    model2.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
    model2.add(MaxPool2D((2, 2)))
    model2.add(Dropout(0.20))
    model2.add(Conv2D(32, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    model2.add(Dropout(0.25))
    model2.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model2.add(Dropout(0.25))
    model2.add(Flatten())
    model2.add(Dense(32, activation='relu'))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.25))
    model2.add(Dense(num_classes, activation='softmax'))

    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=25, # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.20, # Randomly zoom image 
            width_shift_range=0.20,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.20,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images


    model2.compile(optimizer=RMSprop(),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.0001)
    datagen.fit(X_train2)
    h = model2.fit_generator(datagen.flow(X_train2, y_train2, batch_size = 1024),
                                  epochs = epochs, validation_data = (X_val2 , y_val2),
                                  steps_per_epoch=10,  # // batch_size
                                  callbacks=[learning_rate_reduction]), 
    
    return (model2, X_train2, y_train2, X_val2, y_val2)

from sklearn.metrics import accuracy_score
y_pred_bagged = model.predict(X_test).argmax(axis=1)
doubtMap = [[1,2], [1,7], [4,9], [6,0], [7,2] ]
#doubts:[(1,2), (1,7), (4,9), (0,6), (7,2) ]
for doubt in doubtMap:
    (model2, X_train2, y_train2, X_val2, y_val2) = classifier(doubt, epochs = 20)
    y_pred2 = np.array([doubt[x] for x in model2.predict(X_test).argmax(axis=1).tolist()])
    for x in doubt: y_pred_bagged = np.where(y_pred_bagged==x, y_pred2, y_pred_bagged)
    
print(confusion_matrix(y_test2.argmax(axis =1 ), model2.predict(X_test2).argmax(axis = 1)))
y_pred2 = np.array([doubtMap[2][x] for x in model2.predict(X_test).argmax(axis=1).tolist()])

doubtMap[2]
from sklearn.metrics import accuracy_score
accuracy_score(y_test.argmax(axis = 1), y_pred_bagged)

#y_pred = model.predict(X_test).argmax(axis=1)
confusion_matrix(y_test.argmax(axis =1), y_pred_bagged)

confusion_matrix(y_test.argmax(axis =1), y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test.argmax(axis = 1), np.where(y_pred == 2, y_pred2, y_pred))


from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, Conv2D, MaxPooling2D, MaxPool2D
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

num_classes = 10
batch_size = 64
epochs = 20
input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Dense(32, activation='relu', input_shape=input_shape))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30, # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.20, # Randomly zoom image 
        width_shift_range=0.20,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.20,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images



model.compile(optimizer=RMSprop(),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)


#history=model.fit(X_train, train_labels, validation_split=0.05,
#                 epochs=10, batch_size=200)
model.save_weights('initial2')


from keras.utils.np_utils import to_categorical #One hot 


datagen.fit(X_train)
h = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
                              epochs = 50, validation_data = (X_test , y_test),
                              steps_per_epoch=500,  # // batch_size
                       ), 



wrong = [y_test.argmax(axis =1 ) != model.predict(X_test).argmax(axis = 1)]
X_wrong = X_test[wrong]
#y_wrong = y_test.argmax(axis = 1)[wrong]
y_wrong = y_pred[wrong]
images_and_labels = list(zip(X_wrong.reshape(-1,28,28), y_wrong))
for index, (image, label) in enumerate(images_and_labels[:20]):
    plt.subplot(5,4,index+1)
    plt.axis('off')
    plt.imshow(image, plt.cm.gray_r, interpolation='nearest')
    plt.title('Test: ' + str(label))
    



def shift(X, row = 1, col = 1):
    X = np.array(X)
    X = X.reshape((X.shape[0], 28, 28))
    if(row>0):
        X[:,row:28, :] = X[:, :28-row, :]
        X[:,0:row, :] = 0
    if(row<0):
        X[:,:28+row,:] = X[:, -row:28, :]
        X[:,28+row:,:] = 0
        
    if(col>0):
        X[:,:,col:28] = X[:,:,:28-col]
        X[:,:,0:col] = 0
    if(col<0):
        X[:,:,:28+col] = X[:,:,-col:28]
        X[:,:,28+col:] = 0
        
    return X.reshape((X.shape[0], 784))



(X_train, y_train, X_val, y_val, X_test, y_test) = prepareMNISTdata()
X_train = shift(X_train, 3, -3)
i = 5
plt.imshow(X_train[i].reshape((28, 28)))
print(y_train[i])


np.concatenate((X_train, X_train)).shape
train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv').values
X_train = np.array(train[:,1:])/255
X_train2 = np.array(train[:,1:])/255
y_train =  np.array(train[:,0])

val = pd.read_csv('../input/mnist-in-csv/mnist_test.csv').values
X_val = np.array(val[:,1:])/255
y_val =  np.array(val[:,0])



pd.read_csv('../input/mnist-in-csv/mnist_train.csv').head()
0.7 + np.random.random()/2
X_train.mean()
import pandas as pd
#y_pred = model.predict(X_test).argmax(axis = 1)
y_pred = y_pred_bagged
images_and_labels = list(zip(X_test.reshape(-1,28,28), y_pred))
for index, (image, label) in enumerate(images_and_labels[:8]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image, plt.cm.gray_r, interpolation='nearest')
    plt.title('Test: ' + str(label))
    


d = {'Index': Index, 'Labels': y_pred}
df = pd.DataFrame(data=d)
df.reset_index(drop=True, inplace=True)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe
create_download_link(df)
