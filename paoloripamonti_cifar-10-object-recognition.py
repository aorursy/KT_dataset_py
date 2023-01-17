# Keras

import keras

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping



# Metrics

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from skimage.transform import resize



# Utility

import time

import os

import itertools

import requests

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import requests

from io import BytesIO

from IPython import display



import warnings

warnings.simplefilter("ignore", UserWarning)
batch_size = 32

num_classes = 10

epochs = 100

data_augmentation = True

num_predictions = 20

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_cifar10.h5'

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
for i in range(0, 9):

    plt.subplot(3,3,i + 1)

    plt.imshow((x_train[i]))

# show the plot

plt.show()
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
#create the model

model = Sequential()



#Layer 1

model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = x_train.shape[1:]))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Flatten())  

model.add(Dense(512, activation = 'relu', kernel_initializer = 'uniform'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation = 'softmax'))



model.summary()
# initiate RMSprop optimizer

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



# Let's train the model using RMSprop

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255
# define early stopping callback

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=5, verbose=1, mode='auto')

callbacks_list = [earlystop]
%%time

history = model.fit(x_train, y_train,

              batch_size=batch_size,

              epochs=epochs,

              validation_data=(x_test, y_test),

              callbacks=callbacks_list,

              shuffle=True)
# Loss Curves

plt.figure(figsize=[14,10])

plt.subplot(211)

plt.plot(history.history['loss'],'r',linewidth=3.0)

plt.plot(history.history['val_loss'],'b',linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Loss',fontsize=16)

plt.title('Loss Curves',fontsize=16)

 

# Accuracy Curves

plt.figure(figsize=[14,10])

plt.subplot(212)

plt.plot(history.history['acc'],'r',linewidth=3.0)

plt.plot(history.history['val_acc'],'b',linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)
# Save model and weights

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)
def predict(img):

    start_at = time.time()

    prediction = model.predict(np.asarray([img]))

    proba = np.max(prediction)

    label = classes[np.argmax(prediction)]

    

    return {

            "label": label,

            "confidence": proba,

            "elapsed_time": time.time() - start_at

        }
idx = np.random.randint(x_test.shape[0])

plt.imshow(x_test[idx])

predict(x_test[idx])
scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
predictions = model.predict(x_test, verbose=1)
y_true, y_pred = [],[]

for idx, prediction in enumerate(predictions): 

    y_true.append(classes[np.argmax(y_test[idx])])

    y_pred.append(classes[np.argmax(prediction)])
print(classification_report(y_pred, y_true))
def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(11, 11))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=30)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)

    plt.yticks(tick_marks, classes, fontsize=15)



    fmt = '.2f'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label', fontsize=25)

    plt.xlabel('Predicted label', fontsize=25)

    plt.tight_layout()



    plt.show()
cnf_matrix = confusion_matrix(y_pred, y_true)

cnf_matrix = cnf_matrix.astype(float) / cnf_matrix.sum(axis=1)[:, np.newaxis]

plot_confusion_matrix(cnf_matrix, classes)
def predict_by_url(url):

    response = requests.get(url)

    original = Image.open(BytesIO(response.content))

    img = resize(np.asarray(original), (32, 32), anti_aliasing=True)

    return predict(img)    
url1 = "https://www.ttnews.com/sites/default/files/images/articles/usa-truck_0.jpg"

print(predict_by_url(url1))

display.Image(url= url1)
url2 = "https://a57.foxnews.com/media2.foxnews.com/BrightCove/694940094001/2019/02/13/931/524/694940094001_6001731668001_6001731174001-vs.jpg?ve=1&tl=1"

print(predict_by_url(url2))

display.Image(url= url2)
url3 = "https://robbreportedit.files.wordpress.com/2018/03/the-hinckley-sport-boat_water.jpg?w=1024"

print(predict_by_url(url3))

display.Image(url= url3)
url4 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQXh3S5Vzu0Vx-yRTdo4UfbQsuhFtaqjiMAlKrTcFmN9JgSSiD_Lw"

print(predict_by_url(url4))

display.Image(url= url4)
url5 = "https://cdn.theatlantic.com/assets/media/img/mt/2018/07/AP_18191475093432-1/lead_720_405.jpg?mod=1533691454"

print(predict_by_url(url5))

display.Image(url= url5)