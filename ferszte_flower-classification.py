import os

import numpy as np 

import pandas as pd 

import seaborn as sns

from PIL import Image

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout

from keras.layers import BatchNormalization, Flatten, MaxPooling2D

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
data = os.listdir('../input/flowers/flowers/')

n_classes = len(data)

print(n_classes,'classes', ':', data)
X_data = []

y_data = []



# Extract images and their corresponding labels

for classe in data :

    for image in os.listdir('../input/flowers/flowers/'+str(classe)):

        if image.endswith('.jpg'):

            img = Image.open('../input/flowers/flowers/'+str(classe)+'/'+image)

            img = img.resize((224,224),Image.ANTIALIAS)

            

            # Convert img to np array

            img = np.array(img)

            classe_index = data.index(classe)

            X_data.append(img)

            y_data.append(classe_index)
# Count samples per class

sns.countplot(y_data)
X = np.array(X_data)

X = X.astype('float32')/255.0

print(X.shape)
y = np.array(y_data)

y = to_categorical(y_data,n_classes)

print(y.shape)
n_samples = X.shape[0]

img_dim = (224,224,3)

batch_size = 64

epochs = 50

n_val_samples = 0.2 * n_samples

n_train_samples = 0.8 * n_samples
model = Sequential()

model.add(Conv2D(32, (5,5),padding = 'same',activation ='relu',

                 input_shape = img_dim))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(64,(3,3),padding = 'same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, (3,3),padding = 'same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, (3,3),padding = 'same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dense(n_classes, activation = "softmax"))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='rmsprop')
model.summary()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
batch_size = 64

epochs = 50

n_val_samples = 0.2 * n_samples

n_train_samples = 0.8 * n_samples
train_datagen = ImageDataGenerator(horizontal_flip=True,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   rotation_range=20)

                                  

train_generator = train_datagen.flow(x=X_train,

                                     batch_size=batch_size,

                                     y=y_train)                                
early_stop = EarlyStopping(patience=5,monitor='val_acc')



hist = model.fit_generator(train_generator,

                 callbacks=[early_stop],

                 validation_data=(X_test,y_test),

                 validation_steps=n_val_samples//batch_size,

                 steps_per_epoch=n_samples//batch_size, 

                 epochs=epochs,verbose=1)
# Plot acc and val_acc curves

plt.title('Accuracy')

plt.plot(hist.history['val_acc'])

plt.plot(hist.history['acc'])

plt.legend(['test','train'])

plt.show()



# Plot loss and val_loss curves

plt.title('Loss')

plt.plot(hist.history['val_loss'])

plt.plot(hist.history['loss'])

plt.legend(['test','train'])

plt.show()
# Make predictions from test set

y_pred = model.predict_classes(X_test)

y_true = np.argmax(y_test,axis=1)



# Plot confusion matrix

plt.figure(figsize=(10,10))

plt.title('Confusion matrix')

sns.heatmap(confusion_matrix(y_true,y_pred),annot=True,linewidth=2)
from sklearn.metrics import classification_report



# Compute precision recall & f1-score

print('Classification report : \n', classification_report(y_true, y_pred))
j = 0

X_misclassified = []

wrong_labels = []

true_labels =[]



# Get 20 misclassified samples 

for pred, true_classe in zip(y_pred,y_true):

    if pred != true_classe:

        wrong_labels.append(pred)

        true_labels.append(true_classe)

        X_misclassified.append(X_test[j])

        

        if len(wrong_labels)==20:

            break

    j+=1

        

# Plot misclassified samples and their true labels   

plt.figure(figsize=(10,10))

for i in range(0,20):

    plt.subplot(5,4,i+1)

    plt.title('Predicted : ' +str(data[wrong_labels[i]])

              +'\n True label : '+str(data[true_labels[i]]))

    plt.axis('off')

    plt.imshow(X_misclassified[i])

plt.tight_layout()
