import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import itertools

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint

from keras.applications.resnet50 import ResNet50

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D

import random as rn

import tensorflow as tf

import cv2                  

from tqdm import tqdm

import os                   

print(os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images'))

print(os.listdir('../input/resnet50'))
X=[]

Y=[]

IMG_SIZE=50

parasitized_dir='../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized'

uninfected_dir='../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected'
categories = ['Parasitized','Uninfected']

print("labels of images are " ,categories )
def assign_label(cell_type):

    return cell_type
def build_train_data(cell_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        try:

            label=assign_label(cell_type)

            path = os.path.join(DIR,img)

            img = cv2.imread(path,cv2.IMREAD_COLOR)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

            X.append(np.array(img))

            Y.append(str(label))

        except:

            print("")
build_train_data('Parasitized',parasitized_dir)

print(len(X))
build_train_data('Uninfected',uninfected_dir)

print(len(X))
def find_categories_count(Y):

    categories_count={

    'Parasitized':0,

    'Uninfected':0

    }

    for i in Y:

         categories_count[i]=categories_count[i]+1

    return categories_count
def find_class_label(data):

    match={

    0:'Parasitized',

    1:'Uninfected'

        }

    return match[data]
print(find_categories_count(Y))
height=[]

categories_count=find_categories_count(Y)

for i in categories_count:

    height.append(categories_count[i])
plt.bar([x for x in categories],height)
fig,ax=plt.subplots(3,3)

fig.set_size_inches(20,20)

for i in range(3):

    for j in range (3):

        l=rn.randint(0,len(Y))

        ax[i,j].imshow(X[l])

        ax[i,j].set_title('cell_type: '+Y[l])

        

plt.tight_layout()
def covert_categories_to_numeric(data):

    y=[]

    match={

    'Parasitized':0,

    'Uninfected':1

    }

    for i in  data:

        y.append(match[i])

    return np.array(y)
Y=covert_categories_to_numeric(Y)

print(Y)
Y=to_categorical(Y,2)
X=np.array(X)

X=X/255
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=48)

x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.1,random_state=48)


print('There are %d total image categories.' % len(categories))

print('There are %s total cell images.\n' % X.shape[0])

print('There are %d training cell images.' % x_train.shape[0])

print('There are %d validation cell images.' % x_valid.shape[0])

print('There are %d testing cell images.'% x_test.shape[0])
plt.bar(['total_data','training_data','validation_data','testing_data'],[X.shape[0],x_train.shape[0],x_valid.shape[0],x_test.shape[0]])
height=[]

data=[]

for i in np.argmax(y_train,axis=1):

    data.append(find_class_label(i))

categories_count=find_categories_count(data)

for i in categories_count:

    height.append(categories_count[i])

plt.bar([x for x in categories],height)
y_train.shape[0]
height=[]

data=[]

for i in np.argmax(y_valid,axis=1):

    data.append(find_class_label(i))

categories_count=find_categories_count(data)

for i in categories_count:

    height.append(categories_count[i])

plt.bar([x for x in categories],height)
y_valid.shape[0]
height=[]

data=[]

for i in np.argmax(y_test,axis=1):

    data.append(find_class_label(i))

categories_count=find_categories_count(data)

for i in categories_count:

    height.append(categories_count[i])

plt.bar([x for x in categories],height)
y_test.shape[0]
np.random.seed(42)

rn.seed(42)

tf.set_random_seed(42)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
benchmark_checkpoints = ModelCheckpoint(filepath='weights.best.from_bench.hdf5', 

                               verbose=1, save_best_only=True)
bench_model = Sequential()

bench_model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 

                        input_shape=(50,50, 3)))

bench_model.add(MaxPooling2D(pool_size=2))

bench_model.add(Flatten())

bench_model.add(Dense(2, activation='softmax'))
bench_model.summary()
bench_model.compile(loss='categorical_crossentropy', 

                    optimizer=Adam(lr=0.1, decay=1e-6), 

                  metrics=['accuracy'])
bench_results = bench_model.fit(x_train, y_train, batch_size=32,

                   epochs=10,validation_data=(x_valid,y_valid), callbacks=[benchmark_checkpoints], verbose=1)
pred=bench_model.predict(x_test)

bench_pred=np.argmax(pred ,axis=1)
plt.plot(bench_results.history['loss'])

plt.plot(bench_results.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])

plt.show()
plt.plot(bench_results.history['acc'])

plt.plot(bench_results.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])

plt.show()
mis_classification_count=0

for i in range(0,len(bench_pred)):

    if bench_pred[i]!=np.argmax(y_test[i]):

        mis_classification_count+=1

        

print(mis_classification_count)
test_accuracy = 100*(y_test.shape[0]-mis_classification_count)/len(bench_pred)

print('Test accuracy: %.4f%%' % test_accuracy)
plot_confusion_matrix(confusion_matrix(np.argmax(y_test ,axis=1), bench_pred), classes=categories,

                      title='Confusion matrix')
print(classification_report( np.argmax(y_test ,axis=1),bench_pred ,target_names =categories ))
sequential_Checkpoints=ModelCheckpoint(filepath='weights.best.from_sequential_model.hdf5', 

                               verbose=1, save_best_only=True)
#creating sequential model

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
sequential_results = model.fit(x_train, y_train, batch_size=32,

                     epochs=10,validation_data=(x_valid,y_valid),callbacks=[sequential_Checkpoints], verbose=1)
plt.plot(sequential_results.history['loss'])

plt.plot(sequential_results.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])

plt.show()
plt.plot(sequential_results.history['acc'])

plt.plot(sequential_results.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])

plt.show()
model.load_weights('weights.best.from_sequential_model.hdf5')
pred=model.predict(x_test)

sequential_model_pred=np.argmax(pred,axis=1)
mis_classification_count=0

mis_classification_index=[]

for i in range(0,len(sequential_model_pred)):

    if sequential_model_pred[i]!=np.argmax(y_test[i]):

        mis_classification_count+=1

        mis_classification_index.append(i)

print(mis_classification_count)     
test_accuracy = 100*(y_test.shape[0]-mis_classification_count)/len(sequential_model_pred)

print('Test accuracy: %.4f%%' % test_accuracy)
plot_confusion_matrix(confusion_matrix(np.argmax(y_test ,axis=1), sequential_model_pred), classes=categories,

                      title='Confusion matrix')
print(classification_report(np.argmax(y_test ,axis=1), sequential_model_pred ,target_names =categories ))
fig,ax=plt.subplots(3,3)

fig.set_size_inches(15,15)

for i in range(3):

    for j in range (3):

        l= rn.randint(0,len(y_test))

        ax[i,j].imshow(x_test[l])

        ax[i,j].set_title('predicted_label: '+find_class_label(sequential_model_pred[l])+'/original_label: '+find_class_label(np.argmax(y_test[l])))

        

plt.tight_layout()
resnet50_checkpoints = ModelCheckpoint(filepath='weights.best.from_resnet50.hdf5', 

                               verbose=1, save_best_only=True)
model = Sequential()

model.add(ResNet50(include_top = False, weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape = (50,50,3)))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['accuracy'])
resnet50_results = model.fit(x_train, y_train, batch_size=32,

                     epochs=10,validation_data=(x_valid,y_valid),callbacks=[resnet50_checkpoints ], verbose=1)
plt.plot(resnet50_results.history['loss'])

plt.plot(resnet50_results.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])

plt.show()
plt.plot(resnet50_results.history['acc'])

plt.plot(resnet50_results.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'validation'])

plt.show()
model.load_weights('weights.best.from_resnet50.hdf5')
resnet50_pred=np.argmax(model.predict(x_test),axis=1)
mis_classification_count=0

mis_classification_index=[]

for i in range(0,len(resnet50_pred)):

    if resnet50_pred[i]!=np.argmax(y_test[i]):

        mis_classification_count+=1

        mis_classification_index.append(i)

print(mis_classification_count)     
test_accuracy = 100*(y_test.shape[0]-mis_classification_count)/len(resnet50_pred)

print('Test accuracy: %.4f%%' % test_accuracy)
plot_confusion_matrix(confusion_matrix(np.argmax(y_test ,axis=1), resnet50_pred), classes=categories,

                      title='Confusion matrix')
print(classification_report(np.argmax(y_test ,axis=1), resnet50_pred ,target_names =categories ))
fig,ax=plt.subplots(3,3)

fig.set_size_inches(20,20)

for i in range(3):

    for j in range (3):

        l= rn.randint(0,len(y_test))

        ax[i,j].imshow(x_test[l])

        ax[i,j].set_title('predicted_label: '+find_class_label(resnet50_pred[l])+'/original_label: '+find_class_label(np.argmax(y_test[l])))

        

plt.tight_layout()