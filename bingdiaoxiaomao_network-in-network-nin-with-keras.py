import keras
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPool2D,GlobalAveragePooling2D,AveragePooling2D
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
%matplotlib inline
batch_size = 128
epochs = 30
num_classes = 10
weight_decay = 1e-6
nets = 15
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
labels = train['label']
train = train.drop(['label'],axis=1)
X = train.values
test = test.values
print(X.shape)
print(labels.shape)
print(test.shape)
X = X/255.
test = test/255.
X = np.reshape(X,(-1,28,28,1))
test = np.reshape(test,(-1,28,28,1))
labels = keras.utils.to_categorical(labels,10)
print(X.shape)
print(test.shape)
print(labels.shape)
plt.figure(figsize=(10,10))
for i in range(10):
    for j in range(10):
        plt.subplot(10,10,i*10+j+1)
        plt.imshow(X[np.argmax(labels,axis=1)==i][j].reshape(28,28),cmap=plt.cm.gray)
        plt.axis('off')
plt.subplots_adjust(wspace=-0.1,hspace=-0.1)
plt.show()    
learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_delta=1e-5)
earlystopping = EarlyStopping(monitor='val_loss',
                              patience=3,
                              verbose=1,
                              mode='auto')
def build_model():
    model = Sequential()
    
    model.add(Conv2D(192,(5,5),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal',input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(160,(1,1),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(96,(1,1),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same'))
    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(192,(5,5),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1,1),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1,1),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same'))
    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(192,(3,3),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(192,(1,1),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(10,(1,1),padding='same',kernel_regularizer=l2(weight_decay),kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))  
    
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    adam = optimizers.rmsprop()
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model
models = []
for i in range(nets):
    models.append(build_model())
print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=False,
                             width_shift_range=0.10,
                             height_shift_range=0.10,
                             fill_mode='constant',
                             cval=0,
                             rotation_range=10,
                             zoom_range=0.1)
histories = []
for i in range(nets):
    x_train,x_valid,y_train,y_valid = train_test_split(X,labels,test_size=0.2)
    datagen.fit(x_train)
    history = models[i].fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                                  steps_per_epoch=int(len(x_train)/batch_size),
                                  epochs=epochs,
                                  callbacks=[learning_rate_reduction,earlystopping],
                                  validation_data=(x_valid,y_valid),
                                  verbose=1)
    histories.append(history)
    print('Network in Network {0:d}: Epocks={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}'.format(i+1,max(history.epoch),max(history.history['acc']),max(history.history['val_acc'])))
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
for i in range(nets):
    history=histories[i]
    epoch_range = 1 + np.arange(len(history.history['acc']))
    plt.plot(epoch_range,history.history['loss'],'g-',label='Training loss')
    plt.plot(epoch_range,history.history['val_loss'],'r--',label='Validation loss')
    plt.legend(loc='best',shadow=True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([0,epochs])
    plt.ylim([0,0.2])


plt.subplot(2,1,2)
for i in range(nets):
    history=histories[i]
    epoch_range = 1 + np.arange(len(history.history['acc']))
    plt.plot(epoch_range,history.history['acc'],'g-',label='Training accuracy')
    plt.plot(epoch_range,history.history['val_acc'],'r--',label='Validation accuracy')
    plt.legend(loc='best',shadow=True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim([0,epochs])
    plt.ylim([0.95,1])
plt.show()
def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
Y_pred = np.zeros_like(y_valid)
for i in range(nets):
    model = models[i]
    Y_pred += model.predict(x_valid,batch_size=128)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_valid,axis = 1) 
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = range(10))
errors = (Y_pred_classes != Y_true)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = x_valid[errors]
def display_errors(errors_index,img_errors,pred_errors,obs_errors):
    n = 0
    nrows = 2
    ncols = 3
    fig,ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(12,12))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)),cmap=plt.cm.gray)
            ax[row,col].set_title('Predicted label:{}\nTrue label:{}'.format(pred_errors[error],obs_errors[error]))
            n += 1
    plt.show()
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_dela_errors[-6:]
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
results = np.zeros((len(test),10),dtype='float')
for i in range(nets):
    model = models[i]
    results += model.predict(test,batch_size=128,verbose=1)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)
