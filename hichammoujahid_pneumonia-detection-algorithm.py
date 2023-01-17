import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import cv2
%matplotlib inline


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras import layers
from keras.applications import *
from keras.preprocessing.image import load_img
import random
#from tensorflow.keras.applications import EfficientNetB7
#1. Creating and compiling the model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import tensorflow as tf
import time
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import tensorflow as tf
import time
from keras.preprocessing import image
import random
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import *
import itertools
import matplotlib.pyplot as plt
def get_model_transfer(original="", unfreeze_last=4, last_activation='sigmoid', num_labels=2,
              acc=['acc'], loss='categorical_crossentropy',optimizer="adam", ):
  

    if original!="":
        for layer in original.layers[:-unfreeze_last]:
            layer.trainable = False
        
        model = Sequential()
        model.add(original)
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_labels, activation=last_activation))
        model.compile(loss=loss,optimizer=optimizer,metrics=acc)
        return model
    else: 
        print("Please specify an original model !")
        return False

def train_model(model, train_gen, valid_gen, epochs=10, steps_per_epoch=100, my_callbacks=""):
    if isinstance(my_callbacks, str):
        history = model.fit_generator(train_gen,
                                  validation_data=valid_gen,
                                  epochs=epochs,
                                  validation_steps=valid_gen.samples/(valid_gen.batch_size*5),
                                  verbose=1,
                                  steps_per_epoch=steps_per_epoch)
    else:
        history = model.fit_generator(train_gen,
                                  validation_data=valid_gen,
                                  epochs=epochs,
                                  validation_steps=valid_gen.samples/(valid_gen.batch_size*5),
                                  steps_per_epoch=steps_per_epoch,
                                  verbose=1,
                                  callbacks=my_callbacks)
        
    return history


def show_accuracy(history, acc='acc', val_acc='val_acc', loss='loss', val_loss='val_loss'):
    acc = history.history[acc]
    val_acc = history.history[val_acc]
    loss = history.history[loss]
    val_loss = history.history[val_loss]
    epochs = range(len(acc))


    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    line1, = plt.plot(epochs, acc,'b',label="Training acc")
    line2, = plt.plot(epochs, val_acc,'r',label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend(handles=[line1, line2], loc='lower right')
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")

    plt.subplot(122)
    line3, = plt.plot(epochs, loss,'b',label="Training loss")
    line4, = plt.plot(epochs, val_loss,'r',label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend(handles=[line3, line4], loc='upper right')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()

def plot_confusion_matrix(validation_generator, predictions, normalize=False, title='Confusion matrix',
                          classes=['PNEUMONIA', 'NORMAL'], cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    Y_pred = predictions
    #Confution Matrix and Classification Report
    y_pred = np.argmax(Y_pred, axis=1)
    cm=confusion_matrix(validation_generator.classes, y_pred)
    
    print('Classification Report')
    print(classification_report(validation_generator.classes, y_pred, target_names=classes))
    
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#plot_confusion_matrix(cm=confusion_matrix(validation_generator.classes, y_pred), classes=target_names, title='Confusion Matrix')

def show_predictions(validation_datagen, directory='../input/chest-xray-pneumonia/chest_xray/chest_xray/test',
                                 target_size=(224, 224),batchsize=10, class_mode='categorical', shuffle=False):
    # Create a generator for prediction
    validation_generator = validation_datagen.flow_from_directory(directory,target_size=target_size,
                                                                batch_size=batchsize,class_mode=class_mode,shuffle=False)
    # Get the filenames from the generator
    fnames = validation_generator.filenames

    # Get the ground truth from generator
    ground_truth = validation_generator.classes

    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v,k) for k,v in label2index.items())

    # Get the predictions from the model using the generator
    predictions = model.predict_generator(validation_generator,
                                          steps=validation_generator.samples/validation_generator.batch_size,
                                          verbose=1)
    predicted_classes = np.argmax(predictions,axis=1)
    #show predicted label of first image print(idx2label[np.argmax(predictions[0])])
    #show original label of first image print(fnames[0].split('/')[0])

    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

    n=0
    plt.figure(figsize=(15,10))
    for i in random.sample(range(len(np.argmax(predictions,axis=1))), 20):
    #pred_class = np.argmax(predictions[errors[i]])
    #pred_label = idx2label[pred_class]
        ax = plt.subplot(5,4,n+1)
        col = 'red' if i in errors else 'green'
        original = load_img('{}/{}'.format('../input/chest-xray-pneumonia/chest_xray/chest_xray/test',fnames[i]))
        plt.imshow(original)
        plt.title('{} : {} : {:.3f}'.format(i,idx2label[np.argmax(predictions[i])], predictions[i][np.argmax(predictions[i])]), color=col)
        plt.axis('off')
        n+=1
    plt.show()
    return (validation_generator, predictions)
    
#print(os.environ)
def check_tpu_statue():
    if 'TPU_NAME' not in os.environ:
        return False
    else:
        return True
#get_model_transfer(original=VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))).summary();
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=10,width_shift_range=0.2,
                                    height_shift_range=0.2,horizontal_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 10

train_generator = train_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/train',
                                            target_size=(224, 224), batch_size=train_batchsize, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory('../input/chest-xray-pneumonia/chest_xray/chest_xray/test',
                                target_size=(224, 224), batch_size=val_batchsize, class_mode='categorical', shuffle=False)
F = [1,1,2,3]
n = 3
while (abs(F[n-2]/F[n-3]-F[n]/F[n-1])>10**(-16)):
    u = F[n]+F[n-1]
    F.extend([u])
    n +=1
    
print(F[n]/F[n-1])
start_time = time.time()

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=0.0001, verbose=1,patience=5)
callbacks=[checkpoint,early]
    
steps = (train_generator.samples/(train_generator.batch_size*5))
model = get_model_transfer(original=VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
                          optimizer=optimizers.RMSprop(lr=1e-4))
history = train_model(model, train_generator, validation_generator, epochs=50, steps_per_epoch=steps, my_callbacks=callbacks)
print("Time of trainning is (hh:mm:ss): %s" % time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))


show_accuracy(history, acc='acc', val_acc='val_acc', loss='loss', val_loss='val_loss')
val_generator, predictions = show_predictions(validation_datagen, directory='../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
#binary_predictions = []
#threshold = thresholds[np.argmax(precisions >= 0.85)]
#for i in predictions:
    #if i >= threshold:
        #binary_predictions.append(1)
    #else:
       #binary_predictions.append(0) 


print("[loss,  accuracy] = ",model.evaluate(val_generator))
#print('Accuracy on testing set:', accuracy_score(binary_predictions, ground_truth))
#print('Precision on testing set:', precision_score(binary_predictions, y_test))
#print('Recall on testing set:', recall_score(binary_predictions, y_test))
plot_confusion_matrix(val_generator, predictions)
start_time = time.time() 

checkpoint = ModelCheckpoint("vgg19_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=0.0001, verbose=1,patience=5)
callbacks=[checkpoint,early]    

steps = (train_generator.samples/(train_generator.batch_size*5))
model = get_model_transfer(original=VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
                                  optimizer=optimizers.RMSprop(lr=1e-4))
history = train_model(model, train_generator, validation_generator, epochs=50, steps_per_epoch=steps, my_callbacks=callbacks)
print("Time of trainning is (hh:mm:ss): %s" % time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
show_accuracy(history, acc='acc', val_acc='val_acc', loss='loss', val_loss='val_loss')
val_generator, predictions = show_predictions(validation_datagen, directory='../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
print("[loss,  accuracy] = ",model.evaluate(val_generator))
plot_confusion_matrix(val_generator, predictions)

start_time = time.time() 

checkpoint = ModelCheckpoint("NASNetMobile.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=0.0001, verbose=1,patience=5)
callbacks=[checkpoint,early]    

steps = (train_generator.samples/(train_generator.batch_size*5))
model = get_model_transfer(original=VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
                                  optimizer=optimizers.RMSprop(lr=1e-4))
history = train_model(model, train_generator, validation_generator, epochs=50, steps_per_epoch=steps, my_callbacks=callbacks)
print("Time of trainning is (hh:mm:ss): %s" % time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
show_accuracy(history, acc='acc', val_acc='val_acc', loss='loss', val_loss='val_loss')
val_generator, predictions = show_predictions(validation_datagen, directory='../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
print("[loss,  accuracy] = ",model.evaluate(val_generator))
plot_confusion_matrix(val_generator, predictions)
start_time = time.time() 

checkpoint = ModelCheckpoint("ResNet152V2.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=0.0001, verbose=1,patience=5)
callbacks=[checkpoint,early]    

steps = (train_generator.samples/(train_generator.batch_size*5))
model = get_model_transfer(original=ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
                           optimizer=optimizers.RMSprop(lr=1e-4))
history = train_model(model, train_generator, validation_generator, epochs=50, steps_per_epoch=steps, my_callbacks=callbacks)
print("Time of trainning is (hh:mm:ss): %s" % time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
show_accuracy(history, acc='acc', val_acc='val_acc', loss='loss', val_loss='val_loss')
val_generator, predictions = show_predictions(validation_datagen, directory='../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
print("[loss,  accuracy] = ",model.evaluate(val_generator))
plot_confusion_matrix(val_generator, predictions)
start_time = time.time() 

checkpoint = ModelCheckpoint("InceptionResNetV2.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=0.0001, verbose=1,patience=5)
callbacks=[checkpoint,early]    

steps = (train_generator.samples/(train_generator.batch_size*5))
model = get_model_transfer(original=InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
                           optimizer=optimizers.RMSprop(lr=1e-4), unfreeze_last=10)
history = train_model(model, train_generator, validation_generator, epochs=50, steps_per_epoch=steps, my_callbacks=callbacks)
print("Time of trainning is (hh:mm:ss): %s" % time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
show_accuracy(history, acc='acc', val_acc='val_acc', loss='loss', val_loss='val_loss')
val_generator, predictions = show_predictions(validation_datagen, directory='../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
print("[loss,  accuracy] = ",model.evaluate(val_generator))
plot_confusion_matrix(val_generator, predictions)