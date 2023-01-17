import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pickle
from keras.metrics import categorical_crossentropy
import pandas as pd
from keras.models import Model,load_model
import keras
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Activation,Flatten,MaxPool2D,Conv2D,Dropout
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import itertools
%matplotlib inline
img_width, img_height=224,224
train_data_dir = '../input/datafiles/train/train'
validation_data_dir = '../input/datafiles/validation/validation'
test_data_dir = '../input/datafiles/test/test'
nb_train_samples=125
nb_validation_samples=31
epochs=80
batch_size=64


if K.image_data_format() == 'channel_first':
    input_shape = (3,img_width, img_height)
else:
     input_shape = (img_width, img_height, 3)
        
train_datagen = ImageDataGenerator(rescale= 1. / 255,
                                   rotation_range=90,
                                   shear_range = 0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
        

train_batches = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width,img_height), 
                classes=['dogs','cats'], 
                batch_size=64)

test_batches = ImageDataGenerator().flow_from_directory(
               test_data_dir,
               target_size=(img_width,img_height),
               classes=['dogs','cats'], 
               batch_size=64)

valid_batches = train_datagen.flow_from_directory(
               validation_data_dir, 
               target_size=(img_width,img_height),
               classes=['dogs','cats'],
               batch_size=64)
mobile = keras.applications.mobilenet.MobileNet()
x = mobile.layers[-6].output
predictions = Dense(2, activation='softmax')(x) 
model = Model(inputs = mobile.input, outputs = predictions)
model.summary()
for layer in model.layers[:-5]:
    layer.trainable=False
model.compile(Adam(lr=0.001), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
import os
# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'MobileNet'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
callbacks = [checkpoint]
h = model.fit_generator(train_batches,
                    steps_per_epoch =125,
                    validation_data= valid_batches,
                    validation_steps = 31,
                    epochs = epochs,
                    callbacks=callbacks)
model.save('mobilenet.h5')
model.save_weights('mobilenet.hdf5')
f=open('mobileneth.pckl','wb')
pickle.dump(h.history,f)
f.close()
epoch_nums = range(1, epochs+1)
training_loss = h.history["loss"]
validation_loss = h.history["val_loss"]
plt.plot(epoch_nums , training_loss)
plt.plot(epoch_nums , validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training','validation'], loc='upper right')
plt.show()
bst_val_score = max(hist['val_accuracy'])
bst_val_score
# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
test_imgs,test_labels = next(test_batches)
plots(test_imgs,titles=test_labels)
test_labels = test_labels[:,0]
test_labels

predictions = model.predict_generator(test_batches,steps=1,verbose=0).round()
predictions
from sklearn.metrics import classification_report
print(classification_report(test_labels, predictions[:,0]))

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',):
    
    print('Confusion Matrix')
    print(cm)

    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm = confusion_matrix(test_labels, predictions[:,0])
cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')