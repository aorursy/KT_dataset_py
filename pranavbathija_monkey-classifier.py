import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.applications import InceptionV3,Xception
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense,Input,Flatten,Dropout
from keras.applications.inception_v3 import preprocess_input
from keras import models, layers, optimizers
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2

cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
labels = pd.read_csv("../input/10-monkey-species/monkey_labels.txt", names=cols, skiprows=1)
labels
img_h=150
img_w=150
img_c=3
bs=32

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen=train_datagen.flow_from_directory(
    '../input/10-monkey-species/training/training/',
    target_size=(img_h,img_w),
    batch_size=bs,
    class_mode='categorical',
    seed=1
)
val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen=val_datagen.flow_from_directory(
    '../input/10-monkey-species/validation/validation',
    target_size=(img_h,img_w),
    batch_size=bs,
    class_mode='categorical',
    seed=1
)
# next(val_gen)[0].shape
data_len=train_gen.samples
base_model=InceptionV3(weights='imagenet',include_top=False)

base_model.summary()
for layer in base_model.layers:
        layer.trainable = False
X=base_model.output
X = layers.GlobalAveragePooling2D()(X)
X=Dense(512,activation='relu')(X)
X=Dropout(0.5)(X)
X=Dense(512,activation='relu')(X)
X=Dropout(0.5)(X)
out=Dense(10,activation='softmax')(X)
model=Model(inputs=base_model.input,outputs=out)
model.summary()
callbacks=[
    EarlyStopping(
        patience=4,
        monitor='val_accuracy',
    ),
    
    ReduceLROnPlateau(monitor='loss',
                     factor=0.1,
                     patience=2,
                     cooldown=2,
                     verbose=1)
]
batches_per_epoch = data_len//bs
lr_decay = (1./0.8 -1)/batches_per_epoch
opt = Adam(lr=1e-3,beta_1=0.9,beta_2=0.999,decay=lr_decay)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit_generator(train_gen,
        epochs=15,
        validation_data=val_gen,
        steps_per_epoch=batches_per_epoch,                    
        callbacks=callbacks)
train_gen1=train_datagen.flow_from_directory(
    '../input/10-monkey-species/training/training/',
    target_size=(280,280),
    batch_size=bs,
    class_mode='categorical',
    seed=1
)
val_gen1=val_datagen.flow_from_directory(
    '../input/10-monkey-species/validation/validation',
    target_size=(280,280),
    batch_size=bs,
    class_mode='categorical',
    seed=1
)
history=model.fit_generator(train_gen1,
        epochs=15,
        validation_data=val_gen1,
        steps_per_epoch=batches_per_epoch,                    
        callbacks=callbacks)
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.title('Training and val accuracy')
plt.plot(epochs,acc,'blue',label='Training acc')
plt.plot(epochs,val_acc,'red',label='val acc')
plt.legend()

plt.figure()
plt.title('Training and val losses')
plt.plot(epochs,loss,'blue',label='Training loss')
plt.plot(epochs,val_loss,'red',label='val loss')
plt.legend()
plt.show()
temp_val_gen=val_datagen.flow_from_directory(
    '../input/10-monkey-species/validation/validation',
    target_size=(280,280),
    batch_size=272,
    class_mode='categorical',
    seed=1
)
    
x_val,y_true=next(temp_val_gen)
y_pred=model.predict(x_val)
y_pred.shape
# predictions = [i.argmax() for i in y_pred]
# y_true = [i.argmax() for i in test_labels]
# predictions
y_pred=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_true,axis=1)
def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)
        cm = np.round(cm,2)
        

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass))
    plt.show()
lbls = labels['Common Name']
cm = confusion_matrix(y_pred, y_true)
plot_confusion_matrix(cm, normalize=True, target_names=lbls)
from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average=None)
cm
