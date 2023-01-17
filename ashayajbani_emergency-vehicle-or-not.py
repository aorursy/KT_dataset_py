import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import style
import seaborn as sns
import os

import sklearn
import os 
from PIL import Image
from sklearn.model_selection import train_test_split,StratifiedKFold
  
import os,cv2
from IPython.display import Image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D , BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.initializers import glorot_normal,glorot_uniform, he_normal, he_uniform
from keras.optimizers import Adamax,Adam, Adadelta, Adagrad, RMSprop, Nadam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.regularizers import l2,l1
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.applications.vgg16 import VGG16
from keras.applications import NASNetLarge
from keras import optimizers
test_labels = pd.read_csv("../input/jantahackcomputervision/test_vc2kHdQ.csv")
submission = pd.read_csv("../input/jantahackcomputervision/ss.csv")
train_labels = pd.read_csv("../input/jantahackcomputervision/train_SOaYf6m/train.csv")


train_labels["emergency_or_not"] = train_labels["emergency_or_not"].astype(str)
submission["emergency_or_not"] = submission["emergency_or_not"].astype(str)

random = [ '0' if i%4==0 else '1' for i in range(706)]
submission["emergency_or_not"]  = random
train_labels.head(5)
test_labels.head()
# Looking at class distribution
train_labels["emergency_or_not"].value_counts().plot(kind="bar");

print(train_labels["emergency_or_not"].value_counts())
from PIL import Image
def view_imgs(df,rows,cols):
    IMAGE_DIR ="../input/jantahackcomputervision/train_SOaYf6m/images/"
    axes=[]
    fig=plt.figure(figsize=(20,12))    
    for i in range(rows*cols):
        idx = np.random.randint(len(df), size=1)[0]
        image_name , label = df.loc[idx,"image_names"],df.loc[idx,"emergency_or_not"]
        image = Image.open(IMAGE_DIR+image_name)
        label = "emergency" if label=='1' else "non-emergency"
        axes.append( fig.add_subplot(rows, cols, i+1))
        subplot_title=("Category :"+str(label))
        axes[-1].set_title(subplot_title)  
        plt.imshow(image)
    fig.tight_layout()  
    plt.show()
        
        
view_imgs(train_labels,2,4)
# Shuffling the dataframe
train_labels = train_labels.sample(frac=1).reset_index()

datagen = ImageDataGenerator(rescale=1./255)
batch_size=64
img_dir = "../input/jantahackcomputervision/train_SOaYf6m/images"

train_generator=datagen.flow_from_dataframe(
    dataframe=train_labels[:1318],
    directory=img_dir,
    x_col='image_names',
    class_mode=None,
    batch_size=batch_size,
    target_size=(224,224),
    shuffle=False
)

# class_mode is none since we only need the bottle_neck features 

validation_generator = datagen.flow_from_dataframe(
    dataframe=train_labels[1318:],
    directory=img_dir,x_col='image_names',
    class_mode=None,
    batch_size=32,
    target_size=(224,224),
    shuffle=False
)
vgg16 = VGG16(weights="imagenet",include_top=False,input_shape=(224,224,3))

for layers in vgg16.layers:
    layers.trainable = False
    

# Looking at the model architecture
vgg16.summary()
bottleneck_features_train = vgg16.predict(train_generator,verbose=1)
bottleneck_features_validation = vgg16.predict(validation_generator,verbose=1)
print(bottleneck_features_train.shape,bottleneck_features_validation.shape)
train_data = bottleneck_features_train.copy()
train_class = train_labels.loc[:1317,"emergency_or_not"]

validation_data = bottleneck_features_validation.copy()
validation_class = train_labels.loc[1318:,"emergency_or_not"]

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(2048,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Using checkpoints to save the best model.

from keras.callbacks import ModelCheckpoint
filename = "./bottle_neck_best_wts.hdf5"
checks = ModelCheckpoint(filename,monitor="val_accuracy",verbose=1,
                         save_best_only=True,mode="max",save_weights_only=True)

model.fit(
    train_data, 
    train_class,
    epochs=2,
    batch_size=32,
    validation_data=(validation_data, validation_class),
    callbacks=[checks]
)

# After instantiating the VGG base and loading its weights, we add our previously trained fully-connected classifier on top of it

vgg = VGG16(weights="imagenet",include_top=False,input_shape=(224,224,3))
for layer in vgg.layers[:-4]:
    layer.trainable = False

final_model = Sequential()
final_model.add(vgg)

# Trained Classifier
top_model = Sequential()
top_model.add(Flatten(input_shape=final_model.output_shape[1:]))
top_model.add(Dense(2048,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1,activation="sigmoid"))
top_model.summary()

# note that it is necessary to start with a fully-trained classifier, including the top classifier in order to do fine-tuning

top_model_weights_path = "../input/cv-av-models/bottle_neck_best_wts.hdf5"
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
final_model.add(top_model)
final_model.summary()
final_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
# prepare data augmentation configuration

def return_data_generators(img_width,img_height,train_labels,batch_size=16,class_mode="binary"):
    """
    img_width,img_height : - image dimension to be resized to this sizes during data loading
    
    returns ImageDatagenerators for training and testing purpose
    
    """
    
    
    train_labels = train_labels.sample(frac=1).reset_index()

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)


    img_dir = "../input/jantahackcomputervision/train_SOaYf6m/images"


    train_generator = train_datagen.flow_from_dataframe(dataframe=train_labels[:1318],directory=img_dir,x_col='image_names',
                                                y_col='emergency_or_not',class_mode=class_mode,batch_size=batch_size,
                                                target_size=(img_width,img_height),shuffle=True)

    validation_generator = test_datagen.flow_from_dataframe(dataframe=train_labels[1318:],directory=img_dir,x_col='image_names',
                                                    y_col='emergency_or_not',class_mode=class_mode,batch_size=batch_size,
                                                    target_size=(img_width,img_height),shuffle=True)
    
    test_generator = test_datagen.flow_from_dataframe(dataframe=test_labels,directory=img_dir,x_col='image_names',
                                                class_mode=None,batch_size=batch_size,
                                                target_size=(img_width,img_height),shuffle=False)
    
    return train_generator,validation_generator,test_generator
batch_size = 32
epochs = 30

train_generator,validation_generator,test_generator = return_data_generators(224,224,train_labels,batch_size)

# Checkpoints
filename = "./fine_tuned.hdf5"

checks = ModelCheckpoint(filename,monitor="val_accuracy",verbose=1,
                         save_best_only=True,mode="max")

# fine-tune the model
history_finetune = final_model.fit_generator(train_generator,epochs=epochs,
                              validation_data=validation_generator,
                             callbacks=[checks],verbose=1)
## TRY THIS FUNCTION ONCE

def plot_loss_acc(history,epochs,filename):
    fig = plt.figure(figsize=(14,9))

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
    
    epoch = list(range(1,epochs+1,1))
    losses = history.history["loss"]
    val_losses = history.history["val_loss"]

    accuracies = history.history["accuracy"]
    val_accs = history.history["val_accuracy"]

    
    ax1.plot(epoch, accuracies, label="acc")
    ax1.plot(epoch, val_accs, label="val_accuracy")
    ax1.legend(loc=2)
    ax1.set_ylabel("Accuracy")
    
    ax2.plot(epoch,losses, label="loss")
    ax2.plot(epoch,val_losses, label="val_loss")
    ax2.legend(loc=2)
    ax2.set_xlabel("No of epochs")
    ax2.set_ylabel("Loss")

    
    
    plt.savefig("./"+filename+".png", dpi=300, bbox_inches='tight')
    plt.show()

    
plot_loss_acc(history_finetune,30,"keras_finetune_vgg")
# VGG16 with l2 regularization

# Instantiating the VGG base and loading its weights 

vgg = VGG16(weights="imagenet",include_top=False,input_shape=(224,224,3))
for layer in vgg.layers:
    layer.trainable = False

    
model_l2 = Sequential()
model_l2.add(vgg)

## Trained Classifier
top_model = Sequential()
top_model.add(Flatten(input_shape=model_l2.output_shape[1:]))
top_model.add(Dense(2048,activation='relu',kernel_initializer=glorot_uniform(), kernel_regularizer=l2()))
top_model.add(Dropout(0.5))
top_model.add(Dense(2,activation="softmax",kernel_initializer=glorot_uniform(), kernel_regularizer=l2()))

# top_model_weights_path = "../input/cv-av-models/bottle_neck_best_wts.hdf5"
# top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model_l2.add(top_model)
model_l2.summary()

model_l2.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Checkpoints
filename = "./fine_tuned_l2_augnmentation.hdf5"

checks = ModelCheckpoint(filename,monitor="val_accuracy",verbose=1,
                         save_best_only=True,mode="max")

train_generator,validation_generator,test_generator = return_data_generators(224,224,train_labels,32,"categorical")

# fine-tune the model
history_l2 = model_l2.fit_generator(train_generator,epochs=30,validation_data=validation_generator,verbose=1,callbacks=[checks])
plot_loss_acc(history_l2,30,"vgg_l2")
nasnet = NASNetLarge(weights = "imagenet", input_shape = (331, 331, 3), include_top = False)
for layer in nasnet.layers:
    layer.trainable = False

nasnet_model = Sequential()
 

## Adding a covolutional base with l2 regularzer and FC layer on top of it
nasnet_model.add(nasnet)

nasnet_model.add(Conv2D(1024, (3, 3), activation = "relu"))
nasnet_model.add(BatchNormalization())
nasnet_model.add(MaxPooling2D(2, 2))
 
nasnet_model.add(Flatten())
  
nasnet_model.add(Dense(2048, activation = "relu",kernel_regularizer=l2()))
nasnet_model.add(BatchNormalization())
nasnet_model.add(Dropout(0.5))
 
nasnet_model.add(Dense(256, activation = "relu",kernel_regularizer=l2()))
nasnet_model.add(BatchNormalization()) 
nasnet_model.add(Dropout(0.5))
 
nasnet_model.add(Dense(1, activation = "sigmoid"))
 
nasnet_model.summary()
# Compiling and training the model

# Checkpoints
filename = "./nasnet_l2.hdf5"

train_generator,validation_generator,test_generator = return_data_generators(331,331,train_labels,32)

checks = ModelCheckpoint(filename,monitor="val_accuracy",verbose=1,
                         save_best_only=True,mode="max")

epochs = 20
nasnet_model.compile(loss = 'binary_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy']
              )

history_nasnet = nasnet_model.fit_generator(train_generator,epochs=epochs,
                              validation_data=validation_generator,
                             callbacks=[checks],verbose=1)
plot_loss_acc(history_nasnet,20,"nasnet_l2")
train_dataset = train_labels.copy()
test_dataset = test_labels.copy()

train_image_name = list(train_dataset["image_names"])
test_image_name = list(test_dataset["image_names"])

print(len(train_image_name),len(test_image_name))

train_images = list()
test_images = list()
train_image_res = list()

dirname = "../input/jantahackcomputervision/train_SOaYf6m/images"

for i,filename in enumerate(train_image_name):
    img = cv2.imread(os.path.join(dirname,filename))
    y_val = int(train_dataset[train_dataset.image_names == filename]["emergency_or_not"])
    train_image_res.append(y_val)
    train_images.append(img)
    #print(i+1)


for i,filename in enumerate(test_image_name):
    img = cv2.imread(os.path.join(dirname,filename))
    test_images.append(img)
    #print(i+1)
    

train_images = np.array(train_images)
test_images = np.array(test_images)
train_image_res = np.array(train_image_res)
    
print(train_images.shape,test_images.shape,train_image_res.shape)
# Creating fold_model

fold_model = Sequential()
fold_model.add(Conv2D(32,(3,3),input_shape=(224, 224,3)))
fold_model.add(Activation('relu'))
fold_model.add(MaxPooling2D(pool_size=(2, 2)))

fold_model.add(Conv2D(32,(3,3)))
fold_model.add(Activation('relu'))
fold_model.add(MaxPooling2D(pool_size=(2, 2)))

fold_model.add(Conv2D(128,(3,3)))
fold_model.add(Activation('relu'))
fold_model.add(MaxPooling2D(pool_size=(2, 2)))

fold_model.add(Conv2D(512,(5,5)))
fold_model.add(Activation('relu'))
fold_model.add(MaxPooling2D(pool_size=(2, 2)))

fold_model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
fold_model.add(Dense(512))
fold_model.add(Activation('relu'))
fold_model.add(Dropout(0.5))
fold_model.add(Dense(2))
fold_model.add(Activation('softmax'))

fold_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


filename = "./kfold.hdf5"
checkpoint = ModelCheckpoint(filename, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping_monitor = EarlyStopping(monitor='val_accuracy',patience=7)
callbacks_list = [early_stopping_monitor, checkpoint]

# Training and predictions
K = 5

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state = 7)
new_submission = pd.DataFrame()

X = train_images
Y = train_image_res
X_test = test_images

j = 1

y_valid = None 
X_valid = None

for (train_data_image, test_data_image) in skf.split(X, Y):
    
    y_train, y_valid = Y[train_data_image], Y[test_data_image]
    X_train, X_valid = X[train_data_image], X[test_data_image]
    
    y_train = to_categorical(y_train, num_classes=2)
    y_valid = to_categorical(y_valid, num_classes=2)
    
    fold_model_history = fold_model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                        epochs=25,callbacks=[callbacks_list],verbose=1)
    
    new_submission["predict" + str(j)] = np.argmax(fold_model.predict(X_test,verbose=1), axis = 1)
    
    j = j + 1
submission = pd.DataFrame()
submission["image_names"] = test_image_name
submission["emergency_or_not"] = new_submission.mode(axis=1)
submission.to_csv("./sub.csv",index = False)
submission.head()