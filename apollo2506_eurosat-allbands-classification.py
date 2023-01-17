import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, cv2, json, random, itertools, rasterio

from tqdm import tqdm
from IPython.display import SVG
from tensorflow.keras.utils import plot_model, model_to_dot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical, Sequence
from sklearn.metrics import f1_score

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Add, Input, Conv2D, Dropout, Activation, BatchNormalization, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Flatten, Dense)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import *

from tensorflow.keras.preprocessing.image import img_to_array, load_img
def show_final_history(history):
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    
    ax[0].set_title('Loss')
    ax[1].set_title('Accuracy')
    
    ax[0].plot(history.history['loss'], 'r-', label='Training Loss')
    ax[0].plot(history.history['val_loss'], 'g-', label='Validation Loss')
    ax[1].plot(history.history['accuracy'], 'r-', label='Training Accuracy')
    ax[1].plot(history.history['val_accuracy'], 'g-', label='Validation Accuracy')
    
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='lower right')
    
    plt.show();
    pass
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm.max()/2.0
    
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        
        plt.text(j,i, format(cm[i,j], fmt),
                horizontalalignment = 'center',
                color = "white" if cm[i,j] > thresh else "black")
        pass
    
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.grid(False);
    pass
with open("../input/eurosat-dataset/EuroSATallBands/label_map.json","r") as f:
    class_names_encoded = json.load(f)
    pass

class_names = list(class_names_encoded.keys())
num_classes = len(class_names)
class_names_encoded
basePath = "../input/eurosat-dataset/EuroSATallBands"

def data_generator(csv_file, num_classes, batch_size = 10, target_size = 64):
    
    df = pd.read_csv(csv_file)
    df.drop(columns=df.columns[0], inplace = True)
    num_samples = df.shape[0]
    
    while True:
        
        for offset in range(0, num_samples, batch_size):
            batch_samples_idx = df.index[offset:offset+batch_size]
            
            X, y = [], []
            
            for i in batch_samples_idx:
                img_name = df.loc[i,'Filename']
                label = df.loc[i,'Label']
                
                src = rasterio.open(os.path.join(basePath,img_name))
                arr_b, arr_g, arr_r, arr_nir = src.read(2),src.read(3), src.read(4), src.read(8) 
                
                arr_b, arr_g   = np.array(arr_b, dtype=np.float32), np.array(arr_g, dtype=np.float32)
                arr_r, arr_nir = np.array(arr_r, dtype=np.float32), np.array(arr_nir, dtype=np.float32)
                
                arr_b = (arr_b - arr_b.min())/(arr_b.max() - arr_b.min())
                arr_g = (arr_g - arr_g.min())/(arr_g.max() - arr_g.min())
                arr_r = (arr_r - arr_r.min())/(arr_r.max() - arr_r.min())
                arr_nir = (arr_nir - arr_nir.min())/(arr_nir.max() - arr_nir.min())
                
                rgb_nir = np.dstack((arr_r,arr_g,arr_b,arr_nir))
                
                X.append(rgb_nir)
                y.append(label)
                pass
            
            X = np.array(X)
            y = np.array(y)
            y = to_categorical(y, num_classes = num_classes)
            
            yield X, y
            pass
        pass
    pass
train_generator = data_generator(csv_file = "../input/eurosat-dataset/EuroSATallBands/train.csv", num_classes = 10, batch_size = 10)
val_generator = data_generator(csv_file = "../input/eurosat-dataset/EuroSATallBands/validation.csv", num_classes = 10, batch_size = 10)
def conv_block(X,k,filters,stage,block,s=2):
    
    conv_base_name = 'conv_' + str(stage) + block + '_branch'
    bn_base_name = 'bn_' + str(stage) + block + '_branch'
    ac_base_name = 'ac_' + str(stage) + block + '_branch'
    
    F1 = filters
    
    X = Conv2D(filters=F1, kernel_size=(k,k), strides=(s,s),
              padding='same', name=conv_base_name+'2a')(X)
    X = BatchNormalization(name=bn_base_name+'2a')(X)
    X = Activation("relu", name=ac_base_name+'2a')(X)
    
    return X
    pass
def conv_model(input_shape, classes):
    
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((5,5),name="zero_padding_1")(X_input)
    
    X = Conv2D(16,(3,3), strides=(2,2), name='conv1', padding="same")(X)
    X = BatchNormalization(name="bn_conv1")(X)
    
    X = conv_block(X,3,32,2,block='A',s=1)
    X = MaxPooling2D((2,2), name="max_pooling_2")(X)
    X = Dropout(0.25, name="dropout_2")(X)
    
    X = conv_block(X,5,32,3,block='A',s=2)
    X = MaxPooling2D((3,3), name="max_pooling_3")(X)
    X = Dropout(0.5, name="dropout_3")(X)
    
    X = conv_block(X,3,64,4,block='A',s=1)
    X = MaxPooling2D((2,2), name="max_pooling_4")(X)
    X = Dropout(0.25, name="dropout_4")(X)
    
    X = Flatten(name="flatten_1")(X)
    X = Dense(64,name="dense_1")(X)
    X = Activation("relu",name="dense_relu_1")(X)
    X = Dropout(0.25,name="dense_dropout_1")(X)
    
    X = Dense(128,name="dense_2")(X)
    X = Activation("relu", name="dense_relu_2")(X)
    
    X = Dense(classes,activation="softmax",name="fc"+str(classes))(X)
    
    model = Model(inputs=X_input, outputs=X, name="Basic_Conv_model")
    
    return model
    pass
model = conv_model(input_shape = (64,64,4), classes = 10)
plot_model(model, to_file="allbands.png",show_shapes=True,show_layer_names=True)
SVG(model_to_dot(model).create(prog="dot", format='svg'))
model.summary()
checkpoint = ModelCheckpoint("allbands_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
logs = TensorBoard("logs")
train_df = pd.read_csv("../input/eurosat-dataset/EuroSATallBands/train.csv")
train_labels = train_df.loc[:,'Label']
train_labels = np.array(train_labels)

num_train_samples = train_labels.shape[0]

val_df = pd.read_csv("../input/eurosat-dataset/EuroSATallBands/validation.csv")
val_labels = val_df.loc[:,'Label']
val_labels = np.array(val_labels)

num_val_samples = val_labels.shape[0]

num_train_samples, num_val_samples
train_labels_encoded = to_categorical(train_labels,num_classes=10)

classTotals = train_labels_encoded.sum(axis=0)
classWeight = {}

for i in range(len(classTotals)):
    classWeight[i] = classTotals[i]/classTotals.max()
    pass

classWeight
opt = Adam(lr=1e-5)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights("../input/eurosat-allbands-classification/allbands_weights.h5")
epochs = 30
batchSize = 100

history = model.fit(train_generator,
                   steps_per_epoch = num_train_samples//batchSize,
                   epochs = epochs,
                   verbose = 1,
                   validation_data = val_generator,
                   validation_steps = num_val_samples//batchSize,
                   callbacks = [checkpoint, logs],
                   class_weight = classWeight
                   )
show_final_history(history)
def obtain_tif_images(csv_file):
    
    df = pd.read_csv(csv_file)
    df.drop(columns=df.columns[0], inplace=True)
    num_samples = df.shape[0]
    
    X, y = [], []
    
    for i in tqdm(range(num_samples)):
        
        label = df.loc[i,'Label']
        tif_name = df.loc[i,'Filename']
        tif_img = rasterio.open(os.path.join(basePath, tif_name))
        
        arr_b, arr_g, arr_r, arr_nir = tif_img.read(2),tif_img.read(3), tif_img.read(4), tif_img.read(8) 

        arr_b, arr_g   = np.array(arr_b, dtype=np.float32), np.array(arr_g, dtype=np.float32)
        arr_r, arr_nir = np.array(arr_r, dtype=np.float32), np.array(arr_nir, dtype=np.float32)

        arr_b = (arr_b - arr_b.min())/(arr_b.max() - arr_b.min())
        arr_g = (arr_g - arr_g.min())/(arr_g.max() - arr_g.min())
        arr_r = (arr_r - arr_r.min())/(arr_r.max() - arr_r.min())
        arr_nir = (arr_nir - arr_nir.min())/(arr_nir.max() - arr_nir.min())

        rgb_nir = np.dstack((arr_r,arr_g,arr_b,arr_nir))

        X.append(rgb_nir)
        y.append(label)
        
        pass
    
    X = np.array(X)
    y = np.array(y)
    
    return X,y
    pass
test_tifs, test_labels = obtain_tif_images(csv_file="../input/eurosat-dataset/EuroSATallBands/test.csv")

test_labels_encoded = to_categorical(test_labels, num_classes = len(class_names))

test_tifs.shape, test_labels.shape, test_labels_encoded.shape
test_pred = model.predict(test_tifs)
test_pred = np.argmax(test_pred, axis=1)
test_pred.shape
cnf_mat = confusion_matrix(test_labels, test_pred)

plt.figure()
plot_confusion_matrix(cnf_mat, classes=class_names)
plt.grid(False);
plt.show();
model.save("allbands_v2.h5")
model_test = load_model("./allbands_v2.h5")

model_test.summary()
model_test.load_weights("./allbands_weights.h5")
test_pred_2 = model.predict(test_tifs)
test_pred_2 = np.argmax(test_pred_2, axis=1)
test_pred_2.shape
for f1,class_name in zip(f1_score(test_labels, test_pred_2, average=None), class_names):
    print("Class name: {}, F1 score: {:.3f}".format(class_name, f1))
    pass
cnf_mat = confusion_matrix(test_labels, test_pred_2)

plt.figure()
plot_confusion_matrix(cnf_mat, classes=class_names)
plt.grid(False);
plt.show();
val_tifs, val_labels = obtain_tif_images(csv_file="../input/eurosat-dataset/EuroSATallBands/validation.csv")

val_labels_encoded = to_categorical(val_labels, num_classes = len(class_names))

val_tifs.shape, val_labels.shape, val_labels_encoded.shape
val_pred = model.predict(val_tifs)
val_pred = np.argmax(val_pred, axis=1)
val_pred.shape
for f1,class_name in zip(f1_score(val_labels, val_pred, average=None), class_names):
    print("Class name: {}, F1 score: {:.3f}".format(class_name, f1))
    pass
cnf_mat = confusion_matrix(val_labels, val_pred)

plt.figure()
plot_confusion_matrix(cnf_mat, classes=class_names)
plt.grid(False);
plt.show();