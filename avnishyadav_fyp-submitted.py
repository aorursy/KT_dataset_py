#!pip install tensorflow==2.0.0-beta1
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

%matplotlib inline

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import math
import seaborn as sns
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, Input, Softmax, Add, Activation

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
!pip install split-folders tqdm
import split_folders
split_folders.ratio('../input/', output="/content/output", seed=1337, ratio=(.7, .15, .15)) # default values
len(os.listdir('/content/output/test/NOR'))
import os
import shutil
from random import sample

#dest = '/content/output/test/NOR/'
files = os.listdir('/content/output/train/NOR')
for file in sample(files,45000):
    os.remove('/content/output/train/NOR/'+file)
    #shutil.move('/content/output/train/NOR/'+file, dest + file)
    
files = os.listdir('/content/output/test/NOR')
for file in sample(files,45000):
    os.remove('/content/output/test/NOR/'+file)

files = os.listdir('/content/output/val/NOR')
for file in sample(files,9000):
    os.remove('/content/output/val/NOR/'+file)
    
files = os.listdir('/content/output/test/NOR')
for file in sample(files,9000):
    os.remove('/content/output/test/NOR/'+file)
PATH = os.path.join(os.path.dirname('/content/images1'))

train_dir = os.path.join(PATH, '/content/output/train')
validation_dir = os.path.join(PATH, '/content/output/val')
test_dir = os.path.join(PATH, '/content/output/test')
train_apc_dir = os.path.join(train_dir, 'APC')  
train_lbb_dir = os.path.join(train_dir, 'LBB')  
train_nor_dir = os.path.join(train_dir, 'NOR')  
train_pab_dir = os.path.join(train_dir, 'PAB')  
train_pvc_dir = os.path.join(train_dir, 'PVC')  
train_rbb_dir = os.path.join(train_dir, 'RBB')  
train_veb_dir = os.path.join(train_dir, 'VEB')  
train_vfw_dir = os.path.join(train_dir, 'VFW')  

validation_apc_dir = os.path.join(validation_dir, 'APC')  
validation_lbb_dir = os.path.join(validation_dir, 'LBB')  
validation_nor_dir = os.path.join(validation_dir, 'NOR')  
validation_pab_dir = os.path.join(validation_dir, 'PAB')  
validation_pvc_dir = os.path.join(validation_dir, 'PVC')  
validation_rbb_dir = os.path.join(validation_dir, 'RBB')  
validation_veb_dir = os.path.join(validation_dir, 'VEB')  
validation_vfw_dir = os.path.join(validation_dir, 'VFW')  


num_apc_tr = len(os.listdir('/content/output/train/APC'))
num_lbb_tr = len(os.listdir('/content/output/train/LBB'))
num_nor_tr = len(os.listdir('/content/output/train/NOR'))
#num_pab_tr = len(os.listdir('/content/output/train/PAB'))
num_pvc_tr = len(os.listdir('/content/output/train/PVC'))
num_rbb_tr = len(os.listdir('/content/output/train/RBB'))
#num_veb_tr = len(os.listdir('/content/output/train/VEB'))
#num_vfw_tr = len(os.listdir('/content/output/train/VFW'))

num_apc_val = len(os.listdir('/content/output/val/APC'))
num_lbb_val = len(os.listdir('/content/output/val/LBB'))
num_nor_val = len(os.listdir('/content/output/val/NOR'))
#num_pab_val = len(os.listdir('/content/output/val/PAB'))
num_pvc_val = len(os.listdir('/content/output/val/PVC'))
num_rbb_val = len(os.listdir('/content/output/val/RBB'))
#num_veb_val = len(os.listdir('/content/output/val/VEB'))
#num_vfw_val = len(os.listdir('/content/output/val/VFW'))


total_train = num_apc_tr + num_lbb_tr + num_nor_tr  + num_pvc_tr + num_rbb_tr  
total_val = num_apc_val + num_lbb_val + num_nor_val + num_pvc_val + num_rbb_val  

print(total_train)
batch_size = 256
epochs = 5
IMG_HEIGHT = 128
IMG_WIDTH = 128

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           color_mode="grayscale",
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')
print(train_data_gen.classes[10000:10010])
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              color_mode="grayscale",
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           color_mode="grayscale",
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')
model = Sequential([
    Conv2D(128, (3,3),strides = (1,1), activation='elu', input_shape = (IMG_HEIGHT, IMG_WIDTH ,1),kernel_initializer='glorot_uniform'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='elu', strides = (1,1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides= (2,2)),
    Conv2D(128, (3,3), activation='elu', strides = (1,1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides= (2,2)),
    Conv2D(128, (3,3), activation='elu', strides = (1,1)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='elu', strides = (1,1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), strides= (2,2)),
    Conv2D(32, (3,3), activation='elu', strides = (1,1)),
    BatchNormalization(),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='elu'),
    Dense(5, activation='softmax')
])

def exp_decay(epoch):
    initial_lrate = 0.0008
    k = 0.75
    t = total_train//(10000 * batch_size)  # every epoch we do n_obs/batch_size iteration
    lrate = initial_lrate * math.exp(-k*t)
    return lrate

lrate = LearningRateScheduler(exp_decay)
adam = Adam(lr = 0.0008, beta_1 = 0.9, beta_2 = 0.999)
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = adam,
    metrics = ['accuracy']
)
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True, dpi = 60)
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 2,
                                             restore_best_weights=True)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[cp_callback, early_stop]
)
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
# Python script for confusion matrix creation. 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
y_test_preds = model.predict_generator(generator = test_data_gen)
y_test_preds = [np.argmax(i) for i in y_test_preds ]
y_test = test_data_gen.classes
#y_test = list(test_data_gen.class_indices.keys())   


results = confusion_matrix(y_test, y_test_preds) 
  
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test, y_test_preds))
print ('Report : ')
print (classification_report(y_test, y_test_preds))
def plot_cm(y_true, y_pred, figsize=(10,10)):
    label = list(test_data_gen.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    #cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm = pd.DataFrame(cm_perc, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    ax.xaxis.set_ticklabels(label);
    ax.yaxis.set_ticklabels(label);
plot_cm(y_test, y_test_preds)
model.save('2DCNN_without_stft.h5')
from tensorflow.keras.models import load_model
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = load_model('/content/model.h5')
#model.summary()

  
# open method used to open different extension image file 
#im = plt.imread('/content/images1/VFW/VFW102100641.png') 

test_loss, test_acc = model.evaluate(test_data_gen, verbose=2)
#Removing any folders
!rm -rf /content/output/
y = model.predict_generator(generator = test_data_gen)
test_steps_per_epoch = np.math.ceil(test_data_gen.samples / test_data_gen.batch_size)

predictions = model.predict_generator(test_data_gen, steps=test_steps_per_epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data_gen.classes
class_labels = list(test_data_gen.class_indices.keys())   
#!pip install scipy
from sklearn import preprocessing, metrics
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   
