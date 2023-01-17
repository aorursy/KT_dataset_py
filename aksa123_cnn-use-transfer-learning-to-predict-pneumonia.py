# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input/chest-xray-pneumonia/chest_xray'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.layers import Input, Lambda, Dense, Flatten,Dropout
from keras.models import Model
import tensorflow as tf
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt #for plotting things
import cv2
train_n = "../input/chest-xray-pneumonia/chest_xray/train/NORMAL/"
train_p = "../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/"
#Normal pic 
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)

norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))

sic_pic =  os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = cv2.imread(norm_pic_address)
sic_load = cv2.imread(sic_address)

#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
IMAGE_SIZE = [224, 224]

train_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
valid_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/val'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False

  # useful for getting number of classes
folders = glob('/kaggle/input/chest-xray-pneumonia/chest_xray/train/*')
folders
x = Flatten()(vgg.output)
x = Dense(400, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(400, activation="relu")(x)
x = Dropout(0.5)(x)
prediction = Dense(1, activation='sigmoid')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()
# tell the model what cost and optimization method to use
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(
  loss='binary_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

val_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
val_set = val_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/val',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary', shuffle = False)
filepath="model.{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy',save_weights_only=False, verbose=1, save_best_only=False, mode='max')
r = model.fit_generator(
  training_set,
  validation_data=val_set,
  epochs=5,
  verbose =1,
  steps_per_epoch=len(training_set),
  validation_steps=len(val_set),
  callbacks = [checkpoint]
)
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
import tensorflow as tf

from keras.models import load_model

best_model = load_model('./model.04-0.17.h5')

### START CODE HERE ### (1 line)
preds = best_model.evaluate_generator(test_set, steps = 624)
### END CODE HERE ###
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
test_set.reset()
Y_pred = best_model.predict_generator(test_set, (624//32)+1)
predict_labels = Y_pred
predict_labels[predict_labels<0.5] = 0
predict_labels[predict_labels>=0.5] = 1
predict_labels
orig_labels = test_set.labels
predict_labels.shape, orig_labels.shape
a = predict_labels.reshape(-1)
fnames = test_set.filenames
path = '../input/chest-xray-pneumonia/chest_xray/test/'
ind_misclassified = np.where(a!=orig_labels)[0][0:5]
for ind in ind_misclassified:
    print("Original Label: " + str(orig_labels[ind]))
    print("Predicted Label: " + str(a[ind]))
    print("Filename: " + fnames[ind])
    img = cv2.imread(path+fnames[ind])
    plt.figure()
    plt.imshow(img)
    plt.title(fnames[ind])
    plt.show()
cm  = confusion_matrix(orig_labels, predict_labels)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()
tn, fp, fn, tp = cm.ravel()
print(tn,fp, fn, tp)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
acc = (tp+tn)/(tn+fp+fn+tp)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
print("Accuracy of the model is {:.2f}".format(acc))
predicted_class_indices=np.argmax(Y_pred,axis=1)
labels = (test_set.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
predictions = [labels2[k] for k in predicted_class_indices]
print(predicted_class_indices)
print (labels2)
print (predictions)