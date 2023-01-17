!pip install -q efficientnet
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import sys
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
import scikitplot as skplt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,precision_recall_curve
import efficientnet.tfkeras as efn
dataset = pd.read_csv('../input/resized-2015-2019-blindness-detection-images/labels/trainLabels19.csv')

dataset
names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Proliferate DR']
print(dataset['diagnosis'].value_counts())
sns.barplot(x=names,y=dataset.diagnosis.value_counts().sort_index())
dataset1 = pd.read_csv('../input/resized-2015-2019-blindness-detection-images/labels/trainLabels15.csv')
dataset1.columns = ['id_code','diagnosis']
dataset1
print(dataset1['diagnosis'].value_counts())
sns.barplot(x=names,y=dataset1.diagnosis.value_counts().sort_index())
#Now we will take 900 images in total for each class. So to complete the 900 images we will take majority of images from 'dataset' 
#and if necessary take the rest of the required images from 'dataset1'

#index  Final_Img_count   Image taken from dataset 1
# 0          900                   (0)
# 1          900                 (530)
# 2          900                   (0)
# 3          900                 (707)
# 4          900                 (605)


level_1 = dataset1[dataset1.diagnosis == 1].sample(n=530)

level_3 = dataset1[dataset1.diagnosis == 3].sample(n=707)

level_4 = dataset1[dataset1.diagnosis == 4].sample(n=605)
level_1.shape , level_3.shape, level_4.shape
level_0 = dataset[dataset.diagnosis == 0].sample(n=900)
level_0
level_2 = dataset[dataset.diagnosis == 2].sample(n=900)
level_2
dataset= dataset[dataset['diagnosis']>0]
dataset= dataset[dataset['diagnosis'] != 2]
print(dataset['diagnosis'].value_counts())
dataset = pd.concat([level_0,level_2,dataset])
dataset=dataset.sample(frac=1)
print(dataset['diagnosis'].value_counts())
dataset
dataset1 = pd.concat([level_1,level_3, level_4])
dataset1=dataset1.sample(frac=1)

print(dataset1['diagnosis'].value_counts())
dataset1
images = []
for i, image_id in enumerate(tqdm(dataset.id_code)):
    im = cv2.imread(f'../input/resized-2015-2019-blindness-detection-images/resized train 19/{image_id}.jpg')
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (128, 128))
    images.append(im)

images
for i, image_id in enumerate(tqdm(dataset1.id_code)):
    im = cv2.imread(f'../input/resized-2015-2019-blindness-detection-images/resized train 15/{image_id}.jpg')
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (128, 128))
    images.append(im)

images
# random image from imported data
plt.imshow(images[-30])
plt.show()
# This function will act as a filter for the image data

def load_colorfilter(image, sigmaX=10):
    #image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = crop_image_from_gray(image)
    #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX),-4 ,128)
    return image
for i in range(len(images)):
    output = load_colorfilter(images[i])
    images[i] = output
# image after filtering
plt.imshow(images[-30])
plt.show()
images = np.array(images)
images.shape
dataset = pd.concat([dataset,dataset1])
print(dataset['diagnosis'].value_counts())

sns.barplot(x=names,y=dataset.diagnosis.value_counts().sort_index())
X = images/255.0
y = dataset.diagnosis.values
X, y
# Cleaning some RAM memory space
del images,level_1,level_3, level_4, level_0, dataset1
# Applying image augmentation
sys.stdout.flush()
aug = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.2, \
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,\
    horizontal_flip=True, fill_mode="nearest")
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#X_train.shape, X_test.shape, y_train.shape, y_test.shape

#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1)#
#X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
#######################################################################################

X, X_test, y, y_test = train_test_split(X,y,test_size=0.1, stratify = y)
X.shape, X_test.shape, y.shape, y_test.shape
# Function defined to plot the curves during training

def display_training_curves(training, validation, title, subplot):
    
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
    plt.show()
BS = 32       #Batch size
accuracy = []

############ USING STRATIFIED K-FOLD CROSS VALIDATION TECHNIQUE ##########

skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X,y)

for train, test in skf.split(X,y):
    model2 = tf.keras.Sequential([
        efn.EfficientNetB5(
            input_shape=(128,128, 3),
            weights='imagenet',
            include_top=False
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # Compiling the model
    model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001),loss='sparse_categorical_crossentropy',metrics=['acc'])
    
    # Training
    history = model2.fit_generator(aug.flow(X[train], y[train], batch_size=BS),
    validation_data=(X[test], y[test]),
    epochs=80, verbose = 1)

    # Evaluate score
    acc=model2.evaluate(X[test], y[test])
    accuracy.append(acc[1])
    
    # Plotting traning curves
    display_training_curves(
    history.history['loss'], 
    history.history['val_loss'], 
    'loss', 211)
    
    display_training_curves(
    history.history['acc'], 
    history.history['val_acc'], 
    'accuracy', 212)
# we can see the minimum and maximum validation accuracy received after training on the training dataset
accuracy
# thus we can assume the mean accuracy of the model on the training set to be:
a=sum(accuracy)/len(accuracy)
print(f'Mean evaluated accuracy of model : {a}')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model2).create(prog='dot', format='svg'))
#predicting training labels
y_train_pred = model2.predict_classes(X)

#Accuracy of train prediction
print('\nAccuracy of training data prediction : {:.2f}\n'.format(accuracy_score(y, y_train_pred)))

#confusion matrix for training set
confusion = confusion_matrix(y, y_train_pred)
print('Confusion Matrix of training data prediction \n')
print(confusion)
# Visualizing confusion matrix for train data
skplt.metrics.plot_confusion_matrix(y, y_train_pred, figsize=(8, 8))
plt.show()
#Classification report 
print('\nClassification Report of training set : \n')
print(classification_report(y, y_train_pred, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate DR']))
y_pred = model2.predict_classes(X_test)
y_pred
y_test
# Accuracy of test prediction
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
# Confusion matrix of the test data
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)
# Visualizing confusion matrix for test data
skplt.metrics.plot_confusion_matrix(y_test, y_pred, figsize=(8, 8))
plt.show()
#Classification report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate DR']))