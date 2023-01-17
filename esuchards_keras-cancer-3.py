# Import required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import os
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from keras.callbacks import Callback
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_curve,auc, confusion_matrix
print ("Number of train files:",len(os.listdir("../input/train")))
print ("Number of test files:",len(os.listdir("../input/test")))

dftrain=pd.read_csv("../input/train_labels.csv",dtype=str)
dftrain.head()
print("Counts of negative and postive labels in training data:")
dftrain.groupby(['label']).count()
def add_ext(id):
    return id+".tif"

dftrain["id"]=dftrain["id"].apply(add_ext)

def addpath(col):
    return '../input/train/' + col 

dftrain['Path']=dftrain['id'].apply(addpath)
dftrain.head()
## function to plot historgrams

def plothist(plot_img,axnum):
    color = ('b','g','r')
    for j,col in enumerate(color):
         histr = cv2.calcHist([plot_img],[j],None,[256],[0,256])
         ax[axnum,i].plot(histr,color = col)
         ax[axnum,i].set_xlim([0,256])
         ax[axnum,i].set_xlabel("Pixel Values")
         ax[axnum,0].set_ylabel("# of Pixels")
    return 
## print a sample of the images
nums = [76, 46, 69, 20, 17] # random.sample(range(1, 100), 5)
num_pics = len(nums)
f,ax = plt.subplots(3,num_pics,figsize=(15,15))

for i in range(5):
    img = plt.imread(dftrain.iloc[nums[i]]['Path'])
   # ax[i].imshow(img)
   # ax[i].set_title(dfdata.iloc[i]['label'],fontweight="bold", size=20)
    ax[0,i].imshow(img)
    ax[0,i].set_title(dftrain.iloc[i]['label'],fontweight="bold", size=20)
    # Create a Rectangle patch
    rect = patches.Rectangle((32,32),32,32,linewidth=3,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax[0,i].add_patch(rect)
    ## plot histograms of full image and cancer patch
    plothist(img,1)
    plothist(img[32:64, 32:64],2)
    
plt.show() 
## use flow from directory
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.2)
batch_size = 20
image_size = (96,96)

train_generator=datagen.flow_from_dataframe(
dataframe=dftrain,
directory="../input/train/",
x_col="id",
y_col="label",
subset="training",
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode='categorical', #class_mode="binary",
target_size=image_size)

validation_generator=datagen.flow_from_dataframe(
dataframe=dftrain,
directory="../input/train/",
x_col="id",
y_col="label",
subset="validation",
batch_size=batch_size,
seed=42,
shuffle=True,
class_mode='categorical', #class_mode="binary",
target_size=image_size)
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()

model.compile('Adam', loss = "binary_crossentropy", metrics=["accuracy"])
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.3


model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(MaxPooling2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(2, activation = "softmax"))

model.summary()

model.compile(Adam(0.0001), loss = "binary_crossentropy", metrics=["accuracy"])

trainstep=train_generator.n//train_generator.batch_size
valstep=validation_generator.n//validation_generator.batch_size

filepath="weights-best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=trainstep,
                    validation_data=validation_generator,
                    validation_steps=valstep,
                    epochs=20,
                    callbacks=[checkpoint]
)
# plot learning curves
filepath="weights-best.hdf5"
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
## Create test generator and evaluate model 

model.load_weights(filepath) #load saved weights
test_datagen=ImageDataGenerator(rescale=1./255)

test_generator=datagen.flow_from_dataframe(
dataframe=dftrain,
directory="../input/train/",
x_col="id",
y_col="label",
subset="validation",
batch_size=5,   # want to divide num samples evenly 
seed=42,
shuffle=False,  # don't shuffle
class_mode='categorical', #class_mode="binary",
target_size=image_size)

scores = model.evaluate_generator(test_generator)
print('Test loss:', round(100*scores[0],2))
print('Test accuracy:', round(100*scores[1],2))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

test_labels = test_generator.classes
y_preds = model.predict_generator(test_generator,verbose=1,steps=test_generator.n/5)

y_pred_keras=np.argmax(y_preds, axis=-1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
print('AUC score :', + auc_keras)

from sklearn.metrics import classification_report
print(classification_report(test_labels, y_pred_keras))
# plot ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
classes=list((test_generator.class_indices).values())
cm=confusion_matrix(test_labels,y_pred_keras)

plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.imshow(cm, cmap=plt.cm.Blues)
print(cm)
len(test_generator.filenames)
test_frame=pd.DataFrame({'id':(test_generator.filenames)})
test_frame['true_label']=test_labels
test_frame['predicted'] = y_pred_keras
test_frame['no_cancer'] = y_preds[:,0]
test_frame['cancer'] = y_preds[:,1]
test_frame['Path']=test_frame['id'].apply(addpath)
incorrect_preds=(test_frame[test_frame.true_label != test_frame.predicted]).head()
no_cancer=incorrect_preds.nlargest(3, columns='no_cancer')
no_cancer
cancer=(incorrect_preds.nlargest(3, columns='cancer'))[(incorrect_preds.true_label==0)]
cancer_list=cancer['id'].values.tolist()
cancer
## plot the "most incorrect" images

f,ax = plt.subplots(1,2,figsize=(15,15))

for i in range(len(cancer)):
    img = plt.imread(cancer.iloc[i]['Path'])
    ax[i].imshow(img)
    ax[i].set_title(cancer.iloc[i]['true_label'],fontweight="bold", size=20)  
    
plt.show() 

f,ax = plt.subplots(1,2,figsize=(15,15))

for i in range(2):
    img = plt.imread(no_cancer.iloc[i]['Path'])
    ax[i].imshow(img)
    ax[i].set_title(no_cancer.iloc[i]['true_label'],fontweight="bold", size=20)  

## look at the "most correct" submission
no_cancer_true=test_frame.nlargest(3, columns='no_cancer')
cancer_true=test_frame.nlargest(3, columns='cancer')
no_cancer_true.head()
cancer_true.head()
## plot the "most incorrect" images

f,ax = plt.subplots(1,3,figsize=(15,15))

for i in range(3):
    img = plt.imread(cancer_true.iloc[i]['Path'])
    ax[i].imshow(img)
    ax[i].set_title(cancer_true.iloc[i]['true_label'],fontweight="bold", size=20)  
    
plt.show() 

f,ax = plt.subplots(1,3,figsize=(15,15))

for i in range(3):
    img = plt.imread(no_cancer_true.iloc[i]['Path'])
    ax[i].imshow(img)
    ax[i].set_title(no_cancer_true.iloc[i]['true_label'],fontweight="bold", size=20)  

test_results=pd.DataFrame({'id':os.listdir("../input/test/")})
test_datagen=ImageDataGenerator(rescale=1./255)

submit_generator=datagen.flow_from_dataframe(
dataframe=test_results,
directory="../input/test/",
x_col="id",
batch_size=2,   # want to divide num samples evenly 
shuffle=False,  # don't shuffle
class_mode=None,
target_size=image_size)
## use 0.5 as threshold to assign to class 0 or 1 
y_test_prob=model.predict_generator(submit_generator,verbose=1,steps=submit_generator.n/2)
y_test_pred=np.argmax(y_test_prob, axis=-1)  #y_test_prob.round()
results = pd.DataFrame({'id':(submit_generator.filenames)})

def remove_ext(id):
    return (id.split('.'))[0]
results['id']=results['id'].apply(remove_ext)
results['label'] = y_test_pred
results.to_csv("submission.csv",index=False)
results.head()