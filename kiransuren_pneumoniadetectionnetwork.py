import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import IPython.display as display
import PIL
import os
import glob
#Get all relevant directories
dataset_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/"
train_dir = dataset_dir + "train/"
val_dir = dataset_dir + "val/"
test_dir = dataset_dir + "test/"

#Get list of directories to all images in training, validation and testing samples
TRAIN_SAMPLES = glob.glob(train_dir+"*/*.jpeg")
VAL_SAMPLES = glob.glob(val_dir+"*/*.jpeg")
TEST_SAMPLES = glob.glob(test_dir+"*/*.jpeg")

#Find number of training, validation and testing samples
TRAIN_NUM = len(glob.glob(train_dir+"*/*.jpeg"))
VAL_NUM = len(glob.glob(val_dir+"*/*.jpeg"))
TEST_NUM = len(glob.glob(test_dir+"*/*.jpeg"))
#Find average height and width of all images in dataset
tW = 0
tH = 0
for i in glob.glob(dataset_dir+"*/*/*.jpeg"):
    image = PIL.Image.open(i)
    width,height = image.size
    tW += width
    tH += height
avg_width = round(tW/(TRAIN_NUM+VAL_NUM+TEST_NUM))
avg_height = round(tH/(TRAIN_NUM+VAL_NUM+TEST_NUM))
print("Average Width of Images", avg_width)
print("Average Heigth of Images", avg_height)
#Further dimension reduction of images
avg_width = round(avg_width/5)
avg_height = round(avg_height/5)
print("Average Width of Images", avg_width)
print("Average Heigth of Images", avg_height)
ndir = glob.glob(train_dir+"NORMAL/*.jpeg")
pdir = glob.glob(train_dir+"PNEUMONIA/*.jpeg")[0:len(ndir)]
print(len(ndir))
print(len(pdir))

b=0
v=0

for i in pdir:
    if "bacteria" in i.lower():
        b +=1
    elif "virus" in i.lower():
        v +=1
print("Bacteria: ", b) 
print("Virus: ", v) 
#Ratio is similar to actual training set
import random
print(len(pdir))
print(len(ndir))
final =pdir + ndir
print(final[-1].split("/")[-2])
random.shuffle(final)
final[0:30]
#Load all image data into the dataset along with its label
CLASSES =  ["NORMAL","PNEUMONIA"]  #0 will represent Normal, 1 will represent Pneumonia
#dirDataset = tf.data.Dataset.from_tensor_slices(final)  #FOR EQUAL DATASET

def decodeImage(i):
    img = tf.io.read_file(i)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [avg_width,avg_height])
    print(tf.strings.split(i, "/")[-2])
    """
        if (CLASSES[1] == tf.strings.split(i, "/")[-2]): #pneumonia positive
        label = 1
        return img, label
    else:
        label = 0
        return img, label
    """
    if (CLASSES[1] == tf.strings.split(i, "/")[-2]): #pneumonia positive
        label = [0,1]
        return img, label
    else:
        label = [1,0]
        return img, label

imageDataset = dirDataset.map(decodeImage)
#Quick undo of all the transformations show all the data is still clean and correct
for x,y in imageDataset.shuffle(400).take(6):
    print(y)
    x = tf.image.convert_image_dtype(x, tf.uint8)
    x = PIL.Image.fromarray(x.numpy())
    display.display(x)

#From the below result, everything looks clean, the labels are such where the [True False] is Normal and [False True] is Pneumonia
#More data preprocessing
BATCH_SIZE = 10
imageDataset = imageDataset.shuffle(500).batch(BATCH_SIZE)
subImageDataset = imageDataset.take(1)
for x,y in subImageDataset:
    print(y)
#Create the Model

from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, SeparableConv2D, BatchNormalization

LAMBDA = 0.5
DROPOUT_RATE = 0.7


class ConvModel(tf.keras.Model):
    
  def __init__(self):
    super(ConvModel, self).__init__()
    
    self.conv1 = Conv2D(64, 3, activation='relu', padding="same", input_shape=(avg_width, avg_height, 3))
    self.conv2 = Conv2D(64, 3, activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.conv3 = Conv2D(64, 3, activation='relu', padding="same", kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.bnormA = BatchNormalization()
    self.poolA = self.pool1 = MaxPool2D((2,2))
     
    self.dconv1  = SeparableConv2D(130, (3,3), 1, padding = "same", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.bnorm1 = BatchNormalization()
    self.dconv2 = SeparableConv2D(130, (3,3), 1, padding = "same", activation='relu',kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.bnorm2 = BatchNormalization()
    self.pool1 = MaxPool2D((2,2))
    
    self.dconv3  = SeparableConv2D(260, (3,3), 1, padding = "same", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.bnorm3 = BatchNormalization()
    self.dconv4 = SeparableConv2D(260, (3,3), 1, padding = "same", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.bnorm4 = BatchNormalization()
    self.pool2 = MaxPool2D((2,2))
    
    self.dconv5  = SeparableConv2D(600, (3,3), 1, padding = "same", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.bnorm5 = BatchNormalization()
    self.dconv6 = SeparableConv2D(600, (3,3), 1, padding = "same", activation='relu', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.bnorm6 = BatchNormalization()
    self.pool3 = MaxPool2D((2,2))
    
    self.flatten = Flatten()
    self.d1 = Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.drop1 = Dropout(0.7)
    self.d2 = Dense(250, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(LAMBDA))
    self.drop2 = Dropout(0.5)
    self.d5 = Dense(2,activation='softmax')
    
  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.bnormA(x)
    x = self.poolA(x)
    
    x = self.dconv1(x)
    x = self.bnorm1(x)
    x = self.dconv2(x)
    x = self.bnorm2(x)
    x = self.pool1(x)
    
    x = self.dconv3(x)
    x = self.bnorm3(x)
    x = self.dconv4(x)
    x = self.bnorm4(x)
    x = self.pool2(x)
    
    x = self.dconv5(x)
    x = self.bnorm5(x)
    x = self.dconv6(x)
    x = self.bnorm6(x)
    x = self.pool3(x)
    
    
    x = self.flatten(x)
    x = self.d1(x)
    x = self.drop1(x)
    x = self.d2(x)
    x = self.drop2(x)
    
    x = self.d5(x)
    
    return x

model = ConvModel()
#Define loss function and optimzer

LEARNING_RATE = 0.0001  #0.0001


#loss_fn = tf.keras.losses.BinaryCrossentropy()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate =LEARNING_RATE)    #learning_rate=0.005

#Define loss and accuracy metrics

train_loss = tf.keras.metrics.CategoricalCrossentropy(name="train_loss")
train_accuracy= tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")
val_loss = tf.keras.metrics.CategoricalCrossentropy(name="val_loss")
val_accuracy= tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")
test_loss = tf.keras.metrics.CategoricalCrossentropy(name="test_loss")
test_accuracy= tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
"""
train_loss = tf.keras.metrics.BinaryCrossentropy(name="train_loss")
train_accuracy= tf.keras.metrics.BinaryAccuracy(name="train_accuracy")
val_loss = tf.keras.metrics.BinaryCrossentropy(name="val_loss")
val_accuracy= tf.keras.metrics.BinaryAccuracy(name="val_accuracy")
test_loss = tf.keras.metrics.BinaryCrossentropy(name="test_loss")
test_accuracy= tf.keras.metrics.BinaryAccuracy(name="test_accuracy")
"""
#Train Model

#Store Training History
loss_history = []
accuracy_history = []

@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape: #record gradients
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)    #Take gradient of loss function with respect to trainable variables of model
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) #Apply gradients to model using optimizer
    
    train_loss.update_state(labels, predictions)
    train_accuracy.update_state(labels,predictions)
    
EPOCHS = 50

print("<--------BEGIN TRAINING-------->\n\n\n")
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    #batchNum = 0
    for imageBatch, labelBatch in imageDataset:
        #batchNum += 1
        train_step(imageBatch, labelBatch)
        #print("Epoch {} || Batch: {}    |  Loss: {}    | Accuracy: {} ".format(epoch+1, batchNum, train_loss.result(), train_accuracy.result() * 100))
    loss_history.append(train_loss.result())
    accuracy_history.append(train_accuracy.result()*100)
    print("Epoch {} || Loss: {}    | Accuracy: {} ".format(epoch+1, train_loss.result(), train_accuracy.result() * 100))
    model.save_weights('/training_weights/checkpoint{}'.format(epoch+1))
    
print("\n\n\n<--------END TRAINING-------->\n\n\n")
from sklearn.metrics import confusion_matrix

#Load all test image data into the dataset along with its label
CLASSES =  ["NORMAL","PNEUMONIA"]  #0 will represent Normal, 1 will represent Pneumonia
dirValDataset = tf.data.Dataset.list_files(val_dir+"*/*.jpeg")

#Create testDataset
valDataset = dirValDataset.map(decodeImage)
valDataset = valDataset.shuffle(600).batch(10)

valImgDataset = valDataset.map(lambda x,y: x)
valLabelDataset = valDataset.map(lambda x,y: y)

valLabelDataset = valLabelDataset.unbatch()
out_labels = []
for y in valLabelDataset:
    if y.numpy()[1]: #Second element is true, pneumonia
        out_labels.append(1)
    else:
        out_labels.append(0)
        
preds = model.predict(valImgDataset)

maxed_preds = []
for i in range(len(preds)):
    
    if (list(preds[i]).index(max(preds[i])) == 1): #pneumonia
        maxed_preds.append(1)
    else:
        maxed_preds.append(0)
    
"""
cm  = confusion_matrix(out_labels, maxed_preds)

# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix")
print("True Positive: ", tp, "\n")
print("False Positive: ", fp, "\n")
print("True Negative: ", tn, "\n")
print("False Negative: ", fn, "\n")

precision = tp/(tp+fp) #how many predicted true positives out of positves predicted
recall = tp/(tp+fn) #How many predicted true positives out of total positives in datset

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
"""

from sklearn.metrics import classification_report
print(classification_report(maxed_preds, out_labels))

from sklearn.metrics import confusion_matrix

#Load all test image data into the dataset along with its label
CLASSES =  ["NORMAL","PNEUMONIA"]  #0 will represent Normal, 1 will represent Pneumonia
dirTestDataset = tf.data.Dataset.list_files(test_dir+"*/*.jpeg")

#Create testDataset
testDataset = dirTestDataset.map(decodeImage)
testDataset = testDataset.shuffle(600).batch(10)

testImgDataset = testDataset.map(lambda x,y: x)
testLabelDataset = testDataset.map(lambda x,y: y)

testLabelDataset = testLabelDataset.unbatch()
out_labels = []
for y in testLabelDataset:
    if y.numpy()[1]: #Second element is true, pneumonia
        out_labels.append(1)
    else:
        out_labels.append(0)

preds = model.predict(testImgDataset)

maxed_preds = []
for i in range(len(preds)):
    
    if (list(preds[i]).index(max(preds[i])) == 1): #pneumonia
        maxed_preds.append(1)
    else:
        maxed_preds.append(0)
"""
cm  = confusion_matrix(out_labels, maxed_preds)

# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix")
print("True Positive: ", tp, "\n")
print("False Positive: ", fp, "\n")
print("True Negative: ", tn, "\n")
print("False Negative: ", fn, "\n")

precision = tp/(tp+fp) #how many predicted true positives out of positves predicted
recall = tp/(tp+fn) #How many predicted true positives out of total positives in datset

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
"""

#print("Loss: {}    | Accuracy: {} ".format(test_loss.result(), test_accuracy.result() * 100))
from sklearn.metrics import classification_report
print(classification_report(maxed_preds, out_labels))

