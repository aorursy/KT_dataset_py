# run only if you want to create class activation mappings for visualization
# installing old version of scipy
# warning will restart runtime on google colab
!pip install -I scipy==1.2.*
!pip install -q kaggle
# make sure the kaggle.json file is in the working directory
# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json
# List available datasets.
!kaggle datasets list -s Xrays
# downloading dataset
!kaggle datasets download jbeltranleon/xrays-chest-299-small
# downloading trained models
!kaggle datasets download SmolBoi96/CheXNet-Ensemble-Model
# Unzip the data
!unzip -qq -n xrays-chest-299-small.zip
!unzip -qq -n CheXNet-Ensemble-Model.zip
# Switch directory and show its content
!cd 299_small && ls
import shutil

shutil.rmtree('input_299_small')
shutil.rmtree('sample_data')

#shutil.rmtree('keras-vis')
import os

base_dir = '299_small'

# Directory to our training data
train_folder = os.path.join(base_dir, 'train')

# Directory to our validation data
val_folder = os.path.join(base_dir, 'val')

# Directory to our validation data
test_folder = os.path.join(base_dir, 'test')

# List folders and number of files
print("Directory, Number of files")
for root, subdirs, files in os.walk(base_dir):
    print(root, len(files))
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0


atelectasis_dir = '299_small/train/atelectasis'
atelectasis_fnames = os.listdir(atelectasis_dir)
cardiomegaly_dir = '299_small/train/cardiomegaly'
cardiomegaly_fnames = os.listdir(cardiomegaly_dir)
consolidation_dir = '299_small/train/consolidation'
consolidation_fnames = os.listdir(consolidation_dir)
effusion_dir = '299_small/train/effusion'
effusion_fnames = os.listdir(effusion_dir)
infiltration_dir = '299_small/train/infiltration'
infiltration_fnames = os.listdir(infiltration_dir)
mass_dir = '299_small/train/mass'
mass_fnames = os.listdir(mass_dir)
nodule_dir = '299_small/train/nodule'
nodule_fnames = os.listdir(nodule_dir)
nofinding_dir = '299_small/train/no_finding'
nofinding_fnames = os.listdir(nofinding_dir)
pneumothorax_dir = '299_small/train/pneumothorax'
pneumothorax_fnames = os.listdir(pneumothorax_dir)

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(15, 15)
# showing Pneumothorax and No Finding pix
pic_index += 8
next_nofinding_pix = [os.path.join(nofinding_dir, fname) 
                for fname in nofinding_fnames[pic_index-8:pic_index]]
next_pneumothorax_pix = [os.path.join(pneumothorax_dir, fname) 
                for fname in pneumothorax_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_nofinding_pix+next_pneumothorax_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)
    
    img = mpimg.imread(img_path)
    #print(img.shape)
    plt.imshow(img)
    
print(255*img)
print(img_path)
print(img_path.endswith(".jpg"))
# Plot class distribution chart

import pandas as pd

count_classes = pd.Series([len(atelectasis_fnames),len(cardiomegaly_fnames),len(consolidation_fnames),len(effusion_fnames),len(infiltration_fnames),len(mass_fnames),len(nofinding_fnames),len(nodule_fnames),len(pneumothorax_fnames)])
count_classes.plot(kind = 'bar')
plt.title("Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
import os
from imblearn.over_sampling import ADASYN
from sklearn.datasets import make_classification
from PIL import Image
import numpy as np
from numpy import asarray

path, dirs, files = next(os.walk("299_small/test/no_finding"))
X = np.zeros((139, 299,299))
for i in range(139):
  image = Image.open("299_small/test/no_finding/"+files[i])
  y = asarray(image)
  #print(i)
  dim = y.shape
  if len(y.shape) == 2:
    X[i]=y
  if len(y.shape) == 3:
    X[i]=y[:,:,0]
  #print(i)
  #print(X[i].shape)

print(X.shape)

y = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
#z = np.ones(30)
#y = np.concatenate(y,z)
print(y.shape)
#from imblearn.over_sampling import SMOTE
ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_sample(X.reshape(X.shape[0], -1), y)
print(X_res.shape)
X_res = X_res.reshape(X_res.shape[0], 299,299)
print(X_res.shape)

c = 32

for i in range (50,66):
  #print(y_res[i])
  i = i + c
  img = Image.fromarray(X_res[i])
  sp = plt.subplot(4, 4, i - 49 - c)
  sp.axis('Off')
  plt.imshow(img)
img = Image.fromarray(X_res[49 + 36 + 9 + 1])
plt.imshow(img)
img = Image.fromarray(X_res[49 + 36 + 9])
plt.imshow(img)
import os
from imblearn.over_sampling import RandomOverSampler
from PIL import Image
import numpy as np

pathall, dirsall, filesall = next(os.walk("299_small/train"))
k = 0
X = np.ones((10000, 299,299))
y = np.zeros(10000)
for i in range(9):
  path, dirs, files = next(os.walk("299_small/train/" + dirsall[i]))
  for j in range(len(files)):
    image = Image.open("299_small/train/" + dirsall[i]+"/"+files[j])
    y[k] = i
    image = np.asarray(image)
  #print(i)
  #print(y.shape)
    if hasattr(image, 'shape'):
      #print('hi1')
      if len(image.shape) == 2:
        #print('hi2')
        X[k]=image
        k = k +1
      if len(image.shape) == 3:
        #print('hi3')
        X[k]=image[:,:,0]
        k = k +1
    if k >= 10000:
      break
  if k >= 10000:
    break
  #print(i)
  #print(X[i].shape)
    

print(X.shape)
print(y.shape)
LABELS = ["atelectasis","cardiomegaly","consolidation","effusion","infiltration","mass","no_finding","nodule","pneumothorax"]
size = [2697,699,838,2531,6109,1368,6400,1731,1404]
import numpy as np

for j in range(9):
  path, dirs, files = next(os.walk("299_small/train/" + LABELS[j]))
  for i in range(size[6]-size[j]):
    k = np.random.randint(0, len(files)-1)
    shutil.copy("299_small/train/" + LABELS[j]+"/"+files[k], "299_small/train/" + LABELS[j]+"/"+str(i)+".png")
#shutil.rmtree("299_small/train/atelectasisnew")
for i in range(9):
  path, dirs, files = next(os.walk("299_small/train/"+LABELS[i]))
  print(len(files))
from keras.preprocessing.image import ImageDataGenerator

# Batch size
bs = 16

# All images will be resized to this value
image_size = (299, 299)

# All images will be rescaled by 1./255. We apply data augmentation here.
train_datagen = ImageDataGenerator(rescale=1./255,
                                   brightness_range= [0.5,1.5],
                                   zoom_range=0.2)

val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 16 using train_datagen generator
print("Preparing generator for train dataset")
train_generator = train_datagen.flow_from_directory(
    directory= train_folder, # This is the source directory for training images 
    target_size=image_size, # All images will be resized to value set in image_size
    batch_size=bs,
    class_mode='categorical')

# Flow validation images in batches of 16 using val_datagen generator
print("Preparing generator for validation dataset")
val_generator = val_datagen.flow_from_directory(
    directory= val_folder, 
    target_size=image_size,
    batch_size=bs,
    class_mode='categorical')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D
from tensorflow.keras.applications.densenet import DenseNet121

# Here we specify the input shape of our data 
# This should match the size of images ('image_size') along with the number of channels (3)
input_shape = (299, 299, 3)

# Define the number of classes
num_classes = 14

input_img = Input(shape=input_shape)

model = DenseNet121(include_top=False, weights=None, input_tensor=input_img, input_shape=input_shape, pooling='avg', classes = num_classes)
temp = model.layers[-1].output

predictions = Dense(14, activation='softmax',name = 'last')(temp)

model = Model(inputs=input_img, outputs=predictions)
# loading weights from model trained on ChestXray14 for transfer learning 
model_path = 'brucechou1983_CheXNet_Keras_0.3.0_weights.h5'
model.load_weights( model_path )
from tensorflow.keras import optimizers
#from keras import regularizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
temp = model.layers[-2].output

num_classes = 9
predictions = Dense(num_classes, activation='softmax',name = 'last')(temp)

model = Model(inputs=input_img, outputs=predictions)
# train only last layer first
for layer in model.layers:
    if layer.name != 'last':
        layer.trainable = False
model.summary()
from tensorflow.keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint

Checkpointer = ModelCheckpoint('balanced.hdf5', monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit(
        train_generator, # train generator has 23777 train images
        steps_per_epoch=train_generator.samples // bs + 1,
        epochs=20,
        validation_data=val_generator, # validation generator has 5948 validation images
        validation_steps=val_generator.samples // bs + 1,
        callbacks=[Checkpointer]
)
from tensorflow.keras.models import load_model

model_path = 'balanced.hdf5'
model = load_model( model_path )
# unfreeze all weights and train whole model

for layer in model.layers:
  layer.trainable = True
model.summary()

from tensorflow.keras import optimizers
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
Checkpointer = ModelCheckpoint('balanced2.hdf5', monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit(
        train_generator, # train generator has 23777 train images
        steps_per_epoch=train_generator.samples // bs + 1,
        epochs=20,
        validation_data=val_generator, # validation generator has 5948 validation images
        validation_steps=val_generator.samples // bs + 1,
        callbacks=[Checkpointer]
)
from tensorflow.keras.callbacks import ModelCheckpoint

Checkpointer = ModelCheckpoint('CheXNet.hdf5', monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit_generator(
        train_generator, # train generator has 23777 train images
        steps_per_epoch=train_generator.samples // bs + 1,
        epochs=20,
        validation_data=val_generator, # validation generator has 5948 validation images
        validation_steps=val_generator.samples // bs + 1,
        callbacks=[Checkpointer]
)
from tensorflow.keras.models import load_model

model_path = 'CheXNet.hdf5'
model = load_model( model_path )
# unfreeze all weights and train whole model

for layer in model.layers:
  layer.trainable = True
model.summary()

from tensorflow.keras import optimizers
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
Checkpointer = ModelCheckpoint('CheXNet2.hdf5', monitor='val_accuracy', save_best_only=True, verbose=1)
history = model.fit_generator(
        train_generator, # train generator has 23777 train images
        steps_per_epoch=train_generator.samples // bs + 1,
        epochs=20,
        validation_data=val_generator, # validation generator has 5948 validation images
        validation_steps=val_generator.samples // bs + 1,
        callbacks=[Checkpointer]
)
from tensorflow.keras.models import load_model

model_path = 'CheXNet2.hdf5'
model = load_model( model_path )
scores = model.evaluate_generator(train_generator, steps=train_generator.samples // train_generator.batch_size + 1, verbose=1)
print('Train loss:', scores[0])
print('Train accuracy:', scores[1])
scores = model.evaluate_generator(val_generator, steps=val_generator.samples // val_generator.batch_size + 1, verbose=1)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])
test_datagen = ImageDataGenerator(rescale=1./255)

print("Preparing generator for test dataset")
test_generator = test_datagen.flow_from_directory(
    directory= test_folder, 
    target_size=image_size,
    batch_size=16,
    shuffle=False,
    class_mode='categorical')

scores = model.evaluate_generator(test_generator, steps=test_generator.samples // test_generator.batch_size + 1, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

model_path = 'CheXNet2.hdf5'
model = load_model( model_path )
from tensorflow.keras.preprocessing.image import ImageDataGenerator


test_datagen = ImageDataGenerator(rescale=1./255)

print("Preparing generator for test dataset")
test_generator = test_datagen.flow_from_directory(
    directory= test_folder, 
    target_size=(299,299),
    batch_size=16,
    shuffle=False,
    class_mode='categorical')

#scores = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size + 1, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

input_shape = (299, 299, 3)
num_classes = 9
nb_samples = 7433
bs = 16

y_test = pd.get_dummies(pd.Series(test_generator.classes))
y_pred =  model.predict(test_generator,steps = nb_samples//bs + 1, verbose=1)
y_test = y_test.to_numpy()
y_preds = [np.argmax(i) for i in y_pred]
y_preds = np.asarray(y_preds)
y_tests = [np.argmax(i) for i in y_test]
y_tests = np.asarray(y_tests)

cnf_matrix1 = confusion_matrix(test_generator.classes, y_preds)

LABELS = ["atelectasis","cardiomegaly","consolidation","effusion","infiltration","mass","no_finding","nodule","pneumothorax"]

plt.figure(figsize=(12, 12))
sns.heatmap(cnf_matrix1, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix (Unnormalized)")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
cnf_matrix2 = confusion_matrix(y_tests, y_preds, normalize = 'true')

plt.figure(figsize=(12, 12))
sns.heatmap(cnf_matrix2, xticklabels=LABELS, yticklabels=LABELS, annot=True,fmt="0.3f",annot_kws={"size": 15});
plt.title("Confusion matrix for 'Augmented' Model (Row Normalized)")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
cnf_matrix3 = confusion_matrix(y_tests, y_preds, normalize = 'pred')

plt.figure(figsize=(12, 12))
sns.heatmap(cnf_matrix3, xticklabels=LABELS, yticklabels=LABELS, annot=True);
plt.title("Confusion matrix (Column Normalized)")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(num_classes):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.title("ROC for 'Augmented' Model")
plt.legend(loc='best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

AUCsum = 0
for i in range(num_classes):
    print("AUC of class {0} = {1:0.4f}".format(LABELS[i], roc_auc[i]))
    AUCsum +=roc_auc[i]
print("Average ROC AUC = {0:0.4f}".format(AUCsum/9))
from sklearn.metrics import precision_recall_curve

# Compute PR curve and area for each class
precision = dict()
recall = dict()
pr_auc = dict()

f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')

for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
    pr_auc[i] = auc(recall[i], precision[i])
for i in range(num_classes):
    plt.plot(recall[i], precision[i],
             label='PR curve of class {0} (area = {1:0.4f})'
             ''.format(i, pr_auc[i]))
plt.title("Precision Recall Curve for 'Augmented' Model")
plt.legend(loc=(0.05, -1.05), prop=dict(size=14))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.show()
AUCsum = 0
for i in range(num_classes):
    print("AUC of class {0} = {1:0.4f}".format(LABELS[i], pr_auc[i]))
    AUCsum +=pr_auc[i]
print("Average PR AUC = {0:0.4f}".format(AUCsum/9))
# getting the indices of the images for each class that is classified correctly and classified wrongly

print(y_tests[0])
print(y_preds[0])
print(y_tests[0 + 1])
print(y_preds[0 + 1])
actelectasis_idx = 1
print("Top")
print(y_tests[843])
print(y_preds[843])
print(y_tests[843 + 2])
print(y_preds[843 + 2])
cardiomegaly_idx = 2
print("Top")
print(y_tests[1062])
print(y_preds[1062])
print(y_tests[1062 + 145])
print(y_preds[1062 + 145])
consolidation_idx = 145
print("Bottom")
print(y_tests[1324])
print(y_preds[1324])
print(y_tests[1324 + 1])
print(y_preds[1324 + 1])
effusion_idx = 1
print("Bottom")
print(y_tests[2115])
print(y_preds[2115])
print(y_tests[2115 + 1])
print(y_preds[2115 + 1])
infiltration_idx = 1
print("Bottom")
print(y_tests[4025])
print(y_preds[4025])
print(y_tests[4025 + 1])
print(y_preds[4025 + 1])
mass_idx = 1
print("Top")
print(y_tests[4453])
print(y_preds[4453])
print(y_tests[4453 + 1])
print(y_preds[4453 + 1])
nofinding_idx = 1
print("Bottom")
print(y_tests[6453])
print(y_preds[6453])
print(y_tests[6453 + 19])
print(y_preds[6453 + 19])
nodule_idx = 19
print("Bottom")
print(y_tests[6994])
print(y_preds[6994])
print(y_tests[6994 + 438])
print(y_preds[6994 + 438])
pneumothorax_idx = 438
print("Bottom")
#print(np.where(y_preds == 7))
#print(np.where(y_preds == 2))
from PIL import Image
from vis.visualization import visualize_cam, overlay

def visualize_top_convolution(model,image_batch):
    layer_idx=-1

    # credit: https://github.com/raghakot/keras-vis/blob/master/applications/self_driving/visualize_attention.ipynb
    heatmap = visualize_cam(model, layer_idx=-1, filter_indices=None, seed_input=image_batch, grad_modifier=None)
    img = image_batch.squeeze()

    # credit: (gaussian filter for a better UI) http://bradsliz.com/2017-12-21-saliency-maps/
    import scipy.ndimage as ndimage
    smooth_heatmap = ndimage.gaussian_filter(heatmap[:,:,2], sigma=5)

    nn = 5
    fig = plt.figure(figsize=(20,20))
    a = fig.add_subplot(1, nn, 1)
    plt.imshow(img)
    a.set_title("original",fontsize=10)
    plt.axis('off')
    a = fig.add_subplot(1, nn, 2)
    plt.imshow(overlay(img, heatmap, alpha=0.7))
    a.set_title("heatmap",fontsize=10)
    plt.axis('off')
    a = fig.add_subplot(1, nn, 3)
    plt.imshow(img)
    plt.imshow(smooth_heatmap, alpha=0.7)
    a.set_title("heatmap/gaussian",fontsize=10)
    plt.axis('off')
    plt.show()
filenames = os.listdir(atelectasis_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[actelectasis_idx]

path1 = os.path.join(atelectasis_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(atelectasis_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**Atelectasis Correct Prediction**")
visualize_top_convolution(model,image1)
print("**Atelectasis mistaken for Cardiomegaly**")
visualize_top_convolution(model,image2)
filenames = os.listdir(cardiomegaly_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[cardiomegaly_idx]

path1 = os.path.join(cardiomegaly_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(cardiomegaly_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**Cardiomegaly Correct Prediction**")
visualize_top_convolution(model,image1)
print("**Cardiomegaly mistaken for No Finding**")
visualize_top_convolution(model,image2)
filenames = os.listdir(consolidation_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[consolidation_idx]

path1 = os.path.join(consolidation_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(consolidation_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**Consolidation Correct Prediction**")
visualize_top_convolution(model,image2)
print("**Consolidation mistaken for Infiltration**")
visualize_top_convolution(model,image1)
filenames = os.listdir(effusion_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[effusion_idx]

path1 = os.path.join(effusion_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(effusion_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**Effusion Correct Prediction**")
visualize_top_convolution(model,image2)
print("**Effusion mistaken for No Finding**")
visualize_top_convolution(model,image1)
filenames = os.listdir(infiltration_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[infiltration_idx]

path1 = os.path.join(infiltration_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(infiltration_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**Infiltration Correct Prediction**")
visualize_top_convolution(model,image2)
print("**Infiltration mistaken for No Finding**")
visualize_top_convolution(model,image1)
filenames = os.listdir(mass_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[mass_idx]

path1 = os.path.join(mass_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(mass_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**Mass Correct Prediction**")
visualize_top_convolution(model,image1)
print("**Mass mistaken for Effusion**")
visualize_top_convolution(model,image2)
filenames = os.listdir(nofinding_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[nofinding_idx]

path1 = os.path.join(nofinding_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(nofinding_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**No Finding Correct Prediction**")
visualize_top_convolution(model,image2)
print("**No Finding mistaken for Cardiomegaly**")
visualize_top_convolution(model,image1)
filenames = os.listdir(nodule_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[nodule_idx]

path1 = os.path.join(nodule_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(nodule_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**Nodule Correct Prediction**")
visualize_top_convolution(model,image2)
print("**Nodule mistaken for No Finding**")
visualize_top_convolution(model,image1)
filenames = os.listdir(pneumothorax_dir)
filenames.sort()
filename1 = filenames[0]
filename2 = filenames[pneumothorax_idx]

path1 = os.path.join(pneumothorax_dir, filename1)
image1 = mpimg.imread(path1)
image1 = np.stack((image1,image1, image1), axis=2)

path2 = os.path.join(pneumothorax_dir, filename2)
image2 = mpimg.imread(path2)
image2 = np.stack((image2,image2, image2), axis=2)

print("**Pneumothorax Correct Prediction**")
visualize_top_convolution(model,image2)
print("**Pneumothorax mistaken for Effusion**")
visualize_top_convolution(model,image1)