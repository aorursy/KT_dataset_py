#Import libraries
# Select TensorFlow 2.X, numpy, pandas, and the dataset
#%tensorflow_version 2.x

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import backend, models, layers, optimizers, regularizers
from tensorflow.keras.utils import to_categorical
import numpy as np
from numpy import expand_dims 

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from IPython.display import display 
from PIL import Image 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array 
from keras.preprocessing.image import array_to_img
from keras.applications.vgg19 import preprocess_input
from keras.applications.inception_v3 import InceptionV3 

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import os, shutil, datetime
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Input 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense 
from keras.layers import Flatten 
from keras.layers import Dropout 
from keras.optimizers import SGD, Adam
import pandas as pd
from IPython.core.display import display, HTML

import os
from statistics import mean, median
from numpy.random import seed
# Select TensorFlow 2.X, numpy, pandas, and the dataset
#%tensorflow_version 2.x

display(HTML("<style>.container { width:100% !important; }</style>"))
# Helper Function(s)
def plot_eval(Model, hist):
    import matplotlib.pyplot as plt
    
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(1, len(loss) + 1)
    
    f = plt.figure(figsize=(12,6))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    
    plt.clf()   # clear figure
    plt.subplot(121)
    plt.plot(epochs, loss, color = 'blue', label='Training Loss')
    plt.plot(epochs, val_loss, color = 'orange', label='Validation Loss')
    plt.title(Model + ' Cross Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    
    plt.subplot(122)
    plt.plot(epochs, acc, color = 'blue', label='Training Accuracy')
    plt.plot(epochs, val_acc, color = 'orange', label='Validation Accuracy')
    plt.title(Model + ' Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
# from google.colab import drive
# drive.mount('/k')
# import os
# os.listdir('/k/My Drive/data/train/ALBATROSS')
seed(1234)
base       = '/kaggle/input/100-bird-species/'
train_dir  = 'train'
test_dir   = 'test'
valid_dir  = 'valid'

# Model Constants
epochs = 50
b_size = 32  # 19
img_width =  224
img_height = 224

# Define callback functions to monitor early stopping and LR
callbacks_list = [
  EarlyStopping(
        monitor = 'accuracy', 
        patience = 10, 
        mode = 'auto',
        restore_best_weights=True),
  ReduceLROnPlateau( 
      monitor='accuracy',
      factor=0.1, 
      patience=1)
]


# Count images for each species
def cntSamples(directory):
    specs = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs.sort()
        for name in dirs:
            if name not in specs:
                specs.append(name)

    # file counts for each species 
    nums = []
    for b in specs:
        path = os.path.join(directory,b)
        num_files = len(os.listdir(path))
        nums.append(num_files)
 
    # Create Dictionary
    adict = {specs[i]:nums[i] for i in range(len(specs))}
    return adict

# Count labels in train, valid and test
DIR_TEST =  base + 'test'
DIR_TRAIN = base + 'train'
DIR_VALID = base + 'valid'


testDict =  cntSamples(DIR_TEST)
trainDict = cntSamples(DIR_TRAIN)
validDict = cntSamples(DIR_VALID)
train_tbl = pd.DataFrame.from_dict(trainDict, 
                orient='index', dtype=None, columns=['Images'])

num_classes = len(trainDict)
label_index = list(range(num_classes))
#print(num_classes)
train_tbl.insert(0,'Label Index',label_index, True)

display(HTML(train_tbl.to_html()))
from statistics import mean, median
from IPython.display import display, HTML

# Create dictionary of dataset partitions statistics. 
data = {'Images': [sum(trainDict.values()), 
                   sum(testDict.values()), 
                   sum(validDict.values())],
        
        'Species': [len(trainDict.values()), 
                    len(testDict.values()), 
                    len(validDict.values())],
        
        'Minimum': [min(trainDict.values()), 
                    min(testDict.values()), 
                    min(validDict.values())],
        
        'Maximum': [max(trainDict.values()), 
                    max(testDict.values()), 
                    max(validDict.values())],
        
        'Median': [int(median(trainDict.values())), 
                   int(median(testDict.values())), 
                   int(median(validDict.values()))]}
image_cnt  = train_tbl['Images']
image_indx = train_tbl['Label Index']
y_pos = range(len(image_indx))
plt.bar(y_pos, image_cnt, align='center', alpha=0.5)
plt.xlabel("Class Number (0-189)")
plt.ylabel('Number of Images')
plt.title('Train Data: Number of Images in each Class (Species)')

plt.show()
# Creates pandas DataFrame. 
lblSummary = pd.DataFrame(data, index =['Train', 'Test', 'Valid']) 
  
# print the data 
#print(lblSummary)
display(HTML(lblSummary.to_html()))
# Example images
from numpy import expand_dims, reshape
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.preprocessing.image import array_to_img

from keras.preprocessing.image import ImageDataGenerator 
import keras
import matplotlib.pyplot as plt
import numpy as np

def getKeybyValue(LabelDict, value):
    listItems = LabelDict.items()
    for item in listItems:
        if item[1] == value:
            return item[0]
    
    return None

seed(1234)

def pltFourImages(dir):
    datagen2 = ImageDataGenerator()
    it2 = datagen2.flow_from_directory(
            dir,
            target_size=(224, 224),
            batch_size=4,
            class_mode='binary')

    labDict = it2.class_indices
    batchX, batchy = it2.next() 
    num_img = batchX.shape[0]
    imgs = [array_to_img(batchX[i]) for i in range(num_img)]
    indx = [int(batchy[i]) for i in range(len(batchy))]
    labs = [getKeybyValue(labDict, i) for i in indx]
   
    # settings
    h, w = 10, 10        
    nrows, ncols = 2, 2  
    figsize = [18,12]     

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi = 80)

    # plot image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        axi.imshow(imgs[i], aspect = 'auto')

        # write Label as title
        axi.set_title(labs[i])

    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.2, wspace=0.2)
    plt.show()
    return
# Print four images from each of train, test and valid
for d in ['train', 'test', 'valid']:
    print('\n\nImages from ', d)
    pltFourImages(base + d)
if K.image_data_format() == 'channels_first':
    input_shape = (3,img_width, img_height)
else:
    input_shape = (img_width, img_height,3)
    
# create a data generator 
datagen = ImageDataGenerator(rescale = 1.0 / 255)

# Train Generator for InceptionV3
train_generator = datagen.flow_from_directory(
    base + train_dir , 
    target_size = (img_width, img_height),
    class_mode='categorical', 
    batch_size=b_size) 

# Validation Generator 
validation_generator = datagen.flow_from_directory(
    base + valid_dir, 
    target_size = (img_width, img_height),
    class_mode='categorical', 
    batch_size=b_size) 

# Test Generator 
test_generator = datagen.flow_from_directory(
    base + test_dir, 
    target_size = (img_width, img_height),
    class_mode='categorical',
    shuffle = False,
    batch_size=b_size) 
# Build the model.

backend.clear_session()
input_shape = (img_width, img_height, 3)

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# Input Tensor
input_tensor = Input(shape=input_shape)  

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

flat_1 = Flatten()(base_model.output)

# Add a fully-connected layer consisting of 2 x Dense(512)
x = Dense(512, activation='relu')(flat_1)
x = Dense(512, activation='relu')(x)

# Logistic layer -- for ever-changing number of classes
predictions = Dense(num_classes, activation='softmax')(x)

# Model
model_nda = Model(inputs=base_model.input, outputs=predictions)
print("InceptionV3 layers in model: ", len(model_nda.layers))
#print(model_nda.summary())

# Compile
from keras.optimizers import SGD
model_nda.compile(optimizer=SGD(lr=0.0015, momentum=0.8), 
              loss='categorical_crossentropy', metrics = ['accuracy'])
start = datetime.datetime.now()
# Train
epochs = 100 # Should yield ~98 percent accuracy
history_nda = model_nda.fit_generator(train_generator, 
                            steps_per_epoch=len(train_generator)//b_size, 
                            validation_data=validation_generator, 
                            validation_steps=len(validation_generator)//b_size, 
                            epochs=epochs, 
                            #callbacks = callbacks_list,
                            verbose=2)

end = datetime.datetime.now()
elapsed = end - start
print ('InceptionV3 Model Training complete. Elapsed: ', elapsed)
# Evaluate
loss, accuracy = model_nda.evaluate_generator(test_generator) 
 
# Report
hist_nda = pd.DataFrame(history_nda.history)
e_exe = hist_nda.shape[0]

print("\n\n        InceptionV3\n\n")
print("  Epochs completed: ", e_exe)
print("              Loss: {0:.4f}".format(loss))
print("          Accuracy: {0:.4f} % ".format(accuracy * 100.0))
print("           Elapsed:",elapsed)
plot_eval("InceptionV3 Model", hist_nda)
import math
# Create a special test generator
# Prediction Generator 

pred_generator = datagen.flow_from_directory(
    base + test_dir, 
    target_size = (img_width, img_height),
    class_mode='categorical',
    shuffle = False,
    batch_size=b_size) 

pred = model_nda.predict_generator(pred_generator, steps= math.ceil(950/b_size))
from sklearn.metrics import classification_report
pred_generator.reset
species = list(train_tbl.index)

pred_labs = np.argmax(pred,axis=1)

print(pred_labs)
#test_generator.reset()
pred_generator.reset
repDict = classification_report(pred_generator.labels, pred_labs, target_names=species, output_dict = True)
report = classification_report(pred_generator.labels, pred_labs, target_names=species)
print(report)
pred_generator.reset
errors = np.where(pred_labs != pred_generator.classes)[0]
data = []
for i in errors:
   j = pred_labs[i]
   data.append([pred_generator.filenames[i], train_tbl.index.values[j]])    
  
# Create the pandas DataFrame 
misl_tbl = pd.DataFrame(data, columns = ['File Name', 'Mis-Labeled AS'])
  
display(HTML(misl_tbl.to_html()))
if K.image_data_format() == 'channels_first':
    input_shape = (3,img_width, img_height)
else:
    input_shape = (img_width, img_height,3)
    
# create a data generator 
datagen = ImageDataGenerator(rescale = 1.0 / 255)

# Train Generator
train_generator = datagen.flow_from_directory(
    base + train_dir , 
    target_size = (img_width, img_height),
    class_mode='categorical', 
    batch_size=b_size) 

# Validation Generator 
validation_generator = datagen.flow_from_directory(
    base + valid_dir, 
    target_size = (img_width, img_height),
    class_mode='categorical', 
    batch_size=b_size) 

# Test Generator
test_generator = datagen.flow_from_directory(
    base + test_dir, 
    target_size = (img_width, img_height),
    class_mode='categorical', 
    batch_size=b_size) 
# InceptionResNetV2
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.utils.vis_utils import plot_model
import numpy as np

backend.clear_session()
input_shape = (img_width, img_height, 3)
input_tensor = Input(shape = input_shape)

model = InceptionResNetV2(include_top=False, weights='imagenet', 
            input_tensor=input_tensor, input_shape=input_shape)

# Freeze the pre-trained layers
cnt_trainable = 0
cnt_untrainable = 0
set_trainable = False
for layer in model.layers:
    if  layer.name == 'block8_1_mixed':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
        cnt_trainable += 1
    else:
        layer.trainable = False
        cnt_untrainable += 1
        
print("  Trainable layers: ", cnt_trainable)
print("Untrainable layers: ", cnt_untrainable)
        

#print(model.summary())

x = Flatten()(model.output)
x = Dense(512, activation = 'relu') (x)
x = Dense(512, activation = 'relu') (x)
output = Dense(num_classes, activation = 'softmax') (x)


# Model
model_irn = Model(inputs=model.inputs, outputs=output)
print("IRNV2 layers in model: ", len(model_irn.layers))


#print(model_irn.summary())
start = datetime.datetime.now()
epochs = 150 # should yield ~ 93 percent accuracy
b_size = 25  # 32

# Compile
model_irn.compile(optimizer=SGD(lr=0.001, momentum=0.9), 
              loss='categorical_crossentropy', 
              metrics = ['accuracy'])

# Train
history_irn = model_irn.fit_generator(train_generator, 
                            steps_per_epoch=len(train_generator)//b_size, 
                            validation_data=validation_generator, 
                            validation_steps=len(validation_generator)//b_size, 
                            epochs=epochs, 
                            #callbacks = callbacks_list, 
                            verbose=2)

end = datetime.datetime.now()
elapsed = end - start
print ('InceptionResNetV2 training complete. Elapsed: ', elapsed)
# Evaluate
loss, accuracy = model_irn.evaluate_generator(test_generator, 
                    steps=len(test_generator), verbose=2) 
 
# Report
hist_irn = pd.DataFrame(history_irn.history)
e_exe = hist_irn.shape[0]

print("\n\n         InceptionResNetV2 Model \n\n")
print("  Epochs completed: ", e_exe)
print("              Loss: {0:.4f}".format(loss))
print("          Accuracy: {0:.4f} %".format(accuracy * 100.0))
print("           Elapsed:", elapsed,"\n\n")

plot_eval("InceptionResNetV2 Model", hist_irn)
import math
# Create a special test generator
# Prediction Generator 

pred_generator = datagen.flow_from_directory(
    base + test_dir, 
    target_size = (img_width, img_height),
    class_mode='categorical',
    shuffle = False,
    batch_size=b_size) 



pred = model_irn.predict_generator(pred_generator, steps= math.ceil(950/b_size))

from sklearn.metrics import classification_report
pred_generator.reset
species = list(train_tbl.index)

pred_labs = np.argmax(pred,axis=1)

test_generator.reset()

report = classification_report(pred_generator.labels, pred_labs, target_names=species)
print(report)
pred_generator.reset
errors = np.where(pred_labs != pred_generator.classes)[0]
data = []
for i in errors:
   j = pred_labs[i]
   data.append([pred_generator.filenames[i], train_tbl.index.values[j]])    
  
# Create the pandas DataFrame 
misl_tbl = pd.DataFrame(data, columns = ['File Name', 'Mis-Labeled AS'])
  
display(HTML(misl_tbl.to_html()))
