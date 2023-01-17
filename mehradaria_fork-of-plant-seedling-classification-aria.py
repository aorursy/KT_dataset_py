import numpy as np # MATRIX OPERATIONS
import pandas as pd # EFFICIENT DATA STRUCTURES
import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS
import math # MATHEMATICAL OPERATIONS
import cv2 # IMAGE PROCESSING - OPENCV
from glob import glob # FILE OPERATIONS
import itertools # Efficient looping

# KERAS AND SKLEARN MODULES
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# GLOBAL VARIABLES
scale = 70 # px to scale
seed = 7 # fixing random
path_to_images = '../input/plant-seedlings-classification/train/*/*.png'
images = glob(path_to_images)
trainingset = []
traininglabels = []
num = len(images)
count = 1
#READING IMAGES AND RESIZING THEM
for i in images:
    print(str(count)+'/'+str(num),end='\r')
    # Get image (with resizing)
    trainingset.append(cv2.resize(cv2.imread(i),(scale,scale)))
    # Get image label (folder name)
    traininglabels.append(i.split('/')[-2])
    count=count+1
trainingset = np.asarray(trainingset) # Train images set
traininglabels = pd.DataFrame(traininglabels) # Train labels set
# Show some example images
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(trainingset[i])
new_train = []
sets = []; getEx = True
for i in trainingset:
        # Use gaussian blur
    blurr = cv2.GaussianBlur(i,(5,5),0)
        # Convert to HSV image
    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
    
    # Create mask (parameters - green color range)
    lower = (25,40,50)
    upper = (75,255,255)
    mask = cv2.inRange(hsv,lower,upper)
    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
    
       # Create bool mask
    boolean = mask>0
    
        # Apply the mask
    new = np.zeros_like(i,np.uint8) # Create empty image
    new[boolean] = i[boolean] # Apply boolean mask to the origin image
    
    new_train.append(new) # Append image without backgroung
    
     # Show examples
    if getEx:
        plt.subplot(2,3,1);plt.imshow(i) # ORIGINAL
        plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED
        plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED
        plt.subplot(2,3,4);plt.imshow(mask) # MASKED
        plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED
        plt.subplot(2,3,6);plt.imshow(new) # NEW PROCESSED IMAGE without background
        plt.show()
        getEx = False
new_train = np.asarray(new_train)


print('Most of the background removed:')
    
# Show sample result
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(new_train[i])
# Encode labels and create classes
labels = preprocessing.LabelEncoder()
labels.fit(traininglabels[0])
print('Classes: '+str(labels.classes_))
encodedlabels = labels.transform(traininglabels[0])

# Make labels categorical
clearalllabels = np_utils.to_categorical(encodedlabels)
classes = clearalllabels.shape[1]
print("Number of classes: " + str(classes))
    
# Plot of label types numbers
traininglabels[0].value_counts().plot(kind='pie')
new_train = new_train/255 # Normalize input [0...255] to [0...1]

x_train,x_test,y_train,y_test = train_test_split(new_train,clearalllabels,test_size=0.1,random_state=seed,stratify=clearalllabels)
print('Train Shape: {}'.format(x_train.shape))
generator = ImageDataGenerator(
    rotation_range = 180, # randomly rotate images in the range
    zoom_range = 0.1, # Randomly zoom image 
    width_shift_range = 0.1, # randomly shift images horizontally
    height_shift_range = 0.1, # randomly shift images vertically 
    horizontal_flip = True, # randomly flip images horizontally
    vertical_flip = True # randomly flip images vertically
)
generator.fit(x_train)
np.random.seed(seed) # Fix seed

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# learning rate reduction
lrr = ReduceLROnPlateau(monitor='val_acc', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.00001)

# checkpoints
filepath='weights.best_{epoch:02d}-{val_acc:.2f}.h5'
checkpoints = ModelCheckpoint(filepath, monitor='val_acc', 
                              verbose=1, save_best_only=True, mode='max')
filepath='weights.last_auto4.h5'
checkpoints_full = ModelCheckpoint(filepath, monitor='val_acc', 
                                 verbose=1, save_best_only=False, mode='max')

# all callbacks
callbacks_list = [checkpoints, lrr, checkpoints_full]

# fit model
#hist = model.fit_generator(generator.flow(x_train, y_train, batch_size=75),
#                            epochs=35,
#                            validation_data=(x_test, y_test),
#                            steps_per_epoch=x_train.shape[0],
#                            callbacks=callbacks_list
#                           )

# Evaluate model
# LOADING MODEL
model.load_weights("../input/weights/weights.best_17-0.96.hdf5") # best fitting model
dataset = np.load("../input/plantrecomodels/Data.npz") # Training and validation datasets
data = dict(zip(("x_train","x_test","y_train", "y_test"), (dataset[k] for k in dataset)))
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

print(model.evaluate(x_train, y_train))  # Evaluate on train set
print(model.evaluate(x_test, y_test))  # Evaluate on test set
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
predY = model.predict(x_test)
predYClasses = np.argmax(predY, axis = 1) 
trueY = np.argmax(y_test, axis = 1) 

# confusion matrix
confusionMTX = confusion_matrix(trueY, predYClasses) 

# plot the confusion matrix
plot_confusion_matrix(confusionMTX, classes = labels.classes_)
path_to_test = '../input/plant-seedlings-classification/test/*.png'
pics = glob(path_to_test)

testimages = []
tests = []
count=1
num = len(pics)

# Obtain images and resizing, obtain labels
for i in pics:
    print(str(count)+'/'+str(num),end='\r')
    tests.append(i.split('/')[-1]) # Images id's
    testimages.append(cv2.resize(cv2.imread(i),(scale,scale)))
    count = count + 1

testimages = np.asarray(testimages) # Train images set 

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(testimages[i])
newtestimages = []
sets = []
getEx = True
for i in testimages:
        # Use gaussian blur
    blurr = cv2.GaussianBlur(i,(5,5),0)
        # Convert to HSV image
    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
    
        # Create mask (parameters - green color range)
    lower = (25,40,50)
    upper = (75,255,255)
    mask = cv2.inRange(hsv,lower,upper)
    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
    
        # Create bool mask
    boolean = mask>0
    
        # Apply the mask
    masking = np.zeros_like(i,np.uint8) # Create empty image
    masking[boolean] = i[boolean] # Apply boolean mask to the origin image
    
    # Append image without backgroung
    newtestimages.append(masking)
    
        # Show examples
    if getEx:
        plt.subplot(2,3,1);plt.imshow(i) # Show the original image
        plt.subplot(2,3,2);plt.imshow(blurr) # Blur image
        plt.subplot(2,3,3);plt.imshow(hsv) # HSV image
        plt.subplot(2,3,4);plt.imshow(mask) # Mask
        plt.subplot(2,3,5);plt.imshow(boolean) # Boolean mask
        plt.subplot(2,3,6);plt.imshow(masking) # Image without background
        plt.show()
        getEx=False

newtestimages = np.asarray(newtestimages)

# OTHER MASKED IMAGES
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(newtestimages[i])
newtestimages=newtestimages/255
prediction = model.predict(newtestimages)

# Write prediction result to a file
pred = np.argmax(prediction,axis=1)
predStr = labels.classes_[pred]

result = {'file':tests,'species':predStr}
result = pd.DataFrame(result)
result.to_csv("Prediction.csv",index=False)
print('Prediction result saved as Prediction.csv')