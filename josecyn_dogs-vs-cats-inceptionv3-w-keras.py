# defines

DIR_IMG_TRAIN = '../input/train'
DIR_IMG_TEST = '../input/test'
DIR_TMP_DATA = '.'
DIR_OUTPUT = '.'

IMG_SHAPE = (150,150,3)
# imports

import os
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from scipy.misc import imsave
import numpy as np
import time

%matplotlib inline

sample_img_path = os.path.join(DIR_IMG_TRAIN,'cat.5.jpg')
sample_img = plt.imread(sample_img_path)
plt.imshow(sample_img)
print ('Image have shape: {}'.format(sample_img.shape))
# dog example
sample_img_path = os.path.join(DIR_IMG_TRAIN,'dog.0.jpg')
sample_img = plt.imread(sample_img_path)
plt.imshow(sample_img)
print ('Image have shape: {}'.format(sample_img.shape))
# as long as images have different sizes, let's resize!

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def loadAndResizeImage (img, w, h):
    '''
    loads the image in 'img' path and returns a PIL image of size (w,h)
    '''
    return image.load_img (img, target_size=(w,h))

# dog example resized
dog_index = 0
sample_img_path = os.path.join(DIR_IMG_TRAIN,'dog.'+str(dog_index)+'.jpg')
sample_img = loadAndResizeImage (sample_img_path, IMG_SHAPE[0], IMG_SHAPE[1])
plt.imshow(sample_img)
print ('Dog #'+str(dog_index)+' as image:')
from PIL import Image

def getLabel (X):
    """
    Return the label for an element based on the filename: 
        dog -> 1 
        cat -> 0
    """
    if ('dog' in X):
        return 1
    else:
        return 0
    
def getLabels (X):
    """
    Returns an array that contains the label for each X
    """
    return np.array([getLabel(X[i]) for i in range(len(X))])

def getLabelFromScore (score):
    """
    Returns the label based on the probability
    if score >= 0.5, return 'dog'
    else return 'cat'
    """
    if (score >=0.5):
        return 'dog'
    else:
        return 'cat'

def normalizedArrayFromImageInPath (image_path, img_shape):
    """
    returns an the image in 'image' path normalized in an np array
    """
    img = loadAndResizeImage (image_path, img_shape[0], img_shape[1])
    return image.img_to_array(img) / 255.

def loadResizeNormalizeImages (basepath, path_array, img_shape):
    """
    Loads the images from the path 
    and returns them in an array
    """
    images = np.empty ((len(path_array), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    for i in range (len(path_array)):
        images[i] = normalizedArrayFromImageInPath (os.path.join(basepath,path_array[i]), img_shape)
    return images
import os
import numpy as np

train_imgs = os.listdir(DIR_IMG_TRAIN)

train_examples = 1000
train_x = train_imgs[:train_examples]
train_y = getLabels (train_x)

print ("Number of training examples = {}".format(len(train_x)))
print ("Number of training labels = {}".format(len(train_y)))

validation_examples = 100
validation_x = train_imgs[train_examples:train_examples+validation_examples]
validation_y = getLabels (validation_x)

print ("Number of validation examples = {}".format (len(validation_x)))
print ("Number of validation labels = {}".format(len(validation_y)))
img_train_x = loadResizeNormalizeImages (DIR_IMG_TRAIN, train_x, IMG_SHAPE)
np.save (os.path.join(DIR_TMP_DATA,'train_x'), img_train_x)
print (img_train_x.shape)
print ('Input X for training saved!')

img_validation_x = loadResizeNormalizeImages (DIR_IMG_TRAIN, validation_x, IMG_SHAPE)
np.save (os.path.join(DIR_TMP_DATA,'validation_x'), img_validation_x)
print (img_validation_x.shape)
print ('Input X for validation saved!')

print ('Checkpoint 1')
def dogsVsCatsProportion (labels):
    cats = 0
    dogs = 0
    for label in labels:
        if (label == 1):
            dogs = dogs + 1
        else:
            cats = cats + 1

    cats_percent = (cats/len(labels))*100.0
    dogs_percent = 100.0 - cats_percent

    return dogs_percent, cats_percent
img_train_x = np.load (os.path.join(DIR_TMP_DATA,'train_x')+'.npy')
img_validation_x = np.load (os.path.join(DIR_TMP_DATA,'validation_x')+'.npy')

print ('Input X train and validation loaded!')

dogs, cats = dogsVsCatsProportion (train_y)
print ("--> Training set: cats represent "+str(cats)+"% of the total")
print ("    Dogs represent "+str(dogs))

dogs, cats = dogsVsCatsProportion (validation_y)
print ("--> Validation set: cats represent "+str(cats)+"% of the total")
print ("    Dogs represent "+str(dogs))
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# add dropout
x = Dropout (0.5)(x)
# one fully connected layer more
x = Dense(256, activation='relu')(x)
# add dropout
x = Dropout (0.5)(x)
# one fully connected layer more
x = Dense(32, activation='relu')(x)
# and a logistic layer --
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# train the model on the new data for a few epochs
model.fit (x=img_train_x, y=train_y, 
           batch_size=16, epochs=6, 
           validation_data=(img_validation_x, validation_y))
print ('Saving the model...')
model.save(os.path.join(DIR_TMP_DATA,'model.h5'))
print ('Model saved! Filename: {}'.format (os.path.join(DIR_TMP_DATA,'model.h5')))
print ('Checkpoint 2')
from keras.models import load_model
model = load_model(os.path.join(DIR_TMP_DATA,'model.h5'))
print ('Model loaded!')
# Evaluation with test images the model didn't see before

test_examples = 250

idx_test_example = (-1) * test_examples

test_x = train_imgs[idx_test_example:]

test_y = getLabels (test_x)

print ("Number of test dev examples = "+str(len(test_x)))
print ("Number of test dev labels = "+str(len(test_y)))

print ("First element of test dev set "+test_x[0])
print ("Label of first element of test dev set = "+str(test_y[0]))
img_test_x = loadResizeNormalizeImages (DIR_IMG_TRAIN, test_x, IMG_SHAPE)
print (img_test_x.shape)
print ('Saving test set...')
np.save (os.path.join(DIR_TMP_DATA,'test_x'), img_test_x)
print ('Input X for test saved!')
print ('Checkpoint 3')
img_test_x = np.load (os.path.join(DIR_TMP_DATA,'test_x')+'.npy')
print ('Input X for evaluation loaded!')

# check the proportion dogs vs cats
dogs, cats = dogsVsCatsProportion (test_y)

print ("--> Test set: cats represent "+str(dogs)+"% of the total")
print ("    dogs represent "+str(cats)+"%")
preds = model.evaluate (x=img_test_x, y=test_y, batch_size=10)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
def predictFromPath (img_path, img_size=(150,150)):
    img = normalizedArrayFromImageInPath (img_path, img_size)
    return predictFromImg (img)

def predictFromImg (img):
    x = np.expand_dims(img, axis=0)
    return model.predict(x)[0][0]
img_path = DIR_IMG_TEST+'4.jpg'
display_img = image.load_img(img_path)
start = time.time()
score = predictFromPath (img_path)
end = time.time()
print("Prediction took {:.3f} seconds".format (end - start))
print("It's a {}! (with a score of {}) 0 -> cat / 1 -> dog".format (getLabelFromScore (score), score))
plt.imshow(display_img)
# last test

fig=plt.figure()

plot_test_paths = os.listdir(DIR_IMG_TEST)[50:62]

for num,plot_test_path in enumerate(plot_test_paths):

    y = fig.add_subplot(3,4,num+1)
    orig = image.load_img (DIR_IMG_TEST+plot_test_path)

    prediction = predictFromPath (DIR_IMG_TEST+plot_test_path)
    str_label = getLabelFromScore (prediction)
        
    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
# prepare data for submission

submission_imgs = os.listdir(DIR_IMG_TEST)
submission_x = submission_imgs[:]
print ("Number of submission examples = {}".format(len(submission_x)))

submission_imgs = sorted(submission_imgs, key=lambda x: int(x[:x.index('.')]))
import csv
from tqdm import tqdm

with open(os.path.join(DIR_OUTPUT,'submission.csv'), 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['id', 'label'])
    for elem in tqdm (submission_imgs):
        prediction = predictFromPath (DIR_IMG_TEST+elem)
        filewriter.writerow([elem[:elem.index('.')], "{:.6f}".format(prediction)])
model.summary()