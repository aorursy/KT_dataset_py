# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!ls
!ls ../input/human-protein-atlas-image-classification
!pwd
os.chdir("/kaggle/working")

if True:

    SUBFOLDER = "human-protein-atlas-image-classification/"
    
INPUT_PATH = "/kaggle/input/" + SUBFOLDER 
INPUT_IMAGES_PATH = INPUT_PATH + "train/"
INPUT_IMAGES_TEST_PATH = INPUT_PATH + "test/"

print(INPUT_IMAGES_PATH)
x = 1000
os.listdir(".")
import cv2
test_img = cv2.imread(INPUT_IMAGES_PATH + "65ac91dc-bba7-11e8-b2ba-ac1f6b6435d0_red.png", -1)

%matplotlib inline
import matplotlib.pyplot as plt
plt.axis("off")
plt.imshow(test_img)

np.set_printoptions(threshold=np.nan)
print(test_img.shape)
print(test_img[0].shape)
print(test_img[0])

# for some reasons, the image is saved as greyscale?? 
# we could simply load it as greyscale, then extend it with two dimensions of the things we dont want
# then if we save that, it should work! 
# no, so the image is saved with just one value of intensity. We likely have some additional meta info which enables us to render the image in the appropriate
# color info.  But cv2 does some modification, depending on the flag you pass in when asking to open 

plt.show()
import matplotlib.image as mpimg
mpimg_image = mpimg.imread(INPUT_IMAGES_PATH + "65ac91dc-bba7-11e8-b2ba-ac1f6b6435d0_red.png")
# print(mpimg_image)
imgplot = plt.imshow(mpimg_image, cmap=plt.cm.Reds_r)

# one issue with this is that we want the "inverse" of this image: black and white values should be flipped or swapped (could take an inverse for example)


test_filename = INPUT_IMAGES_PATH + "ac39847a-bbb1-11e8-b2ba-ac1f6b6435d0_red.png"

test_img = cv2.imread( test_filename)

plt.imshow(test_img)
plt.show()

# os.chdir("..")
blur = cv2.GaussianBlur(test_img,(5,5),0)

cv2.imwrite(r"/kaggle/working/sample_out.png", test_img)
plt.imshow(blur)
plt.show()

# files may be implicitly saved somewhere...
# we want to match the images (filenames) with their target labels
# we need to I guess find the log-likelihood
# we still want to present and do some exploratory data analysis
data = pd.read_csv(INPUT_PATH + "train.csv")
data.head()
name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',   
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }

# we could try a simple network:
# which does 
# we could train 28 binary networks, or we could train one network that does 28 classifications at once
# our keras network
# will have the size of the initial input, be the size of the initial layer
#  how does convolution appropriately do the patch? (when everything is linearized; assume it takes care of it for us!)

# get the xtrain data as a vector, as well as the ytrain data as a vector
# how is the loss computed?
# it is not like Pytorch, where you need to compute a loss, then backwards yourself
# instead, it is just fitting X and Y
# presumably, X should be the image data, then Y should be the vector to predict
# actually this is essentially done!

# ok, so let us assume we want it as one hot vectors then!

# ok, so we will create a row of zeros
# given a row, we should make it into a k-hot vector
def create_one_hot(row):
    size = len(name_label_dict)
#     we can make a numpy array, then pass it to pandas
    vector = np.zeros((size))
    for label in row.loc["Target"]:
        vector[label] = 1
    return vector

    pass

# we could also: just get all the labels; then, just make stuff with that. Then, we would just join or zip everything back together at the end!

# for row in data.head().iterrows():
#     print(row)
#     print(create_one_hot(row))

# data.loc["Target"]
data.columns
# loc is used to look up values where it is
# data.loc['Id']
size = len(name_label_dict)
y_train = []
for data_elts in data["Target"]:
#     we can make a numpy array, then pass it to pandas
    vector = np.zeros((size))
    indices = [int(elt) for elt in data_elts.split()]
#     vector[X = 1]
#     print(type(data_elts))
    for elt in indices:
        vector[elt] = 1

#     print(data_elts)
#     print(vector)
    y_train.append(vector)
    
# now, we can actually begin adding it into the neural network
# we can marry them together
# we have both the input dims as well as the output dims! 

# we could have something like this. But we could also have an input layer as defined by Keras
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape_img)) 
# model.add(Dense(5000, activation='relu', input_dim=X_train.shape[1]))
y_train = np.array(y_train)

# now, we want to have both the X and Y
# actually, they don't need to be explicitly joined together!
# print(y_train)
print(y_train.shape)
print(data["Target"][1])
from sklearn.preprocessing import MultiLabelBinarizer
import keras
# this class inherits from the dataset sampling class
class MyDataGenerator(keras.utils.Sequence):
#     we need to provide a mapping, listing all the ids in both the training and validation set
# we also need a dictinary mapping each class to its labels. Here, the implementation is a little different, since we need to adapt it to work with a LIST of labels as opposed to a single label
    def __init__(self, data_dict, batch_size=32, dim=(512,512), n_channels=1,
                 n_classes=28, shuffle=True, train=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = np.arange(0, len(data_dict))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data_frame = data_dict
        self.total_length = len(data_dict)
        self.train_ds = train
        
        labels = list(map(str, np.arange(0,28)))

        mlb = MultiLabelBinarizer(classes=labels)
        
        if self.train_ds:
            x = self.data_frame["Target"].apply(lambda x: x.split())
        else:
            x = self.data_frame["Predicted"].apply(lambda x: str(x).split())
            
        self.label_vectors = mlb.fit_transform(x)

    
    def __len__(self):
      'Denotes the number of batches per epoch'
#       print (int(np.ceil(len(self.list_IDs) / self.batch_size)))
      return int(np.ceil(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
                
    # we need to keep a counter of the data! 
    def __data_generation(self, list_IDs_temp, index):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      y = np.empty((self.batch_size, self.n_classes), dtype=int)

#         y should probably be k-dimensional as well! 

      # Generate data
      # we will need to build something that can fetch all the addresses/files
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
#         load the data from the disk...
          if self.train_ds:
              X[i,] = cv2.imread(INPUT_IMAGES_PATH + self.data_frame.iloc[i]["Id"] + "_green.png", 0)[..., np.newaxis]
          else:
              X[i,] = cv2.imread(INPUT_IMAGES_TEST_PATH + self.data_frame.iloc[i]["Id"] + "_green.png", 0)[..., np.newaxis]
          # load classes!!
          y[i] = self.label_vectors[i]

            
#             we can rewrite this in terms of efficient list level operations! 
# we would need a function that batches a read however


#           mlb.fit_transform([])  
#       for the final batch, we should be vary of making it too large or too small!
#       print("length is")
#       print(self.__len__())
#       print(index)
      
#         this is the last batch, and we have an uneven amount
      remainder = None
      if (index == self.__len__()-1 and self.total_length % self.batch_size != 0):
        remainder = self.total_length % self.batch_size
        
        print("last")
#         pass
    
      return X[:remainder], y[:remainder]


    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#       print(indexes)
#       print(len(indexes))

      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]

      # Generate data
      
      X, y = self.__data_generation(list_IDs_temp, index)

      return X, y
# We can simply pass in an array from 0 to len(data), since we are relying on pandas to do the indexing anyways
from sklearn.model_selection import train_test_split
train_data, valid_data = train_test_split(data)
# Parameters
params = {'dim': (512,512),
          'batch_size': 32,
          'n_classes': 28,
          'n_channels': 1,
          'shuffle': True}

# Datasets
training_generator = MyDataGenerator(train_data[:], **params)
valid_generator = MyDataGenerator(valid_data[:], **params)


# for batch in training_generator:
#     print(batch[0].shape)
# raise Error()
# print(len(training_generator))

# for index,batch in enumerate(training_generator):
#     print(index)
print(485*64)
# WARNING: we can ask for things outside the range of it!! 
for index,elt in enumerate(training_generator[485]):
    print(elt.shape)
#     print(X, y)
# print(training_generator[485])
import keras.backend as K
# credits to: https://www.kaggle.com/rejpalcz/cnn-128x128x4-keras-from-scratch-lb-0-328
THRESHOLD = 0.05
import tensorflow as tf

def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)


#  you should be on the right area to write things
with open("my_file.txt", "w") as file:
    file.write("my content")
    
# anyways, now we want to do all the data 

# # load 5 images into memory
# import cv2
# from tqdm import tqdm 
# data_pics = []
# for i,example in enumerate(tqdm(data["Id"])):
# #     even here, we must make sure to batch it appropriately!
# #     IOW, just making a list of all of them will be difficult!
# #     if i > 3000:
# #         break
#     full_path = INPUT_IMAGES_PATH + example + "_green.png"
# #     print(full_path)
#     img_data = cv2.imread(full_path, 0)
#     data_pics.append(img_data)
# #     print(img_data)
# #     break
    
# # we should try doing some batching; keras and stuff should do this for you automatically, so long as you implement one of their datasets
# numLength = 3000

# X_train = np.array(data_pics) # this will cause an error; conversion of data_pics into an array (i.e. having two compies will be problematic!)
# del data_pics
# print("god heer")
# expanded_X_train = X_train[..., np.newaxis]
# del X_train
# print("added a new axis")

# expanded_X_train = expanded_X_train[0:numLength]
# print("cut up the expanded_X_train")

# # print(expanded_X_train.shape)
# y_train = y_train[0:len(expanded_X_train)]
# # print(expanded_X_train)
# # print(expanded_X_train.shape) 
# print(y_train.shape) 
import keras.backend as K
# credits to: https://www.kaggle.com/rejpalcz/cnn-128x128x4-keras-from-scratch-lb-0-328
THRESHOLD = 0.05
import tensorflow as tf

def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)


#  you should be on the right area to write things
with open("my_file.txt", "w") as file:
    file.write("my content")
    
# anyways, now we want to do all the data 

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout
model = Sequential()

# Dense probably wont work nicely at all! 
# model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))

# input_dims = X_train[0].shape + (1,)
input_dims = (512,512,1)
print(type(input_dims))
print(input_dims)
# print(type(X_train[0].shape))
# we could do 2D or otger convolution
# now we understand: both why they have green and 4 other loaders, as well as why we have batch_first and so forth
# the input dims are for a specific example
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))

# model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
# model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))
# model.add(Conv2D(16, (3, 3), activation='relu', padding='valid'))

model.add(Conv2D(16, (3, 3), strides = 11, activation='relu', padding='valid'))
model.add(Dropout(0.5))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Dropout(0.5))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))
model.add(Conv2D(16, (3, 3), activation='relu', padding='valid', input_shape=input_dims))

model.add(Flatten())
# we want to get the 

model.add(Dense(y_train.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
# model.add(Activation('sigmoid'))
### Regular SGD 
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='binary_crossentropy',
#               optimizer=sgd)
###

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy','acc',f1] )
print(model.summary())

# preds = model.predict(X_test)
# Note that we cannot use too large a batch size! 64*32*512*512 is just over 5gb!!
!pwd
LIMIT_STEPS = None
model.fit_generator(generator=training_generator, epochs=10,
                    steps_per_epoch=LIMIT_STEPS , validation_data=valid_generator,  verbose = 1, 
                    validation_steps=LIMIT_STEPS )

# model.save('my_model_nov26.h5')

model.save('my_model_nov30_hard_sig.h5')
!ls .
!pwd
from keras.models import load_model

model = load_model('../input/csc420-pea-working-model-evaluate-generator/my_model_nov26.h5')

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy', f1])
print(model.summary())
# model.fit_generator(generator=training_generator, epochs=1,  verbose = 1)
model.evaluate_generator(generator=training_generator,  verbose = 1)
def f1_score(predictions, true_labels, threshold):
    '''computes the f1_score for a given set of predictions and the true labels, and threshold'''
#     find true positives, false negatives and so forth; these depend on the threshold!
# they have some fancy softmax to do it without using np where and so forth! and that's fine! 
# ah, so they just provide an alternate formulation of the cost fnction!! 

#     true_positive = # wherever they agree and they are both 1 (AND they exceed the threshold)!
#     true_negative = # whether they agree and they are both 0 
    
    
#     y_pred = K.round(y_pred)
#     find y_pred by checking it vs the vector of thresholds 
# do the vector comparison! 

    predictions[predictions > threshold] = 1
    y_pred = predictions
    y_true = true_labels
#     y_pred =  

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

#     return score

    
    pass

# find optimal f1_score: we also want to find the threshold which is best
def find_optimal_f1_threshold(predictions, true_labels):
    params = 0.5*np.ones(len(name_label_dict))

#     call to least squares here
# least squares
#     p should be a 28-dim vector
    error = lambda p: (f1_score(predictions,true_labels,p)
                                      ) # flatten the arrays before concatenating them!
#     so this now will tell us what we want vs what we get
#     f1 score will give us a score, while the true score will give us the wd*(p-0.5)

    #     so we can get out the solutions to this problem, as well as the covaraince stuff 
    import scipy
    
    p, success = scipy.optimize.leastsq(error, params) 
    
    return p

# then, we apply the threshold to get the actual classifications
#  the threshold is done via already making classifications! hence, we need to back up a bit!
params = {'dim': (512,512),
          'batch_size': 64,
          'n_classes': 28,
          'n_channels': 1,
          'shuffle': True}
evaluation_generator = MyDataGenerator(data, **params)


# submit = pd.read_csv(DIR + '/sample_submission.csv')
# P = np.zeros((pathsTest.shape[0], 28))
# for i in tqdm(range(len(testg))):
#     images, labels = testg[i]
#     score = bestModel.predict(images)
#     P[i*BATCH_SIZE:i*BATCH_SIZE+score.shape[0]] = score
model.evaluate_generator(generator=evaluation_generator, verbose = 1)

# we just want to get all the predictions out now!! 

# my_predictions =model.predict_generator(generator=evaluation_generator, verbose = 1)

# smaller evaluation generator
# evaluation_generator = MyDataGenerator(data[0:1000], **params)
# my_predictions =model.predict_generator(generator=evaluation_generator, verbose = 1)

#CONSIDER  we are also randomizing the batch data, so we should get the actual order that it was randomized as well!
print(len(evaluation_generator))
print(len(training_generator))

485*32*2
# now, let us examine the predictions, potentially needing to sort them!
# we really just need to turn it into a string, amenable for the predictions!! 
LIMIT_SIZE = None
# print(my_predictions)
labels = list(map(str, np.arange(0,28)))

mlb = MultiLabelBinarizer(classes=labels)
x = data["Target"].apply(lambda x: x.split())[:LIMIT_SIZE]

label_vectors = mlb.fit_transform(x)
print(len(my_predictions))
print(len(label_vectors))
# ideally: what we have is the following: both a list of thresholds for all the classes, as well as
# the scores for al of the classes
# find_optimal_f1_threshold(my_predictions,label_vectors )
from tqdm import tqdm

lastFullValPred = np.empty((0, 28))
lastFullValLabels = np.empty((0, 28))
for i in tqdm(range(len(evaluation_generator))): 
    im, lbl = evaluation_generator[i]
    scores = model.predict(im)
    lastFullValPred = np.append(lastFullValPred, scores, axis=0)
    lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
print(lastFullValPred.shape, lastFullValLabels.shape)


from sklearn.metrics import f1_score as off1
rng = np.arange(0, 0.00, 0.01)
rng = np.array([0])
print(rng.shape)
# print(rng)

rng = np.concatenate((rng, np.arange(0.01, 1,0.01)), None )
rng = np.arange(0.001, 1, 0.01)
print(rng)
f1s = np.zeros((rng.shape[0], 28))

row_sums = lastFullValPred.sum(axis=1)
print(row_sums[1])
print("rybbubg")
normalized_lastFullValPred = lastFullValPred/row_sums[:, np.newaxis]
# print(normalized_lastFullValPred[0:1])
# print(np.sum(normalized_lastFullValPred[0]))
for j,t in enumerate(tqdm(rng)):
    for i in range(28):
#         print(p)
        
        p = np.array(normalized_lastFullValPred[:,i]>t, dtype=np.int8)
        scoref1 = off1(lastFullValLabels[:,i], p, average='binary')
        f1s[j,i] = scoref1
        

print('Individual F1-scores for each class:')
print(np.max(f1s, axis=0))
print('Macro F1-score CV =', np.mean(np.max(f1s, axis=0)))



plt.plot(rng, f1s)
T = np.empty(28)
for i in range(28):
    T[i] = rng[np.where(f1s[:,i] == np.max(f1s[:,i]))[0][0]]
print('Probability threshold maximizing CV F1-score for each class:')
print(T)
def getTestDataset():
    
    path_to_test = INPUT_PATH  + '/test/'
    test_data = pd.read_csv(INPUT_PATH + '/sample_submission.csv')

    paths = []
    labels = []
    
    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)
pathsTest, labelsTest = getTestDataset()
BATCH_SIZE = 64

params = {'dim': (512,512),
          'batch_size': 64,
          'n_classes': 28,
          'n_channels': 1,
          'shuffle': True, 
         'train': False}

test_data = pd.read_csv(INPUT_PATH + '/sample_submission.csv')
print(test_data)

testg = MyDataGenerator(test_data, **params)
submit = pd.read_csv(INPUT_PATH + '/sample_submission.csv')
P = np.zeros((len(test_data), 28))
for i in tqdm(range(len(testg))):
    images, labels = testg[i]
    score = model.predict(images)
#     print(score)
#     print(score.shape)
#     score = np.
    row_sums = score.sum(axis=1)
    score = score / row_sums[:, np.newaxis]
    
#     score = score / np.sum(score)
#     print(score)
#     print(np.sum(score, axis=1))
    P[i*BATCH_SIZE:i*BATCH_SIZE+score.shape[0]] = score

PP = np.array(P)
prediction = []
# submit['Predicted'] = []
for row in tqdm(range(submit.shape[0])):
#     print("ok")
    str_label = ''
    
    for col in range(PP.shape[1]):
        if(PP[row, col] < T[col]):
#             print(T[col])
            str_label += ''
        else:
            str_label += str(col) + ' '
    prediction.append(str_label.strip())
#     print("these are preds")
#     print(prediction)
#     break
    
submit['Predicted'] = np.array(prediction)
submit.to_csv('csc420_scratch.csv', index=False)
print(submit.to_csv)