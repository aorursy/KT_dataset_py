import os
import numpy as np
!unzip  ../input/gazing-with-ml.zip 
data_path = 'data/'
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Reshape
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD, RMSprop, Adam

import matplotlib.pylab as plt
%matplotlib inline
def ConvBlock(layers, model, filters, activation):
    for i in range(layers): 
        model.add(Conv2D(filters, kernel_size=[3,3], activation=activation , padding='same'))  # Edit Filter Size if Required
def MaxPBlock(layers, model):
    for i in range(layers): 
        model.add(MaxPooling2D(pool_size=(2,2) , strides=(2,2) , padding='same'))

def NNBlock(layers, model ,activation , dropout):
    model.add(Dense(layers, activation=activation))
    model.add(Dropout(dropout))
    
def Architecture():
    model = Sequential()
    model.add(Lambda(lambda x : x, input_shape=(424,424,3))) # Depending on size of input image modify number
########################################### Input Your Code Here #############################################################################    
    ConvBlock( 1 , model , 32, 'relu')
    ConvBlock( 1 , model , 32, 'relu')
    MaxPBlock(1 , model)
    ConvBlock( 1 , model , 32, 'relu')
    ConvBlock( 1 , model , 32, 'relu')
    MaxPBlock(1 , model)


##############################################################################################################################################  
    model.add(Flatten())
########################################### Input Your Code Here #############################################################################    
    NNBlock( 3 , model , 'softmax' ,0.5 )
    model.add(Dense(3, activation = 'softmax'))
    return model

  
optimizer = RMSprop(lr=1e-6)
model = Architecture()
model.compile(loss='mean_squared_error', optimizer=optimizer)
from random import shuffle
from scipy.misc import imresize  

class data_getter:    

    def __init__(self, path):    
        self.path = path 
        self.train_path = path + "train"
        self.val_path = path + "valid"
        self.test_path = path + "test"
        
        def get_paths(directory):
            return [f for f in os.listdir(directory)]
        
        self.training_images_paths = get_paths(self.train_path)
        self.validation_images_paths = get_paths(self.val_path)
        self.test_images_paths = get_paths(self.test_path)    
        
        #read Train and valid csv
        def get_all_solutions():
            import csv
            all_solutions = {}
            with open(data_path+'train_valid.csv', 'r') as f:
                reader = csv.reader(f, delimiter=",")
                for i, line in enumerate(reader):
                    all_solutions[line[0]] = [float(x) for x in line[1:]]
            return all_solutions
        
        self.all_solutions = get_all_solutions()

    def get_id(self,fname):
        return fname.replace(".jpg","").replace("data/train/","").replace("data/valid/","")
        
    def find_label(self,val):
        return self.all_solutions[val]
        
fetcher = data_getter(data_path)
print(fetcher.train_path)
def process_images(path):
    arr = np.zeros(shape=(1,424,424,3))
    arr[0] = img
    return arr
path = fetcher.train_path + '/' + fetcher.training_images_paths[0]

## Image Before processing
img = plt.imread(path)
plt.imshow(img)
plt.show()

## Image after processing
imgp = process_images([path])
imgp = imgp.reshape((424,424,3)) # input size here
plt.imshow(np.uint8(imgp))
plt.show()

def BatchGenerator(getter):
    while 1:
        for f in getter.training_images_paths:
            X_train = process_images([getter.train_path + '/' + fname for fname in [f]])
            id_ = getter.get_id(f)
            y_train = np.array(getter.find_label(id_))
            y_train = np.reshape(y_train,(1,3))
            yield (X_train, y_train)
            
def ValBatchGenerator(getter):
    while 1:
        for f in getter.validation_images_paths:
            X_train = process_images([getter.val_path + '/' + fname for fname in [f]])
            id_ = getter.get_id(f)
            y_train = np.array(getter.find_label(id_))
            y_train = np.reshape(y_train,(1,3))
            yield (X_train, y_train)
                        
# Edit batch size and epochs per batch 
batch_size = 32
epochs_batch = 30
steps_to_take = int(len(fetcher.training_images_paths)/batch_size)
val_steps_to_take = int(len(fetcher.validation_images_paths)/batch_size)


hist = model.fit_generator(BatchGenerator(fetcher),
                    samples_per_epoch=steps_to_take, 
                    nb_epoch=epochs_batch,
                    validation_data=ValBatchGenerator(fetcher),
                    nb_val_samples=val_steps_to_take,
                    verbose=1
                   )
def TestBatchGenerator(getter):
    while 1:
        for f in getter.test_images_paths:
            X_train = process_images([getter.test_path + '/' + fname for fname in [f]])
            yield (X_train)

predictions = model.predict_generator(TestBatchGenerator(fetcher),
                       val_samples = len(fetcher.test_images_paths),
                        max_q_size = 32)
# Code to see accuracy 
noofpred = 0 
for f in fetcher.training_images_paths:
         img = process_images([fetcher.train_path + '/' + fname for fname in [f]])
         output=model.predict(img)
         pred_label = [np.argmax(output)]
         id_ = fetcher.get_id(f) 
         y_train = fetcher.find_label(id_)
         if y_train==pred_label:
            noofpred += 1
totalpred=int(len(fetcher.training_images_paths))
accuracy = noofpred/totalpred
print(accuracy)
!rm -r data
with open('submission_1.csv','w') as outfile:
    outfile.write("Id,Catergory\n")
    for i in range(len(fetcher.test_images_paths)):
        id_ = (fetcher.get_id(fetcher.test_images_paths[i]))
        pred =[np.argmax(predictions[i])]
        outline = id_ + "," + ",".join(str(pred[0]))
        outfile.write(outline + "\n")
!cat submission_1.csv