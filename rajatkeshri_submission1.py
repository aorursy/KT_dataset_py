import os
import numpy as np

from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Reshape
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD, RMSprop, Adam

import matplotlib.pylab as plt
%matplotlib inline

from random import shuffle
from scipy.misc import imresize 

!unzip  ../input/gazing-with-ml.zip 
train='data/train'
test='data/test'
valid='data/valid'

data_path='data/'
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
    model.add(Lambda(lambda x : x, input_shape=(3,106,106))) # Depending on size of input image modify number
########################################### Input Your Code Here #############################################################################    
# example for convolution    ConvBlock( no of layers , model , no of filters , type of activation )
# and for max pooling        MaxPBlock(no of layers , model)
    ConvBlock(2,model,64,'relu')
    MaxPBlock(1,model)
    ConvBlock(2,model,128,'relu')
    MaxPBlock(1,model)
    ConvBlock(3,model,256,'relu')
    MaxPBlock(1,model)
    ConvBlock(3,model,512,'relu')
    MaxPBlock(1,model)
    """ConvBlock(3,model,512,'relu')
    MaxPBlock(1,model)"""

##############################################################################################################################################  
    model.add(Flatten())
########################################### Input Your Code Here #############################################################################    
# example    NNBlock( no of layers , model , type of activation , dropout )

    NNBlock(128,model,'relu',0.5)
    NNBlock(128,model,'relu',0.5)
    
    model.add(Dense(3, activation = 'softmax'))
    return model

  
optimizer = RMSprop(lr=1e-6)
model = Architecture()
model.compile(loss='mean_squared_error', optimizer=optimizer)
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
def process_images(paths):
    """
    Import image at 'paths', decode, centre crop and prepare for batching. 
    """
    count = len(paths)
    arr = np.zeros(shape=(count,3,106,106))
    for c, path in enumerate(paths):
        img = plt.imread(path).T
        img = img[:,106:106*3,106:106*3] #crop 424x424 -> 212x212
        img = imresize(img,size=(106,106,3),interp="cubic").T # downsample to half res
        arr[c] = img
    return arr
## Print some before/after processing images

#process_images([fetcher.train_path + '/' + fetcher.training_images_paths[100]])
im = plt.imread(fetcher.train_path + '/' + fetcher.training_images_paths[0])
# print(im.shape)

plt.imshow(im)
plt.show()
im = im.T[:,106:106*3,106:106*3] #crop 424x424 -> 212x212
# im = imresize(im,size=(106,106,3),interp="cubic").T # downsample to half res
# print(im.shape)
plt.imshow(im.T)
# Create generator that yields (current features X, current labels y)
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
epochs_batch = 20
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
print(predictions)
noofpred = 0
count=0
for f in fetcher.training_images_paths:
    
    img = process_images([fetcher.train_path + '/' + fname for fname in [f]])
    output=model.predict(img)
    count=count+1
    if count%1000==0:
        print(count)
    pred_label = [np.argmax(output)]
    id_ = fetcher.get_id(f)
    y_train = fetcher.find_label(id_)
    if y_train==pred_label:
        noofpred += 1
        print(noofpred)
totalpred=int(len(fetcher.training_images_paths))
accuracy = noofpred/totalpred
print(accuracy)
plt.figure(figsize=(12,8))
plt.plot(hist.epoch,hist.history['loss'],label='Test')
plt.plot(hist.epoch,hist.history['val_loss'],label='Validation',linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.show()
predictions.shape
#header = open('all_zeros_benchmark.csv','r').readlines()[0]

with open('submission_1.csv','w') as outfile:
    #outfile.write(header)
    for i in range(len(fetcher.test_images_paths)):
        id_ = (fetcher.get_id(fetcher.test_images_paths[i]))
        pred = predictions[i]
        outline = id_ + "," + ",".join([str(x) for x in pred])
        outfile.write(outline + "\n")
!rm -r data
with open('submission_1.csv','w') as outfile:
    outfile.write("Id,Catergory\n")
    for i in range(len(fetcher.test_images_paths)):
        id_ = (fetcher.get_id(fetcher.test_images_paths[i]))
        pred =[np.argmax(predictions[i])]
        outline = id_ + "," + ",".join(str(pred[0]))
        outfile.write(outline + "\n")
!cat submission_1.csv
