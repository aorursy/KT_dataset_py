# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_vehicles_dir = "../input/train/train/vehicles/"
train_vehicles_names = os.listdir(train_vehicles_dir)
train_vehicles = np.array([cv2.imread(train_vehicles_dir + image_name) for image_name in train_vehicles_names])

train_vehicles.shape # print out 6014,64,64,3
# A list in python
list_of_images = ["image_1","image_2","image_3"]

# In python we can iterate over a list just like this
print("*"*20 + "Usual Python version" + "*"*20)
for image in list_of_images : 
    print(image)
    
# In C++, it would normally be like : 
#     for (int i = 0; i++; i<n_images)
#     {
#         cout << list_of_images[i] << endl;
#     }
    
# Python make it simpler by the use of iterators, the above python instructions are equivalent to :
print("*"*20 + "Detailled Python version" + "*"*16)
# create an iterator object from that iterable
iter_obj = iter(list_of_images)

# infinite loop
while True:
    try:
        # get the next item
        image = next(iter_obj)
        print(image)
        # do something with element
    except StopIteration:
        # if StopIteration is raised, break from loop
        break
# Way of defining an iterator 
class PowTwo:
    def __init__(self, max = 0):
        self.max = max

    def __iter__(self):  # We need to define this method
        self.n = 0
        return self

    def __next__(self):  # And this one to clearly states that this class does the job of an iterator
        if self.n > self.max:
            raise StopIteration

        result = 2 ** self.n
        self.n += 1
        return result
 
# You can use it like this 
PowTwo_instance = PowTwo(3)
iterator = iter(PowTwo_instance)
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator)) # raise StopIteration error cause 3 is the max
# Same thing with a generator
def PowTwoGen(max = 0):
    n = 0
    while n < max:
        yield 2 ** n
        n += 1
        
# You can use it like this
PowTwo_func = PowTwoGen(3)
iterator = iter(PowTwo_func)
print(next(iterator))
print(next(iterator))
print(next(iterator))

# Or equivalently : 
for i in PowTwoGen(3) : 
    print(i)      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Generator
generator = ImageDataGenerator()

# Generating images
batch_size = 2
for i in generator.flow(train_vehicles, batch_size=batch_size) :
    for image_index in range(batch_size) : 
        plt.imshow(np.reshape(i[image_index], (64,64,3)))
        plt.show()
    break  # You need to break at some point otherwise it will loop forever

# Generator
generator = ImageDataGenerator()
train_generator = generator.flow_from_directory("../input/train/train/", target_size=(64, 64), batch_size=batch_size)

for batch in train_generator:
    for image_index in range(batch_size) : 
        plt.imshow(np.reshape(batch[0][image_index], (64,64,3)))
        plt.show()
    break  # You need to break at some point otherwise it will loop forever    
