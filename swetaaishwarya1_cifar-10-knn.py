# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Cifar-10 KNN code
import sys 
sys.executable
import numpy as np
import os
#pickle for serialisation and de-serialisation
import pickle
import matplotlib.pyplot as plt
#For embedded Visualisation 
#from vega3 import VegaLite
#API for display tools
from IPython.display import clear_output

#The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. 
#Each of these files is a Python "pickled" object produced with Pickle. 
#Here is a python3 routine which will open such a file and return a dictionary:
rel_path = "/kaggle/input/my-cifar/"
#deserilaizing the data 
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


X = unpickle(rel_path + 'data_batch_1')
img_data = X[b'data']
img_label_orig = img_label = X[b'labels']
img_label = np.array(img_label).reshape(-1, 1)
#Now that our data and labels are loaded lets check their presence 
print(img_data)
print('shape', img_data.shape)
#Perfect the shape is 10000 * 3072 ,that is 10k examples ,each example has 32* 32 image which is 1024 pixels ,hence each pixel has three levels (r,g,b)
#we get 1024 * 3 = 3072 values for a sigle image out of 10k datapoints.

#Now, let's look into labels in data_batch_1
print(img_label)
print('shape', img_label.shape)

# the data to be processed is in dictionary form ,namely key,value where key is image_label and value is image_data 

#Now lets sample the data into test, train for our data model fitting 
test_X = unpickle(rel_path + 'test_batch');
test_data = test_X[b'data']
test_label = test_X[b'labels']
test_label = np.array(test_label).reshape(-1, 1)
#Let's check the Test dataset 
print(test_data)
print('shape', test_data.shape)

# Now ,lets sample the first few images of data  from  data_batch_1
sample_img_data = img_data[0:10, :]
print(sample_img_data)
print('shape', sample_img_data.shape)
# We, obtain 10*3072 data matrice 
#The dataset has image names in the batches.meta file, lets get the file to display the image
batch = unpickle(rel_path + 'batches.meta');
meta = batch[b'label_names']
print(meta)

#Now, let check into hownto display image 
#We can use show_image.py file of python
from PIL import Image
import numpy as np
from IPython.display import display

def default_label_fn(i, original):
    return original


def show_img(img_arr, label_arr, meta, index, label_fn=default_label_fn):
    """
        Given a numpy array of image from CIFAR-10 labels this method transform the data so that PIL can read and show
        the image.
        Check here how CIFAR encodes the image http://www.cs.toronto.edu/~kriz/cifar.html
    """
    
    one_img = img_arr[index,:]
    # Assume image size is 32 x 32. First 1024 px is r, next 1024 px is g, last 1024 px is b from the (r,g b) channel
    r = one_img[:1024].reshape(32, 32)
    g = one_img[1024:2048].reshape(32, 32)
    b = one_img[2048:]. reshape(32, 32)
    rgb = np.dstack([r, g, b])
    img = Image.fromarray(np.array(rgb), 'RGB')
    display(img)
    print(label_fn(index, meta[label_arr[index][0]].decode('utf-8')))
    

for i in range(0, 10):
    show_img(sample_img_data, img_label, meta, i)  
    
#Lets run our KNN algorithm now

#Since this algorith is slow, it takes time to run the model and get predicted outcome.
#Select the number of test data on which you would want to run the model.

    
    
   
from sklearn.neighbors import KNeighborsClassifier
neighbors = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(img_data, img_label_orig)
YPred_soa = neighbors.predict(sample_test_data)
def pred_label_fn(i, original):
    return original + '::' + meta[YPred_soa[i]].decode('utf-8')

for i in range(0, len(YPred_soa)):
    show_img(sample_test_data, test_label, meta, i, label_fn=pred_label_fn)

