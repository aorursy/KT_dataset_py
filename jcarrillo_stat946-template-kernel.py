# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# To import cifar-100 data
import pickle
import numpy as np

with open('../input/train_data/train_data', 'rb') as f:
    train_data = pickle.load(f)
    train_label= pickle.load(f)

# Print training data caracteristics
print('Training images data type: ' + str(type(train_data)))
print('Training images array dimension: ' + str(train_data.shape))
print('Training labels data type: ' + str(type(train_label)))
print('Training labels length: ' + str(len(train_label)))
# Function to extract images in the right format
def getTrainData():
    labels_training = []
    dataImgSet_training = []
    
    # Extract only the first 100 images in the training dataset
    for i in range(100):
            # Get flattened image and label
            img_flat = train_data[i]
            labels_training.append(train_label[i])

            # Extract each color channel of 32x32 image 
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            
            # Set channels last format
            imgFormat = np.array([img_R, img_G, img_B])
            imgFormat = np.transpose(imgFormat, (1, 2, 0))  #Change the shape 3,32,32 to 32,32,3 
            dataImgSet_training.append(imgFormat)
            
    # Convert to numpy array
    dataImgSet_training = np.array(dataImgSet_training)
    labels_training = np.array(labels_training)

    return dataImgSet_training, labels_training
# Call getTrainData() function
X_train, y_train = getTrainData()
# Plot some images
from matplotlib import pyplot
from PIL import Image

# Create a grid of 3x3 images
print("Some images")
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(Image.fromarray(X_train[i]))
    
pyplot.show()
# Here you create and train your model, then you use it to predict in the test dataset
# and most likely you will output a .csv table with the predicted labels

# To save and download the .csv you have to run the entire notebook by pressing 'Commit'
# at the top of this Kernel, it will save the .csv so you can download it later on
# and use it as your submission
# As an example I will create a random submission file and store it as .csv 

# Vector of labels probability (random)
results = np.random.rand(10000, 100)

# Select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="labels")

# Add image id column
submission = pd.concat([pd.Series(range(10000),name = "ids"),results],axis = 1)

# Export to .csv
submission.to_csv("my_predictions.csv",index=False)
