import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import svm
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing the data to pandas dataframes
train_digits = pd.read_csv('../input/train.csv')
test_images = pd.read_csv('../input/test.csv')
#splitting the data (keeping the data to 2000 observations to save time)
train_labels = train_digits.iloc[0:2000,0]
train_images = train_digits.iloc[0:2000,1:]
train_X, test_X, train_y, test_y = train_test_split(train_images, train_labels,
                                                    test_size=0.33, random_state=42)

#I'm going to take a quick look at one of the rows of the dataframe. 
#Each row of the matrix is an image that is 28x28 pixels, so we should have 784 columns. 
#if we reshape the an observation, one row of data, into a 28x28 matrix we can see image.
#here is a quick function to print the image correcty
def display(image):
    single_image = image.reshape((28,28))
    plt.imshow(single_image, cmap=cm.Pastel1_r)
    
display(train_images.iloc[10].values)

#Note that these images aren't actually black and white (0,1). They are gray-scale (0-255). You can 
#see this clearly with the funky pastels
#Let's standardize the greyscale between 1 and 0. 
train_X = train_X.apply(lambda x : x*(1/255))
test_X = test_X.apply(lambda x : x*(1/255))
#running just a simple support vector machine (SVM) classifier on the data. 

clf = svm.SVC()
clf.fit(train_X, train_y.values.ravel())
clf.score(test_X,test_y)
test_images = test_images.apply(lambda x : x*(1/255))
results = clf.predict(test_images)
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)