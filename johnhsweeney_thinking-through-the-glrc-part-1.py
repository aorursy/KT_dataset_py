import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Image
from IPython.core.display import HTML 
from urllib import request
# read in the list of train and test photos

# Kaggle version
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Let's make sure that went as expected...  There should be 117703 test photos and 1225029 train photos

print("Number of photos for testing = ", test.shape[0])
print("Number of photos for training = ", train.shape[0])

# A utility for displaaying thumbnails of images
# Taken from the very nice Kernel by Gabriel Preda
# "Google Landmark Recogn. Challenge Data Exploration"

def displayLandmarkImagesLarge(urls):
    
    imageStyle = "height: 150px; margin: 2px; float: left; border: 1px solid blue;"
    imagesList = ''.join([f"<img style='{imageStyle}' src='{u}' />" for _, u in urls.iteritems()])

    display(HTML(imagesList))
np.random.seed(42) # my favorite "random" seed, change it if you want other images

urls = [] # start with an empty list...

for i in range (20):
    urls.append(train.iloc[np.random.randint(0,1225029),1])
    
urls = pd.Series(urls)

displayLandmarkImagesLarge(urls) # Thanks Gabriel!

np.random.seed(1960) # guess how old I am..., change it if you want other images

urls = [] # start with an empty list...

for i in range (20):
    urls.append(test.iloc[np.random.randint(0,117703),1])
    
urls = pd.Series(urls)

displayLandmarkImagesLarge(urls) # Thanks again Gabriel!
# Review some of the VERY small images in the training set

# A handful of 10x15 images

urls = []
urls.append('https://lh4.googleusercontent.com/-2aSO8nzNfeY/Tne4ZHE2ZKI/AAAAAAAAC1Y/5eRU1tfinQI/s15/')
urls.append('https://lh3.googleusercontent.com/-SXCAgqmUSCY/TKKFZqwVxxI/AAAAAAAADbw/H440k4K4rlY/w11-h15/')
urls.append('https://lh5.googleusercontent.com/-wgFxt042p-4/SkdfH8QuuWI/AAAAAAAADuw/F2hmQxBuVdc/s15/')
urls.append('https://lh4.googleusercontent.com/-qyGFiv31etQ/RzHbxBZUXkI/AAAAAAAABV8/vsSEsNKwQLM/s15/')
urls.append('https://lh4.googleusercontent.com/-1GaFiQamJmU/SYdLH0vdjBI/AAAAAAAAFck/2vwbPssj1xg/s15/')

urls = pd.Series(urls)

displayLandmarkImagesLarge(urls)

# Review some of the VERY small images in the training set

# A handful of 10x15 images

urls = []
urls.append('https://lh4.googleusercontent.com/-2aSO8nzNfeY/Tne4ZHE2ZKI/AAAAAAAAC1Y/5eRU1tfinQI/')
urls.append('https://lh3.googleusercontent.com/-SXCAgqmUSCY/TKKFZqwVxxI/AAAAAAAADbw/H440k4K4rlY/')
urls.append('https://lh5.googleusercontent.com/-wgFxt042p-4/SkdfH8QuuWI/AAAAAAAADuw/F2hmQxBuVdc/')
urls.append('https://lh4.googleusercontent.com/-qyGFiv31etQ/RzHbxBZUXkI/AAAAAAAABV8/vsSEsNKwQLM/')
urls.append('https://lh4.googleusercontent.com/-1GaFiQamJmU/SYdLH0vdjBI/AAAAAAAAFck/2vwbPssj1xg/')

urls = pd.Series(urls)

displayLandmarkImagesLarge(urls)
