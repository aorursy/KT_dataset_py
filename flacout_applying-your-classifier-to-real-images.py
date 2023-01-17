import pandas as pd

import numpy as np

import matplotlib.pyplot as plt, matplotlib.image as mpimg

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.svm import SVC
labeled_images = pd.read_csv('../input/train.csv')

images = labeled_images.iloc[:,1:]

labels = labeled_images.iloc[:,:1]

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
pca = PCA(n_components=75, whiten=True)

train_pca = pca.fit_transform(train_images)

test_pca = pca.transform(test_images)



svc = SVC().fit(train_pca, train_labels)

print("train", svc.score(train_pca,train_labels))

print("test", svc.score(test_pca,test_labels))
from PIL import Image



pathToImage = '0_2828.png'



image = Image.open(pathToImage, 'r')

image = image.resize((28,28,),Image.ANTIALIAS) # I resize the image to fit my classifier

# the getdata method return RBG values so I only keep one of these values

# the image was in black and white anyway

pixels = list(image.getdata())

blackPixels, _, _ = zip(*pixels) 

blackPixels = np.array(blackPixels)



# I invert the values because my classifier was trained like that.

whitePixels = 255 - blackPixels

print( "prediction is :", svc.predict( pca.transform(whitePixels)) )



# then I reshape my array to display it with imshow

whitePixels = whitePixels.reshape((28,28))

plt.imshow(whitePixels,cmap='gray')