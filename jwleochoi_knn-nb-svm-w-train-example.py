# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import bson

import io

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB
# read bson file into pandas DataFrame



data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))



n = 82 #cols of data in train_example set

X_ids = np.zeros((n,1)).astype(int)

Y = np.zeros((n,1)).astype(int) #category_id for each row

X_images = np.zeros((n,180,180,3)) #m images are 180 by 180 by 3



print("Examples:", n)

print("Dimensions of Y: ",Y.shape)

print("Dimensions of X_images: ",X_images.shape)
# prod_to_category = dict()

i = 0

for c, d in enumerate(data):

    X_ids[i] = d['_id'] 

    Y[i] = d['category_id'] 

    for e, pic in enumerate(d['imgs']):

        picture = imread(io.BytesIO(pic['picture']))

    X_images[i] = picture #add only the last image 

    i+=1

    

    #show update every 10 images

    if c > 0 and c % 10 == 0:

        print("[INFO] processed {}/{}".format(c, 82))
# flatten images

X_flat = X_images.reshape(X_images.shape[0], -1)

X_flat = X_flat/255
# partition the data into training and testing splits, using 75%

# of the data for training and the remaining 25% for testing

(trainRI, testRI, trainRL, testRL) = train_test_split(

    X_flat, Y, test_size=0.25, random_state=42)
# train and evaluate a k-NN classifer on the raw pixel intensities

print("[INFO] evaluating raw pixel accuracy...")

model = KNeighborsClassifier(n_jobs=-1)

model.fit(trainRI, trainRL)

acc = model.score(testRI, testRL)

print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
model.predict(testRI)
testRL
# train and evaluate a Gaussian Naive Bayes classifer on the raw pixel intensities

### Training a model

gnb = GaussianNB()

gnb = gnb.fit(trainRI,trainRL)



### Prediction result

acc = gnb.score(testRI, testRL)

print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
gnb.predict(testRI)
testRL
# Now, your classifier is 'svm'

from sklearn import svm

# kernel: specifies the kernel type to be used in the algorithm (linear, poly, rbf, sgmoid, precomputed)

# C: penalty parameter C of the error term

print("Support Vector Machine(SVM)")

clf = svm.SVC(kernel='linear', C=1).fit(trainRI, trainRL)

print(clf.score(testRI, testRL))
clf.predict(testRI)
testRL