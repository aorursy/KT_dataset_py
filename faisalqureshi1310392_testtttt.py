import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import skimage.transform as transform
import sklearn.cross_validation as cv
from sklearn.svm import SVC

from skimage import img_as_ubyte
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.filters import sobel

def get_data():
    dataset = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    target = dataset[[0]].values.ravel()
    train = dataset.iloc[:,1:].values
    target = target.astype(np.uint8)
    train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
    test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
    return train,target,test

def preprocess_and_rotate(X,num):
    
    images_45 = []
    images_315 = []
    original = []

    for i in range(0,num):

        thresh = threshold_otsu(X[i][0])
        X[i][0] = X[i][0] > thresh

        noisy_image = img_as_ubyte(X[i][0])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        X[i][0] = median(noisy_image, disk(1))
        
        #corners_image = sobel(X[i][0])
        images_45.append(transform.rotate(X[i][0],45).flatten())
        images_315.append(transform.rotate(X[i][0],315).flatten())
        original.append(X[i][0].flatten())

    return np.array(images_45, 'float64'), np.array(images_315, 'float64'), np.array(original, 'float64')

train,target,test = get_data()
images_45_train, images_315_train, orginal_images_train = preprocess_and_rotate(train,42000)
#images_45_test, images_315_test, orginal_images_test = preprocess_and_rotate(test,28000)

model1 = RandomForestClassifier(n_estimators=150)
model2 = RandomForestClassifier(n_estimators=150)
model3 = RandomForestClassifier(n_estimators=150)
model1.fit(images_45_train,target)
model2.fit(images_315_train,target)
model3.fit(orginal_images_train,target)

ensemble = VotingClassifier(estimators=[('45', model1), ('315', model2), ('0', model3)], voting='hard', weights=[1,1,2])

print(np.mean(cv.cross_val_score(ensemble, orginal_images_train, target)))
