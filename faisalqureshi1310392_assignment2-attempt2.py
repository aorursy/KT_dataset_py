import pandas as pd
import numpy as np
import sklearn.cross_validation as cv
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier

from skimage import img_as_ubyte
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk

def get_train_data():
	dataset = pd.read_csv("../input/train.csv")
	target = dataset[[0]].values.ravel()
	train = dataset.iloc[:,1:].values
	target = target.astype(np.uint8)
	train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
	return train,target

def preprocess(X,num):
    flat = []
    for i in range(0,num):
        thresh = threshold_otsu(X[i][0])
        X[i][0] = X[i][0] > thresh
        noisy_image = img_as_ubyte(train[i][0])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        X[i][0] = median(noisy_image, disk(1))
        flat.append(X[i][0].flatten())
    X_preprocessed = np.array(flat, 'float64')
    return X_preprocessed

train,target = get_train_data()
X_preprocessed = preprocess(train,42000)

num_estimators = 0
while (num_estimators<150):
    num_estimators+=10
    model = RandomForestClassifier(n_estimators=num_estimators)
    print(np.mean(cv.cross_val_score(model, X_preprocessed, target)))

print("Done")

