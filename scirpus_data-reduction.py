import numpy as np

import pandas as pd

import pywt

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline
directory = '../input/'

train = pd.read_csv(directory+'train.csv')

test = pd.read_csv(directory+'test.csv')

features = train.columns[1:]

trainlabels = train.label 

alldata = pd.concat([train[features],test[features]])
def MungeData(im):

    coeffs = pywt.dwt2(data=im, wavelet='haar')

    return coeffs[0].ravel()



reduceddata = None

for i in range(alldata.shape[0]):

    if(i % 5000 == 0):

        print(i)

    im = alldata.iloc[i][features]

    im = im.reshape(28, 28)

    imwavelet = MungeData(im)

    if(reduceddata is None):

        reduceddata = np.zeros((alldata.shape[0], len(imwavelet)))

    reduceddata[i, :] = imwavelet
plt.imshow(train.iloc[0][features].reshape(28,28), cmap='gray')
plt.imshow(reduceddata[0].reshape(14,14), cmap='gray')
PCA_COMPONENTS = 50

pca = PCA(n_components=PCA_COMPONENTS)

pcadata = pca.fit_transform(reduceddata)
from sklearn.cross_validation import StratifiedKFold

skf = StratifiedKFold(trainlabels,n_folds=10)

avg = 0

for train_index, test_index in skf:

    knn = KNeighborsClassifier(n_neighbors=8,p=2)

    knn.fit(pcadata[:trainlabels.shape[0]][train_index],trainlabels[train_index])

    a = (accuracy_score(trainlabels[test_index], knn.predict(pcadata[:trainlabels.shape[0]][test_index])))

    print(a)

    avg+=a

print('AverageScore',avg/10.)
knn.fit(pcadata[:trainlabels.shape[0]],trainlabels)

preds = knn.predict(pcadata[trainlabels.shape[0]:])

testsubmission = pd.DataFrame({'ImageId': range(1,

                                preds.shape[0]+1),

                               'Label': preds})

testsubmission.to_csv('./knnsubmission.csv', index=False)