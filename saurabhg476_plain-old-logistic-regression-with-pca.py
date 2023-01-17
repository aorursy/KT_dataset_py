%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
def model(X, variance):
    '''

    :param variance(float): amount of variance in data to be kept after applying PCA
    :param X: shape (m,n)
    :return:
    '''

    m = X.shape[0]
    n = X.shape[1]

    #mean normalization
    mean = X.mean(axis=0,keepdims=True)
    range_ = X.ptp(axis=0).reshape(1,n)

    X = (X - mean)/range_

    sigma = (1/m) *np.dot(X.T,X)
    (u,s,v) = np.linalg.svd(sigma)

    #selecting k i.e. number of principal components
    total_sum = np.sum(s)
    running_sum = 0.0

    for i, value in enumerate(s,start=1):
        running_sum += value
        if running_sum/total_sum >= variance:
            break

    k = i
    u_reduce = u[:,:k]

    Z = np.dot(X,u_reduce)


    ###printing k
    #print(k)
    ###printing approximations
    X_approx = np.dot(Z,u_reduce.T)
    print(X_approx*range_ + mean)

    return Z, k
y = train['label']
X = train.iloc[:,1:]

pca = PCA()
pca.fit(X)
for i,value in enumerate(pca.explained_variance_ratio_.cumsum(),start=1):
    if(value > 0.99):
        print(i,value)
        break


pca = PCA(n_components=331)
pca.fit(X)
X_new = pca.transform(X)

model = LogisticRegression(solver='sag')
model.fit(X_new,y)
test_new = pca.transform(test)
predict = model.predict(test_new)
print(predict.shape)

submission = {
    "ImageId": np.arange(1,predict.shape[0]+1),
    "Label": predict
}
pd.DataFrame.from_dict(submission).to_csv("submission.csv",index=False)
