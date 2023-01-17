# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
img_data=np.load("/kaggle/input/eval-lab-4-f464/train.npy",allow_pickle=True);

img_pred=np.load("/kaggle/input/eval-lab-4-f464/test.npy",allow_pickle=True);
img_pred[0]
img_pred[2][1].shape
img_pred.shape
for i in range(2275):

    img_data[i][1] = img_data[i][1] / 255.0 # use 0...1 scale

    img_data[i][1] = img_data[i][1].reshape(50*50*3)



img_data[0][1].shape
for i in range(976):

    img_pred[i][1] = img_pred[i][1] / 255.0 # use 0...1 scale

    img_pred[i][1] = img_pred[i][1].reshape(50*50*3)

    
from matplotlib import pyplot as plt

def plot_pixels(data, title, colors=None, N=10000):

    if colors is None:

        colors = data

    

    # choose a random subset

    rng = np.random.RandomState(0)

    i = rng.permutation(data.shape[0])[:N]

    colors = colors[i]

    R, G, B = data[i].T

    

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].scatter(R, G, color=colors, marker='.')

    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))



    ax[1].scatter(R, B, color=colors, marker='.')

    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))



    fig.suptitle(title, size=20);

plot_pixels(img_data[0][1], title='Input color space: 16 million possible colors')
img_data[0]
labels=[]



for i in range(2275):

    labels.append(img_data[i][0])

#labels



u_labels=set(labels)

u_labels=list(u_labels)

len(u_labels)
#So we need 19 clusters for our data!

u_labels
img_data[0][1].shape

# X_new=img_data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
from sklearn.svm import LinearSVC

from sklearn.datasets import make_classification



X=[]

y=[]

X_test = []



for i in range(2275):

    X.append(img_data[i][1])

#     img_data[i][1].shape[0]

    y.append(img_data[i][0])

    

for i in range(976):

    X_test.append(img_pred[i][1])



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(u_labels)



list(le.classes_)



y=le.transform(y) 

y

#list(le.inverse_transform([2, 2, 1]))



X[0].shape
clf = LinearSVC(random_state=0, tol=1e-5)

clf.fit(X, y)  

print(clf.predict(img_data[0][1]))
X[0].shape
X_new = np.asarray(X)
X_test_new = np.asarray(X_test)
X_new.shape
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()



gnb.fit(X_new, y)
y_pred = gnb.predict(X_test_new)
y_pred
ids = []

for i in range(976):

    ids.append(img_pred[i][0])
y_pred_new = le.inverse_transform(y_pred)
y_pred_new
df = pd.DataFrame({'ImageId': ids,'Celebrity':y_pred_new })
df.head()
for i in range(2275):

    print(y[i],y_pred[i])
df.to_csv('First.csv',index=False)