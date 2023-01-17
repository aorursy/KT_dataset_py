# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# print(df.isna().values.any())

# print(X_test.isna().values.any())

label = df.label

X = df.drop(['label'],axis=1)

X.shape
scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)



X_test = scaler.transform(X_test)
pca2 = PCA(n_components=331)

pca2.fit(X)

X_new2 = pca2.transform(X)



pca3 = PCA(n_components=3)

pca3.fit(X)

X_new3 = pca3.transform(X)

plt.scatter(X_new2[:,0],X_new2[:,1],c=df.label,alpha=0.1)

plt.title('Digitizer Plot in 2D')

plt.xlabel('X1')

plt.ylabel('X2')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_new3[:,0],X_new3[:,1], zs=X_new3[:,2],c=df.label)

plt.show()
pca = PCA().fit(X)

qsum = np.cumsum(pca.explained_variance_ratio_) #list of the cumulative sum of the variance

threshold = 0.99 #threshold of variance retention



num_features = np.argmax(qsum>threshold) + 1 #number of features needed to retain variance within the threshold

print(num_features)
plt.plot(qsum)

plt.xlabel('Number of Components')

plt.ylabel('Cumulative Explained Variance')

plt.xlim(0,600)

plt.grid(True)

plt.show()
pca533 = PCA(n_components=533)

pca533.fit(X)

X_new533 = pca533.transform(X)



X_test_new = pca533.transform(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X_new533, label, test_size=0.2, random_state=1)

clf = MLPClassifier(hidden_layer_sizes=(100,50,25),solver='lbfgs',alpha=1, random_state=1).fit(X_train, y_train)

clf.predict(X_valid)

print(clf.score(X_valid,y_valid))
test_predict = clf.predict(X_test_new)
test_predict
len(test_predict)


submission = pd.Series(test_predict,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),submission],axis = 1)

submission.to_csv("final_submission.csv",index=False)

submission.head()