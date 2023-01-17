%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

from IPython.core.pylabtools import figsize 
X, y = make_blobs(n_samples=50, centers=2,random_state=0, cluster_std=0.60)
X
y
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
xfit = np.linspace(-1, 3.5)

plt.figure(figsize=(8,8))

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:

    plt.plot(xfit, m * xfit + b, '-k')
xfit = np.linspace(-1, 3.5)

plt.figure(figsize=(8,8))

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')



for m, b, d in [(1, 0.65, 0.4), (0.5, 1.6, 0.6), (-0.2, 2.9, 0.3)]:

    yfit = m * xfit + b

    plt.plot(xfit, yfit, '-k')

    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',

                     color='#AAAAAA', alpha=0.4)
from sklearn.svm import SVC # "Support vector classifier"

model = SVC(kernel='linear')

model.fit(X, y)
X, y = make_blobs(n_samples=50, n_features=3,centers=2,random_state=0, cluster_std=0.60)

from mpl_toolkits import mplot3d

ax = plt.subplot(projection='3d')

ax.scatter3D(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('r')

plt.show()
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
import scipy.io as scio
x_train_data = scio.loadmat("/kaggle/input/brain-signal/x_train.mat")
y_train_data = scio.loadmat("/kaggle/input/brain-signal/y_train.mat")
x_train_data
x_train_a = x_train_data["x_train"]
x_train_a.shape
x_train_a[0]
df = pd.DataFrame(x_train_a[0])

df
df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)

df2
df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)

df3 = df2

for i in range(1,1152):

    df = pd.DataFrame(x_train_a[i])

    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)

    df3 = pd.merge(df2,df3,"outer",left_index=True,right_index=True)

x_train = df3

x_train
y_train_data
y_train_a = y_train_data["y_train"]
y_train = pd.DataFrame(y_train_a)

y_train
train = pd.merge(y_train,x_train,"outer",left_index=True,right_index=True)

train
import os

os.environ['KERAS_BACKEND']='tensorflow'

from sklearn.model_selection import train_test_split

import eli5

X_train1,X_test1,Y_train1,Y_test1 = train_test_split(x_train,y_train,

                                                 test_size=0.3,shuffle=True,random_state = 133)

print(X_train1.shape,Y_train1.shape,X_test1.shape,Y_test1.shape)
from sklearn import svm

clf = svm.SVC(kernel='linear')

clf.fit(X_train1,Y_train1)

clf.score(X_test1,Y_test1)
from sklearn import svm

clf = svm.SVC(C=0.1,kernel='linear')

clf.fit(X_train1,Y_train1)

clf.score(X_test1,Y_test1)
from sklearn import svm

clf = svm.SVC()

clf.fit(X_train1,Y_train1)

clf.score(X_test1,Y_test1)
new_col = [i for i in range(1,1153)]
x_train_ch1 = x_train.iloc[:,0::3]

x_train_ch1.columns = new_col

x_train_ch1
x_train_ch2 = x_train.iloc[:,1::3]

x_train_ch3 = x_train.iloc[:,2::3]

x_train_ch2.columns = new_col

x_train_ch3.columns = new_col
import matplotlib.pyplot as plt

plt.figure(figsize(20,8))

x_train_ch1.iloc[0,:].plot(kind="line")

plt.show()
plt.figure(figsize(20,8))

x_train_ch2.iloc[0,:].plot(kind="line")

plt.show()
plt.figure(figsize(20,8))

x_train_ch3.iloc[0,:].plot(kind="line")

plt.show()
plt.figure(figsize(20,8))

x_train_ch1.iloc[0,:].plot(kind="line")

x_train_ch2.iloc[0,:].plot(kind="line")

x_train_ch3.iloc[0,:].plot(kind="line")

plt.show()
plt.figure(figsize(20,8))

x_train_ch1.iloc[1,:].plot(kind="line")

plt.show()
plt.figure(figsize(20,8))

x_train_ch2.iloc[1,:].plot(kind="line")

plt.show()
plt.figure(figsize(20,8))

x_train_ch3.iloc[1,:].plot(kind="line")

plt.show()
plt.figure(figsize(20,8))

x_train_ch1.iloc[1,:].plot(kind="line")

x_train_ch2.iloc[1,:].plot(kind="line")

x_train_ch3.iloc[1,:].plot(kind="line")

plt.show()
plt.figure(figsize(20,8))

x_train_ch1.iloc[0,:].plot(kind="line")

x_train_ch1.iloc[1,:].plot(kind="line")

plt.show()