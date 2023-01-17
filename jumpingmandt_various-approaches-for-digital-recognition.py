# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import os



from sklearn.model_selection import train_test_split, KFold



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# check the files inside the folder

arr = os.listdir('../input/digit-recognizer/')

print (arr)
!nvidia-smi
# Check Python Version

!python --version
# Check CUDA/cuDNN Version

!nvcc -V && which nvcc
# INSTALL RAPIDS FROM KAGGLE DATASET.

import sys

!rsync -ah --progress ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!rsync -ah --progress /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
!conda activate rapids

!conda install -c rapidsai dask-cudf -y
# LOAD LIBRARIES

import cudf,cuml

from sklearn.model_selection import train_test_split, KFold

from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

print('cuDF Version:', cudf.__version__)

print('cuML version',cuml.__version__)
# LOAD TRAINING DATA

train = cudf.read_csv('../input/digit-recognizer/train.csv')

print('train shape =', train.shape )

train.head()
# VISUALIZE DATA

samples = train.iloc[5000:5030,1:].to_pandas().values

plt.figure(figsize=(15,4.5))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    plt.imshow(samples[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
from sklearn.model_selection import train_test_split, KFold



# Train test split # CREATE 20% VALIDATION SET

X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,1:], train.iloc[:,0],\

        test_size=0.2, random_state=42)
from cuml.neighbors import KNeighborsClassifier, NearestNeighbors



# GRID SEARCH FOR OPTIMAL K

accs = []

for k in range(3,22):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    # Better to use knn.predict() but cuML v0.11.0 has bug

    # y_hat = knn.predict(X_test)

    y_hat_p = knn.predict_proba(X_test)

    acc = (y_hat_p.to_pandas().values.argmax(axis=1)==y_test.to_array() ).sum()/y_test.shape[0]

    #print(k,acc)

    print(k,', ',end='')

    accs.append(acc)
# PLOT GRID SEARCH RESULTS

plt.figure(figsize=(15,5))

plt.plot(range(3,22),accs)

plt.title('MNIST kNN k value versus validation acc')

plt.show()
# GRID SEARCH USING CROSS VALIDATION

for k in range(3,6):

    print('k =',k)

    oof = np.zeros(len(train))

    skf = KFold(n_splits=7, shuffle=True, random_state=42)

    for i, (idxT, idxV) in enumerate( skf.split(train.iloc[:,1:], train.iloc[:,0]) ):

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(train.iloc[idxT,1:], train.iloc[idxT,0])

        # Better to use knn.predict() but cuML v0.11.0 has bug

        # y_hat = knn.predict(train.iloc[idxV,1:])

        y_hat_p = knn.predict_proba(train.iloc[idxV,1:])

        oof[idxV] =  y_hat_p.to_pandas().values.argmax(axis=1)

        acc = ( oof[idxV]==train.iloc[idxV,0].to_array() ).sum()/len(idxV)

        print(' fold =',i,'acc =',acc)

    acc = ( oof==train.iloc[:,0].to_array() ).sum()/len(train)

    print(' kNN with k =',k,'ACC =',acc)
# LOAD TEST DATA

test = cudf.read_csv('../input/digit-recognizer/test.csv')

print('test shape =', test.shape )

test.head()
# FIT KNN MODEL

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(train.iloc[:,1:785], train.iloc[:,0])
%%time

# PREDICT TEST DATA

# Better to use knn.predict() but cuML v0.11.0 has bug

# y_hat = knn.predict(test)

y_hat_p = knn.predict_proba(test)

y_hat = y_hat_p.to_pandas().values.argmax(axis=1)
print('Pandas Version:', pd.__version__)
# SAVE PREDICTIONS TO CSV

sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

sub.Label = y_hat

sub.to_csv('submission_cuML.csv',index=False)
sub
y_test = cudf.read_csv('../input/mnist-real/submission.csv')

print('test shape =', y_test.shape )

y_test.head()
y_test.Label
# compare the y_test and submission

from sklearn.metrics import classification_report

print('test accuracy:', knn.score(X_test,y_test))
sub2 = pd.DataFrame(columns = ['ImageId','Label'])

sub2.ImageId = sub.ImageId

sub2.Label = sub.Label

sub.to_csv('submission_cuML.csv',index=False)
sub2