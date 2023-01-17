!nvidia-smi
!cat /usr/local/cuda/version.txt
## Passing Y as input while conda asks for confirmation, we use yes command

!yes Y | conda install faiss-gpu cudatoolkit=10.0 -c pytorch
!wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2

!tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'lib/*'

!tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'site-packages/*'

!cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/

# !export LD_LIBRARY_PATH="/kaggle/working/lib/" 

!cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/
!apt search openblas

!yes Y | apt install libopenblas-dev 

# !printf '%s\n' 0 | update-alternatives --config libblas.so.3 << 0

# !apt-get install libopenblas-dev 

!rm -rf  ./*
import numpy as np

import pandas as pd

from keras.datasets import mnist

import matplotlib.pyplot as plt

from tsnecuda import TSNE

from sklearn.decomposition import PCA

%matplotlib inline
# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train=X_train.reshape(60000,-1)

X_test=X_test.reshape(len(X_test),-1)

X_train.shape
# load data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train=X_train.reshape(60000,-1)/255

X_test=X_test.reshape(len(X_test),-1)/255

X=np.concatenate((X_train, X_test), axis=0)

y=np.concatenate((y_train, y_test), axis=0)
plt.imshow(X[1116].reshape(28,28))
tsne = TSNE(n_components=2, perplexity=42, n_iter=200000)

train_reduced = tsne.fit_transform(X_train)

plt.figure(figsize=(8,6))

plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_train, alpha=0.5,

            cmap=plt.cm.get_cmap('nipy_spectral', 10))



plt.colorbar()

plt.show()
from sklearn.multioutput import MultiOutputRegressor

import xgboost



xgb = xgboost.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.09,

                 max_depth=6,

                 min_child_weight=1.5,

                 n_estimators=5000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42,

                 objective ='reg:squarederror',

                 tree_method='gpu_hist',

                 predictor='cpu_predictor')

xgbModel=MultiOutputRegressor(xgb)



xgbModel.fit(X_train, train_reduced)

trainPred=xgbModel.predict(X_train)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from math import sqrt

print("Score: ",xgbModel.score(X_train, train_reduced))

print("MAE: ",mean_absolute_error(trainPred,train_reduced))

print("MSE: ",(mean_squared_error(trainPred,train_reduced)))

print("RMSE: ",sqrt(mean_squared_error(trainPred,train_reduced)))
plt.figure(figsize=(8,6))

plt.scatter(trainPred[:, 0], trainPred[:, 1], c=y_train, alpha=0.5,

            cmap=plt.cm.get_cmap('nipy_spectral', 10))



plt.colorbar()

plt.show()
testPred=xgbModel.predict(X_test)

plt.figure(figsize=(8,6))

plt.scatter(testPred[:, 0], testPred[:, 1], c=y_test, alpha=0.5,

            cmap=plt.cm.get_cmap('nipy_spectral', 10))



plt.colorbar()

plt.show()
import pickle

import gzip

with gzip.GzipFile('./xgb(regression)-42-12000-scale-all.pgz', 'w') as f:

    pickle.dump(xgbModel, f)
X=np.concatenate((X_train, X_test), axis=0)

y=np.concatenate((y_train, y_test), axis=0)

pred=xgbModel.predict(X)



pdData = pd.DataFrame(pred, columns = ["x1", "x2"])

pdData["y"]=y

pdData.to_csv('./Result-xgb-42-12000-scale-all.csv',index=False)
#讀取資料

data = pd.read_csv("./Result-xgb-42-12000-scale-all.csv") #load the dataset

X=data[['x1','x2']].values

y=data[['y']].values.reshape(-1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score



# fit model no training data

xgbModel = XGBClassifier(tree_method='gpu_hist',predictor='cpu_predictor')

xgbModel.fit(X_train, y_train)
# make predictions for test data

y_pred = xgbModel.predict(X_test)

# evaluate predictions

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
import pickle

import gzip

with gzip.GzipFile('./xgb(classfication)-42-12000-scale-all.pgz', 'w') as f:

    pickle.dump(xgbModel, f)