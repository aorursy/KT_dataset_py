# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
heart_data = pd.read_csv("../input/heart.csv")
heart_data.head()
heart_data.describe()
heart_data['restecg'].unique()
X = heart_data.iloc[:,0:13].values

Y = heart_data.iloc[:,13].values
X.shape, Y.shape
import numpy as np

from sklearn.decomposition import PCA

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale

%matplotlib inline
scaled_X = scale(X)
pca = PCA(n_components=13)



pca.fit(X)
def showVarianceRatio(pca):

    exp_ratio_var = pca.explained_variance_ratio_

    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    plt.plot(var1)
exp_ratio_var = pca.explained_variance_ratio_

exp_ratio_var
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
pca2 = PCA(n_components=3)

pca2.fit(X)

X1=pca2.fit_transform(X)

X1.shape
showVarianceRatio(pca2)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip((0, 1, 2, 3),

                        ('blue', 'red', 'green', 'black')):

        plt.scatter(X1[Y==lab, 0],

                    X1[Y==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower center')

    plt.tight_layout()

    plt.show()
from sklearn.model_selection import train_test_split



train_X,test_X,train_y, test_y = train_test_split(X1,Y, test_size=0.2, random_state=42)

train2_X,val_X,train2_y, val_y = train_test_split(train_X,train_y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
#scaler = StandardScaler()

#scaler.fit(X1)
#NN_train_x = scaler.transform(X1)

#NN_test_x = scaler.transform(test_X.values)
train2_X.shape
clf = MLPClassifier(solver='sgd',hidden_layer_sizes=(train2_X.shape[0],), random_state=1, max_iter=250, learning_rate_init=0.0001)
clf.fit(train2_X, train2_y)
#NN_text_x = scaler.transform(np_train_X)
yhat = clf.predict(val_X)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
#print ("accuracy = ", accuracy_score(test_y.values, YHAT))

print("accuracy = {0:.2f}".format(accuracy_score(val_y, yhat)))

print ("precision_score = {0:.2f}".format(precision_score(val_y, yhat)))

print ("recall_score = {0:.2f}".format(recall_score(val_y, yhat)))

print ("f1_score = {0:.2f}".format(f1_score(val_y, yhat)))
yhat_test = clf.predict(test_X)
print("accuracy = {0:.2f}".format(accuracy_score(test_y, yhat_test)))

print ("precision_score = {0:.2f}".format(precision_score(test_y, yhat_test)))

print ("recall_score = {0:.2f}".format(recall_score(test_y, yhat_test)))

print ("f1_score = {0:.2f}".format(f1_score(test_y, yhat_test)))