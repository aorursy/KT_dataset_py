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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt, matplotlib.image as mpimg

import time 

import warnings

from sklearn import svm

from sklearn.model_selection import GridSearchCV

import seaborn as sb

from sklearn.decomposition import PCA 

from sklearn.preprocessing import StandardScaler



%matplotlib inline

warnings.filterwarnings('ignore')



print(os.listdir("../input"))
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

label = data['label']

data= data.drop('label',axis=1)

data.head()

train_data = data

train_label = label
sc = StandardScaler().fit(train_data)

X_std_train = sc.transform(train_data)

X_std_test = sc.transform(test_data)

pca = PCA().fit(X_std_train)

var_per = pca.explained_variance_ratio_

cum_var_per = pca.explained_variance_ratio_.cumsum()
plt.figure(figsize=(30,10))

ind = np.arange(len(var_per))

plt.bar(ind,var_per)

plt.xlabel('n_components')

plt.ylabel('Variance')
n_comp=len(cum_var_per[cum_var_per <= 0.90])

print("Keeping 90% Info with ",n_comp," components")

pca = PCA(n_components=n_comp)

train_pca_b = pca.fit_transform(X_std_train)

test_pca_b = pca.transform(X_std_test)

print("Shape before PCA for Train: ",X_std_train.shape)

print("Shape after PCA for Train: ",train_pca_b.shape)

print("Shape before PCA for Test: ",X_std_test.shape)

print("Shape after PCA for Test: ",test_pca_b.shape)
clf = svm.SVC(C=10,gamma=0.001,random_state=42)

start_time = time.time()

clf.fit(train_pca_b, label.values.ravel())

fittime = time.time() - start_time

print("Time consumed to fit model: ",time.strftime("%H:%M:%S", time.gmtime(fittime)))
start_time = time.time()

result = clf.predict(test_pca_b)

print("Accuracy for Binary(PCA): ",result)

elapsed_time = time.time() - start_time

print("Time consumed to predict: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
data_to_submit = pd.DataFrame({

    'ImageId':test_data.index.values+1,

    'Label':result

})
data_to_submit.to_csv('result.csv', index = False)
print(os.listdir(".."))