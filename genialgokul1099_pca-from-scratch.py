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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mlp

import seaborn as sns

df = pd.read_csv("../input/crowdedness-at-the-campus-gym/data.csv")
columns = df.columns.tolist()

print(columns)
df.shape
df.corr()
correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation,vmax=1,square = True,annot=True,cmap='cubehelix')

plt.title("correlation")
df.drop('date',axis=1,inplace=True)

df.describe()
correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation,vmax=1,square = True,annot=True,cmap='cubehelix')

plt.title("correlation")
X = df.iloc[:,:-1].values

Y = df.iloc[:,-1].values
np.shape(X)

np.shape(Y)
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)

X_std
X_std.shape[0] - 1

mean_vec = np.mean(X_std,axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec))/(X_std.shape[0]-1)

print(cov_mat)
plt.figure(figsize=(8,8))

sns.heatmap(cov_mat,vmax=1,square=True,annot=True,cmap='cubehelix')

plt.title("correlation between different features")
eig_val,eig_vec = np.linalg.eig(cov_mat)

eig_pairs  = [(np.abs(eig_val[i]),eig_vec[:,i])for i in range(len(eig_val))]

for i in eig_pairs:

    print(i)

    print()
eig_pairs.sort(key=lambda x:x[0],reverse=True)

for i in eig_pairs:

    print(i)

    print()

tot = sum(eig_val)

var_exp = [(i/tot)*100 for i in sorted(eig_val,reverse=True)]

with plt.style.context('dark_background'):

    plt.figure(figsize=(6,4))

    plt.bar(range(9),var_exp,alpha=0.5,align='center',

           label = 'individual explained variance')

    plt.ylabel("Explained variance ratio")

    plt.xlabel("Principal components")

    plt.legend(loc='best')

    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(9,1),

                     eig_pairs[1][1].reshape(9,1)))

print(matrix_w)
Y = X_std.dot(matrix_w)

Y