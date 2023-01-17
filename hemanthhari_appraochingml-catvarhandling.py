# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train= pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

data_test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

data_test.head()
from sklearn.preprocessing import LabelEncoder

data_train.loc[:,'ord_2']=data_train.ord_2.fillna("NONE")

lbl=LabelEncoder()

data_train.loc[:,'ord_2']= lbl.fit_transform(data_train.ord_2.values)
data_train.ord_2.value_counts()
data_train
exam=np.array([[0,0,1],[1,0,0],[1,0,1]])

exam.nbytes

from scipy import sparse

sparse_mat = sparse.csr_matrix(exam)

sparse_mat.data.nbytes
sparse_mat.indptr.nbytes
exam= np.random.randint(1000,size=100000)

from sklearn.preprocessing import OneHotEncoder

ohe= OneHotEncoder()

ohe_exam = ohe.fit_transform(exam.reshape(-1,1))

print(f"Size of array:{exam.data.nbytes}")

print(f"Size of Sparse array:{ohe_exam.data.nbytes}")


