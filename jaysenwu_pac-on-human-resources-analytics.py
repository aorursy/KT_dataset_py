# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.corr()
df['sales'].unique()
sales_mapping = {'sales':1, 'accounting':2, 'hr':3, 'technical':4, 'support':5, 'management':6,

       'IT':7, 'product_mng':8, 'marketing':9, 'RandD':10}

df['sales'] = df['sales'].map(sales_mapping)
df['salary'].unique()
sally_mapping = {"low": 1, "medium": 2, "high": 3}

df['salary'] = df['salary'].map(sally_mapping)
df.head()
y = df['left'].values
cols = df.columns.tolist()

cols.insert(0, cols.pop(cols.index('left')))

cols
X = df.iloc[:,1:9].values

X
np.shape(X)
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlim(0,7,1)

plt.xlabel('Number of components')

plt.ylabel('Cumulative explained variance')
from sklearn.decomposition import PCA 

sklearn_pca = PCA(n_components=6)

Y_sklearn = sklearn_pca.fit_transform(X_std)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train.shape
from sklearn.svm import SVC

model=SVC()

model.fit(X_train,y_train)

model.score(X_train,y_train)
model.predict(X_test)

model.score(X_test,y_test)