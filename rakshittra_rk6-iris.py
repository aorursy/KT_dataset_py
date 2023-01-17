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



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

plt.style.use('ggplot')



df=pd.read_csv('/kaggle/input/iris/Iris.csv')



df=df.drop(['Id'],axis=1)



from sklearn.cluster import KMeans

from sklearn import neighbors

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn import svm



df.dropna(inplace=True)



y=df['Species']

X=df.drop(['Species'],axis=1)



X=preprocessing.scale(X)



X_train,X_test,y_train,y_test=train_test_split(X,y)



clf=neighbors.KNeighborsClassifier(n_neighbors=15)



clf.fit(X_train,y_train)



accuracy=clf.score(X_test,y_test)



print(accuracy)