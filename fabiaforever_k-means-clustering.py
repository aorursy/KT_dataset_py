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
#importing the required packages

from sklearn.cluster import KMeans
#Reading the dataset

iris = pd.read_csv("../input/Iris.csv")
iris.info()
iris.head()
iris.describe()
iris_matrix = pd.DataFrame.as_matrix(iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])
kmeans = KMeans(n_clusters=3)
kmeans.fit(iris_matrix)
kmeans.cluster_centers_
kmeans.labels_