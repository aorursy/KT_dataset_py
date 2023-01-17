# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import MiniBatchKMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
if __name__ == "__main__":



    dataset = pd.read_csv('/kaggle/input/candy.csv')

    print(dataset.head(10))



    X = dataset.drop('competitorname', axis=1)



    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)

    print("Total de centros: " , len(kmeans.cluster_centers_))

    print("="*64)

    print(kmeans.predict(X))



    dataset['group'] = kmeans.predict(X)



    print(dataset)



    #implementacion_k_means