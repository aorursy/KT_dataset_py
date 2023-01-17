# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_excel(os.path.join(dirname, filename));



# limit to categorical data using df.select_dtypes()

# label encoding

X = df.select_dtypes(include=[object])

X.head(2)

X.shape

X.columns

le = preprocessing.LabelEncoder()

X_2 = X.apply(le.fit_transform)

X_2.head()

print(X_2)





#one hot encoding 

enc = preprocessing.OneHotEncoder()

enc.fit(X_2)

onehotlabels = enc.transform(X_2).toarray()

onehotlabels.shape

onehotlabels

type(onehotlabels)

print(onehotlabels[2])



# normalize the other values 

# Normalize total_bedrooms column

X_Age = np.array(df['Age'])

normalized_X_Age = preprocessing.normalize([X_Age])

print(normalized_X_Age)

normalized_X_Age_c = np.array(normalized_X_Age).reshape(15, 1)

print(normalized_X_Age_c)



X_Inc = np.array(df['Income'])

normalized_X_Inc = preprocessing.normalize([X_Inc])

print(normalized_X_Inc)

normalized_X_Inc_c = np.array(normalized_X_Age).reshape(15, 1)

print(normalized_X_Inc_c)





X_new = np.concatenate((normalized_X_Age_c,normalized_X_Inc_c, X_2),axis=1)

print(X_new)



# applying elbow algo to find the value of K

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(X_new)

    kmeanModel.fit(X_new)

    distortions.append(sum(np.min(cdist(X_new, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_new.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()



# we find there is no strong relation between the data by K - Means - Still let us try to seggrigate the data into clusters taking K = 3 



kmeans = KMeans(n_clusters=3) # You want cluster the passenger records into 2: Survived or Not survived

kmeans.fit(X_new)

cluster = np.array(kmeans.labels_).reshape(15, 1)

print(cluster)



final_solution =  np.concatenate((df,cluster),axis=1)



print(final_solution)



# Any results you write to the current directory are saved as output.