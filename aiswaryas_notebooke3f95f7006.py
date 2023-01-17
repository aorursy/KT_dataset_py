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

dataset = pd.read_csv("../input/train.csv") 

test = pd.read_csv("../input/test.csv")



#Drop the first column 'Id' since it just has serial numbers. Not useful in the prediction process.

index = dataset.iloc[:,0]

dataset = dataset.iloc[:,1:]

test = test.iloc[:, 1:]
dataset.head()
print(dataset.columns)
# Is the data categorical or numerical?

print(dataset.dtypes)
# Some data visualization

# How many of each cover is present?

# labels = dataset.iloc[:,54]

labels = dataset.groupby('Cover_Type').size()

print(labels)
# We want to cluster the data. According to the given variables we want to see which tree

# cluster the data best fits.

# This would be a KNN or Kmeans.  So KNN is a supervised learning algorithm because it takes

# in labeled data and figures out for a new datapoint which cluster it is most like.

# Kmeans however is unsupervised and learns the clusters by itself.

# Here we have labeled data so I will first use a KNN.

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()



X = dataset.iloc[:,0:54]

Y = dataset.iloc[:,54]



model.fit(X,Y)



#pred = model.predict(test).reshape(-1,1)



from sklearn import metrics

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, Y, cv=5, scoring='f1_macro')

print(np.mean(scores))

pred = model.predict(test)

pred
c1 = pd.DataFrame(index)

c2 = pd.DataFrame({'Cover_Type': pred})

res = (pd.concat([c1, c2], axis=1))

res.to_csv('ouput.csv', index=False)
pred