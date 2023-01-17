# load modules



import numpy as np

import pandas as pd



from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA
# load data



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

target = train["label"]

train = train.drop("label",1)
# PCA magic



pca = PCA(n_components=40)

pca.fit(train)

transform_train = pca.transform(train)

transform_test = pca.transform(test)
# kNN magic



clf = KNeighborsClassifier(n_neighbors=5)

clf.fit(transform_train, target)

results=clf.predict(transform_test)
# generate submission



np.savetxt('results.csv', 

           np.c_[range(1,len(test)+1),results], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')