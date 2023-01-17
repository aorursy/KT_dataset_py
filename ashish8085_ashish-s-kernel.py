import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd 

from sklearn.ensemble import RandomForestClassifier



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

target = train["label"]

train = train.drop("label",1)



clf = RandomForestClassifier(n_estimators = 100, n_jobs=1, criterion="gini")

clf.fit(train, target)

results=clf.predict(test)



# prepare submit file



np.savetxt('results.csv', 

           np.c_[range(1,len(test)+1),results], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')
