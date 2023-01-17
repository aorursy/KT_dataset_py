from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import numpy as np

from numpy import savetxt

data = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv").values

x = data.iloc[:,1:].values

y = data[[0]].values.ravel()

rf = RandomForestClassifier(n_estimators=100)

rf.fit(x,y)

predict = rf.predict(test)

np.savetxt('final_submission.csv', np.c_[range(1,len(test)+1),predict], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')




