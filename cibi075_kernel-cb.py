import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing



df_csv = pd.read_csv('../input/train.csv')

Y_train = pd.read_csv('../input/test.csv')

Y=df_csv.label

X=df_csv.iloc[:,1:]





X = preprocessing.normalize(X)

x = StandardScaler().fit_transform(X)







neigh = KNeighborsClassifier(n_neighbors=3,algorithm="auto")

model=neigh.fit(x, Y) 



results = model.predict(Y_train)





submission = pd.read_csv('../input/sample_submission.csv')



submission['Label'] = results





submission.to_csv('Digit_submission.csv',index=False)