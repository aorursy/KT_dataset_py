# Import needed libraries

from sklearn import svm

import numpy as np

import pandas as pd
# Get data

dataset = pd.read_csv('../input/train.csv')

df = pd.DataFrame(dataset)

df.head()
x = df.iloc[:,1:]

y = df.iloc[:,0]

print(x.shape, y.shape)
clf = svm.SVC(gamma='scale')

clf.fit(x, y)
test = pd.DataFrame(pd.read_csv('../input/test.csv'))
prediction = clf.predict(test)
submission = pd.DataFrame(columns=['ImageId', 'Label'])

submission.ImageId = np.arange(1,28001)

submission.Label = prediction

submission.to_csv('submission.csv', index=False)