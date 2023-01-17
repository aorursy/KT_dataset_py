import pandas as pd

from sklearn import svm

import numpy as np

classifier = svm.SVC(gamma=0.001)

dataset = pd.read_csv("../input/train.csv")

train_y = np.array(dataset.iloc[:, 0])

train_x = np.array(dataset.iloc[:, 1:])

classifier.fit(train_x, train_y)



test_x = pd.read_csv("../input/test.csv")



result = classifier.predict(np.array(test_x))

sm = pd.DataFrame({'ImageId': range(1, len(result) + 1), 'Label':result})

sm.to_csv('result.csv', index = False)