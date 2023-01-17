import numpy as np

import pandas as pd 

from subprocess import check_output

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

%matplotlib inline 
# read data sets from csv files.

train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')
# print Test set

print('Train set:')

train_set.head(10)
print('Test set:')

test_set.head(10)
# slice to input data and label

trX = train_set.iloc[:, 1:].as_matrix()[0:100]

trY = train_set.label.tolist()[0:100]

teX = test_set.as_matrix()
# example of data set

example = np.array(trX[0]).reshape(28,28)

plt.imshow(example, cmap='gray')

plt.show()
# create and train svm

from sklearn import svm

from sklearn.multiclass import OneVsRestClassifier



model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

model.fit(trX, trY)
# predict test svm

prY = model.predict(teX)

prY_proba = model.predict_proba(teX)



print('Predict Label:')

print(prY)

print('\nPredict Label Probability:')

print(prY_proba)
# save result to csv

result = pd.DataFrame({"ImageId": range(1, len(prY)+1), "Label": prY})

result.to_csv('output.csv', index=False, header=True)

result.head(10)