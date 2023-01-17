%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
labeled_group = train.groupby('label')



total_observation = len(train['label'])

for label in range(0,10):

    print('Relative frequency of {} = {:.3f} %'.format(label,labeled_group['label'].get_group(label).count()/total_observation*100))
train.shape
from sklearn.cross_validation import StratifiedShuffleSplit



X = train.drop(['label'], axis = 1)

y = train['label']



sss = StratifiedShuffleSplit(y, n_iter=3, test_size=0.2)



for train_index, test_index in sss:

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]    
print(X_train.shape)

print(X_test.shape)
print(y_train.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



#k_range = range(1,26)

#scores = []



# We are going to use differnt values of K as choose the best one as per accuracy_score

#for k in k_range:

    #knn = KNeighborsClassifier(n_neighbors=k)

    #knn.fit(X_train,y_train)

    #y_pred = knn.predict(X_test)

    #scores.append(accuracy_score(y_test,y_pred))

    #print('k {} completed'.format(k))

    



# Plotting testing accuracy

#plt.plot(k_range,scores)

#plt.xlabel('Value of K')

#plt.ylabel('Testing accuracy')
test.head()
print(X.shape)

print(y.shape)

print(test.shape)
# Is takes more than 1200 seconds to run so kaggle kernel automatically killed.

# More efficient way is to process the data in samll batches.



# Model training on entire train data

# final prediction for test data

#knn = KNeighborsClassifier(n_neighbors=5)

#knn.fit(X,y)

#y_pred = knn.predict(test)
# save submission to csv

#pd.DataFrame({"ImageId": list(range(1,len(test)+1)),"Label": y_pred}).to_csv('Digit_recogniser.csv', index=False,header=True)