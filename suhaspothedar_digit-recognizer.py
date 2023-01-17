import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

TEST = pd.read_csv('../input/test.csv') 

TRAIN = pd.read_csv('../input/train.csv')

X = TRAIN.drop('label', axis = 1 )

y = TRAIN['label']

#print(train)



#print(test)

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)
# X_pred = knn.predict(X_train)

# accuracy = knn.accuracy(X_test, X_pred)
# print(accuracy)


# y_pred = knn.predict(y_train)
# new_accuracy = knn.accuracy(y_test, y_pred)
# print(new_accuracy)

predictions = knn.predict(TEST)
pd.DataFrame(predictions).index += 1
print(pd.DataFrame(predictions).iloc[:,:1])
pd.DataFrame(predictions).iloc[:,:1].to_csv('out.csv', encoding = 'utf-8', sep = ',')
