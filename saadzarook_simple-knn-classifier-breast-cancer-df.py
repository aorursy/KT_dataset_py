import pandas as pd

import numpy as np

from sklearn import preprocessing, model_selection, neighbors
df = pd.read_csv('../input/breast-cancer-wisconsin.data.txt')

df.head()
df.replace('?', -99999, inplace=True) #Replace the missing values with an outlier number so that the algorith will identify it as an outlier

df.drop(['id'], 1, inplace=True) #id column is not useful in making predictions

df.head()
X = np.array(df.drop((['class']), 1))

y = np.array(df['class'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)
example = np.array([8,3,1,2,3,1,4,4,4])

example = example.reshape(1, -1)

predictions = clf.predict(example)

print("Class:"+ str(predictions))