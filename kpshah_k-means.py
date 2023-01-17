import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing, neighbors

from sklearn.model_selection import train_test_split



#

df = pd.read_csv('../input/voice.csv')

df.replace('male', 0, inplace=True)

df.replace('female', 1, inplace=True)



X = np.array(df.drop('label', 1))

y = np.array(df['label'])

clf = neighbors.KNeighborsClassifier()



for step in range(1,10):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=step/10)

    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test,y_test)

    print(accuracy, step/10)