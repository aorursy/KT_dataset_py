import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split



X = np.array(data.drop(columns=['target_class']))

y = np.array(data['target_class'])



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



forest = RandomForestClassifier(n_estimators=50, max_depth=8,

                                  random_state=0)

tree = DecisionTreeClassifier(random_state=0)

#forest_r = RandomForestRegressor(n_estimators=1000, max_depth=2,

#                                  random_state=0)



print("train with all indexes:")

forest.fit(X_train,y_train)

tree.fit(X_train,y_train)

print("tree score:",tree.score(X_test,y_test))

print("forest score:",forest.score(X_test,y_test))



indices=[i for i, x in enumerate(y_test) if x ==1]

print("tree score if watched:",tree.score(X_test[indices],y_test[indices]))

print("forest score if watched:",forest.score(X_test[indices],y_test[indices]))



indices=[i for i, x in enumerate(y_test) if x ==0]

print("tree score if not watched:",tree.score(X_test[indices],y_test[indices]))

print("forest score if not watched:",forest.score(X_test[indices],y_test[indices]))
np.unique(y, return_counts=True)
new_ind=np.append(np.where(y_train==1)[0], np.where(y_train==0)[0][0:len(np.where(y_train==1)[0])])

new_ind.shape



print("\ntrain with special indexes:")

forest.fit(X_train[new_ind],y_train[new_ind])

tree.fit(X_train[new_ind],y_train[new_ind])

print("tree score:",tree.score(X_test,y_test))

print("forest score:",forest.score(X_test,y_test))



indices=[i for i, x in enumerate(y_test) if x ==1]

print("tree score if watched:",tree.score(X_test[indices],y_test[indices]))

print("forest score if watched:",forest.score(X_test[indices],y_test[indices]))



indices=[i for i, x in enumerate(y_test) if x ==0]

print("tree score if not watched:",tree.score(X_test[indices],y_test[indices]))

print("forest score if not watched:",forest.score(X_test[indices],y_test[indices]))