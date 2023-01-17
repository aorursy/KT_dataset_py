# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

mush= pd.read_csv('../input/mushrooms.csv', delimiter=',')

# Any results you write to the current directory are saved as output.
%matplotlib inline
mush.head()
mush.describe()
mush.info()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
mush1= mush.drop('class',axis=1)
mush1.head()
pred= list(mush.columns.values)

mushnew = list(pred)

mushnew.remove('class')



print(mushnew)
for i in mushnew:

    mush[i] = le.fit_transform(mush[i])
from sklearn.cross_validation import train_test_split
X= mush[mushnew]

y= mush['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.count()
y_test.count()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2) # for p and e
knn.fit(X_train,y_train)
predictions =knn.predict (X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print('The accuracy of the knn classifier is {:.2f} on training data'.format(knn.score(X_train, y_train)))

print('The accuracy of the knn classifier is {:.2f} on test data'.format(knn.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=40)
rf.fit(X_train,y_train)
pr =rf.predict(X_test)
print(classification_report(y_test,pr))
print('The accuracy of the RF classifier is {:.2f} on training data'.format(rf.score(X_train, y_train)))

print('The accuracy of the RF classifier is {:.2f} on test data'.format(rf.score(X_test, y_test)))
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]
# Print the feature ranking

print("Feature ranking:")



for f in range(mush1.shape[1]):

    print("%d. feature %d %s (%f)" % (f+1 , indices[f], mush1.columns[indices[f]],

                                      tree.feature_importances_[indices[f]]))
plt.figure(figsize=(10, 6))

index = np.arange(len(mushnew))

plt.bar(index,importances, color='blue')

plt.xlabel('features')

plt.ylabel('importance')

plt.title('Feature importance')

plt.tight_layout()