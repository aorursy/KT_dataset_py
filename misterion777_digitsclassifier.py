import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

TRAIN_PATH = "../input/train.csv"

TEST_PATH = "../input/test.csv"



train_data = pd.read_csv(TRAIN_PATH)
train = train_data.drop(['label'],axis=1)

X = train.values / 255
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, train_data['label'], test_size=0.1, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X, train_data['label'])
accuracy_score(y_test, knn.predict(X_test))

from sklearn.svm import SVC

# C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

# gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# kernel=['rbf','linear']

# hyper={'kernel':kernel,'C':C,'gamma':gamma}



svc = SVC(random_state=42, C= 0.7, gamma=0.2, kernel='rbf')

# svc = GridSearchCV(svc, param_grid=hyper,cv=4,verbose=5)

svc.fit(X_train, y_train)
accuracy_score(y_test, svc.predict(X_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(random_state=42)



params = {'n_estimators':[90,100,110,120,130],

            'max_depth': [3,4,5,6]

}



rf = GridSearchCV(rf, param_grid=params, cv=4,verbose=5)



rf.fit(X_train, y_train)
rf.best_params_
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)
accuracy_score(y_test, rf.predict(X_test))
data = pd.read_csv(TEST_PATH)

test_X = data / 255

test_X.head()
predicted = knn.predict(test_X)
submission = pd.DataFrame({"Label": predicted})

submission.index.name = 'ImageId'

submission.index +=1
submission.to_csv("submission.csv")
# lol = pd.read_csv("../output/submission.csv")