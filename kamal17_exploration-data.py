# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
cover = pd.read_csv("../input/covtype.csv")

cover.info()

cover["Cover_Type"].describe()
cover["SlopeSmall"] = [np.nan if x > 30 else x for x in cover['Slope']]

cover.isnull().sum()
cover['SlopeSmall'].fillna(cover.SlopeSmall.mean(),inplace=True)

cover.isnull().sum()
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel



clf = ExtraTreesClassifier()

clf = clf.fit(X, Y)

model = SelectFromModel(clf, prefit=True)

New_features = model.transform(X)

model.get_support()

New_features.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(New_features, Y, test_size=0.30, random_state=42)
X_train.shape

X_test.shape
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=3, weights='uniform')

model.fit(X_train, y_train)

model.score(X_train,y_train)
from sklearn import metrics



y_pred = model.predict(X_test)

y_pred.shape

y_test
acc_test = metrics.accuracy_score(y_test,y_pred)

acc_test