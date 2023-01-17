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
import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import  classification_report, accuracy_score
df = pd.read_csv('../input/mushrooms.csv')
df.head()
df.info()
df['class'].value_counts()
df.describe()
#data is not skewed and its all categorical 

#so lets trasform categorical data

labelEncoder = preprocessing.LabelEncoder()

for col in df.columns:

    df[col] = labelEncoder.fit_transform(df[col])

df.head()
df.info()
# Train Test Split

X = df.drop('class', axis=1)

y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
y_train.head()
for i in range(1,40):

    random_forest = RandomForestClassifier(n_estimators=i)

    random_forest.fit(X_train, y_train)

    pred = random_forest.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("accuracy  for {} is {}".format(i, acc))
#lets try different classifiers

models = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),

          'Random Forest': RandomForestClassifier(n_estimators=20), 

          'K-Nearest Neighbors':KNeighborsClassifier(n_neighbors=1)}

for k,v in models.items():

    mod = v

    mod.fit(X_train, y_train)

    pred = mod.predict(X_test)

    print('Results for: ' + str(k) + '\n')

    print(classification_report(y_test, pred))

    acc = accuracy_score(y_test, pred)

    print(acc)

    print('\n' + '\n')
#find important features using random forest

random_forest = RandomForestClassifier(n_estimators= 20)

random_forest.fit(X_train, y_train)

pred = random_forest.predict(X_test)

feature_importance = random_forest.feature_importances_

feat_sort_imp = pd.Series(feature_importance,index=X.columns).sort_values

feat_sort_imp