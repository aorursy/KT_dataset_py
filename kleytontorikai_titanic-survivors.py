# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('/kaggle/input/titanic/train.csv')

X.head()
# Select categorical columns with relatively low cardinality (arbitrary)

low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 

                        X[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X = X[my_cols].copy()



# One-hot encode the data

X = pd.get_dummies(X, drop_first=True)



# Can't just drop NaN rows because the test set might contain NaN values

X.fillna(value=-1, inplace=True)



X.head()
y = X.Survived

X.drop(['Survived'], axis=1, inplace=True)

X.drop(['PassengerId'], axis=1, inplace=True)

model_cols = X.columns



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train.head()
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.linear_model import LogisticRegression



log_reg_model = LogisticRegression(solver='liblinear')

model = log_reg_model

model.fit(X_train, y_train)

predictions = model.predict(X_test)



from xgboost import XGBClassifier

xgb_model = XGBClassifier()

model = xgb_model

model.fit(X_train, y_train)

predictions = model.predict(X_test)



from sklearn.ensemble import RandomForestClassifier

rand_forest_model = RandomForestClassifier(n_estimators = 200, criterion = 'entropy')

model = rand_forest_model

model.fit(X_train, y_train)

predictions = model.predict(X_test)



from sklearn.svm import SVC

svm_model = SVC(kernel = 'rbf', gamma='scale')

model = svm_model

model.fit(X_train, y_train)

predictions = model.predict(X_test)



from sklearn.neighbors import KNeighborsClassifier

# The Euclidean distance is the minkowski metric with p=2.

knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

model = knn_model

model.fit(X_train, y_train)

predictions = model.predict(X_test)
# Applying k-Fold Cross Validation

models = ['log_reg_model', 'xgb_model', 'rand_forest_model', 'svm_model', 'knn_model']



from sklearn.model_selection import cross_val_score

for model in models:

    classifier = eval(model)

    accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10)

    print(f'{model}: accuracies mean = {accuracies.mean():.2f} +- {accuracies.std():.2f}')
# Read test set for submission

X_sub = pd.read_csv('/kaggle/input/titanic/test.csv')



passenger_IDs = X_sub['PassengerId']



# Apply the same transformations on X_sub as in X_train

# One-hot encode

X_sub = pd.get_dummies(X_sub, drop_first=True)

# Select the same columns except Survived

X_sub = X_sub[model_cols].copy()

# Replace NaN for -1

X_sub.fillna(value=-1, inplace=True)

# Feature Scaling

X_sub = sc_X.transform(X_sub)
model = svm_model

predictions = model.predict(X_sub)

predictions = pd.DataFrame(predictions)

submission = pd.concat([passenger_IDs, predictions[0]], axis=1, keys=['PassengerId', 'Survived'])

submission.head()
submission.to_csv('/kaggle/working/submission.csv', index=False)