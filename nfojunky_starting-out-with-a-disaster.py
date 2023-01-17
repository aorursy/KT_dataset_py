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
# Importing the dataset

dataset = pd.read_csv('../input/train.csv')

dataset_test = pd.read_csv('../input/test.csv')
# Taking care of missing data

dataset = dataset.fillna(dataset.mean())

dataset = dataset.fillna(method='pad')

dataset_test = dataset_test.fillna(dataset.mean())

dataset_test = dataset_test.fillna(method='pad')
X_train = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values

y_train = dataset.iloc[:, 1].values



X_test = dataset_test.iloc[:, [1,3,4,5,6,8,10]].values
# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

sex_encoder = LabelEncoder()

X_train[:, 1] = sex_encoder.fit_transform(X_train[:, 1])

X_test[:, 1] = sex_encoder.fit_transform(X_test[:, 1])



embark_encoder = LabelEncoder()

X_train[:, 6] =    embark_encoder.fit_transform(X_train[:, 6])

X_test[:, 6] = embark_encoder.fit_transform(X_test[:, 6])
# Use OneHotEncoder to remove cardinality bias

onehotencoder = OneHotEncoder(categorical_features = [6])

X_train = onehotencoder.fit_transform(X_train).toarray()

X_test = onehotencoder.fit_transform(X_test).toarray()



# Trim first column to prevent dummy variable trap

# (onehot array gets prepended on the left)

X_train = X_train[:, 1:]

X_test = X_test[:, 1:]
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
X_train
# Fitting classifier to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)
dataset_test.insert(1, "Survived", y_pred)
y_pred
dataset_test.to_csv('prediction.csv')