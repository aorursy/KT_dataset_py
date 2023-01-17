import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.svm import SVC
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

    # Split train-test DataSets into X & y

train_X = train.iloc[:, [2, 4, 5, 9]].values
train_y = train.iloc[:, 1].values
test_X = test.iloc[:, [1, 3, 4, 8]].values

    # Replacing null values ​​(NaN) with the average of each DataSet X

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(train_X[:, 2:3])
train_X[:, 2:3] = imputer.transform(train_X[:, 2:3])
imputer = imputer.fit(test_X[:, [2,3]])
test_X[:, [2,3]] = imputer.transform(test_X[:, [2,3]])


    # Encoding the "Sex" Field

le_X = preprocessing.LabelEncoder()
train_X[:, 1] = le_X.fit_transform(train_X[:, 1])
test_X[:, 1] = le_X.fit_transform(test_X[:, 1])

onehotencoder = make_column_transformer(( preprocessing.OneHotEncoder(), [1]), remainder = "passthrough")
train_X = onehotencoder.fit_transform(train_X)
test_X = onehotencoder.fit_transform(test_X)

    # Variable scaling

sc_X = preprocessing.StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.fit_transform(test_X)
classifier = SVC(kernel = "poly", random_state = 0)
classifier.fit(train_X, train_y)
pred_y = classifier.predict(test_X)