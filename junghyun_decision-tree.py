import pandas as pd

import numpy as np

from sklearn import tree

train_data = pd.read_csv("train.csv")



# Age 컬럼에 있는 비어있는 row에 Age컬럼의 중앙값을 채워준다.

median_age_train = train_data["Age"].median()

train_data["Age"] = train_data["Age"].fillna(median_age_train)



# Embarked 컬럼에 있는 값들 중에 가장 많이 나온 밸류를 비어있는 row에 채워준다.

mst_frq_embarked_train = train_data["Embarked"].value_counts().index[0]

train_data["Embarked"] = train_data["Embarked"].fillna(mst_frq_embarked_train)



# Embarked 컬럼에 있는 문자열을 int형으로 바꿔준다

Ports = list(enumerate(np.unique(train_data['Embarked'])))

Ports_dict = { name : i for i, name in Ports }

train_data["Embarked"] = train_data["Embarked"].map(lambda x: Ports_dict[x]).astype(int)



# Sex 컬럼에 있는 값들을 Gender 컬럼을 새로 만들어 0,1로 바꿔준다

train_data["Gender"] = 0

train_data["Gender"][train_data["Sex"] == 'male'] = 1



train_data.head()

# train_data.info()

# train_data.describe()
test_data = pd.read_csv("test.csv")



# Age 컬럼에 있는 비어있는 row에 Age컬럼의 중앙값을 채워준다.

median_age_test = test_data["Age"].median()

test_data["Age"] = test_data["Age"].fillna(median_age_test)



# Embarked 컬럼에 있는 값들 중에 가장 많이 나온 밸류를 비어있는 row에 채워준다.

mst_frq_embarked_test = test_data["Embarked"].value_counts().index[0]

test_data["Embarked"] = test_data["Embarked"].fillna(mst_frq_embarked_test)



# Embarked 컬럼에 있는 문자열을 int형으로 바꿔준다

test_data["Embarked"] = test_data["Embarked"].map(lambda x: Ports_dict[x]).astype(int)



# Sex 컬럼에 있는 값들을 Gender 컬럼을 새로 만들어 0,1로 바꿔준다

test_data["Gender"] = 0

test_data["Gender"][test_data["Sex"] == 'male'] = 1



# Fare 컬럼에 비어있는 로우에 중앙값으로 채워준다

test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())



test_data["Survived"] = np.NaN



test_data.head()

test_data.info()

test_data.describe() 
X_train = train_data[["Pclass", "Gender", "Age", "Fare", "Embarked", "SibSp", "Parch"]]

y_train = train_data["Survived"]



X_test = test_data[["Pclass", "Gender", "Age", "Fare", "Embarked", "SibSp", "Parch"]]

y_test = test_data["Survived"]
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)



print(clf.feature_importances_)

print(clf.score(X_train, y_train))
clf.predict(X_test)



test_data["Survived"] = clf.predict(X_test)
test_data = pd.DataFrame(test_data, columns=["PassengerId", "Survived"])

test_data
# from IPython.display import Image