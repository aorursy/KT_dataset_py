import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

X_test = test
train.head()
X_test.head()
train.isna().sum()
X_test.isna().sum()
mean_age = np.hstack((train["Age"].dropna(), X_test["Age"].dropna())).mean()

print("Mean age:", mean_age)



train["Age"].fillna(mean_age, inplace=True)

X_test["Age"].fillna(mean_age, inplace=True)



train["Cabin"].fillna("None", inplace=True)

X_test["Cabin"].fillna("None", inplace=True)



train["Embarked"].fillna("None", inplace=True)



X_test["Fare"].fillna(0, inplace=True)
train.isna().sum()
X_test.isna().sum()
X_train = train.drop(["Survived", "PassengerId"], axis=1)

y_train = train["Survived"]

X_test.drop("PassengerId", axis=1, inplace=True)
X_train.dtypes
y_train.dtypes
X_train.corr()
train.head()
numeric_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

text_cols = ["Name"] # preprocess names

categorical_cols = list(set(X_train.columns) - set(numeric_cols) - set(text_cols))
categorical_cols
X_train_text = X_train[text_cols]

X_test_text = X_test[text_cols]



X_train_cat = X_train[categorical_cols]

X_test_cat = X_test[categorical_cols]
print(len(X_train_text))
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(min_df=10, max_df=100)

matrix = vectorizer.fit_transform(list(X_train_text["Name"]))
matrix.shape
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=1)

name_preds = kmeans.fit_predict(matrix.toarray())
name_preds.shape
X_train_name = kmeans.predict(vectorizer.transform(list(X_train_text["Name"])).toarray())

X_test_name = kmeans.predict(vectorizer.transform(list(X_test_text["Name"])).toarray())
X_train_name.shape, X_test_name.shape
from sklearn.feature_extraction import DictVectorizer as DV
encoder = DV(sparse=False)

X_train_cat_oh = encoder.fit_transform(X_train_cat.astype(str).T.to_dict().values())

X_test_cat_oh = encoder.transform(X_test_cat.astype(str).T.to_dict().values())
from sklearn.preprocessing import Normalizer
transformer = Normalizer()

X_train_num = transformer.fit_transform(X_train[numeric_cols])

X_test_num = transformer.transform(X_test[numeric_cols])
X_train_num.shape, X_train_cat_oh.shape, X_train_name.reshape((X_train_num.shape[0], 1)).shape
X_train_preprocessed = np.hstack((X_train_num, X_train_cat_oh, X_train_name.reshape((X_train_num.shape[0], 1))))

X_test_preprocessed = np.hstack((X_test_num, X_test_cat_oh, X_test_name.reshape((X_test_num.shape[0], 1))))
X_train_preprocessed.shape, X_test_preprocessed.shape
model_score = []
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
dtc = DecisionTreeClassifier()

dtc_score = cross_val_score(dtc, X_train_preprocessed, y_train).mean()

model_score.append(("DecisionTreeClassifier", dtc_score))
dtc_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
rf_classifier = RandomForestClassifier()

rfc_score = cross_val_score(rf_classifier, X_train_preprocessed, y_train).mean()

model_score.append(("RandomForestClassifier", rfc_score))
print(rfc_score)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr_score = cross_val_score(lr, X_train_preprocessed, y_train).mean()

model_score.append(("LogisticRegression", lr_score))
print(lr_score)
import xgboost
xgb = xgboost.XGBClassifier()

xgb_score = cross_val_score(xgb, X_train_preprocessed, y_train).mean()

model_score.append(("XGBClassifier", xgb_score))
print(xgb_score)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn_score = cross_val_score(knn, X_train_preprocessed, y_train).mean()

model_score.append(("KNeighborsClassifier", knn_score))
print(knn_score)
for name, score in model_score:

    print(score, name)
classifier = RandomForestClassifier()

classifier.fit(X_train_preprocessed, y_train)
test.shape, X_test.shape, X_test_preprocessed.shape
y_predicted = classifier.predict(X_test_preprocessed)
length = X_train.shape[0] + 1

with open("/kaggle/working/answer.csv", "w") as file:

    file.write("PassengerId,Survived\n")

    for i, j in zip(np.arange(length, length + y_predicted.shape[0]), y_predicted):

        file.write("{},{}\n".format(str(i), str(j)))