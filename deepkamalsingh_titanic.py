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
train_data= pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
train_data.info()
import matplotlib.pyplot as plt

train_data.hist(bins=50,figsize=(10,12))

plt.show()
corr_matrix = train_data.corr()

corr_matrix["Survived"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["Survived", "Age", "Fare", "SibSp", "Parch"]

scatter_matrix(train_data[attributes], figsize=(20,20))
train_data["AgeBucket"] = train_data["Age"] // 15 * 15

train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
train_data["RelativeOnBoard"] = train_data["SibSp"] + train_data["Parch"]

train_data[["RelativeOnBoard", "Survived"]].groupby(['RelativeOnBoard']).mean()
# median = train_data["Age"].median()

# train_data["Age"].fillna(median, inplace=True)

train_data["AgeBucket"] = pd.cut(train_data["Age"], bins=[0, 20, 40, 60, np.inf], labels=['most survived', 'less survived', 'least survived', 'definately survivied'])

train_data["RelativeOnBoard"] = pd.cut(train_data["SibSp"]+train_data["Parch"], bins=[-1, 2,  3, 6, np.inf], labels=["half of them survived", "most of them survived", "some of them survived", "none of them survived"])
train_data["AgeBucket"].value_counts()

# train_data["RelativesOnBoard"].value_counts()
train_data["RelativeOnBoard"].value_counts()
# train_data_1 = train_data.drop(["Survived"], axis=1)

train_data_1_label = train_data["Survived"].copy()

print(train_data_1_label)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data
test_data.info()
test_data["AgeBucket"] = pd.cut(test_data["Age"], bins=[0, 20, 40, 60, np.inf], labels=['most survived', 'less survived', 'least survived', 'definately survivied'])

test_data["RelativeOnBoard"] = pd.cut(test_data["SibSp"]+test_data["Parch"], bins=[-1, 2,  3, 6, np.inf], labels=["half of them survived", "most of them survived", "some of them survived", "none of them survived"])

print(test_data)
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]

    

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer





num_pipeline = Pipeline([

        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),

        ("imputer", SimpleImputer(strategy="median")),

    ])



num_pipeline.fit_transform(train_data)



class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)

    

from sklearn.preprocessing import OneHotEncoder



cat_pipeline = Pipeline([

        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked", "AgeBucket", "RelativeOnBoard"])),

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OneHotEncoder(sparse=False)),

    ])



cat_pipeline.fit_transform(train_data)



from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline),

    ])



train_data_prepared = preprocess_pipeline.fit_transform(train_data)

train_data_prepared
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_score = cross_val_score(sgd_clf, train_data_prepared, train_data_1_label, cv=10)

print(sgd_score.mean())
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()

knn_score = cross_val_score(knn_clf, train_data_prepared, train_data_1_label, cv=10)

print(knn_score.mean())
from sklearn.svm import SVC

svm_clf = SVC()

svc_score = cross_val_score(svm_clf, train_data_prepared, train_data_1_label, cv=10)

print(svc_score.mean())
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

bayes_score = cross_val_score(model, train_data_prepared, train_data_1_label, cv=10)

print(bayes_score.mean())
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

forest_score = cross_val_score(forest_clf, train_data_prepared, train_data_1_label, cv=10)

print(forest_score.mean())
X_test = preprocess_pipeline.transform(test_data)
forest_clf.fit(train_data_prepared, train_data_1_label)

prediction = forest_clf.predict(X_test)
prediction
df["PassengerId"] = pd.DataFrame(test_data["PassengerId"])

df["Survived"] = pd.DataFrame(prediction) 

# saving the dataframe 

df.to_csv('D:\file31.csv', index=False) 