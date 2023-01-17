# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Load train and test data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
#Sneak peek into data. 

#Personal details of the passengers. Name, sex, age, ticket and cabin fare details.

train_data.head()

#To check the completeness of train data. Cabin (23%), Age (80%), Embarked (99.7%) have data. Replace Age with median values?

train_data.info()
train_data.describe()
train_data["Survived"].value_counts()

train_data["Pclass"].value_counts()

train_data["Sex"].value_counts()

train_data["Embarked"].value_counts()
#Preprocessing pipeline

from sklearn.base import TransformerMixin, BaseEstimator

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
a = [train_data[c].value_counts().index[0] for c in train_data]

print(a)
from sklearn.preprocessing import OneHotEncoder

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                       index = X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
cat_pipeline = Pipeline([

    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),

    ("imputer", MostFrequentImputer()),

    ("cat_encoder", OneHotEncoder(sparse=False)),

])
cat_pipeline.fit_transform(train_data)
#Join numerical and categorical pipelines

from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list = [

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline),

])
X_train = preprocess_pipeline.fit_transform(train_data)

print(X_train[:5])
# Fetch label from train set.

y_train = train_data["Survived"]
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score



svm_clf = SVC(gamma="auto")

svm_clf.fit(X_train, y_train)
svm_scores = cross_val_score(svm_clf, X_train, y_train,cv=10)

print(svm_scores)

svm_scores.mean()
X_test = preprocess_pipeline.transform(test_data)

y_pred = svm_clf.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})

print(output)

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
#Reading training set csv file into train_data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()

women = train_data.loc[train_data.Sex=='female']["Survived"]

women_survival_rate = sum(women)/len(women)

print(women)

print("% of women who survived: ",women_survival_rate)

men = train_data.loc[train_data.Sex == 'male']['Survived']

men_survival_rate = sum(men) / len(men)

print("% of men who survived: ",men_survival_rate)



#Checking test data set

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()

#print(len(test_data))
from sklearn.ensemble import RandomForestClassifier



y =train_data['Survived']

features = ["Pclass", "Sex", "SibSp", "Parch"]

print(train_data[features])

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



ranForest_clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=1)

ranForest_clf.fit(X,y)

predictions = ranForest_clf.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

print(output)

#output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")
#Using Cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(ranForest_clf, X, y, cv = 5)

print(scores)
from sklearn.naive_bayes import GaussianNB



y =train_data['Survived']

features = ["Pclass", "Sex", "SibSp", "Parch"]

print(train_data[features])

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



gaussian_clf = GaussianNB()

gaussian_clf.fit(X,y)

gauss_prediction = gaussian_clf.predict(X_test)



gaussian_output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': gauss_prediction})

print(gaussian_output)



output.to_csv('my_gauss_submission.csv', index=False)

print("Your submission was successfully saved!")
