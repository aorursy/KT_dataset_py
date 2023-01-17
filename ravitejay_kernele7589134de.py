# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def load_data(file):

    file = '../input/'+file+'.csv'

    return pd.read_csv(file)



df_train_set = load_data("train")

df_test_set = load_data("test")
df_train_set.info()
df_train_set.head(5)
pd.pivot_table(df_train_set, values="PassengerId", index="Pclass", columns="Survived",aggfunc='count',

               margins=True)
func = lambda x: 100*x.count()/df_train_set.shape[0]



pd.pivot_table(df_train_set, values="PassengerId", index=["Pclass"], columns="Survived", aggfunc=func,

               margins=True, fill_value=0)
pd.pivot_table(df_train_set, values="PassengerId", index="Embarked", columns="Survived",aggfunc='count',

               margins=True)
df_train_set_final = df_train_set
# Remove the two rows that are missing embarkation information

df_train_set_final = df_train_set_final.dropna(subset=["Embarked"])



# Remove the columns Cabin and Ticket information for this analysis

df_train_set_final.drop(columns=['Cabin','Ticket'], inplace=True)
corr_matrix_train = df_train_set_final.corr()
corr_matrix_train["Survived"].sort_values(ascending=False)
df_train_set_final.boxplot(by='Survived', column=['Fare'], grid = False)
df_train_set_final.info()
df_train_set_final.describe()
# Creating a DataFrame Selector that will pull either categorical or numerical columns

# Data Frame selector



from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):

	def __init__(self, attribute_names):

		self.attribute_names = attribute_names

	def fit(self, X, y=None):

		return self

	def transform(self, X):

		return X[self.attribute_names].values
# Copying the labels into a dataset



y_train = df_train_set_final["Survived"]
from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# One hot encoder for converting the categorical values into binary values

cat_encoder = OneHotEncoder(sparse=False)



# Using median strategy to replace the missing values for Age column

Imputer = SimpleImputer(strategy="median")



# List of numerical and categorical attributes

num_attribs = ["Age","SibSp","Parch","Fare"]

cat_attribs = ["Sex","Embarked","Pclass"]



num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attribs)),

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])

cat_pipeline = Pipeline([

('selector', DataFrameSelector(cat_attribs)),

('cat_encode', OneHotEncoder(sparse=False)),

])

full_pipeline = FeatureUnion(transformer_list=[

("num_pipeline", num_pipeline),

("cat_pipeline", cat_pipeline),

])



# Applying the full pipeline on the training set



X_train = full_pipeline.fit_transform(df_train_set_final)
X_train
X_train.shape
from sklearn.svm import SVC



#svm_clf = SVC(gamma="auto)

svm_clf = SVC(gamma="auto", C=1, degree=1, kernel='rbf')



svm_clf.fit(X_train, y_train)
X_test = full_pipeline.fit_transform(df_test_set)

y_pred = svm_clf.predict(X_test)
from sklearn.model_selection import cross_val_score



svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)

svm_scores.mean()
X_test
submission = pd.DataFrame({

    "PassengerId": df_test_set["PassengerId"],

    "Survived": y_pred

})





submission.head(5)

submission.to_csv('submission.csv', index=False)