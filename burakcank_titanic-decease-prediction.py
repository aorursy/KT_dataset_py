# import the libraries

import numpy as np

import pandas as pd



from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler



# import the dataset

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
# feature selection - exclude passenger name, ticket and cabin for now

train = train.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9, 11]]

test = test.iloc[:, [0, 1, 3, 4, 5, 6, 8, 10]]



train.head()
# split the data into features and labels

X_train = train.drop(columns=["Survived", "PassengerId"])

y_train = train["Survived"]

X_train.head()
# inspect the data

X_train.describe()
# look which values are missing in the train and test sets

X_train.info()
X_train.head()
num_pipeline = Pipeline([

    ("imputer", SimpleImputer()),

    ("mm_scaler", MinMaxScaler())

])



cat_pipeline = Pipeline([

    ("cat_imputer", SimpleImputer(strategy="most_frequent")),

    ("oe", OneHotEncoder())

])



X_train_num = X_train.drop(["Sex", "Embarked"], axis=1)

X_train_cat = X_train[["Sex", "Embarked"]]



full_pipeline = ColumnTransformer([

    ("num", num_pipeline, list(X_train_num.columns)),

    ("cat", cat_pipeline, list(X_train_cat.columns))

])



X_train_prepared = full_pipeline.fit_transform(X_train)

X_train_prepared
sgd_clf = SGDClassifier()



cross_val_score(sgd_clf, X_train_prepared, y_train, cv=5)
test.head()
test_prepared = full_pipeline.transform(test)

test_prepared
# train the model with Stochastic Gradient Descent

sgd_clf.fit(X_train_prepared, y_train)

preds = sgd_clf.predict(test_prepared)
index = np.arange(892, 1310)

result = np.column_stack((index, preds))



result_df = pd.DataFrame(result, columns = ['PassengerId', 'Survived'])



result_df.head(10)

result_df.to_csv('sgd_clf.csv', index=False)