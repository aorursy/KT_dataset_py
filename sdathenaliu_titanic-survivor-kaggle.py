import numpy as np

import pandas as pd



import os

print(os.listdir("../input")) # on kaggle
# DATA_PATH = os.path.join(".")

DATA_PATH = os.path.join("..", "input") # on kaggle



def load_data(filename, data_path=DATA_PATH):

    file_path = os.path.join(data_path, filename)

    return pd.read_csv(file_path)
train_data = load_data("train.csv")

test_data = load_data("test.csv")
train_data.head()
train_data.info()
train_data.describe()
train_data["Pclass"].value_counts()
train_data["Sex"].value_counts()
train_data["Embarked"].value_counts()
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler



numerical_pipeline = Pipeline([

#         ("select_numeric", DataFrameSelector()),

        ("imputer", SimpleImputer(strategy="median")),

        ("std_scaler", StandardScaler())

    ])
class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OneHotEncoder



categorical_pipeline = Pipeline([

#         ("select_cat", DataFrameSelector(cat_attribs)),

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OneHotEncoder(sparse=False)),

    ])
num_attribs = ["Age", "SibSp", "Parch", "Fare"]

cat_attribs = ["Pclass", "Sex", "Embarked"]



from sklearn.compose import ColumnTransformer

preprocessing_pipeline = ColumnTransformer([

        ("num", numerical_pipeline, num_attribs),

        ("cat", categorical_pipeline, cat_attribs),

    ])
X_train  = preprocessing_pipeline.fit_transform(train_data)

y_train = train_data["Survived"]
X_train
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



forest_clf = RandomForestClassifier(n_estimators=100)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores
forest_scores.mean()
from sklearn.model_selection import GridSearchCV



params_grid = [

    {'n_estimators': [10, 30, 100, 300, 1000]}

]



forest_clf = RandomForestClassifier()

grid_search = GridSearchCV(forest_clf, params_grid, cv=5, 

                           scoring="accuracy", 

                           return_train_score=True)



grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_estimator_
full_pipeline = Pipeline([

    ("preprocessing", preprocessing_pipeline),

    ("random_forest", grid_search.best_estimator_)

])
test_data.head()
full_pipeline.fit(train_data, y_train)
y_test = full_pipeline.predict(test_data)

np.sum(y_test)/y_test.shape[0]
PassengerId = test_data["PassengerId"]

result = pd.DataFrame({

    "PassengerId": pd.Series(PassengerId),

    "Survived": pd.Series(y_test),

})

result.head()
result.info()
result.to_csv("prediction.csv", index=False)