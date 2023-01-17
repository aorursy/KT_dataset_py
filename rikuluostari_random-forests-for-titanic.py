import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib as mpl

import matplotlib.pyplot as plt
# load data



df_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')



# select columns for model

X_train=df_train[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare"]]

y_train=np.array( df_train[["Survived"]])

X_test=df_test[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare"]]



# make indexes

X_train=X_train.set_index('PassengerId')

X_test=X_test.set_index('PassengerId')



# convert to log scale

X_train["Parch"] = np.log(X_train["Parch"]+1)

X_train["Fare"] = np.log(X_train["Fare"]+1)

X_train["SibSp"] = np.log(X_train["SibSp"]+1)



X_test["Parch"] = np.log(X_test["Parch"]+1)

X_test["Fare"] = np.log(X_test["Fare"]+1)

X_test["SibSp"] = np.log(X_test["SibSp"]+1)
X_train.hist()
# pipeline for replacing nans with medians and then scale 



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('std_scaler', MinMaxScaler()),

    ])

# additionally onehot for categoricals



from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



num_attribs = list(["Age","SibSp","Parch","Fare"])

cat_attribs = ["Pclass","Sex"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])

# prepare training and testing data



X_train_prep = full_pipeline.fit_transform(X_train)

X_test_prep = full_pipeline.fit_transform(X_test)
# Search random hyperparameters for random forest



from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=6),

    }



forest_reg = RandomForestClassifier(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=20, cv=5, scoring='neg_mean_squared_error', random_state=42)



rnd_search.fit(X_train_prep, y_train.ravel())

# show the best parameters

rnd_search.best_params_
# fit the model with all data



forest_reg = RandomForestClassifier(n_estimators=100, max_features=5)

forest_reg.fit(X_train_prep, y_train.ravel())
# make predictions

X_test['Survived'] = forest_reg.predict(X_test_prep)

submission = X_test[["Survived"]]

# save results

submission.to_csv('submission.csv')