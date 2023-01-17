import numpy as np

import pandas as pd 



import plotly.express as px

import matplotlib.pyplot as plt



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

import xgboost
df = pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")
class DataProcess(BaseEstimator, TransformerMixin):

    """ This is just a transformer that I will feed into a pipeline  """

    

    def __init__(self):

        self.columns_to_drop = ["Firstname", "Lastname", "PassengerId"]

        self.country_ratio_param = pd.Series(np.nan)  ## This will be se tby fit

        self.has_family_map = pd.Series(np.nan)

    

    def transform(self, X, y=None):

        X_ = X.copy()

        X_["Sex"] = X_["Sex"].map({"M": 0, "F": 1})

        X_["Category"] = X_["Category"].map({"C": 0, "P": 1})

        

        X_["swedish"] = X_["Country"].apply(lambda x: x == "Sweden")

        X_["estonian"] = X_["Country"].apply(lambda x: x == "Estonia")

        X_["Country"] = X_["Country"].map(self.country_ratio_param)

        

        X_["has_family"] = X_["Lastname"].map(self.has_family_map)

        

        X_.drop(self.columns_to_drop, inplace=True, axis=1)

        

        assert not X_.isna().any().any(), f"Missing values found: {X_.isna().any()}"

        

        return X_

    

    def fit(self, X, y):

        """ There is not anything to fit here """

        X_ = X.copy()

        X_["Survived"] = y

        self.country_ratio_param = df.groupby("Country")["Survived"].apply(lambda x: x.sum()/x.shape[0]).sort_values()

        self.has_family_map = df.groupby("Lastname", as_index=False)["Firstname"].apply(lambda x: x.shape[0] > 1).set_index("Lastname").squeeze()

    

    def fit_transform(self, X, y):

        self.fit(X, y)

        return self.transform(X)

    

print("Processed DataFrame")

DataProcess().fit_transform(X=df.drop("Survived", axis=1), y=df["Survived"]).head()
print(df.shape)

df.head()
survivors_by_country = df.groupby(["Country", "Survived"])["Age"].count().sort_values().reset_index()

survivors_by_country["Survived"] = survivors_by_country["Survived"].astype(str)

survivors_by_country.rename(columns={"Age": "nPeople"}, inplace=True)

fig = px.bar(survivors_by_country, x="Country", y="nPeople", color="Survived")

fig.layout.yaxis.title = "# People"

fig.layout.title = "# Survivors and Total passenders by country"

fig.show()
country_survivor_ratio = df.groupby("Country")["Survived"].apply(lambda x: x.sum()/x.shape[0]).sort_values()

fig = px.bar(country_survivor_ratio.reset_index(), x="Country", y="Survived", title="Surviving Ratio per Country")

fig.layout.yaxis.title = "Survivor Ratio [%]"

fig.show()
grouped = df.groupby(["Category", "Sex"])["Survived"].sum().reset_index()

px.bar(grouped, y="Survived", x="Category", color="Sex", title="Number of survivors per Category and Gender")
%%time



# I used a bigger param grid on previous commits. 

params = {

 'xgboost__booster': ['gbtree'], 

 'xgboost__colsample_bytree': [0.8], 

 'xgboost__eta': [0.05],

 'xgboost__eval_metric': ['error'], 

 'xgboost__gamma': [0.5], 

 'xgboost__max_depth': [5],

 'xgboost__min_child_weight': [1], 

 'xgboost__n_estimators': [100], 

 'xgboost__subsample': [1.0]}



pipeline = Pipeline([("data_process", DataProcess()), ("xgboost", xgboost.XGBRFClassifier())])

clf = GridSearchCV(pipeline, cv=StratifiedShuffleSplit(6, random_state=1), n_jobs=-1, scoring=["f1", "accuracy"], refit="accuracy",  param_grid=params)



X = df.drop("Survived", axis=1)

y = df["Survived"]

clf.fit(X, y)

print(clf.best_params_)

pd.DataFrame(clf.cv_results_).sort_values("rank_test_f1").head(1).T
# Feature importance

_, ax = plt.subplots(figsize=(20, 5))

xgboost.plot_importance(clf.best_estimator_[-1], ax=ax)

plt.show()
_, ax = plt.subplots(figsize=(30, 30))

xgboost.plot_tree(clf.best_estimator_[-1], ax=ax, num_trees=0)

plt.show()