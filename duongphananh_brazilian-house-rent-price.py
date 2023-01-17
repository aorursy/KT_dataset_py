import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%pylab inline
# Import the scikit-learn methods and models here

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV
data = pd.read_csv("../input/brasilian-houses-to-rent/houses_to_rent_v2.csv").drop(["total (R$)"], axis=1)

data.head()
X_labels = ["city", "area", "rooms", "bathroom", "parking spaces", "floor", "animal", "furniture", "hoa (R$)", "fire insurance (R$)", "property tax (R$)"]

data.dropna()

X = data[X_labels]

y = data["rent amount (R$)"]
data.describe()
data.groupby(["city"]).mean()
sns.heatmap(data.corr(), annot=True)
X = X.drop(["hoa (R$)", "property tax (R$)"], axis=1)

X.head()
sns.pairplot(data=data, hue="city")
print(X["city"].unique())

print(X["animal"].unique())

print(X["furniture"].unique())
print(X["floor"].unique())

# I checked the other columns. Seems to be all fine
# Run this only once, please

X["city"] = X["city"].apply(lambda x: 1 if x == "São Paulo" 

                            else 2 if x == "Porto Alegre" 

                            else 3 if x == "Rio de Janeiro"

                            else 4 if x =="Campinas" else 5)

X["animal"] = X["animal"].apply(lambda x: 1 if x == "acept" else 0) # Gosh the hell is acept

X["furniture"] = X["furniture"].apply(lambda x: 1 if x == "furnished" else 0)

X["floor"] = X["floor"].apply(lambda x: np.nan if x == "-" else x)
X.tail()
y.tail()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

print("There are {} samples in the training set and {} samples in the test set".format(X_train.shape[0], X_test.shape[0]))
"""

pipeline = Pipeline(steps=[("preprocess", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),

                            ("model", RandomForestRegressor(random_state=1))])

grid_params = {

    "model__n_estimators": [140, 160, 180],

    "model__criterion": ["mse"],

    "model__bootstrap": [False],

    "model__max_depth": list(range(5, 21, 5))

}

grid_search = GridSearchCV(estimator=pipeline, param_grid=grid_params, cv=3, verbose=1)

grid_search.fit(X_train, y_train)

grid_search.best_params_

"""
final_model = Pipeline(steps=[("preprocess", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),

                            ("model", RandomForestRegressor(random_state=1,

                                                            bootstrap=False, 

                                                            criterion="mse",

                                                            n_estimators=180,

                                                            max_depth=7))])

scores = cross_validate(final_model, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error")

print(-scores["test_score"].mean())
final_model_mae = Pipeline(steps=[("preprocess", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),

                            ("model", RandomForestRegressor(random_state=1,

                                                            bootstrap=False, 

                                                            criterion="mae",

                                                            n_estimators=60,

                                                            max_depth=16))])

scores_mae = cross_validate(final_model_mae, X_train, y_train, cv=3, scoring="neg_mean_absolute_error")

print(-scores_mae["test_score"].mean())
final_model_mae_fire = Pipeline(steps=[("preprocess", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),

                            ("model", RandomForestRegressor(random_state=1,

                                                            bootstrap=False, 

                                                            criterion="mse",

                                                            n_estimators=200,

                                                            max_depth=15))])

scores_mae_fire = cross_validate(final_model_mae_fire, X_train[["fire insurance (R$)", "city"]], y_train, cv=3, scoring="neg_root_mean_squared_error")

print(-scores_mae_fire["test_score"].mean())