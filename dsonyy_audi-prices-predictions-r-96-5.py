import os

import sys

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



# To reduce output size while working with vscode

%config InlineBackend.figure_format = 'png'



%matplotlib inline



# Display all columns

pd.options.display.max_columns = None



FIGURES_PATH = "plots/"



def save_fig(name, extension="png", resolution=300):

    os.makedirs(FIGURES_PATH, exist_ok=True)

    path = os.path.join(FIGURES_PATH, name + "." + extension)

    # print("Saving figure", name)

    plt.tight_layout()

    plt.savefig(path, format=extension, dpi=resolution)



np.random.seed(42)
AUDI_DATASET_PATH = "../input/used-car-dataset-ford-and-mercedes/audi.csv"



audi_orig = pd.read_csv(AUDI_DATASET_PATH)
audi = audi_orig.copy()

audi
audi.describe()
audi.info()
num_attribs = audi_orig.select_dtypes("number").columns.to_numpy()

cat_attribs = audi_orig.select_dtypes("object").columns.to_numpy()
audi.hist(figsize=(15, 10), bins=30)

save_fig("audi_numerical_hist")

for cat in cat_attribs:

    plt.subplots(figsize=(10, 4))

    sns.countplot(cat, data=audi, order=audi[cat].value_counts().index)

    save_fig(f"audi_{cat}_hist")
from sklearn.model_selection import train_test_split

audi_train, audi_test = train_test_split(audi_orig, random_state=42, test_size=0.2)
print("Train:\t", audi_train.shape)

print("Test:\t", audi_test.shape)
from pandas.plotting import scatter_matrix



attribs = num_attribs



scatter_matrix(audi[attribs], figsize=(12, 10))

save_fig("audi_scatter_matrix")
corr = audi[num_attribs].corr()

corr["price"].sort_values(ascending=False)
audi_corr = audi_train.copy()



columns_search = num_attribs[num_attribs != "price"]

for i in columns_search:

    for j in columns_search:

        if i != j:

            i_num = audi[columns_search].columns.get_loc(i)

            j_num = audi[columns_search].columns.get_loc(j)

            audi_corr[(i_num, j_num)] = audi_corr[i] / audi_corr[j]



correlations = audi_corr.corr()["price"]

correlations = correlations[~correlations.index.isin(num_attribs)].abs().sort_values()

correlations
def divided_attributes(X, min_corr=0.7):

    new_attribs = []

    for i, j in correlations[correlations >= min_corr].index:

        new_attribs.append((X[:, i] / X[:, j]).reshape((-1, 1)))

    new_attribs = np.concatenate(new_attribs, axis=1)

    return np.concatenate((X, new_attribs), axis=1)
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, StandardScaler, OneHotEncoder, Normalizer

from sklearn.compose import ColumnTransformer
X_train = audi_train.drop("price", axis=1)

X_test = audi_test.drop("price", axis=1)

y_train = audi_train[["price"]].to_numpy()

y_test = audi_test[["price"]].to_numpy()
num_attribs = X_train.select_dtypes("number").columns

cat_attribs = X_train.select_dtypes("object").columns
num_pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="median")),

    ("additional_attribs", FunctionTransformer(divided_attributes, kw_args={"min_corr":0.65})),

    ("polynomial_attribs", PolynomialFeatures(degree=2)),

    ("scaler", StandardScaler()),

])



cat_pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="most_frequent")),

    ("encoder", OneHotEncoder(handle_unknown="ignore")),

])



full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_attribs),

    ("cat", cat_pipeline, cat_attribs),

])



label_pipeline = Pipeline([

    ("scaler", StandardScaler()),

])
X_train = full_pipeline.fit_transform(X_train, y_train)

X_test = full_pipeline.transform(X_test)



y_train = label_pipeline.fit_transform(y_train)



y_test = label_pipeline.transform(y_test)
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score



def train_evaluate(model, X_train, y_train, X_test, y_test, cv=10):

    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="neg_mean_absolute_error")

    print("Model:\t", model)

    print("Mean MAE:\t", -scores.mean())

    print("StD MAE:\t", scores.std())
%%time

from sklearn.tree import DecisionTreeRegressor



train_evaluate(DecisionTreeRegressor(), X_train, y_train, X_train, y_train)
"""

%%time

from sklearn.model_selection import GridSearchCV



param_grid = [

    {"max_depth": [12, 15, 17, 20],

     "splitter": ["random", "best"],

     "random_state": [42],

     "min_samples_split": [3, 4, 5, 6]}

]



tree = DecisionTreeRegressor()

grid_search = GridSearchCV(tree, param_grid, cv=10, scoring="neg_mean_absolute_error", verbose=1)

grid_search.fit(X_train, y_train)



print("Best params:\t", grid_search.best_params_)

print("Best MAE:\t", -grid_search.best_score_)

best_tree = grid_search.best_estimator_

"""

best_tree = DecisionTreeRegressor(max_depth=12, min_samples_split=3, splitter="random", random_state=42)

train_evaluate(best_tree, X_train, y_train, X_train, y_train)
%%time

from sklearn.ensemble import RandomForestRegressor



train_evaluate(RandomForestRegressor(n_jobs=16), X_train, y_train.ravel(), X_train, y_train.ravel())
"""

%%time

from sklearn.model_selection import GridSearchCV



param_grid = [

    {"n_estimators": [200],

     "random_state": [42],

     "warm_start": [True, False],

     "oob_score": [True, False],

     "bootstrap": [True, False],

     "min_samples_split": [7, 8]}

]



forest = RandomForestRegressor()

grid_search = GridSearchCV(forest, param_grid, cv=10, scoring="neg_mean_absolute_error", verbose=1, n_jobs=16)

grid_search.fit(X_train, y_train.ravel())



print("Best params:\t", grid_search.best_params_)

print("Best MAE:\t", -grid_search.best_score_)

best_forest = grid_search.best_estimator_

"""

best_forest = RandomForestRegressor(bootstrap=True, min_samples_split=7, n_estimators=200, 

                                    oob_score=True, random_state=42, warm_start=True, n_jobs=16)

train_evaluate(best_forest, X_train, y_train.ravel(), X_train, y_train.ravel())
predictions = best_forest.predict(X_test)
from sklearn.metrics import r2_score, mean_absolute_error

 

final_r2 = r2_score(y_test, predictions)

final_mae = mean_absolute_error(y_test, predictions)



print("Final R²:\t", final_r2)

print("Final MAE:\t", final_mae)