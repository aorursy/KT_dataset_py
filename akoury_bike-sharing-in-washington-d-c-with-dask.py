import seaborn as sns

import dask.dataframe as dd

from sklearn.base import clone

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from dask.distributed import Client

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PowerTransformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import TimeSeriesSplit

from dask_ml.model_selection import RandomizedSearchCV

from dask_ml.preprocessing import OneHotEncoder, PolynomialFeatures

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
PATH = "../input/bike-sharing-dataset/hour.csv"

SPLITS = 4

METRIC = "r2"

SEED = 1

TARGET = "cnt"
client = Client()

client
types = {

    "season": "category",

    "yr": "category",

    "mnth": "category",

    "holiday": "bool",

    "weekday": "category",

    "workingday": "bool",

    "weathersit": "category",

}



df = dd.read_csv(PATH, parse_dates=[1], dtype=types, blocksize="300KB")

df.npartitions
precipitation = dd.read_csv(

    "https://gist.githubusercontent.com/akoury/6fb1897e44aec81cced8843b920bad78/raw/b1161d2c8989d013d6812b224f028587a327c86d/precipitation.csv",

    parse_dates=[1],

)

df = dd.merge(df, precipitation, how="left", on=["dteday", "hr"])

df["precipitation"] = (

    df["precipitation"]

    .mask(df["precipitation"].isnull(), 0)

    .mask(df["precipitation"] > 0, 1)

    .astype(bool)

)

df = df.set_index("dteday")

df.head()
df.divisions
df.dtypes
sample = df.sample(frac=0.15, replace=True, random_state=SEED)
plt.figure(figsize=(16, 8))

sns.distplot(sample[TARGET])
plt.figure(figsize=(16, 8))

grouped = (

    sample.groupby("hr")

    .agg({"registered": "mean", "casual": "mean"})

    .reset_index()

    .compute()

)

sns.lineplot(data=grouped, x="hr", y="registered", palette="husl", label="registered")

sns.lineplot(data=grouped, x="hr", y="casual", palette="husl", label="casual")

plt.xlabel("Hour")

plt.ylabel("Users")
plt.figure(figsize=(16, 8))

sample["mnth"] = sample["mnth"].astype("int")

grouped = (

    sample.groupby("mnth")

    .agg({"registered": "mean", "casual": "mean"})

    .reset_index()

    .compute()

)

sns.lineplot(data=grouped, x="mnth", y="registered", palette="husl", label="registered")

sns.lineplot(data=grouped, x="mnth", y="casual", palette="husl", label="casual")

plt.xlabel("Month")

plt.ylabel("Users")
plt.figure(figsize=(16, 8))

sample["weekday"] = sample["weekday"].astype("int")

grouped = (

    sample.groupby("weekday")

    .agg({"registered": "mean", "casual": "mean"})

    .reset_index()

    .compute()

)

sns.lineplot(

    data=grouped, x="weekday", y="registered", palette="husl", label="registered"

)

sns.lineplot(data=grouped, x="weekday", y="casual", palette="husl", label="casual")

plt.xlabel("Weekday")

plt.ylabel("Users")
plt.figure(figsize=(12, 10))

numeric = sample[

    ["instant", "hum", "atemp", "temp", "windspeed", "casual", "registered", "cnt"]

].compute()

numeric = (numeric - numeric.mean()) / numeric.std()

sns.boxplot(data=numeric, orient="h")
plt.figure(figsize=(13, 13))

sns.heatmap(

    sample.astype(float).corr(),

    cmap="coolwarm",

    center=0,

    square=True,

    annot=True,

    xticklabels=sample.columns,

    yticklabels=sample.columns,

)
df["is_late"] = (df["hr"] > 20) | (df["hr"] < 6)
df = df.drop(["season", "atemp", "casual", "registered"], axis=1)

df["hr"] = df["hr"].astype("category")

df = df.categorize()

train_df = df.loc[:"2012-09-30"]

holdout = df.loc["2012-10-01":]
num_pipeline = Pipeline([("power_transformer", PowerTransformer(method="yeo-johnson", standardize=True))])



categorical_pipeline = Pipeline([("one_hot", OneHotEncoder())])



pipe = Pipeline([

    ("column_transformer", ColumnTransformer([

        ("numerical_pipeline", num_pipeline, ["instant", "hum", "temp", "windspeed"]),

        ("categorical_pipeline", categorical_pipeline, ["yr", "mnth", "hr", "weekday", "weathersit"]),

    ], remainder="passthrough")),

])
X = train_df.drop([TARGET], axis=1)

y = train_df[TARGET]
def fit(X, y, pipe, model, grid):

    pipe = clone(pipe)

    pipe.steps.append(model)

    gridpipe = RandomizedSearchCV(

        pipe,

        grid,

        n_iter=100,

        cv=TimeSeriesSplit(n_splits=SPLITS),

        scoring=METRIC,

        random_state=SEED,

    )



    gridpipe.fit(X, y)



    print("Model: " + str(model[0]))



    print("Best Parameters: " + str(gridpipe.best_params_))

    print("Best Fold Score: " + str(gridpipe.best_score_))



    return gridpipe
model = ("linear_reg", LinearRegression())



grid = {

    "linear_reg__normalize": [True, False],

    "linear_reg__fit_intercept": [True, False],

}



lr_pipe = fit(X, y, pipe, model, grid)
model = ("xgb", XGBRegressor(random_state=SEED))



grid = {

    "xgb__max_depth": [3, 5],

    "xgb__learning_rate": [0.1, 0.2],

    "xgb__n_estimators": [100, 200],

}



xgb_gridpipe = fit(X, y, pipe, model, grid)
model = ("random_forest", RandomForestRegressor(n_estimators=100, random_state=SEED))



grid = {

    "random_forest__max_depth": [80, 100],

    "random_forest__min_samples_leaf": [3, 5],

    "random_forest__min_samples_split": [5, 10],

    "random_forest__max_leaf_nodes": [None, 30],

}



rf_gridpipe = fit(X, y, pipe, model, grid)
final_pipe = xgb_gridpipe

X_test = holdout.drop([TARGET], axis=1)

y_test = holdout[TARGET]



predicted = final_pipe.predict(X_test)

scores = {}

scores["R2"] = r2_score(y_test, predicted)

scores["MAE"] = mean_absolute_error(y_test, predicted)

scores["MSE"] = mean_squared_error(y_test, predicted)

scores
y_test = y_test.compute()

plt.figure(figsize=(11, 9))

plt.scatter(y_test, predicted, alpha=0.3)

plt.ylabel("Predicted")

plt.show()
y_test = y_test.reset_index(drop=True)

predicted = {

    i: predicted[24 * i : (24 * i) + 24].sum() for i in range(len(predicted) // 24)

}



plt.figure(figsize=(19, 9))

ax = sns.lineplot(

    data=y_test.groupby(y_test.index // 24).sum(), color="red", label="Actual"

)

ax = sns.lineplot(

    list(predicted.keys()), list(predicted.values()), color="blue", label="Predicted"

)

plt.xlabel("Day")

plt.ylabel("Users")
client.close()