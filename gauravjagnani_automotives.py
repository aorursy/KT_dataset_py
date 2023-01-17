import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
df = pd.read_csv("/kaggle/input/automobile-dataset/Automobile_data.csv")
df.shape
df.head()
df.info()
df.dtypes.value_counts()
df = df.replace("?", np.nan)
df.isnull().sum()
null_cols = df.columns[df.isnull().any()].values
null_cols
df[null_cols]
df[null_cols].dtypes
# normalized-losses - Replace missing with mean
# num-of-doors - Replace missing with mode
# bore - Replace missing with mean
# stroke - Replace missing with mean
# horsepower - Replace missing with mode
# peak-rpm - Replace missing with mode
# price - Replace missing with mean
for col in ["normalized-losses", "bore",
                        "stroke", "price"]:
    df[col] = df[col].astype(np.number)
    df[col].fillna(df[col].mean(),
                       inplace=True)

df["num-of-doors"].replace(
            {"two": 2, "four": 4},
            inplace=True)
for col in ["horsepower", "peak-rpm",
                            "num-of-doors"]:
    df[col] = df[col].astype(np.number)
    df[col].fillna(df[col].mode()[0],
                        inplace=True)
df[null_cols].isnull().any()
df.describe()
sns.distplot(df["city-mpg"])
sns.distplot(df["highway-mpg"])
numeric_cols = df.select_dtypes(np.number).columns.values
categorical_cols = df.select_dtypes(object).columns.values
print(numeric_cols)
print(categorical_cols)
cols = 5

for i in range(0, len(numeric_cols), cols):
    sns.pairplot(df,
        x_vars=numeric_cols[i: i + cols],
        y_vars=["highway-mpg"])

for i, col in enumerate(categorical_cols):
    plt.figure()
    sns.boxplot(data=df, y=col, x="highway-mpg")
    sns.swarmplot(data=df, y=col, x="highway-mpg")

df["num-of-cylinders"].replace(
	dict(
		four = 159,
		six =   24,
		five =  11,
		eight =  5,
		two =    4,
		twelve = 1,
		three =  1,
	),
    inplace=True
)
df["num-of-cylinders"].head()
df[categorical_cols]
categorical_cols = categorical_cols[
    categorical_cols != "num-of-cylinders"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df[categorical_cols].head()
df.head()
df.describe()
unscaled_cols =	["normalized-losses",
                    "make",
                    "wheel-base",
                    "engine-size",
                    "horsepower",
                    "peak-rpm",
                    "highway-mpg",
                    "price"]

for col in unscaled_cols:
	df[col] = StandardScaler().fit_transform(df[[col]])
# Remove correlated columns
df_X = df.drop(["highway-mpg", "price"], axis=1)
corr_df = df_X.corr().abs()
up_tri = np.triu(
    np.full(corr_df.shape, 1), k=1).astype(bool)
corr_df = corr_df.where(up_tri)
correlated_cols = [col for col in corr_df.columns \
                       if any(corr_df[col] > 0.75)]
correlated_cols
df.drop(correlated_cols, axis=1, inplace=True)
Y = df["price"]
X = df.drop(["price"], axis=1)
ridge = Ridge()
ridge.fit(X, Y)
scores = cross_val_score(ridge, X, Y, cv=5)
scores.mean()
svr = SVR(kernel="linear")
svr.fit(X, Y)
scores = cross_val_score(svr, X, Y, cv=5)
scores.mean()
# gridSearch = GridSearchCV(
#     svr,
#     param_grid=dict(kernel=["linear", "poly", "rbf"]),
#     cv=5
# )
# gridSearch.fit(X, Y)
# gridSearch.best_params_
randomForest = RandomForestRegressor(n_estimators=6)
randomForest.fit(X, Y)
scores = cross_val_score(randomForest, X, Y, cv=5)
scores.mean()
# gridSearch = GridSearchCV(
#     randomForest,
#     param_grid=dict(n_estimators=range(1, 20)),
#     cv=5
# )
# gridSearch.fit(X, Y)
# gridSearch.best_params_
kNeighbors = KNeighborsRegressor(n_neighbors=50)
kNeighbors.fit(X, Y)
scores = cross_val_score(kNeighbors, X, Y, cv=5)
scores.mean()
# gridSearch = GridSearchCV(
#     kNeighbors,
#     param_grid=dict(n_neighbors =range(1, 100)),
#     cv=5
# )
# gridSearch.fit(X, Y)
# gridSearch.best_params_

stackingRegressor = StackingRegressor(
    estimators=(
        ("Ridge", ridge),
        ("SVR", svr),
        ("RandomForestRegressor", randomForest),
        ("KNeighborsRegressor", kNeighbors)
    ),
    final_estimator=SVR(kernel="poly"),
)

scores = cross_val_score(stackingRegressor, X, Y, cv=5)
scores.mean()