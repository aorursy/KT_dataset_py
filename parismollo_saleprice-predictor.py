import pandas as pd



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import seaborn as sns



from xgboost import XGBRegressor
# Set data paths

train_data_path = "../input/home-data-for-ml-course/train.csv"

test_data_path = "../input/home-data-for-ml-course/test.csv"



# Create pandas dataframe

X_full = pd.read_csv(train_data_path, index_col="Id")

X_test_full = pd.read_csv(test_data_path, index_col="Id")
# Remove rows that are missing target fields

X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)

# Set target and predictors dataframes

y = X_full["SalePrice"]

X_full.drop(["SalePrice"],axis=1, inplace=True)
# categorical columns to use on pipeline transformations. Low cardinality and object type columns

categorical_cols = [cname for cname in X_full if X_full[cname].nunique() < 

                    10 and X_full[cname].dtype == "object" ]

# numerical cols

numerical_cols = [cname for cname in X_full if X_full[cname].dtype in ["float64", "int64"]]
used_cols = categorical_cols + numerical_cols

# We will use only the columns that we defined above

X = X_full[used_cols].copy()

X_test = X_test_full[used_cols].copy()
# Numerical transformations...

num_transformer = SimpleImputer(strategy="mean")

# Categorical transformations

categorical_transformations = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="most_frequent")),

    ("onehot", OneHotEncoder(handle_unknown="ignore"))

])

# preprocessing

preprocessor = ColumnTransformer(transformers=[

    ("num", num_transformer, numerical_cols),

    ("categorical", categorical_transformations, categorical_cols)

]) 
def get_score(model):

    # Full pipeline

    full_pipeline = Pipeline(steps=[

        ("preprocessing", preprocessor),

        ("model", model)

    ])

    

    return (-1 * cross_val_score(full_pipeline, X, y, cv=3, scoring="neg_mean_absolute_error")).mean()
results_random_forest = [(n_estimators, get_score(RandomForestRegressor(n_estimators=n_estimators, random_state=42))) for n_estimators in range(100, 500, 50)]

results_random_forest = dict(results_random_forest)



results_xbgr = [(n_estimators, get_score(XGBRegressor(n_estimators=n_estimators, random_state=42))) for n_estimators in range(200, 700, 50)]

results_xbgr = dict(results_xbgr)
sns.lineplot(x=list(results_random_forest.keys()), y=list(results_random_forest.values()), label="RDF - Mean Absolute Error")
sns.lineplot(x=list(results_xbgr.keys()), y=list(results_xbgr.values()), label="XGBR - Mean Absolute Error")
best_estimator_rf = min(results_random_forest, key=results_random_forest.get)

print(f"estimator Random Forest: {best_estimator_rf} with MAE: {results_random_forest[best_estimator_rf]}")
best_estimator_xgbr = min(results_xbgr, key=results_xbgr.get)

print(f"estimator XBGR: {best_estimator_xgbr} with MAE: {results_xbgr[best_estimator_xgbr]}")
# Train model with best estimator on the full dataset

full_pipeline = Pipeline(steps=[

        ("preprocessing", preprocessor),

        ("model", XGBRegressor(n_estimators=best_estimator_xgbr, random_state=42))

    ])

full_pipeline.fit(X, y)

preds = full_pipeline.predict(X_test)
output = pd.DataFrame({"Id": X_test.index, "SalePrice": preds})

output.to_csv("submission.csv", index=False)