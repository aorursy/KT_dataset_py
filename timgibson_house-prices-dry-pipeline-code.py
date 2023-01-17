import category_encoders as ce

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns

import xgboost as xgb



from IPython.display import display, Markdown

from pandas_profiling import ProfileReport

from sklearn.base import BaseEstimator, clone, TransformerMixin

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

from sklearn.impute import SimpleImputer



from sklearn.linear_model import ElasticNetCV

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import (

    GridSearchCV,

    RandomizedSearchCV,

    train_test_split

)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (

    FunctionTransformer,

    OneHotEncoder,

    OrdinalEncoder,

    QuantileTransformer,

)

from sklearn_pandas import DataFrameMapper, gen_features
pd.set_option('display.max_columns', None)

sns.set()



def md(text: str):

    display(Markdown(text))

    

def rmsle(y_true, y_pred):

    return np.sqrt(mean_squared_log_error(y_true, y_pred))



def prepend_keys(dict_: dict, prepend: str) -> dict:

    return {f"{prepend}{key}": value for key, value in dict_.items()}
data_dir = "/kaggle/input/house-prices-advanced-regression-techniques"



data = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col="Id")

test = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col="Id")

sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))



X = data.drop(columns=["SalePrice"])

y = data["SalePrice"]



md(f"The total number of training observations is: `{len(X)}`")



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



md(f"The number of observations with which we do hyperparameter tuning is: `{len(X_train)}`")
profile = ProfileReport(data, minimal=True, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
class FeatureAdder(TransformerMixin, BaseEstimator):

    numeric_transforms = {

        "sale_age": lambda df: df["YrSold"] - df["YearRemodAdd"]

    }

    

    categoric_transforms = {

        "MSSubClass": lambda df: df["MSSubClass"].astype("object"),

    }

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X.assign(**self.numeric_transforms, **self.categoric_transforms)
numeric_cols = set(FeatureAdder().fit_transform(X).select_dtypes(exclude="object").columns)

categoric_cols = set(FeatureAdder().fit_transform(X).select_dtypes(include="object").columns)



print(numeric_cols)

print("=" * 80)

print(categoric_cols)
zero_impute_cols = {"LotFrontage", "MasVnrArea"}

mode_impute_cols = {"Electrical", "Functional", "KitchenQual", "MSZoning", "SaleType", "Utilities"}





# feature_imputer = ColumnTransformer(

#     transformers=[

#         ("num_zero", SimpleImputer(strategy="constant", fill_value=0), list(zero_impute_cols)),

#         ("num_median", SimpleImputer(strategy="median"), list(numeric_cols - zero_impute_cols)),

#         ("cat_mode", SimpleImputer(strategy="most_frequent"), list(mode_impute_cols)),

#         ("cat_unknown", SimpleImputer(strategy="constant", fill_value="Unknown"), list(categoric_cols - mode_impute_cols))

#     ],

#     remainder="drop"

# )



feature_imputer = DataFrameMapper(

    [

        *gen_features(

            columns=[[col] for col in zero_impute_cols],

            classes=[{"class": SimpleImputer, "strategy": "constant", "fill_value": 0}]

        ),

        *gen_features(

            columns=[[col] for col in numeric_cols - zero_impute_cols],

            classes=[{"class": SimpleImputer, "strategy": "median"}]

        ),

        *gen_features(

            columns=[[col] for col in mode_impute_cols],

            classes=[{"class": SimpleImputer, "strategy": "most_frequent"}]

        ),

        *gen_features(

            columns=[[col] for col in categoric_cols - mode_impute_cols],

            classes=[{"class": SimpleImputer, "strategy": "constant", "fill_value": "Unknown"}]

        ),

    ],

    df_out=True

)
target_encode_cols = {

    "Condition1",

    "Condition2",

    "Exterior1st",

    "Exterior2nd",

    "MSSubClass",

    "Neighborhood",

    "SaleType",

}





# TODO: consider adding "BsmtExposure", ["Gd", "Av", "Mn", "No"]

ordinal_encode_categories = ["Ex", "Gd","TA", "Fa", "Po", "Unknown"]

ordinal_encode_cols = {

    "BsmtCond",

    "BsmtQual",

    "ExterCond",

    "ExterQual",

    "FireplaceQu",

    "GarageCond",

    "GarageQual",

    "HeatingQC",

    "KitchenQual",

    "PoolQC",    

}



# let's write them all here once to prevent repeating ourselves

feature_transformers = {

    "ordinal": OrdinalEncoder(categories=[ordinal_encode_categories] * len(ordinal_encode_cols)),

    "normal": QuantileTransformer(n_quantiles=500, output_distribution="normal"),

    "target": ce.target_encoder.TargetEncoder(),

    "onehot": OneHotEncoder(handle_unknown="ignore"),

}



feature_cleaner = ColumnTransformer(

    transformers=[

        (

            "num_normal",

            clone(feature_transformers["normal"]),

            list(numeric_cols)

        ),

        (

            "cat_ordinal",

            clone(feature_transformers["ordinal"]),

            list(ordinal_encode_cols)

        ),

        (

            "cat_target",

            clone(feature_transformers["target"]),

            list(target_encode_cols)

        ),

        (

            "cat_onehot",

            clone(feature_transformers["onehot"]),

            list(categoric_cols - ordinal_encode_cols - target_encode_cols)

        ),

    ],

    remainder="drop"

)
def transform_regressor(regressor):

    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)
pipe = Pipeline(steps=[

    ("feature_adder", FeatureAdder()),

    ("feature_imputer", feature_imputer),

    ("feature_cleaner", feature_cleaner),

    ("model", transform_regressor(ElasticNetCV(cv=5)))

])
feature_param_grid = {

    "feature_cleaner__cat_ordinal": [

        clone(feature_transformers["ordinal"]),

        clone(feature_transformers["onehot"]),

    ],

    "feature_cleaner__cat_target": [

        clone(feature_transformers["target"]),

        clone(feature_transformers["onehot"])

    ],

}



grid = GridSearchCV(pipe, param_grid=feature_param_grid, cv=5, scoring="neg_mean_squared_log_error", n_jobs=-1)

grid.fit(X_train, y_train)
grid.best_params_
md(f"Model score on the validation set: `{rmsle(y_test, grid.predict(X_test)):.3f}`")

md(f"Model score on the train set: `{rmsle(y_train, grid.predict(X_train)):.3f}`")
y_train_trf = np.log1p(y_train)

y_test_trf = np.log1p(y_test)



xgb_pipe_trf = Pipeline(grid.best_estimator_.steps[:-1])

xgb_pipe_trf.fit(X_train, y_train_trf)



X_train_trf = xgb_pipe_trf.transform(X_train)

X_test_trf = xgb_pipe_trf.transform(X_test)
xgb_fit_params = {

    "early_stopping_rounds": 10,  

    "eval_set" : [(X_test_trf, y_test_trf)],

    "verbose": False

}



xgb_model = xgb.XGBRegressor(objective="reg:squaredlogerror")
"""

Credit: http://danielhnyk.cz/how-to-use-xgboost-in-python/

For the randomised parameter grid.

"""



import scipy.stats as st



one_to_left = st.beta(10, 1)  

from_zero_positive = st.expon(0, 50)



xgb_params = {  

    "n_estimators": st.randint(3, 2000),

    "max_depth": st.randint(3, 60),

    "learning_rate": st.uniform(0.05, 0.4),

    "colsample_bytree": one_to_left,

    "subsample": one_to_left,

    "gamma": st.uniform(0, 10),

    "reg_alpha": from_zero_positive,

    "min_child_weight": from_zero_positive,

}



xgb_grid = RandomizedSearchCV(

    xgb_model,

    param_distributions=xgb_params,

    n_iter=20,

    cv=5,

    scoring="neg_mean_squared_log_error",

)

xgb_grid.fit(X_train_trf, y_train_trf, **xgb_fit_params)
xgb_grid.best_params_
xgb_pipe = Pipeline([*xgb_pipe_trf.steps, ("model", transform_regressor(xgb_grid.best_estimator_))])



xgb_pipe.fit(X_train, y_train, **prepend_keys(xgb_fit_params, "model__"))
md(f"Model score on the validation set: `{rmsle(y_test, xgb_pipe.predict(X_test)):.3f}`")

md(f"Model score on the train set: `{rmsle(y_train, xgb_pipe.predict(X_train)):.3f}`")
submission = pd.DataFrame({"SalePrice": grid.predict(test)}, index=test.index)



submission
submission.to_csv("submission.csv")
from sklearn.inspection import permutation_importance



def plot_importance(estimator, X, y):

    """

    Source: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html

    """

    result = permutation_importance(estimator, X, y, n_repeats=10,

                                    random_state=42, n_jobs=-1)

    sorted_idx = result.importances_mean.argsort()



    fig, ax = plt.subplots(figsize=(15, 15))

    ax.boxplot(result.importances[sorted_idx].T,

               vert=False, labels=X_test.columns[sorted_idx])

    ax.set_title("Permutation Importances")

    fig.tight_layout()

    

    return fig
fig = plot_importance(grid, X_test, y_test)