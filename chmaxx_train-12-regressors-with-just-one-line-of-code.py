import pandas as pd



from tabulate import tabulate



from sklearn.datasets import load_boston

from sklearn.datasets import load_diabetes

from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split



# we import the three main functions from the utility script for scoring, training and prediction

from quick_regression import score_models

from quick_regression import train_models

from quick_regression import predict_from_models



BASE = "/kaggle/input"
df = pd.read_csv(f"{BASE}/sample-data/mpg.csv")

# score_models() just expects your training data as a Pandas dataframe and the column name of the target variable

# the function prints out scoring values ("r2" by default) and processing times per classifier

scores_mpg = score_models(df, "mpg")
# the utility script returns a dataframe with a sorted list of scores of 14 classifiers

print(tabulate(scores_mpg, showindex=False, floatfmt=".3f", headers="keys"))
scores_mpg = score_models(df=df, 

                          target_name="mpg", 

                          sample_size=None, 

                          impute_strategy="mean", 

                          scoring_metric="r2", 

                          log_x=False,

                          log_y=False, 

                          verbose=True,

                         )
# the diamonds data set has more than 50k samples which would take a while to crossvalidate on 14 classifiers

# we therefore reduce to 1000 samples

df = pd.read_csv(f"{BASE}/sample-data/diamonds.csv")

scores_diamonds = score_models(df, "price", sample_size=1000, verbose=False)

print()

print(tabulate(scores_diamonds, showindex=False, floatfmt=".3f", headers="keys"))
df = pd.read_csv(f"{BASE}/house-prices-advanced-regression-techniques/train.csv")

scores_ames = score_models(df, "SalePrice", verbose=False)

print(tabulate(scores_ames, showindex=False, floatfmt=".3f", headers="keys"))

print()



# now trying with log transformed target variable y

scores_ames = score_models(df, "SalePrice", log_y=True, verbose=False)

print(tabulate(scores_ames, showindex=False, floatfmt=".3f", headers="keys"))

print()



# now trying with log transformed predictive variables

scores_ames = score_models(df, "SalePrice", log_x=True, log_y=True, verbose=False)

print(tabulate(scores_ames, showindex=False, floatfmt=".3f", headers="keys"))

print()
pipelines = train_models(df, "SalePrice", log_y=True)



df_test = pd.read_csv(f"{BASE}/house-prices-advanced-regression-techniques/test.csv")

predictions = predict_from_models(df_test, pipelines)

predictions.head()
df = pd.read_csv(f"{BASE}/tmdb-box-office-prediction/train.csv")

baseline_tmdb = score_models(df, "revenue", 1000)
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
df = pd.read_csv(f"{BASE}/house-prices-advanced-regression-techniques/train.csv")

X = df.select_dtypes("number").drop("SalePrice", axis=1)

y = df.SalePrice



# using the convenience function make_pipeline() to build a whole data pipeline in just one line of code

pipe = make_pipeline(SimpleImputer(), RobustScaler(), LinearRegression())

print(f"The R2 score is: {cross_val_score(pipe, X, y).mean():.4f}")
num_cols = df.drop("SalePrice", axis=1).select_dtypes("number").columns

cat_cols = df.select_dtypes("object").columns



# we instantiate a first Pipeline, that processes our numerical values

numeric_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer()),

        ('scaler', RobustScaler())])



# the same we do for categorical data

categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),

        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    

# a ColumnTransformer combines the two created pipelines

# each tranformer gets the proper features according to «num_cols» and «cat_cols»

preprocessor = ColumnTransformer(

        transformers=[

            ('num', numeric_transformer, num_cols),

            ('cat', categorical_transformer, cat_cols)])



pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LinearRegression())])



X = df.drop("SalePrice", axis=1)

y = df.SalePrice

print(f"The R2 score is: {cross_val_score(pipe, X, y).mean():.4f}")
