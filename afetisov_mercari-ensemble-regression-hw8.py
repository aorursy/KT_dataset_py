import matplotlib



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.pipeline import make_union, make_pipeline

from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import make_scorer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from lightgbm import LGBMRegressor



%matplotlib inline

sns.set()

plt.rcParams["figure.figsize"] = (20, 10)

pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv('../input/train.tsv', sep='\t', index_col=0)
df.info()
df.head()
df_test = pd.read_csv('../input/test.tsv', sep='\t', index_col=0)
df_test.info()
df_test.head()
sum(df.name.isnull())
df.item_condition_id.unique()
df.category_name.unique().shape
sum(df.category_name.isnull())
df.category_name.fillna("//").str.split("/").apply(lambda x: x[0]).unique().shape
df.category_name.fillna("//").str.split("/").apply(lambda x: x[1]).unique().shape
df.category_name.fillna("//").str.split("/").apply(lambda x: x[2]).unique().shape
df.groupby("category_name").agg({"price": "mean"}).sort_values("price", ascending=False).head(10)
sum(df.brand_name.isnull())
df.brand_name.unique().shape
df.groupby("brand_name").agg({"price": "mean"}).sort_values("price", ascending=False).head(10)
sum(df.price.isnull())
df.price.describe()
df.price.quantile(0.9)
df.price.quantile(0.99)
sum(df.price == 0)
plot = np.log10(df.price + 1).hist(bins=20, log=True)

plot.set_ylabel("Count")

plot.set_xlabel("log10(price)")
df.shipping.unique().shape
df.groupby("shipping").agg({"price": "mean"})
sum(df.item_description.isnull())
class LabelEncoderPipelineFriendly(LabelEncoder):

    def __init__(self, **kwargs):

        super(LabelEncoderPipelineFriendly, self).__init__(**kwargs)

    

    def fit(self, X, y=None):

        """this would allow us to fit the model based on the X input."""

        super(LabelEncoderPipelineFriendly, self).fit(X)

        

    def transform(self, X, y=None):

        return super(LabelEncoderPipelineFriendly, self).transform(X).reshape(-1, 1)



    def fit_transform(self, X, y=None):

        return super(LabelEncoderPipelineFriendly, self).fit(X).transform(X).reshape(-1, 1)
def prepare_data(train, test):

    def get_name_col(df):

        return df["name"]

    

    def get_condition_col(df):

        return df[["item_condition_id"]]

    

    def get_category_col(df):

        return df["category_name"].fillna("None").astype("category")

    

    def get_brand_col(df):

        return df["brand_name"].fillna("None").astype("category")

    

    def get_shipping_col(df):

        return df[["shipping"]]

    

    def get_desc_col(df):

        return df["item_description"].fillna("None")

    

    p = make_union(*[

        make_pipeline(FunctionTransformer(get_name_col, validate=False), 

                      TfidfVectorizer(min_df=15)), # we really don't want to end up with a gazzilion of columns

        make_pipeline(FunctionTransformer(get_condition_col, validate=False),

                      OneHotEncoder()),

        make_pipeline(FunctionTransformer(get_category_col, validate=False),

                      CountVectorizer()),

        make_pipeline(FunctionTransformer(get_brand_col, validate=False),

                      LabelEncoderPipelineFriendly(),

                      OneHotEncoder(sparse=True)),

        make_pipeline(FunctionTransformer(get_shipping_col, validate=False)),

        make_pipeline(FunctionTransformer(get_desc_col, validate=False),

                      TfidfVectorizer(ngram_range=(1, 3), 

                                      stop_words="english", 

                                      max_features=10000))

        ])

    

    train_rows = train.shape[0]

    df = pd.concat([train, test], axis=0)

    transformed = p.fit_transform(df)

    transformed_train, transformed_test = transformed[:train_rows], transformed[train_rows:]

    del df

    return (transformed_train, transformed_test)
X, X_test = prepare_data(df, df_test) # we need both of them to reliably get every categorical level

y = np.array(df.price)

log_y = np.log1p(y)
X.shape, X_test.shape, y.shape
# https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError

def rmsle(h, y): 

    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())

scorer = make_scorer(score_func=rmsle, greater_is_better=False)
# three linear models and one tree

models = [

    ("Lasso", Lasso, {"alpha": [0.1, 0.5], # bigger alpha -> bigger error

                      "random_state": [0],

                      "max_iter": [1000],

                      "tol": [0.001],

                      "selection": ["random"],

                      "fit_intercept": [False]}), # everything is normalized, thanks to TF-IDF

    ("Ridge", Ridge, {"solver": ["lsqr", "sparse_cg"], # svd doesn't support sparse matrices

                      "alpha": [0.1, 0.5, 1],          # cholesky is slow, as well as sag and saga

                      "random_state": [0],

                      "tol": [0.001],

                      "fit_intercept": [False]}),

    ("ElasticNet", ElasticNet, {"alpha": [0.1, 0.5], # bigger alpha -> bigger error

                                "l1_ratio": [0.1, 0.5, 0.9],

                                "random_state": [0],

                                "max_iter": [1000],

                                "tol": [0.001],

                                "selection": ["random"],

                                "fit_intercept": [False]}),

    ("DecisionTreeRegressor", DecisionTreeRegressor, {"max_depth": [3, 7], # more depth -> slower learning rate

                                                      "random_state": [0]})

]
best_models = []



for name, model_class, params in models:

    gs = GridSearchCV(model_class(), params, scoring=scorer, cv=5, n_jobs=1, refit=True)

    gs.fit(X, log_y)

    best_models.append((name, model_class, gs.best_estimator_, gs.best_params_, gs.best_score_))
for model_name, _, estimator, _, _ in best_models:

    predicted = np.expm1(estimator.predict(X_test))

    pd.DataFrame({"price": predicted}, index=df_test.index).to_csv("baseline_{}.csv".format(model_name), sep=",")
lgr = LGBMRegressor(n_jobs=-1, n_estimators=100)

lgr.fit(X, log_y, eval_metric=rmsle)
best_models.append(("LGBMRegressor", LGBMRegressor, lgr, lgr.get_params(), lgr.best_score_))
all_models_preds_train = np.zeros((X.shape[0], len(best_models)))

for i in range(len(best_models)):

    _, _, estimator, _, _ = best_models[i]

    all_models_preds_train[..., i] = estimator.predict(X)
all_models_preds_test = np.zeros((X_test.shape[0], len(best_models)))

for i in range(len(best_models)):

    _, _, estimator, _, _ = best_models[i]

    all_models_preds_test[..., i] = estimator.predict(X_test)
lr = LinearRegression()

lr.fit(all_models_preds_train, log_y)

final_preds = np.expm1(lr.predict(all_models_preds_test))
pd.DataFrame({"price": final_preds}, index=df_test.index).to_csv("ensemble.csv", sep=",")