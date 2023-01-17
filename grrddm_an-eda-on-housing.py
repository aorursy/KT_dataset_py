import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import CategoricalImputer
from collections import defaultdict
plt.rcParams["figure.figsize"] = (12, 6)


class DataframeSelector(BaseEstimator, TransformerMixin):
    """
    ScikitLearn class to select a subset of the columns
    in a pandas dataframe 
    """
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns].values
    
class CategoricalDfImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values from a categorical pd.DataFrame
    
    Parameters
    ----------
    nan_threshold: if specified, columns with a percentage of nans
    greater than nan_threshold will be dropped.
    """
    def __init__(self, missing_values="NaN", nan_threshold=None):
        self.missing_values = missing_values
        self.nan_threshold = nan_threshold
    def fit(self, X, y=None):
        return self
    def transform (self, X):
        # Return a numpy array in the indices where the % of nans is greater than the stablished threshold
        void_cols = np.where(np.count_nonzero(pd.isna(X), axis=0) / X.shape[0] > self.nan_threshold)[0] if self.nan_threshold is not None else []
        Xout = np.empty_like(X, dtype="object")
        for i, col in enumerate(X.T):
            imputer = CategoricalImputer()
            cleaned_col = imputer.fit_transform(col)
            Xout[:, i] = cleaned_col
        Xout = Xout[:, [i for i in range(len(X.T)) if i not in void_cols]]
        return Xout

class LabelNdBinarizer(BaseEstimator, TransformerMixin):
    """
    Transorm a bi-dimensional categorical numpy array ('object')
    into a single OHE matrix.
    """
    def __init__(self):
        self.all_classes_ = []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xout = None
        _, nfeats = X.shape
        for ix in range(nfeats):
            lab_bin = LabelBinarizer()
            bin_X = lab_bin.fit_transform(X[:, ix])
            self.all_classes_.append(lab_bin.classes_)
            if Xout is None:
                Xout = bin_X
            else:
                Xout = np.c_[Xout, bin_X]
        return Xout
houses = pd.read_csv("../input/train.csv", index_col=0)
houses.head()
houses.info()
sns.distplot(houses.SalePrice);
houses.SalePrice.describe()
def plot_price_seg(housevar):
    """
    Function to graph the distribution of `SalePrice` groupped by a
    the categorical variable 'housevar'
    """
    houses_var = houses[housevar].unique()
    for cat in houses_var:
        sns.distplot(houses.query(f"{housevar} == '{cat}'").SalePrice, kde=False,
                     label=cat, norm_hist=True, hist_kws={"alpha":0.3})
    plt.legend();
plot_price_seg("MSZoning")
plot_price_seg("CentralAir")
houses.pivot_table(index="OverallCond",
                   values="SalePrice", aggfunc="mean").plot(kind="bar")
rates = ['Very Excellent', 'Excellent', 'Very Good',
         'Good', 'Above Average', 'Average', 'Below Average',
         'Fair', 'Poor', 'Very Poor'][::-1]
plt.xticks(range(10), rates);
sns.heatmap(houses.pivot_table(index="ExterQual", columns="ExterCond",
                   values="SalePrice", aggfunc="mean"), annot=True, fmt=",.2f", cmap=plt.cm.viridis_r)
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

linreg = Pipeline([
    ("ohebin", OneHotEncoder()),
    ("linreg", LinearRegression())
])

x = houses[["OverallCond"]].values.reshape(-1, 1)
y = houses.SalePrice

mean_mse = -cross_val_score(linreg, x, y, cv=5, scoring="neg_mean_squared_error").mean()
print(f"{np.sqrt(mean_mse):,.2f}")
void_cols = ["SalePrice"]
var_cats = defaultdict(list)
for col, dtype in houses.dtypes.iteritems():
    if col not in void_cols:
        var_cats[dtype.name].append(col)
var_cats = dict(var_cats)
var_cats.keys()
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn_pandas import CategoricalImputer

intpipe = Pipeline([
    ("feat_select", DataframeSelector(var_cats["int64"])),
    ("float_imp", Imputer()),
    ("ohe", OneHotEncoder())
])

catpipe = Pipeline([
    ("cat_select", DataframeSelector(var_cats["object"])),
    ("lab_imp", CategoricalDfImputer(nan_threshold=0.4)),
    ("lab_enc", LabelNdBinarizer())
])

floatpipe = Pipeline([
    ("float_select", DataframeSelector(var_cats["float64"])),
    ("float_imp", Imputer())
])

transform_pipeline = FeatureUnion([
    ("ints", intpipe),
    ("cat", catpipe),
    ("float", floatpipe)
])
houses_x = transform_pipeline.fit_transform(houses.drop("SalePrice", axis=1))
houses_y = houses.SalePrice.values
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()
lreg.fit(houses_x, houses_y)
from sklearn.metrics import mean_squared_log_error
mean_mse = -cross_val_score(lreg, houses_x, houses_y, cv=8, scoring="neg_mean_squared_log_error").mean()
print(f"{np.sqrt(mean_mse):,.2f}")
transform_pipeline.transformer_list[1][1].named_steps["lab_imp"]
houses_x_test = pd.read_csv("../input/test.csv")
houses_x_test_transf = transform_pipeline.transform(houses_x_test)

houses_x.shape, houses_x_test_transf.shape