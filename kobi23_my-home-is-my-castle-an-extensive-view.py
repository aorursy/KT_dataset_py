# tools know to be useful
import numpy as np
import pandas as pd
from scipy import stats
import os

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

import seaborn as sns
cmap = sns.color_palette("Blues")
sns.palplot(sns.color_palette("Blues"))

import warnings
warnings.filterwarnings('ignore')
# the answer to everything - or at least to deterministic randomness
np.random.seed(42)

# for some it is location, location, location - here it is data, data, data
train_set = pd.read_csv("../input/train.csv")
# What's in it
train_set.head()
train_set.describe()
# ...or a forrest of histograms
train_set.hist(bins=50, figsize=(30,20));
#descriptive statistics summary
train_set['SalePrice'].describe()
# what are we aiming for?
# Something positive...
print("Skewness: %f" % train_set['SalePrice'].skew())
# ... and nothing too normal...
print("Kurtosis: %f" % train_set['SalePrice'].kurt())
# ...see ? (histogram)
sns.distplot(train_set['SalePrice']/1000, kde = False, fit=stats.norm)
sns.distplot(train_set['SalePrice']/1000);
# a handy scatter plot grlivarea/saleprice (you can vary the input variable to determine the sales price)
var = 'GrLivArea'
data = pd.concat([train_set['SalePrice'], train_set[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000),alpha=0.1);
# Maybe we check out a select few against each other
from pandas.plotting import scatter_matrix

attributes = ["SalePrice", "GrLivArea", "1stFlrSF", "GarageArea","LotArea"]
scatter_matrix(train_set[attributes], figsize=(20, 15),alpha=0.2);
# (b) scatter plot 1st floor sqft/saleprice
var = 'LotArea' #'1stFlrSF'
data = pd.concat([train_set['SalePrice'], train_set[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', xlim=(0,30000), ylim=(0,600000),alpha=0.2);
g = sns.jointplot(x=var, y='SalePrice',  xlim=(0,25000), ylim=(0,400000), data=data, kind="kde", color="m")
g.plot_joint(plt.scatter, c="grey", s=30, linewidth=1, marker="+", alpha=0.3)
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$");
train_set_num = train_set.select_dtypes(include=[np.number])
# Uncomment only in case you really want to look at al numerical values .... but be careful what you ask for 
#scatter_matrix(train_set_num, figsize=(40, 40),alpha=0.2)

# sometimes it is better to reduce the scope of what to look at....
corr_matrix = train_set_num.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train_set['SalePrice'], train_set[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))

#fig = sns.boxplot(x=var, y="SalePrice", data=data, palette = cmap)
fig = sns.violinplot(x=var, y="SalePrice", data=data)
sns.despine(offset=10, trim=True); 

fig.axis(ymin=0, ymax=800000);


var = 'YearBuilt'
data = pd.concat([train_set['SalePrice'], train_set[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
var = 'GarageCars'
data = pd.concat([train_set['SalePrice'], train_set[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))

#fig = sns.boxplot(x=var, y="SalePrice", data=data, palette = cmap)
fig = sns.violinplot(x=var, y="SalePrice", data=data)
sns.despine(offset=10, trim=True); 

fig.axis(ymin=0, ymax=800000);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_set[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot TotalBsmtSF vs 1stFlrSF
sns.set()
cols = ['SalePrice', 'TotalBsmtSF', '1stFlrSF']
sns.pairplot(train_set[cols], size = 3)
plt.show();
sns.set()
predictor_cols = ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF', 'FullBath', 'YearBuilt']
data = pd.concat([train_set['SalePrice'], train_set[predictor_cols]], axis=1)
sns.pairplot(data, size = 3)
plt.show();
# Definition of the CategoricalEncoder class, copied from 
# Aurélien Géron, 2017, Hands on Machine Learning with Scikit-Learn and TensorFLow (see reference)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
# to show categories per column uncomment following line
# train_set.dtypes 

# Alt 1: Single category
var='SaleCondition'
train_cat = train_set[var]
train_cat_encoded, train_categories = train_cat.factorize()

# Alt 2 split of all categorical columns -- not yet working
#train_cat = train_set.select_dtypes(include=['object']).copy()
#train_cat.info()

# reshape input into 2d array -- used directly on input
train_cat_reshaped = train_cat.values.reshape(-1, 1)

# Categorical Encoder
encoder = CategoricalEncoder(encoding="onehot-dense")
train_1hot = encoder.fit_transform(train_cat.values.reshape(-1, 1))
train_1hot
#missing data
total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#dealing with missing data
train_set = train_set.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_set = train_set.drop(df_train.loc[df_train['Electrical'].isnull()].index)
train_set.isnull().sum().max() #just checking that there's no missing data missing...
# Look at the two point on the lower right...
varX = 'GrLivArea'
varY = 'SalePrice'
data = pd.concat([train_set[varX], train_set[varY]], axis=1)
data.plot.scatter(x=varX, y=varY,  alpha=1);
# IDs of Outliers to be removed
train_set.sort_values(by = 'GrLivArea', ascending = False)[:2]
# 1299
# 524

#deleting points
train_set = train_set.drop(train_set[train_set['Id'] == 1299].index)
train_set = train_set.drop(train_set[train_set['Id'] == 524].index)
# Let's use the ssame measure for everything .... standardizing
from sklearn.preprocessing import StandardScaler

Y = train_set['SalePrice']
Y_scaled = StandardScaler().fit_transform(Y[:,np.newaxis]);
low_range = Y_scaled[Y_scaled[:,0].argsort()][:10]
high_range= Y_scaled[Y_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

cat_attribs = ['SaleCondition','MSZoning','MSSubClass']

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(predictor_cols)),
        ('imputer', Imputer(strategy="median")),
        #('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        #("cat_pipeline", cat_pipeline),
    ])

train_prepared = full_pipeline.fit_transform(train_set)
train_prepared
train_prepared.shape
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_prepared, Y)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()
rf_reg.fit(train_prepared, Y)
# let's try the full pipeline on a few training instances
some_data = train_set.iloc[:5]
some_labels = Y.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Predictions:", rf_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error

lin_housing_predictions = lin_reg.predict(train_prepared)
lin_mse = mean_squared_error(Y, lin_housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
rf_housing_predictions = rf_reg.predict(train_prepared)
rf_mse = mean_squared_error(Y, rf_housing_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_rmse
from sklearn.metrics import mean_absolute_error

rf_mae = mean_absolute_error(Y, rf_housing_predictions)
rf_mae
# Read the test data
test_set = pd.read_csv('../input/test.csv')
# predict
test_prepared = full_pipeline.transform(test_set)
final_predictions = rf_reg.predict(test_prepared)
my_submission = pd.DataFrame({'Id': test_set.Id, 'SalePrice': final_predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)