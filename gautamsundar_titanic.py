# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
train_data.head()
train_data.info()
train_data.describe()
train_data["Survived"].value_counts()
train_data["Pclass"].value_counts()
train_data["Sex"].value_counts()
train_data["Embarked"].value_counts()
plt.hist(train_data["Age"].dropna(), bins=np.arange(0, 85, 5))
plt.xlabel("Passenger age")
# sample a few names to get a sense of the structure 
train_data["Fsize"] = train_data["SibSp"] + train_data["Parch"]
survived = train_data[train_data["Survived"] == 1].groupby("Fsize").size()
survived
perished = train_data[train_data["Survived"] == 0].groupby("Fsize").size()
perished
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
width=0.4
ax.bar(survived.index - (width)/2, survived.values, width=width, color='#68C3A3', align='center', label="Survived")
ax.bar(perished.index + (width)/2, perished.values, width=width, color='#D24D57', align='center', label="Perished")
ax.set_ylabel("Count")
ax.set_xticks(range(0, 11))
ax.set_xlabel("Family Size")
plt.legend(loc=1, prop={'size': 20})
# discretize family size
train_data["Fcat"] = pd.cut(train_data["Fsize"], bins=[0, 1, 4, 10], labels=["Singleton", "Small", "Large"], include_lowest=True)
# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, no need to try to
# understand every line.

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
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names]
class NumFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X["Fsize"] = X["SibSp"] + X["Parch"]
        
#         X["Fcat"] = pd.cut(X["Fsize"], 
#                            bins=[0, 1, 4, 20], 
#                            labels=["Singleton", "Small", "Large"], 
#                            include_lowest=True)
                       
        return X
class CatFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X["Title"] = X["Name"].str.extract(r'\w+,\s+([\w\s]*)\.\s+.*', expand=True)
        X.loc[(X["Title"] == "Mlle") | (X["Title"] == "Ms"), "Title"] = "Miss"
        X.loc[X["Title"] == "Mme", "Title"] = "Mrs"
        rare_titles = ["Dr", "Rev", "Major", "Col", "Sir","Capt", "the Countess", 
                       "Jonkheer", "Don", "Dona", "Lady"]
        X.loc[X["Title"].isin(rare_titles), "Title"] = "Rare" 
                
        return X.drop("Name", axis=1)
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent = pd.Series([X[col].value_counts().index[0] for col in X], 
                                       index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent)
# impute missing values using median
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "Fare", "SibSp", "Parch"])),
    ("num_feature_engineering", NumFeatureEngineering()),
    ("imputer", Imputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])
cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Name", "Pclass", "Sex", "Embarked"])),
    ("cat_feature_engineering", CatFeatureEngineering()),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", CategoricalEncoder(encoding='onehot-dense')),
    ("std_scaler", StandardScaler()),
])
from sklearn.pipeline import FeatureUnion
pre_process_pipeline = FeatureUnion([
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    
])
X_train = pre_process_pipeline.fit_transform(train_data.copy())
X_train
X_train.shape
y_train = train_data["Survived"]
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier


sgd_clf = SGDClassifier(random_state=42)
sgd_scores = cross_val_score(sgd_clf, X_train, y_train, cv=5, scoring="accuracy")
sgd_scores.mean()
def display_CV_scores(scores):
    print("CV scores")
    print("mean:", scores.mean())
    print("standard deviation", scores.std())
from sklearn.svm import SVC

svc = SVC()
svc_scores = cross_val_score(svc, X_train, y_train, cv=10)
display_CV_scores(svc_scores)
from sklearn.svm import LinearSVC

linsvc = LinearSVC()
linsvc_scores = cross_val_score(linsvc, X_train, y_train, cv=10)
display_CV_scores(linsvc_scores)
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
display_CV_scores(forest_scores)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=42)
logreg_scores = cross_val_score(logreg, X_train, y_train, cv=10)
display_CV_scores(logreg_scores)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc_scores = cross_val_score(gbc, X_train, y_train, cv=10)
display_CV_scores(gbc_scores)
from sklearn.model_selection import GridSearchCV

svc_param_grid = [
    {'C': [1, 10, 100], 
     'kernel': ['linear']},
     {'C': [0.1, 10, 10000], 
      'kernel': ['rbf'], 
      'gamma': [0.1, 0.3, 1.0, 3.0]},
]

svc = SVC()

grid_search = GridSearchCV(svc, svc_param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
svc_best = grid_search.best_estimator_
print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
# forest_param_grid = [
#    {'n_estimators': [10, 30, 90], 'max_features': [5, 6, 7]},
#    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [4, 6, 7]},
# ]

# forest = RandomForestClassifier()

# grid_search = GridSearchCV(forest, forest_param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# forest_best = grid_search.best_estimator_
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(mean_score, params)
# gbc_param_grid = [
#    {'n_estimators': [120, 180], 
#     'max_features': [5, 6, 7], 
#     'learning_rate': [0.3, 1],
#    }
   
# ]

# gbc = GradientBoostingClassifier()

# grid_search = GridSearchCV(gbc, gbc_param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# gbc_best = grid_search.best_estimator_
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(mean_score, params)
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint

# svc_distribs = {
#     'C': randint(low=1, high=30), 
#     'kernel': ['linear', 'rbf', 'poly'], 
#     'degree': randint(1, 10),
# }

# svc = SVC()
# rnd_search = RandomizedSearchCV(svc, svc_distribs, n_iter=10, cv=5, scoring="accuracy", random_state=42)
# rnd_search.fit(X_train,y_train)

# svc_best = rnd_search.best_estimator_
# cvres = rnd_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(mean_score, params)
test_data = pd.read_csv("../input/test.csv")
X_test = pre_process_pipeline.transform(test_data)
y_test = svc_best.predict(X_test)
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], 
                                                     "Survived": y_test})
submission.to_csv("titanic_submission_svc.csv", index=False)