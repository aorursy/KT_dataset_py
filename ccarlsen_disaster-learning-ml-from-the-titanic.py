# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
%matplotlib inline

# preprocessing data
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion

# machine learning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier, LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC

# remove warnings to keep notebook clean
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape, test.shape
full_data = pd.concat([train, test])
train.head()
train.dtypes
train.describe()
train.isnull().sum()
train['Cabin'].isnull().sum() / len(train['Cabin'])
train['Embarked'].value_counts()
full_data.Embarked.replace(np.NaN, "S", inplace=True)
plt.subplots(figsize=(10,6))
sns.countplot(y=train['Survived'],order=train['Survived'].value_counts().index)
plt.show()

print("Survival rate: ", train['Survived'].mean()*100, "percent")
train_int = train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_int.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
survived = train[train['Survived'] == 1]
died = train[train['Survived'] == 0]
plt.figure(figsize=[15,15])
current_palette = sns.set_palette(["#e74c3c","#3498db"])
sns.set_style({'axes.grid': False})
plt.subplot(331)
sns.countplot(train['Pclass'], hue=train['Survived'])
plt.subplot(332)
sns.countplot(train['Sex'], hue=train['Survived'])
plt.subplot(333)
sns.distplot(died['Age'].dropna(), bins=50, kde_kws={'shade' : True})
sns.distplot(survived['Age'].dropna(), bins=50,kde_kws={'shade' : True})
young_survived = survived[survived['Age'] < 15]
young_died = died[died['Age'] < 15]
sns.set_style({'axes.grid': False})
plt.subplots(figsize=(10,6))
sns.distplot(young_died['Age'].dropna(), bins=15, kde_kws={'shade' : True})
sns.distplot(young_survived['Age'].dropna(), bins=15,kde_kws={'shade' : True})
train['Family'] = train['SibSp'] + train['Parch']
print(train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean())
sns.countplot(train['Family'], hue=train['Survived'])
full_data['Child'] = full_data['Age'] <= 6
full_data['Child'] = full_data['Child'].astype(int)
full_data['Family'] = full_data['SibSp'] + full_data['Parch']
full_data['Alone'] = full_data['Family'] == 0
full_data['Alone'] = full_data['Alone'].astype(int)
full_data['Small_family'] = (full_data['Family'] > 0) & (full_data['Family'] < 4)
full_data['Small_family'] = full_data['Small_family'].astype(int)
full_data['Large_family'] = full_data['Family'] > 4
full_data['Large_family'] = full_data['Large_family'].astype(int)
dropcols = ['PassengerId', 'Name', 'Ticket', 'Cabin']

full_data_clean = full_data.drop(dropcols, axis=1)
train_labels = train['Survived']
full_data_clean.head()
# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, do not try to understand it (yet).

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
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
num_attribs = ['Age', 'Child', 'Fare', 'Alone', 'Small_family']
cat_attribs = ['Pclass', 'Sex', 'Embarked']
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler())

    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
train_prepared = full_pipeline.fit_transform(full_data[:891])
test_prepared = full_pipeline.transform(full_data[891:])
train_prepared.shape, test_prepared.shape
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=25)

for train_index, test_index in split.split(train_prepared, train_labels):
    X_train = train_prepared[train_index]
    X_test = train_prepared[test_index]
    y_train = train_labels.iloc[train_index]
    y_test = train_labels.iloc[test_index]
names = ['PassiveAgressive', 'SGD', 'LogR', 'NearestCent',
        'MLP', 'Decision Tree', 'Extra Tree', 'Ada Boost',
        'Bagging', 'Random Forest', 'Gaussian Process', 'SVC']

classifiers = [PassiveAggressiveClassifier(),
               SGDClassifier(),
               LogisticRegression(),
               #BernoulliNB(),
               #MultinomialNB(), 
               #KNeighborsClassifier(), 
               #RadiusNeighborsClassifier(), 
               NearestCentroid(), 
               MLPClassifier(),
               DecisionTreeClassifier(),
               ExtraTreeClassifier(),
               AdaBoostClassifier(),
               BaggingClassifier(),
               RandomForestClassifier(),
               GaussianProcessClassifier(),
               SVC()
              ]
results = []

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    results.append([name, scores.mean(), scores.std()])
pd.DataFrame(results, columns=['Name', 'Mean', 'Std']).sort_values(['Mean'])
param_grid = {
    'C': [0.01,0.1,1,10,100,100],
    'kernel':['linear', 'rbf', 'sigmoid'],
    'gamma':[0.001,0.01,0.1,1]
 }
random_search = RandomizedSearchCV(SVC(), param_grid, cv=10, n_iter=72)
random_search.fit(X_train, y_train)
random_search.best_params_, random_search.best_score_
param_logr = {
    'penalty':['l1', 'l2'],
    'C':[0.001,0.01,0.1,1,10,100]
}
random_search = RandomizedSearchCV(LogisticRegression(), param_logr, cv=10, n_iter=12)
random_search.fit(X_train, y_train)
random_search.best_params_, random_search.best_score_
param_ada = {
    'n_estimators':sp_randint(2,100),
    'learning_rate':[0.01,0.1,1,10]
}
random_search = RandomizedSearchCV(AdaBoostClassifier(), param_ada, cv=10, n_iter=100)
random_search.fit(X_train, y_train)
random_search.best_params_, random_search.best_score_
param_rf = {
    'n_estimators':sp_randint(2,50),
    'max_depth':sp_randint(2,50),
}
random_search = RandomizedSearchCV(RandomForestClassifier(), param_rf, cv=10, n_iter=100)
random_search.fit(X_train, y_train)
random_search.best_params_, random_search.best_score_
param_bag = {
    'n_estimators': sp_randint(2,50),
}
random_search = RandomizedSearchCV(BaggingClassifier(), param_bag, cv=10, n_iter=50)
random_search.fit(X_train, y_train)
random_search.best_params_, random_search.best_score_
clf_svc = SVC(probability=True, C=1, gamma=0.1, kernel='rbf')
clf_logr = LogisticRegression(C=0.1, penalty='l2')
clf_ada = AdaBoostClassifier(learning_rate=1, n_estimators=89)
clf_rf = RandomForestClassifier(max_depth=11, n_estimators=36)
clf_bag = BaggingClassifier(n_estimators=14)
clf_mlp = MLPClassifier()
clf_vote = VotingClassifier(
    estimators=[
        ('Bagging', clf_bag),
        ('Random Forest', clf_rf),
        ('MLP', clf_mlp),
        ('SVC', clf_svc),
        ('Logr', clf_logr),
        ('AdaBoost', clf_ada)
    ],
    weights=[2,2,1,2,1,1],
    voting='soft')

clf_vote.fit(X_train,y_train)
scores = cross_val_score(clf_vote, X_test, y_test, cv=10, scoring='accuracy')
print("Voting: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
stack_pred = clf_vote.predict(test_prepared)
submit = pd.DataFrame({'PassengerId' : test.loc[:,'PassengerId'],
                       'Survived': stack_pred.T})
submit.to_csv("submit.csv", index=False)
