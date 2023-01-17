# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
all_data = pd.read_csv("../input/train.csv")

all_data = all_data.drop('Name', axis = 1)

all_data = all_data.drop('Ticket', axis = 1)

all_data = all_data.drop('PassengerId', axis = 1)

all_data = all_data.drop('Cabin', axis = 1)



freq_port = all_data['Embarked'].dropna().mode()[0]

all_data["Embarked"].fillna(freq_port, inplace = True)



# Add new features FamilySize and IsAlone.

all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

all_data['IsAlone'] = 0

all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1



all_data.info()
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
# Definition of the CategoricalEncoder class, copied from PR #9151.

# Just run this cell, or copy it to your code, do not try to understand it (yet).

import numpy as np

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
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

all_data_sex = all_data['Sex']

sex_encoded = encoder.fit_transform(all_data_sex)

encoder.classes_
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

all_data_emb = all_data['Embarked']

emb_encoded = encoder.fit_transform(all_data_emb)

encoder.classes_
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.preprocessing import StandardScaler, Imputer



num_attributes = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']

cat_attributes = ['Sex', 'Embarked']



num_pipe = Pipeline([

    ('selector', DataFrameSelector(num_attributes)),

    ('imputer', Imputer(strategy="median")),

    ('std_scaler', StandardScaler())

])



cat_pipe = Pipeline([

    ('selector', DataFrameSelector(cat_attributes)),

    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense", categories=[['female', 'male'], ['C', 'Q', 'S']], handle_unknown='ignore'))

])



full_pipeline = FeatureUnion(transformer_list=[

    ("num_pip", num_pipe),

    ("cat_pip", cat_pipe),

])



prepared_data = full_pipeline.fit_transform(all_data)
labels = all_data['Survived']
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import StratifiedKFold



skfolds = StratifiedKFold(n_splits=3, random_state=42)



for train_index, test_index in skfolds.split(prepared_data, labels):

    X_train_folds = prepared_data[train_index]

    y_train_folds = (labels[train_index])

    X_test_fold = prepared_data[test_index]

    y_test_fold = (labels[test_index])

    sgd_clr = SGDClassifier(random_state=42,tol=0.1)

    sgd_clr.fit(X_train_folds, y_train_folds)

    y_pred = sgd_clr.predict(X_test_fold)

    n_correct = sum(y_pred == y_test_fold)

    print(n_correct / len(y_pred))
test_data = pd.read_csv("../input/test.csv")

test_data = test_data.drop('Name', axis = 1)

test_data = test_data.drop('Ticket', axis = 1)

test_data = test_data.drop('PassengerId', axis = 1)

test_data = test_data.drop('Cabin', axis = 1)



test_data["Embarked"].fillna(freq_port, inplace = True)



# Add new features FamilySize and IsAlone.

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

test_data['IsAlone'] = 0

test_data.loc[test_data['FamilySize'] == 1, 'IsAlone'] = 1



prepared_test_data = full_pipeline.fit_transform(test_data)



test_predicts = sgd_clr.predict(prepared_test_data)
test_predicts
test_data = pd.read_csv("../input/test.csv")

test_data['PassengerId']
result = np.c_[test_data['PassengerId'], test_predicts]
np.savetxt('./predictions.csv', result, fmt='%i,%i', header="PassengerId,Survived", comments='')
pds = pd.read_csv('predictions.csv')
pds.info()