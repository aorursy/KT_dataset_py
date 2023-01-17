
{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 14966,
      "digest": "sha256:328178db27b4735499af91d40fefd588e49ccc87d1b88a5edbb9fed6a89edb4d"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 50382957,
         "digest": "sha256:7e2b2a5af8f65687add6d864d5841067e23bd435eb1a051be6fe1ea2384946b4"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 222909892,
         "digest": "sha256:59c89b5f9b0c6d94c77d4c3a42986d420aaa7575ac65fcd2c3f5968b3726abfc"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 195204532,
         "digest": "sha256:4017849f9f85133e68a4125e9679775f8e46a17dcdb8c2a52bbe72d0198f5e68"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1522,
         "digest": "sha256:c8b29d62979a416da925e526364a332b13f8d5f43804ae98964de2a60d47c17a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 717,
         "digest": "sha256:12004028a6a740ac35e69f489093b860968cc37b9668f65b1e2f61fd4c4ad25c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 247,
         "digest": "sha256:3f09b9a53dfb03fd34e35d43694c2d38656f7431efce0e6647c47efb5f7b3137"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 408,
         "digest": "sha256:03ed58116b0cb733cc552dc89ef5ea122b6c5cf39ec467f6ad671dc0ba35db0c"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 331594702,
         "digest": "sha256:7844554d9ef75bb3f1d224e166ed12561e78add339448c52a8e5679943b229f1"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 112665095,
         "digest": "sha256:c84072c312680c652a6c518b253687b37a1693af7a1752b9700f394c27eeb40a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 426,
         "digest": "sha256:acf8406fd31d7cd1a5d10608ee0da8c774e6b30d148cdca688324bf6244d6293"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 5494,
         "digest": "sha256:abe41b7b0add0c404b69a76a20e2ed75574a584679184b791a3e0812e75a0a34"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1948,
         "digest": "sha256:3281cc773713413d3c1b4f2e92887d2051cb23972abbf796f8d1a3204ae7a980"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2456434655,
         "digest": "sha256:bf03198f391f55ab1b27d866c137f28faa8b8a0ddc3131e4957cb733ca97c678"
      }
   ]
}


from __future__ import print_function, division, absolute_import
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Embedding, Dense, Dropout, Input, Reshape, Concatenate, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import logging
import numpy as np
from scipy import sparse
from sklearn import base
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import pandas as pd
import tensorflow as tf

from .const import NAN_INT, MIN_EMBEDDING


logger = logging.getLogger('Kaggler')
EMBEDDING_SUFFIX = '_emb'
kfold = KFold(n_splits=5, shuffle=True, random_state=42)


class LabelEncoder(base.BaseEstimator):
    """Label Encoder that groups infrequent values into one label.

    Attributes:
        min_obs (int): minimum number of observation to assign a label.
        label_encoders (list of dict): label encoders for columns
        label_maxes (list of int): maximum of labels for columns
    """

    def __init__(self, min_obs=10):
        """Initialize the OneHotEncoder class object.

        Args:
            min_obs (int): minimum number of observation to assign a label.
        """

        self.min_obs = min_obs

    def __repr__(self):
        return ('LabelEncoder(min_obs={})').format(self.min_obs)

    def _get_label_encoder_and_max(self, x):
        """Return a mapping from values and its maximum of a column to integer labels.

        Args:
            x (pandas.Series): a categorical column to encode.

        Returns:
            (tuple):
                - (dict): mapping from values of features to integers
                - (int): maximum label
        """

        # NaN cannot be used as a key for dict. Impute it with a random
        # integer.
        label_count = x.fillna(NAN_INT).value_counts()
        n_uniq = label_count.shape[0]

        label_count = label_count[label_count >= self.min_obs]
        n_uniq_new = label_count.shape[0]

        # If every label appears more than min_obs, new label starts from 0.
        # Otherwise, new label starts from 1 and 0 is used for all old labels
        # that appear less than min_obs.
        offset = 0 if n_uniq == n_uniq_new else 1

        label_encoder = pd.Series(np.arange(n_uniq_new) + offset,
                                  index=label_count.index)
        max_label = label_encoder.max()
        label_encoder = label_encoder.to_dict()

        return label_encoder, max_label

    def _transform_col(self, x, i):
        """Encode one categorical column into labels.

        Args:
            x (pandas.Series): a categorical column to encode
            i (int): column index

        Returns:
            (pandas.Series): a column with labels.
        """
        return x.fillna(NAN_INT).map(self.label_encoders[i]).fillna(0).astype(int)

    def fit(self, X, y=None):
        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        for i, col in enumerate(X.columns):
            self.label_encoders[i], self.label_maxes[i] = \
                self._get_label_encoder_and_max(X[col])

        return self

    def transform(self, X):
        """Encode categorical columns into label encoded columns

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): label encoded columns
        """

        X = X.copy()
        for i, col in enumerate(X.columns):
            X.loc[:, col] = self._transform_col(X[col], i)

        return X

    def fit_transform(self, X, y=None):
        """Encode categorical columns into label encoded columns

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): label encoded columns
        """

        self.label_encoders = [None] * X.shape[1]
        self.label_maxes = [None] * X.shape[1]

        for i, col in enumerate(X.columns):
            self.label_encoders[i], self.label_maxes[i] = \
                self._get_label_encoder_and_max(X[col])

            X.loc[:, col] = (X[col].fillna(NAN_INT)
                             .map(self.label_encoders[i])
                             .fillna(0))

        return X


class OneHotEncoder(base.BaseEstimator):
    """One-Hot-Encoder that groups infrequent values into one dummy variable.

    Attributes:
        min_obs (int): minimum number of observation to create a dummy variable
        label_encoders (list of (dict, int)): label encoders and their maximums
                                              for columns
    """

    def __init__(self, min_obs=10):
        """Initialize the OneHotEncoder class object.

        Args:
            min_obs (int): minimum number of observations required to create
                a dummy variable
            label_encoder (LabelEncoder): LabelEncoder that transofrm
        """

        self.min_obs = min_obs
        self.label_encoder = LabelEncoder(min_obs)

    def __repr__(self):
        return ('OneHotEncoder(min_obs={})').format(self.min_obs)

    def _transform_col(self, x, i):
        """Encode one categorical column into sparse matrix with one-hot-encoding.

        Args:
            x (pandas.Series): a categorical column to encode
            i (int): column index

        Returns:
            (scipy.sparse.coo_matrix): sparse matrix encoding a categorical
                                       variable into dummy variables
        """

        labels = self.label_encoder._transform_col(x, i)
        label_max = self.label_encoder.label_maxes[i]

        # build row and column index for non-zero values of a sparse matrix
        index = np.array(range(len(labels)))
        i = index[labels > 0]
        j = labels[labels > 0] - 1  # column index starts from 0

        if len(i) > 0:
            return sparse.coo_matrix((np.ones_like(i), (i, j)),
                                     shape=(x.shape[0], label_max))
        else:
            # if there is no non-zero value, return no matrix
            return None

    def fit(self, X, y=None):
        self.label_encoder.fit(X)

        return self

    def transform(self, X):
        """Encode categorical columns into sparse matrix with one-hot-encoding.

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (scipy.sparse.coo_matrix): sparse matrix encoding categorical
                                       variables into dummy variables
        """

        for i, col in enumerate(X.columns):
            X_col = self._transform_col(X[col], i)
            if X_col is not None:
                if i == 0:
                    X_new = X_col
                else:
                    X_new = sparse.hstack((X_new, X_col))

            logger.debug('{} --> {} features'.format(
                col, self.label_encoder.label_maxes[i])
            )

        return X_new

    def fit_transform(self, X, y=None):
        """Encode categorical columns into sparse matrix with one-hot-encoding.

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            sparse matrix encoding categorical variables into dummy variables
        """

        self.label_encoder.fit(X)

        return self.transform(X)


class TargetEncoder(base.BaseEstimator):
    """Target Encoder that encode categorical values into average target values.

    Smoothing and min_samples are added based on olivier's kernel at Kaggle:
    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

    , which is based on Daniele Micci-Barreca (2001):
    https://dl.acm.org/citation.cfm?id=507538

    Attributes:
        target_encoders (list of dict): target encoders for columns
    """

    def __init__(self, smoothing=1, min_samples=10, cv=kfold):
        """Initialize the TargetEncoder class object.

        Args:
            smoothing (int): smoothing effect to balance between the categorical average vs global mean
            min_samples (int): minimum samples to take category average into account
            cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object. default=KFold(5, True, 42)
        """
        assert (min_samples >= 0) and (smoothing >= 0), 'min_samples and smoothing should be positive'
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.cv = cv

    def __repr__(self):
        return('TargetEncoder(smoothing={}, min_samples={}, cv={})'.format(self.smoothing, self.min_samples, self.cv))

    def _get_target_encoder(self, x, y):
        """Return a mapping from categories to average target values.

        Args:
            x (pandas.Series): a categorical column to encode.
            y (pandas.Series): the target column

        Returns:
            (dict): mapping from categories to average target values
        """

        assert len(x) == len(y)

        # NaN cannot be used as a key for dict. So replace it with a random
        # integer
        mean_count = pd.DataFrame({y.name: y, x.name: x.fillna(NAN_INT)}).groupby(x.name)[y.name].agg(['mean', 'count'])
        smoothing = 1 / (1 + np.exp(-(mean_count['count'] - self.min_samples) / self.smoothing))

        mean_count[y.name] = self.target_mean * (1 - smoothing) + mean_count['mean'] * smoothing
        return mean_count[y.name].to_dict()

    def fit(self, X, y):
        """Encode categorical columns into average target values.

        Args:
            X (pandas.DataFrame): categorical columns to encode
            y (pandas.Series): the target column

        Returns:
            (pandas.DataFrame): encoded columns
        """
        self.target_encoders = [None] * X.shape[1]
        self.target_mean = y.mean()

        for i, col in enumerate(X.columns):
            if self.cv is None:
                self.target_encoders[i] = self._get_target_encoder(X[col], y)
            else:
                self.target_encoders[i] = []
                for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X[col], y), 1):
                    self.target_encoders[i].append(self._get_target_encoder(X.loc[i_trn, col], y[i_trn]))

        return self

    def transform(self, X):
        """Encode categorical columns into average target values.

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): encoded columns
        """
        for i, col in enumerate(X.columns):
            if self.cv is None:
                X.loc[:, col] = (X[col].fillna(NAN_INT)
                                       .map(self.target_encoders[i])
                                       .fillna(self.target_mean))
            else:
                for i_enc, target_encoder in enumerate(self.target_encoders[i], 1):
                    if i_enc == 1:
                        x = X[col].fillna(NAN_INT).map(target_encoder).fillna(self.target_mean)
                    else:
                        x += X[col].fillna(NAN_INT).map(target_encoder).fillna(self.target_mean)

                X.loc[:, col] = x / i_enc

        return X.astype(float)

    def fit_transform(self, X, y):
        """Encode categorical columns into average target values.

        Args:
            X (pandas.DataFrame): categorical columns to encode
            y (pandas.Series): the target column

        Returns:
            (pandas.DataFrame): encoded columns
        """
        self.target_encoders = [None] * X.shape[1]
        self.target_mean = y.mean()

        for i, col in enumerate(X.columns):
            if self.cv is None:
                self.target_encoders[i] = self._get_target_encoder(X[col], y)

                X.loc[:, col] = (X[col].fillna(NAN_INT)
                                       .map(self.target_encoders[i])
                                       .fillna(self.target_mean))
            else:
                self.target_encoders[i] = []
                for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X[col], y), 1):
                    target_encoder = self._get_target_encoder(X.loc[i_trn, col], y[i_trn])

                    X.loc[i_val, col] = (X.loc[i_val, col].fillna(NAN_INT)
                                                          .map(target_encoder)
                                                          .fillna(y[i_trn].mean()))

                    self.target_encoders[i].append(target_encoder)

        return X.astype(float)


class EmbeddingEncoder(base.BaseEstimator):
    """EmbeddingEncoder encodes categorical features to numerical embedding features.

    Reference: 'Entity embeddings to handle categories' by Abhishek Thakur
    at https://www.kaggle.com/abhishek/entity-embeddings-to-handle-categories
    """

    def __init__(self, cat_cols, num_cols=[], n_emb=[], min_obs=10, n_epoch=100, batch_size=1024, cv=None,
                 random_state=42):
        """Initialize an EmbeddingEncoder class object.

        Args:
            cat_cols (list of str): the names of categorical features to create embeddings for.
            num_cols (list of str): the names of numerical features to train embeddings with.
            n_emb (int or list of int): the numbers of embedding features used for columns.
            min_obs (int): categories observed less than it will be grouped together before training embeddings
            n_epoch (int): the number of epochs to train a neural network with embedding layer
            batch_size (int): the size of mini-batches in model training
            cv (sklearn.model_selection._BaseKFold): sklearn CV object
            random_state (int): random seed.
        """
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        if isinstance(n_emb, int):
            self.n_emb = [n_emb] * len(cat_cols)
        elif isinstance(n_emb, list):
            if not n_emb:
                self.n_emb = [None] * len(cat_cols)
            else:
                assert len(cat_cols) == len(n_emb)
                self.n_emb = n_emb
        else:
            raise ValueError('n_emb should be int or list')

        self.min_obs = min_obs
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.cv = cv
        self.random_state = random_state

        self.lbe = LabelEncoder(min_obs=self.min_obs)

    @staticmethod
    def auc(y, p):
        return tf.py_function(roc_auc_score, (y, p), tf.double)

    @staticmethod
    def _get_model(X, cat_cols, num_cols, n_uniq, n_emb, output_activation):
        inputs = []
        num_inputs = []
        embeddings = []
        for i, col in enumerate(cat_cols):

            if not n_uniq[i]:
                n_uniq[i] = X[col].nunique()
            if not n_emb[i]:
                n_emb[i] = max(MIN_EMBEDDING, 2 * int(np.log2(n_uniq[i])))

            _input = Input(shape=(1,), name=col)
            _embed = Embedding(input_dim=n_uniq[i], output_dim=n_emb[i], name=col + EMBEDDING_SUFFIX)(_input)
            _embed = Dropout(.2)(_embed)
            _embed = Reshape((n_emb[i],))(_embed)

            inputs.append(_input)
            embeddings.append(_embed)

        if num_cols:
            num_inputs = Input(shape=(len(num_cols),), name='num_inputs')
            merged_input = Concatenate(axis=1)(embeddings + [num_inputs])

            inputs = inputs + [num_inputs]
        else:
            merged_input = Concatenate(axis=1)(embeddings)

        x = BatchNormalization()(merged_input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(.5)(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(.5)(x)
        x = BatchNormalization()(x)
        output = Dense(1, activation=output_activation)(x)

        model = Model(inputs=inputs, outputs=output)

        return model, n_emb, n_uniq

    def fit(self, X, y):
        """Train a neural network model with embedding layers.

        Args:
            X (pandas.DataFrame): categorical features to create embeddings for
            y (pandas.Series): a target variable

        Returns:
            A trained EmbeddingEncoder object.
        """
        is_classification = y.nunique() == 2

        X_cat = self.lbe.fit_transform(X[self.cat_cols])
        if is_classification:
            assert np.isin(y, [0, 1]).all(), 'Target values should be 0 or 1 for classification.'
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics = [self.auc]
            monitor = 'val_auc'
            mode = 'max'
        else:
            output_activation = 'linear'
            loss = 'mse'
            metrics = ['mse']
            monitor = 'val_mse'
            mode = 'min'

        n_uniq = [X_cat[col].nunique() for col in self.cat_cols]
        if self.cv:
            self.embs = []
            n_fold = self.cv.get_n_splits(X)
            for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X, y), 1):
                model, self.n_emb, _ = self._get_model(X_cat, self.cat_cols, self.num_cols, n_uniq, self.n_emb,
                                                       output_activation)
                model.compile(optimizer=Adam(lr=0.01), loss=loss, metrics=metrics)

                features_trn = [X_cat[col][i_trn] for col in self.cat_cols]
                features_val = [X_cat[col][i_val] for col in self.cat_cols]
                if self.num_cols:
                    features_trn += [X[self.num_cols].values[i_trn]]
                    features_val += [X[self.num_cols].values[i_val]]

                es = EarlyStopping(monitor=monitor, min_delta=.001, patience=5, verbose=1, mode=mode,
                                   baseline=None, restore_best_weights=True)
                rlr = ReduceLROnPlateau(monitor=monitor, factor=.5, patience=3, min_lr=1e-6, mode=mode)
                model.fit(x=features_trn,
                          y=y[i_trn],
                          validation_data=(features_val, y[i_val]),
                          epochs=self.n_epoch,
                          batch_size=self.batch_size,
                          callbacks=[es, rlr])

                for i_col, col in enumerate(self.cat_cols):
                    emb = model.get_layer(col + EMBEDDING_SUFFIX).get_weights()[0]
                    if i_cv == 1:
                        self.embs.append(emb / n_fold)
                    else:
                        self.embs[i_col] += emb / n_fold

        else:
            model, self.n_emb, _ = self._get_model(X_cat, self.cat_cols, self.num_cols, n_uniq, self.n_emb,
                                                   output_activation)
            model.compile(optimizer=Adam(lr=0.01), loss=loss, metrics=metrics)

            features = [X_cat[col] for col in self.cat_cols]
            if self.num_cols:
                features += [X[self.num_cols].values]

            es = EarlyStopping(monitor=monitor, min_delta=.001, patience=5, verbose=1, mode=mode,
                               baseline=None, restore_best_weights=True)
            rlr = ReduceLROnPlateau(monitor=monitor, factor=.5, patience=3, min_lr=1e-6, mode=mode)
            model.fit(x=features,
                      y=y,
                      epochs=self.n_epoch,
                      validation_split=.2,
                      batch_size=self.batch_size,
                      callbacks=[es, rlr])

            self.embs = []
            for i, col in enumerate(self.cat_cols):
                self.embs.append(model.get_layer(col + EMBEDDING_SUFFIX).get_weights()[0])
                logger.debug('{}: {}'.format(col, self.embs[i].shape))

    def transform(self, X):
        X_cat = self.lbe.transform(X[self.cat_cols])
        X_emb = []

        for i, col in enumerate(self.cat_cols):
            X_emb.append(self.embs[i][X_cat[col].values])

        return np.hstack(X_emb)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class FrequencyEncoder(base.BaseEstimator):
    """Frequency Encoder that encode categorical values by counting frequencies.

    Attributes:
        frequency_encoders (list of dict): frequency encoders for columns
    """

    def __init__(self, cv=None):
        """Initialize the FrequencyEncoder class object.
        Args:
            cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object
        """
        self.cv = cv

    def __repr__(self):
        return('FrequencyEncoder(cv={})'.format(self.cv))

    def _get_frequency_encoder(self, x):
        """Return a mapping from categories to frequency.

        Args:
            x (pandas.Series): a categorical column to encode.

        Returns:
            (dict): mapping from categories to frequency
        """

        # NaN cannot be used as a key for dict. So replace it with a random
        # integer
        df = pd.DataFrame({x.name: x.fillna('NaN')})
        df[x.name + '_freq'] = df[x.name].map(df[x.name].value_counts())
        return df.groupby(x.name)[x.name + '_freq'].size().to_dict()

    def fit(self, X, y=None):
        """Encode categorical columns into frequency.

        Args:
            X (pandas.DataFrame): categorical columns to encode
            y (pandas.Series, optional): the target column

        Returns:
            (pandas.DataFrame): encoded columns
        """
        self.frequency_encoders = [None] * X.shape[1]

        for i, col in enumerate(X.columns):
            if self.cv is None:
                self.frequency_encoders[i] = self._get_frequency_encoder(X[col])
            else:
                self.frequency_encoders[i] = []
                for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X[col]), 1):
                    self.frequency_encoders[i].append(self._get_frequency_encoder(X.loc[i_trn, col]))

        return self

    def transform(self, X):
        """Encode categorical columns into feature frequency counts.

        Args:
            X (pandas.DataFrame): categorical columns to encode

        Returns:
            (pandas.DataFrame): encoded columns
        """
        for i, col in enumerate(X.columns):
            if self.cv is None:
                X.loc[:, col] = X[col].fillna('NaN').map(self.frequency_encoders[i]).fillna(0)
            else:
                for i_enc, frequency_encoder in enumerate(self.frequency_encoders[i], 1):
                    if i_enc == 1:
                        x = X[col].fillna('NaN').map(frequency_encoder).fillna(0)
                    else:
                        x += X[col].fillna('NaN').map(frequency_encoder).fillna(0)

                X.loc[:, col] = x / i_enc

        return X

    def fit_transform(self, X, y=None):
        """Encode categorical columns into feature frequency counts.

        Args:
            X (pandas.DataFrame): categorical columns to encode
            y (pandas.Series, optional): the target column
        """
        self.frequency_encoders = [None] * X.shape[1]

        for i, col in enumerate(X.columns):
            if self.cv is None:
                self.frequency_encoders[i] = self._get_frequency_encoder(X[col])

                X.loc[:, col] = X[col].fillna('NaN').map(self.frequency_encoders[i]).fillna(0)
            else:
                self.frequency_encoders[i] = []
                for i_cv, (i_trn, i_val) in enumerate(self.cv.split(X[col]), 1):
                    frequency_encoder = self._get_frequency_encoder(X.loc[i_trn, col])

                    X.loc[i_val, col] = X.loc[i_val, col].fillna('NaN').map(frequency_encoder).fillna(0)
                    self.frequency_encoders[i].append(frequency_encoder)

        return X

!pip install pysolr
!pip install transformers
!pip install pandas
from pysolr import Solr
import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForQuestionAnswering
import shutil
import torch
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
stop_words = set(stopwords.words('english'))
title_dict = {}
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("solr_server")
solr = Solr(secret_value_0)
def preprocess(outer_folder, folder_name):
    fname = "../input/CORD-19-research-challenge/"+outer_folder+folder_name + "/"
    contents = os.listdir(fname)
    address = "processed/" + folder_name + "/"
    new_files = []
    result = solr.search("path:"+ folder_name + " path: " + folder_name + "/", rows=40000, **{"fl":"id"})
    dct = set()
    for i in result:
        dct.add(address + i["id"])
    new_docs = set()
    for i in contents:
        d = json.load(open(fname+i))
        id = d['paper_id']
        path = folder_name
        title=d['metadata']['title']
        abstract = ""
        if 'abstract' in d:
            abst = d['abstract']
            for j in abst:
                abstract = abstract + j["text"] + "\n"
        body = ""
        for j in d['body_text']:
            body = body + j["text"] + "\n"
        data = {}
        title_dict[id]=title
        data['title']=title
        data['abstract']=abstract
        data['id']=id
        data['body']=body
        data['path']=path
        with open(address+i,'w') as out:
            json.dump(data,out)
        new_docs.add(address + id)
        #new_files.append(address+i)
    to_add = new_docs.difference(dct)
    to_remove = dct.difference(new_docs)
    return [*to_add,], [*to_remove,]
# Create directory structure for processed files
def modify_directory_structure():
    if not os.path.exists("processed"):
        os.mkdir("processed")

    if not os.path.exists("processed/biorxiv_medrxiv"):
        os.mkdir("processed/biorxiv_medrxiv")
    if not os.path.exists("processed/biorxiv_medrxiv/pdf_json"):
        os.mkdir("processed/biorxiv_medrxiv/pdf_json")

    if not os.path.exists("processed/comm_use_subset"):
        os.mkdir("processed/comm_use_subset")
    if not os.path.exists("processed/comm_use_subset/pdf_json"):
        os.mkdir("processed/comm_use_subset/pdf_json")
    if not os.path.exists("processed/comm_use_subset/pmc_json"):
        os.mkdir("processed/comm_use_subset/pmc_json")

    if not os.path.exists("processed/noncomm_use_subset"):
        os.mkdir("processed/noncomm_use_subset")
    if not os.path.exists("processed/noncomm_use_subset/pdf_json"):
        os.mkdir("processed/noncomm_use_subset/pdf_json")
    if not os.path.exists("processed/noncomm_use_subset/pmc_json"):
        os.mkdir("processed/noncomm_use_subset/pmc_json")

    if not os.path.exists("processed/custom_license"):
        os.mkdir("processed/custom_license")
    if not os.path.exists("processed/custom_license/pdf_json"):
        os.mkdir("processed/custom_license/pdf_json")
    if not os.path.exists("processed/custom_license/pmc_json"):
        os.mkdir("processed/custom_license/pmc_json")

#Updating index in batches of 500 to prevent http request timeout
def update_index(new_files):
    j = 0
    while j < len(new_files): 
        x = []
        for i in new_files[j:j+500]:
            if 'PMC' in i:
                r = json.load(open(i+'.xml.json','r'))
            else:
                r = json.load(open(i+'.json','r'))
            x.append(r)
        j = j + 500
        solr.add(x)
        
# Remove those documents from the index which have been removed from dataset
def clean_index(removable_files):
    j = 0
    while j < len(removable_files): 
        x = ""
        for i in removable_files[j:j+10]:
            x = x + "id: " + i.split('/')[-1] + " "
        j = j + 10 #Update is done in batches of size 10 due to URL size restriction
        solr.delete(q=x)
           
def handle_changes():
    modify_directory_structure()
    new_files = []
    removable_files = []
    n1, r1 = preprocess('biorxiv_medrxiv/','biorxiv_medrxiv/pdf_json')
    new_files = new_files + n1
    removable_files = removable_files + r1
    print(len(new_files))
    print(new_files[:10])
    
    n1, r1 = preprocess('comm_use_subset/','comm_use_subset/pdf_json')
    new_files = new_files + n1
    removable_files = removable_files + r1
    
    n1, r1 = preprocess('comm_use_subset/','comm_use_subset/pmc_json')
    new_files = new_files + n1
    removable_files = removable_files + r1
    print(len(new_files))
    
    
    n1, r1 = preprocess('noncomm_use_subset/','noncomm_use_subset/pdf_json')
    new_files = new_files + n1
    removable_files = removable_files + r1
    
    n1, r1 = preprocess('noncomm_use_subset/','noncomm_use_subset/pmc_json')
    new_files = new_files + n1
    removable_files = removable_files + r1
    print(len(new_files))
    
    
    n1, r1 = preprocess('custom_license/','custom_license/pdf_json')
    new_files = new_files + n1
    removable_files = removable_files + r1
    
    n1, r1 = preprocess('custom_license/','custom_license/pmc_json')
    new_files = new_files + n1
    removable_files = removable_files + r1
    print(len(new_files))
    
    print(str(len(new_files)) + " new files were found")
    print("Modifying search index... This might take some time")
    print(new_files[:10])
    update_index(new_files)
    clean_index(removable_files)
    print("done updating")
handle_changes()
def scraper():
    terms = []
    for i in range(26):
        html = "https://www.medicinenet.com/medications/alpha_" + chr(97+i) + '.htm'
        r = requests.get(html)
        soup = BeautifulSoup(r.content, 'lxml')
        c = soup.find('div', attrs = {'id':'AZ_container'})
        z = c.findAll('li')
        z = [i.text for i in z]
        terms = terms + z
    return terms
!git clone https://github.com/glutanimate/wordlist-medicalterms-en
wordnet_lemmatizer = WordNetLemmatizer()

words = open('wordlist-medicalterms-en/wordlist.txt').readlines()
words = [wordnet_lemmatizer.lemmatize(i.strip()) for i in words]
words = words + scraper() + ['COVID-19']
def query_maker(query_nlp, medical_words):
    """
        Formulates the query to send to Solr server for retrieving relevant documents.
    """
    
    query_words = query_nlp.strip().split()
    query = "(body:"
    essentials = []
    
    for i in query_words:
        if i in stop_words:
            continue
        if i[0]=='+':
            essentials.append(i)
        elif wordnet_lemmatizer.lemmatize(i) in medical_words or i in medical_words or i.lower() in medical_words:
            essentials.append(i)
        else:
            query = query + " " + i
    query = query + ")"
    if query=="(body:)":
        query=""
    for i in essentials:
        if i[0]=='+':
            query = query + " body:" + i[1:] + "^4"
        else:
            query = query + " " + "+body: " + i
    print(query)
    essesntials = [i for i in essentials if i[0]!='+']
    return query, essentials
def get_relevant_docs(query, max_docs=10, show_hits=False):
    """
        Return contents of the relevant documents and score corresponding to a query
    """
    result = solr.search(query, rows=max_docs, **{"fl":"*,score"}) #rows is length of result returned by server
    doc_names = []
    for i in result.docs:
        if i["path"][-1]=='/':
            doc_names.append((str(i["path"])+str(i["id"]),i["score"]))
        else:
            doc_names.append((str(i["path"])+"/"+str(i["id"]),i["score"]))
    docs = []
    if show_hits:
        print(result.hits)
    for i in doc_names:
        if "pmc" in i[0]:
            dname = i[0] + ".xml.json"
        else:
            dname = i[0] + ".json"
        docs.append((json.load(open('processed/'+dname)), i[1]))
    return docs
def get_docs(query):
    q, keywords = query_maker(query, words)
    kw = [[i]+get_synonyms(i) for i in keywords]
    result = get_relevant_docs(q,100,True)
    p_ids = [i[0]["id"] for i in result]
    titles = [i[0]["title"] for i in result]
    abstracts = [i[0]["abstract"] for i in result]
    body_lens = [len(i[0]["body"].split()) for i in result]
    scores = [i[1] for i in result]
    data = {"id":p_ids, "score":scores,"title":titles,"length of body":body_lens, "abstract":abstracts}
    df = pd.DataFrame (data)
    return result, kw, df
# tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")

# model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")



tokenizer = AutoTokenizer.from_pretrained("gdario/biobert_bioasq")

model = AutoModelForQuestionAnswering.from_pretrained("gdario/biobert_bioasq")

# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BertForQuestionAnswering, BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer.add_tokens(['corona'])
tokenizer.encode('assay')
model.resize_token_embeddings(len(tokenizer))



model = model.cuda()
import nltk 
from nltk.corpus import wordnet 


def get_synonyms(word):
    synonyms = [] 
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonyms.append(l.name())
    return synonyms
def has_keywords(context, keywords):
    c2 = context.lower()
    k = 0
    for i in keywords:
        #print(i)
        b = False
        for j in i:
            if j.lower() in c2:
                b = True
                #k = k + 1
            else:
                continue
        if b:
            k = k + 1
    return k > (2*len(keywords))/3
from spacy.lang.en import English # updated
from nltk.corpus import wordnet
import re

nlp = English()
nlp.max_length = 5101418
nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated
pattern = re.compile("^\[[0-9]*\]$")

def answer(context, question):
    #context = context.lower()
    context = context.replace('COVID-19', 'coronavirus disease/COVID-19')
    context = context.replace('MERS-CoV', 'MERS-coronavirus')
    context = context.replace('SARS-CoV-2', 'coronavirus')
    y = tokenizer.encode(question, context, max_length=256)
    sep_index = y.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    #print(num_seg_a)
    num_seg_b = len(y) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    #y = y.cuda()
    with torch.no_grad():
        start_scores, end_scores = model(torch.tensor([y]).cuda(), token_type_ids=torch.tensor([segment_ids]).cuda())
    #start_scores, end_scores = model(torch.tensor([y]), token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    if answer_start<num_seg_a:
        return ()
    else:
        
        fstops = [ i for i in range(len(y)) if y[i] == 119]
        fstops = [num_seg_a] + fstops + [len(y)-1]
        start = 0
        for i in fstops:
            if i + 1 <= answer_start:
                start = i + 1
        for i in fstops:
            if i >= answer_end:
                end = i
                break
        start = max(start, num_seg_a)
        #print("Here")
        return [start_scores[0][answer_start] + end_scores[0][answer_end],tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y[answer_start:answer_end+1])).strip().replace('corona virus', 'coronavirus'), 
                tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(y[start:end+1])).strip().replace('corona virus', 'coronavirus')]           
    
def non_redundant(out, answers):
    for i in answers:
        if out[1]==i[1] or out[2]==i[2]:
            return False
    return True
    
def get_answer(doc, question, keywords):
    docx = nlp(doc["body"].replace('■', ' '))
    sentences = [sent.string.strip() for sent in docx.sents]
    i = 0
    context = ""
    answers = []
    prev_i = 0
    while i < len(sentences):  #Use sliding window to search for answers in the document
        if len(context.split())>=100:
            if has_keywords(context, keywords): #Search for answers only if the context has all the identified keywords
                out = answer(context, question)
                if len(out) > 0 and non_redundant(out, answers) and out[1] not in question:
                    if not pattern.match(out[1]):
                        answers.append(out)
                        prev_i += 2 #Slide the window further if an answer is found
                        out = []
            context = ""
            i = prev_i + 1
            prev_i = i
        else:
            context = context + " " + sentences[i]
            i = i + 1
            if i==len(sentences) and has_keywords(context, keywords):
                out = answer(context, question)
                if len(out) > 0 and non_redundant(out, answers) and out[1] not in question:
                    
                    if not pattern.match(out[1]):
                        answers.append(out)
                        out = []
    answers = answers[1:]
    if len(answers) > 0:
        answers.sort()
        answers.reverse()
        answers = answers[:2]
        answers = [[doc["id"], i] for i in answers]
        return answers
    else:
        return None
retrieval_queries = [ ['Clinical trials to investigate effect viral inhibitors like naproxen on coronavirus patients', 
                       'Clinical trials to investigate effect viral inhibitors like clarithromycin on coronavirus patients', 
                       'Clinical trials to investigate effect viral inhibitors like minocyclineth on coronavirus patients'],
                     ['evaluate complication of antibody dependent enhancement in vaccine recipients'],
                     ['animal models for vaccine evaluation and efficacy in trials and prediction for human vaccine'],
                     ['capabilities to discover a therapeutic for COVID-19',
                      'clinical effectiveness studies to discover therapeutics for COVID-19',
                      'clinical effectiveness studies to include antiviral agents for COVID-19'],
                     ['accelerate production of therapeutics',
                      'timely and equitable distribution of therapeutics to people'],
                     ['efforts targeted at universal coronavirus vaccine'],
                     ['animal models for vaccine evaluation and efficacy in trials and prediction for human vaccine and stadardize challenge studies'],
                     ['develop prophylaxis for COVID-19',
                      'clinical studies and prioritize in healthcare workers for COVID-19'],
                     ['evaluate or assess risk for enhanced disease after vaccination'],
                     ['assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models with therapeutics']
                    ]

questions = [ ['What are the clinical trials to investigate effect of naproxen on coronavirus patients?',
               'What are the clinical trials to investigate effect of clarithromycin on coronavirus patients?',
               'What are the clinical trials to investigate effect of monocyclineth on coronavirus patients?'
              ],
             ['how to evaluate complications of antibody dependent enhancement in vaccine recipients?'],
             ['What are the best animal models and their predictive value for human vaccine for coronavirus?'],
             ['What are the capabilities to discover a therapeutic for COVID-19?',
             'Which are the clinical effectiveness studies to discover therapeutics for COVID-19?',
             'Which are the clinical effectiveness studies to include antiviral agents for COVID-19?'],
             ['How to increase production capacity of therapeutics?',
             'How were therapeutics distributed to the population'],
             ['What are the efforts targeted at universal coronavirus vaccine?'],
             ['What are the efforts to develop animal models and standardize challenge studies?'],
             ['What are the efforts to develop prophylaxis for COVID-19?',
             'What are the efforts to develop clinical studies and prioritize in healthcare workers for COVID-19?'],
             ['What are the approaches to evaluate risk for enhanced disease after vaccination?'],
             ['What are the assays procedures to evaluate vaccine immune response and process development for vaccines with suitable animal models?']            
            ]
answer_set = []
for i in zip(retrieval_queries, questions):
    answers = []
    docs = []
    for j in zip(i[0],i[1]):
        result, keywords, df = get_docs(j[0])
        incl = 0
        for k in range(min(100, len(result))):
            x = get_answer(result[k][0], j[1], keywords)
            torch.cuda.empty_cache()
            if x is not None:
                toadd = []
                for ax in x:
                    adding = True
                    for ans in answers:     
                        if ans[1][1] == ax[1][1] or ans[1][2] == ax[1][2]:
                            adding=False
                    if adding:
                        toadd.append(ax)
                    
                if len(toadd) > 0:
                    incl = incl + 1
                    answers = answers + toadd
                if incl == 7:
                    break
    answer_set.append(answers)
len(os.listdir('processed/custom_license/pdf_json'))
pd.set_option('display.max_colwidth', -1)
def display_answer(result):
    p_ids = [i[0] for i in result]
    titles = [title_dict[i[0]] for i in result]
    answers = [i[1][2] for i in result]
    data = {"id":p_ids,"title":titles, "Answer":answers}
    df = pd.DataFrame (data)
    return df
display_answer(answer_set[0])
display_answer(answer_set[1])
display_answer(answer_set[2])
display_answer(answer_set[3])
display_answer(answer_set[4])
display_answer(answer_set[5])
display_answer(answer_set[6])
display_answer(answer_set[7])
display_answer(answer_set[8])
display_answer(answer_set[9])
shutil.rmtree('processed')
task_questions = [
    'Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.',
    'Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.',
    'Exploration of use of best animal models and their predictive value for a human vaccine.',
    'Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.',
    'Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.',
    'Efforts targeted at a universal coronavirus vaccine.',
    'Efforts to develop animal models and standardize challenge studies',
    'Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers',
    'Approaches to evaluate risk for enhanced disease after vaccination',
    'Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]'
]
def prepare_submission(answer_set):
    spids = []
    stitles = []
    sanswers = []
    squestions = []
    qnum = 0
    for result in answer_set:
        p_ids = [i[0] for i in result]
        titles = [title_dict[i[0]] for i in result]
        answers = [i[1][2] for i in result]
        questions = [task_questions[qnum] for i in result]
        spids = spids + p_ids
        stitles = stitles + titles
        sanswers = sanswers + answers
        squestions = squestions + questions
        qnum = qnum + 1
    data = {"question":squestions,"id":spids,"title":stitles, "Answer":sanswers}
    df = pd.DataFrame (data)
    df.to_csv('submission.csv',index=False)
prepare_submission(answer_set)
