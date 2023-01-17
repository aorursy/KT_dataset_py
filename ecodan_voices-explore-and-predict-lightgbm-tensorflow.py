# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
INPUT_DIR = "../input"
df_train = pd.read_csv(os.path.join(INPUT_DIR, "voice.csv"))
df_train.head(5)
df_train.columns
FEATURE_COLS = ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
       'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
       'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']
TARGET_COL = 'label'
df_train.dtypes
df_train.describe()
df_train['label'].value_counts()
df_train.hist(figsize=(15,15))
corr = df_train.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
for c in FEATURE_COLS:
    plt.figure()
    plt.hist(df_train[df_train[TARGET_COL] == 'male'][c], 10, alpha=0.5, label='male')
    plt.hist(df_train[df_train[TARGET_COL] == 'female'][c], 10, alpha=0.5, label='female')
    plt.legend(loc='upper right')
    plt.title(c)

# change label to numeric
df_train['label'] = df_train['label'].map(lambda x: 1 if x == 'female' else 0)
def train_with_lgbm(df, FCOLS, DCOLS, TGT_COL):
    print('training on columns {0}\ntreating {1} as categorial'.format(FCOLS, DCOLS))
    
    X = np.array(df[FCOLS])
    y = df[TGT_COL].values
    
    print("train dims: {0}".format(X.shape))

    X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.2, random_state = 12)
    X_train, X_valid, y_train, y_valid = train_test_split(X_1, y_1, test_size=0.2, random_state = 12)
    del X, y, X_1, y_1; gc.collect();

    # prepare lgb datasets
    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_valid, label=y_valid) 
    watchlist = [d_train, d_valid]

    # set model params
    params = {
        'objective': 'binary',
        'is_unbalanced': False,
        'boosting': 'gbdt',
    #           'metric': 'rmse',
    #           'metric': 'multi-logloss',
        'num_leaves': 110,
        'max_depth': 11,
        'learning_rate': 0.01,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.8,
        'min_split_gain': 0.01,
        'min_child_samples': 150,
        'min_child_weight': 0.1,
        'verbosity': -1,
        'data_random_seed': 3,
    }

    model = lgb.train(
        params,
        train_set=d_train,
        valid_sets=watchlist,
        feature_name=FCOLS,
        categorical_feature=DCOLS,
        verbose_eval=100,
        num_boost_round=10000,
        early_stopping_rounds=200,
    )

    return model, X_test, y_test
model, X_test, y_test = train_with_lgbm(df_train, FEATURE_COLS, [], TARGET_COL)
def evaluate_model(model, X_test, y_test, FCOLS, is_multiclass, show_LGBM_feature_importance=False):
#     pdb.set_trace()
    if show_LGBM_feature_importance:
        print("Feature Importance:\n{0}".format(pd.DataFrame(index=FCOLS, data=np.sort(model.feature_importance(importance_type='gain')))))

    y_pred = model.predict(X_test)
#     print(y_pred[0:10])
    if is_multiclass:
        # since multiclass predictions give us proba of each label...
        y_pred = np.array([x.argmax() for x in y_pred])
    else:
        # since regression models give a continuous range
        y_pred = np.array([int(round(x)) for x in y_pred])
#     print(y_pred[0:10])

    print('\nConfusion Matrix:\n{0}'.format(confusion_matrix(y_pred, y_test)))
    print('\nAccuracy:\n{0}'.format(accuracy_score(y_pred, y_test)))


    plt.hist([pd.Series(y_test),pd.Series(y_pred)], color=['r','b'])


evaluate_model(model, X_test, y_test, FEATURE_COLS, False, True)
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels=None, batch_size=100):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
def df_to_tf_train_test_split(df, feature_cols, label_col, split_size=0.2):

    msk = np.random.rand(len(df)) < (1.0 - split_size)
    df_trn = df.iloc[msk,:]
    df_val = df.iloc[~msk,:]
    
    print("train dims: {0} | val dims: {1}".format(df_trn.shape, df_val.shape))

    X_train = {}
    for idx, col in enumerate(feature_cols):
        X_train[col] = df_trn[col].values

    y_train = df_trn[label_col]    
    
    X_val = {}
    for idx, col in enumerate(feature_cols):
        X_val[col] = df_val[col].values
    y_val = df_val[label_col]    

    return X_train, y_train, X_val, y_val
def zscore(mean, std):
    def normalizer(x):
        return (x-mean)/std
    return normalizer

def generate_feature_columns(df, feature_cols, discreet_cols, scalar_cols):
    # Feature columns describe how to use the input.
    tf_fcols = []
    for col in df.columns:
        if col in feature_cols:
            if col in discreet_cols:
#                 print("col {0} as cat".format(col))
                if df[col].nunique() > 1000:
                    tf_fcols.append(
                        tf.feature_column.indicator_column(
                            tf.feature_column.categorical_column_with_hash_bucket(
                                key=col, 
                                hash_bucket_size=10,
                            )
                        )
                    )
                else:
                    tf_fcols.append(
                        tf.feature_column.indicator_column(
                            tf.feature_column.categorical_column_with_identity(
                                key=col, 
                                num_buckets=df[col].nunique(),
                                default_value=0,
                            )
                        )
                    )
            else:
                if df[col].dtype == np.int64:
#                     print("col {0} as {1}".format(col, tf.int64))
                    tf_fcols.append(tf.feature_column.numeric_column(
                        key=col, 
                        dtype=tf.int64,
                        normalizer_fn=zscore(df[col].mean(), df[col].std()),
                    ))
                else:
#                     print("col {0} as {1}".format(col, tf.float64))
                    tf_fcols.append(tf.feature_column.numeric_column(
                        key=col, 
                        dtype=tf.float64,
                        normalizer_fn=zscore(df[col].mean(), df[col].std()),
                    ))

                    
    return tf_fcols
X_train, y_train, X_val, y_val = df_to_tf_train_test_split(df_train, FEATURE_COLS, TARGET_COL) 
tf_fcols = generate_feature_columns(df_train, FEATURE_COLS, [], FEATURE_COLS)
classifier = tf.estimator.DNNClassifier(
    feature_columns=tf_fcols,
    hidden_units=[27,9],
    n_classes=2
)
# Train the Model.
classifier.train(
    input_fn=lambda:train_input_fn(
        X_train, 
        y_train, 
        100
    ),
    steps=10000
)
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(X_val, y_val, 100))
