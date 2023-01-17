import os, sys



import numpy as np

from scipy.stats import chi2_contingency

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

df.head()
df.shape
df_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

df_targets.head()
df_targets.shape
df_test_sub = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
def summarize_categoricals(df, show_levels=False, threshold=5):

    """

        Display uniqueness in each column

    """

    data = [[df[c].unique(), len(df[c].unique()), df[c].isnull().sum()] for c in df.columns]

    df_temp = pd.DataFrame(data, index=df.columns,

                           columns=['Levels', 'No. of Levels', 'No. of Missing Values'])

    return df_temp[df_temp['No. of Levels'] <= threshold].iloc[:, 0 if show_levels else 1:]





def return_categoricals(df, threshold=5):

    """

        Returns a list of columns that have less than or equal to

        `threshold` number of unique categorical levels

    """

    return list(filter(lambda c: c if len(df[c].unique()) <= threshold else None,

                       df.columns))





def to_categorical(columns, df):

    """

        Converts the columns passed in `columns` to categorical datatype

    """

    for col in columns:

        df[col] = df[col].astype('category')

    return df
summarize_categoricals(df, show_levels=True, threshold=500)
categorical_columns = return_categoricals(df)
df = to_categorical(categorical_columns, df)
df.info()
df_test_sub = to_categorical(categorical_columns, df_test_sub)
x = df.iloc[:, 1:]

y = df_targets.iloc[:, 1:]



categorical_columns = list(x.select_dtypes(include='category').columns)

numeric_columns = list(x.select_dtypes(exclude='category').columns)
from sklearn.model_selection import train_test_split



data_splits = train_test_split(x, y, test_size=0.15, random_state=0,

                               shuffle=True)

x_train, x_test, y_train, y_test = data_splits



list(map(lambda x: x.shape, [x, y, x_train, x_test,

                             y_train, y_test]))
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline 





numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])



categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore', dtype=np.int))])



## Column Transformer

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_columns),

        ('cat', categorical_transformer, categorical_columns)],

    remainder='passthrough')





## Applying Column Transformer

x_train = preprocessor.fit_transform(x_train)

x_test = preprocessor.transform(x_test)



x_test_sub = preprocessor.transform(df_test_sub.iloc[:, 1:])





## Label encoding

y_train = y_train.to_numpy(dtype=np.int64)

y_test = y_test.to_numpy(dtype=np.int64)





## Save feature names after one-hot encoding for feature importances plots

feature_names = list(preprocessor.named_transformers_['cat'].named_steps['onehot'] \

                     .get_feature_names(input_features=categorical_columns))

feature_names = feature_names + numeric_columns
from sklearn.metrics import log_loss

from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier



xgb = XGBClassifier(tree_method='gpu_hist',

                    predictor='gpu_predictor',

                    random_state=0, n_jobs=-1)



ovr_clf = OneVsRestClassifier(estimator=xgb, n_jobs=-1)



ovr_clf.fit(x_train, y_train)



ovr_probs = ovr_clf.predict_proba(x_test)



log_loss(y_test, ovr_probs)
from sklearn.multioutput import MultiOutputClassifier



mo_clf = MultiOutputClassifier(estimator=xgb, n_jobs=-1)



mo_clf.fit(x_train, y_train)



mo_probs = mo_clf.predict_proba(x_test)



n_classes = y_test.shape[1]

n_test_samples = x_test.shape[0]

mo_probs_pos = np.zeros((n_test_samples, n_classes))



for c in range(n_classes):

    c_probs = mo_probs[c]

    mo_probs_pos[:, c] = c_probs[:, 1]



log_loss(y_test, mo_probs_pos)
from sklearn.multioutput import ClassifierChain

from joblib import Parallel, delayed

import timeit



chains = [ClassifierChain(base_estimator=xgb, order='random')

          for i in range(5)]



chains = Parallel(n_jobs=-1)(delayed(chain.fit)(x_train, y_train) for chain in chains)



chains_ensemble_proba = Parallel(n_jobs=-1)(delayed(chain.predict_proba)(x_test) for chain in chains)



log_loss(y_test, np.array(chains_ensemble_proba).mean(axis=0))
## Final Model

x_train_full = preprocessor.fit_transform(x)

y_train_full = y.to_numpy(dtype=np.int64)



x_test_final = preprocessor.transform(df_test_sub.iloc[:, 1:])



chains = Parallel(n_jobs=-1)(delayed(chain.fit)(x_train_full, y_train_full) for chain in chains)



final_proba = Parallel(n_jobs=-1)(delayed(chain.predict_proba)(x_test_final) for chain in chains)
pd.DataFrame(np.array(final_proba).mean(axis=0),

             index=df_test_sub['sig_id'],

             columns=df_targets.columns[1:]).to_csv('submission.csv')