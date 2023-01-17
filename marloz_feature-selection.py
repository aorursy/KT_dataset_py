import pandas as pd

import numpy as np
df_train = pd.read_csv('../input/train.csv')

print(df_train.shape)

df_train.head()
def check_missing(df):

    return df.isnull().sum().sum()

print(f'Missing values {check_missing(df_train)}')
df_train.dtypes.value_counts()
df_train.target.sum() / df_train.shape[0]
# Drop ID column as it doesn't contain information

df_train.drop('ID_code', axis=1, inplace=True)
df_positive_class = df_train[df_train.target == 1]

df_negative_class = df_train[df_train.target == 0].sample(df_positive_class.shape[0], replace=False, random_state=42)

sample_df = pd.concat([df_positive_class, df_negative_class], axis=0)

sample_df.target.value_counts()
np.random.seed(42)

X = (sample_df.drop('target', axis=1)

     .apply(lambda x: x.astype('float32'))

     .assign(rand=lambda x: np.random.rand(x.shape[0])))

y = sample_df.target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
import time

def score_and_time(model, X):

    ts = time.time()

    preds = model.predict(X)

    elapsed = int((time.time() - ts) * 1000)

    print(f'Score {model.oob_score_}, predicted in {elapsed}')
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



baseline = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)

baseline.fit(X_train, y_train)

score_and_time(baseline, X_test)
importances = pd.Series(baseline.feature_importances_, index=X_train.columns)

important_cols = importances[importances > importances['rand']].index.tolist()

print(f'Number of features with greater than random column importance {len(important_cols)}')

importances.sort_values().plot()
reduced_rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)

reduced_rf.fit(X_train[important_cols], y_train)

score_and_time(reduced_rf, X_test[important_cols])
import scipy.stats

_ = pd.concat([X_train,y_train], axis=1)

cor = pd.DataFrame(np.abs(scipy.stats.spearmanr(_).correlation),

                   columns=_.columns, index=_.columns)['target']

non_rand_corr = cor[cor > cor['rand']].shape[0]

print(f'Number of variables with correlation to target higher than random {non_rand_corr}')

cor[cor.index != 'target'].sort_values().plot()
top20_cols = cor[cor.index != 'target'].sort_values()[-int(cor.shape[0] * .2):].index.tolist()

rf_corr = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)

rf_corr.fit(X_train[top20_cols], y_train)

print(f'Reduced number of columns {len(top20_cols)}')

score_and_time(rf_corr, X_test[top20_cols])
# scikit-learn==0.22.2 or higher required

from sklearn.inspection import permutation_importance

from sklearn.base import BaseEstimator, ClassifierMixin



class FeatureSelector(BaseEstimator, ClassifierMixin):

    

    # This can be tuned to accept kwargs for pi

    def __init__(self, estimator):

        self.estimator = estimator

    

    def fit(self, X, y=None):

        self.estimator.fit(X, y)

        self.important_cols = self._get_important_cols(X, y)

        self.estimator.fit(X[self.important_cols], y)

        return self

        

    def _get_important_cols(self, X, y):

        pi = permutation_importance(self.estimator, X, y, n_repeats=1, n_jobs=-1, random_state=42)

        importances = pd.DataFrame(pi.importances_mean, index=X.columns, columns=['imp'])['imp']

        return importances[importances > importances['rand']].index.tolist()

    

    def predict(self, X):

        return self.estimator.predict(X[self.important_cols])

    

    @property

    def oob_score_(self):

        return self.estimator.oob_score_
from sklearn.pipeline import Pipeline



clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)

fs = FeatureSelector(clf)
fs.fit(X_train, y_train)

print(f'Number of selected features {len(fs.important_cols)}')

score_and_time(fs, X_test)