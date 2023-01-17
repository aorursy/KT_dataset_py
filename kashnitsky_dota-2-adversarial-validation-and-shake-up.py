import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score

import warnings

warnings.filterwarnings('ignore')
PATH_TO_DATA = '../input/'



df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train_features.csv'), 

                                    index_col='match_id_hash')

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')
df = pd.concat([df_train_features, df_test_features])
is_train = [1] * len(df_train_features) + [0] * len(df_test_features)
model = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
%%time

adv_validation_scores = cross_val_score(model, df, is_train, cv=skf, n_jobs=4,

                                       scoring='roc_auc')
adv_validation_scores
model.fit(df, is_train)
import eli5
eli5.show_weights(estimator=model, feature_names=list(df.columns))
from sklearn.model_selection import StratifiedShuffleSplit
skf = StratifiedShuffleSplit(n_splits=150, test_size=10000, random_state=1)
df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')
%%time

scores = cross_val_score(model, df_train_features, df_train_targets['radiant_win'], 

                         cv=skf, n_jobs=4, scoring='roc_auc')
scores = pd.Series(scores)

mean = scores.mean()

lower, upper = mean - 2 * scores.std(), mean + 2 * scores.std()
from matplotlib import pyplot as plt

import seaborn as sns
sns.distplot(scores, bins=10);

plt.vlines(x=[mean], ymin=0, ymax=110, 

           label='mean', linestyles='dashed');

plt.vlines(x=[lower, upper], ymin=0, ymax=110, 

           color='red', label='+/- 2 std',

          linestyles='dashed');

plt.legend();