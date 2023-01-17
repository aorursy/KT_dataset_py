import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
import seaborn as sns
PATH_TO_DATA = Path('../input/flight-delays-fall-2018/')
train_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_train.csv')
test_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_test.csv')
ctb = CatBoostClassifier(random_seed=17, silent=True)
train_df.head()
train_df['UniqueCarrier'].value_counts()
train_df['flight'] = train_df['Origin'] + '-->' + train_df['Dest']

test_df['flight'] = test_df['Origin'] + '-->' + test_df['Dest']
small_ones ={'AS': 'small',

             'YV': 'small',

             'B6': 'small',

             'HP': 'small',

             'F9': 'small',

             'DH': 'small',

             'HA': 'small',

             'TZ': 'small',

             'AQ': 'small'} 



train_df['UniqueCarrier'] = train_df['UniqueCarrier'].replace(small_ones)

test_df['UniqueCarrier'] = test_df['UniqueCarrier'].replace(small_ones)
# train_df['Distance // 100'] = (train_df['Distance'] // 100).astype(str)

# test_df['Distance // 100'] = (test_df['Distance'] // 100).astype(str)
# train_df['Month Day Week'] = train_df['Month'] + ':' + train_df['DayofMonth'] + ':' + train_df['DayOfWeek']

# test_df['Month Day Week'] = test_df['Month'] + ':' + test_df['DayofMonth'] + ':' + test_df['DayOfWeek']
# train_df['Month + Origin'] = train_df['Month'] + ': ' + train_df['Origin']

# test_df['Month + Origin'] = test_df['Month'] + ': ' + test_df['Origin']
# train_df['UC flight'] = train_df['UniqueCarrier'] + ': ' + train_df['flight']

# test_df['UC flight'] = test_df['UniqueCarrier'] + ': ' + test_df['flight']
# train_df['hour'] = ((train_df['DepTime'] // 100) % 24).astype(str)

# test_df['hour'] = ((test_df['DepTime'] // 100) % 24).astype(str)
# train_df['day off'] = train_df['DayOfWeek'].map({'c-1': 0, 'c-2': 0, 'c-3': 0, 'c-4': 0, 'c-5': 0, 'c-6': 1, 'c-7': 1})

# test_df['day off'] = test_df['DayOfWeek'].map({'c-1': 0, 'c-2': 0, 'c-3': 0, 'c-4': 0, 'c-5': 0, 'c-6': 1, 'c-7': 1})
X_train_part, X_valid_part, y_train_part, y_valid = train_test_split(train_df.drop(['dep_delayed_15min'], axis=1).values, 

                                                                train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values, 

                                                                test_size=0.3, 

                                                                random_state=17)

X_test = test_df.values
cat_features = np.where(train_df.drop(['dep_delayed_15min'], axis=1).dtypes == 'object')[0]
%%time

ctb.fit(X_train_part, y_train_part,

        cat_features=cat_features);
roc_auc_score(y_valid, ctb.predict_proba(X_valid_part)[:, 1])
ctb_test_pred = ctb.predict_proba(X_test)[:, 1]
with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 

                             index_col='id')

    sample_sub['dep_delayed_15min'] = ctb_test_pred

    sample_sub.to_csv('submission.csv')
with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 

                             index_col='id')

    sample_sub['dep_delayed_15min'] = ctb_test_pred

    sample_sub.to_csv('ctb_pred.csv')