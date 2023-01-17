import catboost

from catboost import CatBoostClassifier

from catboost import datasets
train_df, test_df = datasets.amazon() 

train_df.shape, test_df.shape
train_df.head()
# test_df.head()
y = train_df['ACTION']

X = train_df.drop(columns='ACTION')
X_test = test_df.drop(columns='id')

SEED = 1
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=SEED)
%%time



params = {'loss_function':'Logloss', # objective function

          'eval_metric':'AUC', # metric

          'verbose': 200, # output to stdout info about training process every 200 iterations

          'random_seed': SEED

         }

cbc_1 = CatBoostClassifier(**params)

cbc_1.fit(X_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)

          eval_set=(X_valid, y_valid), # data to validate on

          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score

          plot=True # True for visualization of the training process (it is not shown in a published kernel - try executing this code)

         );
cat_features = list(range(X.shape[1]))

print(cat_features)
cat_features_names = X.columns # here we specify names of categorical features

cat_features = [X.columns.get_loc(col) for col in cat_features_names]

print(cat_features)
condition = True # here we specify what condition should be satisfied only by the names of categorical features

cat_features_names = [col for col in X.columns if condition]

cat_features = [X.columns.get_loc(col) for col in cat_features_names]

print(cat_features)
%%time



params = {'loss_function':'Logloss',

          'eval_metric':'AUC',

          'cat_features': cat_features,

          'verbose': 200,

          'random_seed': SEED

         }

cbc_2 = CatBoostClassifier(**params)

cbc_2.fit(X_train, y_train,

          eval_set=(X_valid, y_valid),

          use_best_model=True,

          plot=True

         );
%%time



params = {'loss_function':'Logloss',

          'eval_metric':'AUC',

          'cat_features': cat_features,

          'early_stopping_rounds': 200,

          'verbose': 200,

          'random_seed': SEED

         }

cbc_2 = CatBoostClassifier(**params)

cbc_2.fit(X_train, y_train, 

          eval_set=(X_valid, y_valid), 

          use_best_model=True, 

          plot=True

         );
import numpy as np

import warnings

warnings.filterwarnings("ignore")

np.random.seed(SEED)

noise_cols = [f'noise_{i}' for i in range(5)]

for col in noise_cols:

    X_train[col] = y_train * np.random.rand(X_train.shape[0])

    X_valid[col] = np.random.rand(X_valid.shape[0])
X_train.head()
%%time



params = {'loss_function':'Logloss',

          'eval_metric':'AUC',

          'cat_features': cat_features,

          'verbose': 200,

          'random_seed': SEED

         }

cbc_5 = CatBoostClassifier(**params)

cbc_5.fit(X_train, y_train, 

          eval_set=(X_valid, y_valid), 

          use_best_model=True, 

          plot=True

         );

ignored_features = list(range(X_train.shape[1] - 5, X_train.shape[1]))

print(ignored_features)
%%time



params = {'loss_function':'Logloss',

          'eval_metric':'AUC',

          'cat_features': cat_features,

          'ignored_features': ignored_features,

          'early_stopping_rounds': 200,

          'verbose': 200,

          'random_seed': SEED

         }

cbc_6 = CatBoostClassifier(**params)

cbc_6.fit(X_train, y_train, 

          eval_set=(X_valid, y_valid), 

          use_best_model=True, 

          plot=True

         );