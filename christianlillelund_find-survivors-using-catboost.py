import numpy as np

import pandas as pd



pd.reset_option('^display.', silent=True)

pd.set_option('mode.chained_assignment', None)



# Load the full dataset

df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')



# List the first five passengers and their fate

df.head()
# Check for null values

null_value_stats = df.isnull().sum(axis=0)

null_value_stats[null_value_stats != 0]
# Print the data types

print(df.dtypes)



# Delete unused variables

df = df.drop(['PassengerId', 'Country', 'Firstname', 'Lastname'],axis=1)



# Save indices of categorial features (Sex, category)

categorical_features_indices = np.where(df.dtypes != np.int64)[0]



# Show the final dataframe

df.head()
from sklearn.model_selection import train_test_split



# Split X and y

X = df.drop('Survived', axis=1)

y = df.Survived



# Make a train and validation set of the data

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y, random_state=0)
from catboost import CatBoostClassifier, Pool, cv

from sklearn.metrics import accuracy_score



model = CatBoostClassifier(custom_loss=['Accuracy'],

                           random_seed=0,

                           verbose=200)



model.fit(X_train,

          y_train,

          cat_features=categorical_features_indices,

          eval_set=(X_val, y_val),

          plot=True);
# Set log loss as the log function

cv_params = model.get_params()

cv_params.update({ 'loss_function': 'Logloss' })



cv_data = cv(

    Pool(X, y, cat_features=categorical_features_indices),

    cv_params,

    fold_count=5,

    plot=True

)
# Print the best validation accuracy and its iteration

print('Best validation accuracy score: {:.2f}±{:.2f} at step {}'.format(

    np.max(cv_data['test-Accuracy-mean']),

    cv_data['test-Accuracy-std'][np.argmax(cv_data['test-Accuracy-mean'])],

    np.argmax(cv_data['test-Accuracy-mean'])

))
from time import time

params = {

    'iterations': 500,

    'learning_rate': 0.1,

    'eval_metric': 'Accuracy',

    'random_seed': 0,

    'verbose': 200,

    'use_best_model': False

}



train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)

validate_pool = Pool(X_val, y_val, cat_features=categorical_features_indices)



no_stop_model = CatBoostClassifier(**params)

t0 = time()

no_stop_model.fit(train_pool, eval_set=validate_pool)

print("Training time no stopping:", round(time()-t0, 4), "s")
params.update({

    'od_type': 'Iter',

    'od_wait': 40

})

early_stop_model = CatBoostClassifier(**params)

t0 = time()

early_stop_model.fit(train_pool, eval_set=validate_pool);

print("Training time early stopping:", round(time()-t0, 4), "s")
print(f'No stop model tree count: {no_stop_model.tree_count_}')

print(f'No stop model validation accuracy: {accuracy_score(y_val, no_stop_model.predict(X_val))}')

print()

print(f'Early stop model tree count: {early_stop_model.tree_count_}')

print(f'Early stop model validation accuracy: {accuracy_score(y_val, early_stop_model.predict(X_val))}')
params = {

    'iterations': 5,

    'eval_metric': 'Accuracy',

    'random_seed': 0,

    'verbose': 200

}

model_snapshot = CatBoostClassifier(**params)

model_snapshot.fit(train_pool, eval_set=validate_pool, save_snapshot=True)



params.update({

    'iterations': 10,

    'learning_rate': 0.1,

})

model_snapshot = CatBoostClassifier(**params)

model_snapshot.fit(train_pool, eval_set=validate_pool, save_snapshot=True)
model = CatBoostClassifier(iterations=50, random_seed=0, logging_level='Silent').fit(train_pool)

feature_importances = model.get_feature_importance(train_pool)

feature_names = X_train.columns

for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

    print('{}: {}'.format(name, score))
predictions = model.predict(X_val)

predictions_probs = model.predict_proba(X_val)

print(f'Predictions of classes: {predictions[:10]}')

print(f'Prediction of probs: {predictions_probs[:10]}')