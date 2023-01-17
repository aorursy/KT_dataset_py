import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)

X = train_df.drop('Survived', axis=1)
y = train_df.Survived
categorical_features_indices = np.where(X.dtypes != np.float)[0]
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=42)

X_test = test_df
import hyperopt

def hyperopt_objective(params):
    model = CatBoostClassifier(
        l2_leaf_reg=int(params['l2_leaf_reg']),
        max_depth=int(params['max_depth']),
        iterations=150,
        eval_metric='Accuracy',
        random_seed=164530,
        logging_level='Silent',
        od_type = 'IncToDec',
        od_wait = 20
    )
    
    cv_data = cv(
       Pool(X, y, cat_features=categorical_features_indices),
        model.get_params()
    )
    best_accuracy = np.max(cv_data['test-Accuracy-mean'])
    
    return 1 - best_accuracy # as hyperopt minimises
params_space = {
    'l2_leaf_reg': hyperopt.hp.quniform('l2_leaf_reg', 1, 7, 1),
    'max_depth': hyperopt.hp.quniform('max_depth', 2, 8, 1),
}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=150,
    trials=trials,
)

print(best1)
model = CatBoostClassifier(
    max_depth = int(best['max_depth']),
    l2_leaf_reg = int(best['l2_leaf_reg']),
    iterations=750,
    eval_metric='Accuracy',
    random_seed = 164530,
    logging_level='Silent',
    od_type = 'IncToDec',
    od_wait = 20
)
cv_data = cv(Pool(X, y, cat_features=categorical_features_indices), model.get_params())

print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))
model.get_params()

#          0,028 - 0,8238 
#          0,028 - 0,8193
#          0,029 - 0,8036
#baseline - 0,03 - 0,8024
#          0,031 - 0,8047
#          0,032 - 0,8058 - 0.81705
model.fit(X, y, cat_features=categorical_features_indices, logging_level = 'Silent');
submission = pd.DataFrame()
submission['PassengerId'] = X_test['PassengerId']
submission['Survived'] = model.predict(X_test).astype('int')
submission.head()
submission.to_csv('submission.csv', index=False)