import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool, cv

import hyperopt
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_size = train.shape[0] # 891
test_size = test.shape[0] # 418

data = pd.concat([train, test])
data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
age_ref = data.groupby('Title').Age.mean()
data['Age'] = data.apply(lambda r: r.Age if pd.notnull(r.Age) else age_ref[r.Title] , axis=1)
del age_ref
data.loc[(data.PassengerId==1044, 'Fare')] = 14.43
data['Embarked'] = data['Embarked'].fillna('S')
data['Cabin'] = data['Cabin'].fillna('Undefined')
cols = [
    'Pclass',
    'Name',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Ticket',
    'Fare',
    'Cabin',
    'Embarked'
]
X_train = data[:train_size][cols]
Y_train = data[:train_size]['Survived'].astype(int)
X_test = data[train_size:][cols]

categorical_features_indices = [0,1,2,6,8,9]
X_train.head()
train_pool = Pool(X_train, Y_train, cat_features=categorical_features_indices)
'''
def hyperopt_objective(params):
    model = CatBoostClassifier(
        l2_leaf_reg=int(params['l2_leaf_reg']),
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        iterations=500,
        eval_metric='Accuracy',
        od_type='Iter',
        od_wait=40,
        random_seed=42,
        logging_level='Silent',
        allow_writing_files=False
    )
    
    cv_data = cv(
        train_pool,
        model.get_params()
    )
    best_accuracy = np.max(cv_data['test-Accuracy-mean'])    
    
    print(params, best_accuracy)
    return 1 - best_accuracy # as hyperopt minimises

params_space = {
    'l2_leaf_reg': hyperopt.hp.qloguniform('l2_leaf_reg', 0, 2, 1),
    'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
    'depth': hyperopt.hp.choice('depth', [3,4,5,6,8]),
}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=50,
    trials=trials
)

print(best)
'''
#best = {'depth': 6, 'l2_leaf_reg': 1.0, 'learning_rate': 0.07395682681736576} 

model = CatBoostClassifier(
    #l2_leaf_reg=int(best['l2_leaf_reg']),
    #learning_rate=best['learning_rate'],
    #depth=best['depth'],
    depth=3,
    iterations=300,
    eval_metric='Accuracy',
    #od_type='Iter',
    #od_wait=40,
    random_seed=42,
    logging_level='Silent',
    allow_writing_files=False
)

cv_data = cv(
    train_pool,
    model.get_params(),
    fold_count=5
)

print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format(
    np.max(cv_data['test-Accuracy-mean']), 
    cv_data['test-Accuracy-std'][cv_data['test-Accuracy-mean'].idxmax(axis=0)],
    cv_data['test-Accuracy-mean'].idxmax(axis=0)
))
print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))

model.fit(train_pool);
model.score(X_train, Y_train)
feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))
Y_pred = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": data[train_size:]["PassengerId"], 
    "Survived": Y_pred.astype(int)
})
submission.to_csv('submission.csv', index=False)