import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold, cross_validate

from catboost import CatBoostClassifier



import optuna

import itertools



trains = pd.read_csv('../input/train.csv')

tests = pd.read_csv('../input/test.csv')

tests['Survived'] = 0

print('loaded:', trains.shape, tests.shape)
def get_prefixes():

    return ['Mr.', 'Miss.', 'Mrs.', 'Master.', 'Ms.']



def make_prefix_condition(f, prefix):

    return f['Name'].str.contains(prefix)



def get_age_mean(info, prefix):

    means = info['age_means']

    return means[means['Prefix'] == prefix]['Age'].values[0]



def convert(f, info):

    is_train = False

    if info is None:

        is_train = True

        info = {}

    

    # 年齢と敬称

    if is_train:

        age_means = [['Other', f['Age'].median()]]

        for prefix in get_prefixes():

            condition = make_prefix_condition(f, prefix)

            age_means.append([prefix, f[condition]['Age'].median()])

        info['age_means'] = pd.DataFrame(age_means, columns=['Prefix', 'Age'])

    

    f['Prefix'] = 'Mr.'

    for prefix in get_prefixes():

        condition = make_prefix_condition(f, prefix)

        

        f.loc[condition, 'Prefix'] = prefix

        f.loc[f['Age'].isnull() & condition, 'Age'] = get_age_mean(info, prefix)

    f['Age'].fillna(get_age_mean(info, 'Other'), inplace=True)



    # 運賃（テストだけなのでベタ）

    f.loc[f['Fare'].isnull(), 'Fare'] = 11.3

    

    # 乗船場所

    f.loc[f['Embarked'].isnull(), 'Embarked'] = 'C'

    

    # 客室

    f['Cabin'].fillna('U', inplace=True)

    f['CabinFloor'] = f['Cabin'].str[:1]

    f['CabinSize'] = (f['Cabin'].str.count(' ') + 1).astype(np.float32)

    

    # 家族

    f['FamilyGroup'] = f['SibSp'] + f['Parch']

    f.loc[f['FamilyGroup'] > 0, 'FamilyGroup'] = 1

    f['SibSp'] = f['SibSp'].astype(np.float32)

    f['Parch'] = f['Parch'].astype(np.float32)

    f['FamilySize'] = f['SibSp'] + f['Parch'] + 1

    

    return f, info
trains, info = convert(trains, None)

tests, unuse = convert(tests, info)



y = trains['Survived']

x = trains.drop(['PassengerId', 'Survived', 'Name'], axis=1)

test_x = tests.drop(['PassengerId', 'Survived', 'Name'], axis=1)



cat_features = np.where(x.dtypes != float)[0]



cols = x.columns

x = x.values

y = y.values



acc_patterns = set()

submissions = []
def train_by_cat_boost(params, is_trial):

    weights = []

    val_accs = []

    for train_index, val_index in StratifiedKFold(n_splits=4).split(x, y):

        train_x, train_y = x[train_index], y[train_index]

        val_x, val_y = x[val_index], y[val_index]



        model = CatBoostClassifier(

            iterations=576, learning_rate=params['learning_rate'], l2_leaf_reg=params['l2_leaf_reg'],

            depth=params['depth'], eval_metric='Accuracy', one_hot_max_size=params['one_hot_max_size'],

            use_best_model=True)



        model.fit(train_x, train_y, eval_set=(val_x, val_y), cat_features=cat_features, logging_level='Silent')



        val_pred = model.predict(val_x)

        val_acc = accuracy_score(val_y, val_pred)

        val_accs.append(val_acc)

        

        if not is_trial and (val_acc not in acc_patterns):

            acc_patterns.add(val_acc)

            

            test_pred = model.predict(test_x.values)

            

            submission = pd.DataFrame()

            submission['PassengerId'] = tests['PassengerId']

            submission['Survived'] = test_pred

            submission['Acc'] = val_acc

            submissions.append(submission)

            

        print('    ', val_acc, params)

        weight = pd.DataFrame()

        weight['col'] = cols

        weight['weight'] = model.feature_importances_

        weights.append(weight)

    

#     print(pd.concat(weights).groupby('col').mean().sort_values('weight'))

    

    return np.mean(val_accs)



def objective(trial):

    params = {

        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.015),

        'depth': trial.suggest_int('depth', 9, 11),

        'l2_leaf_reg': trial.suggest_categorical('l2_leaf_reg', [9]),

        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 32, 48),

    }

    val_accs = train_by_cat_boost(params, is_trial=False)

    return 1 - val_accs



def make_product():

    learning_rates = [0.01, 0.0125, 0.015]

    depths = [9, 10]

    l2_leaf_regs = [9]

    one_hot_max_size = [32, 36]

    

    return itertools.product(learning_rates, depths, l2_leaf_regs, one_hot_max_size)
# Optuna

# study = optuna.create_study()

# study.optimize(objective, n_trials=32)



# Grid Search

# for p in make_product():

#     params = { 'learning_rate': p[0], 'depth': p[1], 'l2_leaf_reg': p[2], 'one_hot_max_size': p[3] }

#     train_by_cat_boost(params, is_trial=False)



# Single

params = { 'learning_rate': 0.01, 'depth': 10, 'l2_leaf_reg': 9, 'one_hot_max_size': 32 }

train_by_cat_boost(params, is_trial=False)
alls = pd.concat(submissions)

alls.sort_values('Acc', ascending=False, inplace=True)

alls = alls[0:len(tests) * 12]

alls.groupby('Acc', as_index=False).count()[['Acc']]
alls.drop('Acc', axis=1, inplace=True)

means = alls.groupby('PassengerId', as_index=False).mean()

means.loc[means['Survived'] >= 0.5, 'Survived'] = 1.0

means.loc[means['Survived'] < 0.5, 'Survived'] = 0.0

means['Survived'] = means['Survived'].astype(np.int32)

means.to_csv('submissions.csv', index=False)

means.shape