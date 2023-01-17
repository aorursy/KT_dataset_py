import pandas as pd, catboost

train, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')

train['Boy'], test['Boy'] = [(df.Name.str.split().str[1] == 'Master.').astype('int') for df in [train, test]]

train['Surname'], test['Surname'] = [df.Name.str.split(',').str[0] for df in [train, test]]

model = catboost.CatBoostClassifier(one_hot_max_size=4, iterations=100, random_seed=0, verbose=False)

model.fit(train[['Sex', 'Pclass', 'Embarked', 'Boy', 'Surname']].fillna(''), train['Survived'], cat_features=[0, 2, 4])

pred = model.predict(test[['Sex', 'Pclass', 'Embarked', 'Boy', 'Surname']].fillna('')).astype('int')

pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1).to_csv('submission.csv', index=False)
def highlight(value):

    if value >= 0.5:

        style = 'background-color: palegreen'

    else:

        style = 'background-color: pink'

    return style



pd.pivot_table(train, values='Survived', index=['Pclass', 'Embarked'], columns='Sex').style.applymap(highlight)
features = ['Sex', 'Pclass', 'Embarked']



X_train = train[features].fillna('')

y_train = train['Survived']

X_test = test[features].fillna('')



model = catboost.CatBoostClassifier(

    one_hot_max_size=4,

    iterations=100,

    random_seed=0,

    verbose=False,

    eval_metric='Accuracy'

)



pool = catboost.Pool(X_train, y_train, cat_features=[0, 2])

print('To see the Catboost plots, fork this kernel and run it in the editing mode.')

cv_scores = catboost.cv(pool, model.get_params(), fold_count=10, plot=True)

print('CV score: {:.5f}'.format(cv_scores['test-Accuracy-mean'].values[-1]))
model.fit(pool)

pred = model.predict(X_test).astype('int')

output = pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1)

output.to_csv('submission2.csv', index=False)
pd.pivot_table(train, values='Survived', index='Pclass', columns='Boy').style.applymap(highlight)
features = ['Sex', 'Pclass', 'Embarked', 'Boy']



X_train = train[features].fillna('')

y_train = train['Survived']

X_test = test[features].fillna('')



model = catboost.CatBoostClassifier(

    one_hot_max_size=4,

    iterations=100,

    random_seed=0,

    verbose=False,

    eval_metric='Accuracy'

)



pool = catboost.Pool(X_train, y_train, cat_features=[0, 2])

print('To see the Catboost plots, fork this kernel and run it in the editing mode.')

cv_scores = catboost.cv(pool, model.get_params(), fold_count=10, plot=True)

print('CV score: {:.5f}'.format(cv_scores['test-Accuracy-mean'].values[-1]))
model.fit(pool)

pred = model.predict(X_test).astype('int')

output = pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1)

output.to_csv('submission3.csv', index=False)
pd.pivot_table(train, values='Survived', index='Surname')[:15].sort_values('Survived').style.applymap(highlight)
features = ['Sex', 'Pclass', 'Embarked', 'Boy', 'Surname']



X_train = train[features].fillna('')

y_train = train['Survived']

X_test = test[features].fillna('')



model = catboost.CatBoostClassifier(

    one_hot_max_size=4,

    iterations=100,

    random_seed=0,

    verbose=False,

    eval_metric='Accuracy'

)



pool = catboost.Pool(X_train, y_train, cat_features=[0, 2, 4])

print('To see the Catboost plots, fork this kernel and run it in the editing mode.')

cv_scores = catboost.cv(pool, model.get_params(), fold_count=10, plot=True)

print('CV score: {:.5f}'.format(cv_scores['test-Accuracy-mean'].values[-1]))
model.fit(pool)

pred = model.predict(X_test).astype('int')

output = pd.concat([test['PassengerId'], pd.DataFrame(pred, columns=['Survived'])], axis=1)

output.to_csv('submission4.csv', index=False)