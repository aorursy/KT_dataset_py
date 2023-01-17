import pandas as pd

pd.set_option("max_rows", 200)



import numpy as np



np.random.seed(123)



import catboost as cb

from sklearn.model_selection import train_test_split



from sklearn import metrics



import re 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_train.head()
df_train.info()
df_test.info()
df = pd.concat([df_train, df_test], ignore_index=True)
TRAIN_IDX = df[~df['Survived'].isnull()].index

TEST_IDX = df[df['Survived'].isnull()].index
df.isnull().sum()
df.groupby("Pclass")[['Age']].mean()
def fill_missing_values(df):

    df = df.copy()

    df['Age'] = df.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.mean())).round()

    df['Embarked'].fillna(df['Embarked'].mode(), inplace=True) 

    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

    return df



df = fill_missing_values(df)
def fix_types(df):

    df = df.copy()

    df['Embarked'] = df['Embarked'].astype(str)

    df['Pclass'] = df['Pclass'].astype(int)

    df['SibSp'] = df['SibSp'].astype(int)

    df['Parch'] = df['Parch'].astype(int)

    df['Age'] = df['Age'].astype(int)

    return df



df = fix_types(df)
CAT_FEATURES = ['Embarked', 'Pclass', 'Sex', 'SibSp', 'Parch']

NUM_FEATURES = ['Fare', 'Age']
X = df.loc[TRAIN_IDX, CAT_FEATURES + NUM_FEATURES].copy()

y = df.loc[TRAIN_IDX, 'Survived'].copy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = cb.CatBoostClassifier(custom_metric='Accuracy', use_best_model=True)



clf.fit(

    X_train, 

    y=y_train, 

    cat_features=CAT_FEATURES,

    eval_set=[(X_test, y_test)], 

    verbose=False,

    plot=True, 

    early_stopping_rounds=20,

)
clf.get_feature_importance(prettified=True)
def cross_validation(df, cat_features, num_features, params=None):

    if params is None:

        params = {

            'iterations': 90,

            'loss_function': 'Logloss',

            'verbose': False,

            'depth': 6,

            'eval_metric': 'Accuracy',

        }

    

    df = df.copy()

    

    X = df.loc[TRAIN_IDX, cat_features + num_features]

    y = df.loc[TRAIN_IDX, 'Survived']

    

    pool = cb.Pool(X, label=y, cat_features=cat_features)

    scores = cb.cv(pool, params=params, fold_count=5, seed=42, stratified=True)

    

    acc_mean = scores.iloc[-1]['test-Accuracy-mean']

    std_mean = scores.iloc[-1]['test-Accuracy-std']

    

    return acc_mean, std_mean
BASELINE_ACC, BASELINE_STD = cross_validation(df, CAT_FEATURES, NUM_FEATURES)

print('Baseline Accuracy:', BASELINE_ACC, '+-', BASELINE_STD)
def beat_baseline(df, cat_features, num_features):

    global BASELINE_ACC

    global BASELINE_STD

    

    acc, std = cross_validation(df, cat_features, num_features)

    

    print('New Accuracy:', acc, '+-', std)

    print('Diff:', acc - BASELINE_ACC)

    

    if acc > BASELINE_ACC:

        # update baseline

        BASELINE_ACC = acc

        BASELINE_STD = std

        print('IMPROVE')
df['Title'] = df['Name'].str.extract(r"([A-Z][a-z]+\.)")

df['Title'].head()
def survived_perc(df, col):

    dff = df.loc[TRAIN_IDX].copy()

    dff = pd.crosstab(dff[col], dff['Survived'], margins=True)

    dff['Survived(%)'] = dff[1] / dff['All']

    dff = dff.drop("All").sort_values(by='Survived(%)', ascending=False)

    return dff
titles = survived_perc(df, 'Title')

titles
Mrs = {v: 'Mrs.' for v in ['Sir.', 'Countess.', 'Ms.', 'Mme.', 'Lady.', 'Mlle.', 'Mrs.', 'Dona.']}

Mr = {v: 'Mr.' for v in ['Jonkheer.', 'Don.', 'Rev.', 'Capt.']}

Master = {v: 'Master.' for v in ['Master.', 'Col.', 'Major.', 'Dr.']}



remap = {**Mr, **Mrs, **Master}



df['Title:Remap'] = df['Title'].map(lambda x: remap.get(x, x))

df['Title:Remap'].value_counts()
beat_baseline(df, ['Embarked', 'Pclass', 'Sex', 'Title:Remap', 'SibSp', 'Parch'], ['Fare', 'Age'])
df['Ticket:Number'] = df['Ticket'].str.extract(r"(\d{3,})").fillna(0).astype(int)
beat_baseline(df, ['Embarked', 'Pclass', 'Sex', 'Title:Remap', 'Ticket:Number', 'SibSp', 'Parch'], ['Fare', 'Age'])
CAT_FEATURES = ['Embarked', 'Pclass', 'Sex', 'Title:Remap', 'Ticket:Number', 'SibSp', 'Parch']

NUM_FEATURES = ['Fare', 'Age']
X = df.loc[TRAIN_IDX, CAT_FEATURES + NUM_FEATURES]

y = df.loc[TRAIN_IDX, 'Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
params = {

            'iterations': 1000,

            'loss_function': 'Logloss',

            'verbose': False,

            'depth': 6,

            'custom_metric': ['Accuracy'],

        }
clf = cb.CatBoostClassifier(**params, use_best_model=True)



clf.fit(

    X_train, 

    y=y_train, 

    cat_features=CAT_FEATURES,

    eval_set=[(X_test, y_test)], 

    verbose=False,

    plot=True, 

    early_stopping_rounds=20,

)



y_pred = clf.predict(X_test)



acc = metrics.accuracy_score(y_test, y_pred)

print('Accuracy on test set split:', acc)

print('Best iteration:', clf.best_iteration_)
params['iterations'] = 75



cross_validation(df, CAT_FEATURES, NUM_FEATURES, params=params)
clf = cb.CatBoostClassifier(**params)



clf.fit(

    X, 

    y=y, 

    cat_features=CAT_FEATURES,

    verbose=False,

)
submission = df.loc[TEST_IDX]
submission['Survived'] = clf.predict(submission[CAT_FEATURES + NUM_FEATURES])

submission['Survived'] = submission['Survived'].astype(int)
submission[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)