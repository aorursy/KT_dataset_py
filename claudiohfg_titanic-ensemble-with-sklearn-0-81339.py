from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

from mlens.visualization import corrmat



from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler



import re

from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# I didn't like the outcome of sklearn.preprocessing.LabelEncoder so I made my own label encoder

def label_encode(df, column_name):

    ordered_column = np.sort(df[column_name].unique())

    df[column_name] = df[column_name].map(

        dict(zip(np.sort(df[column_name].unique()),[x for x in range(len(df[column_name].unique()))]))

    )

    return df



# This functions helps us get insights on the relation between variables

def compare(df,column_name, with_table=False, with_graph=True, compare_to='Survived'):

    if with_table:

        print(df[df[compare_to] < 3].groupby([compare_to,column_name]).size().sort_index())

    if with_graph:

        g = sns.FacetGrid(df, col=compare_to).map(sns.distplot, column_name)



# This function display the correlation of all variables to the target label

def show_correlation(df, column_name='Survived'):

    return df.corr()[column_name].apply(abs).sort_values(na_position='first').reset_index()



# This function helps us find the outliers

def get_IQR(df, column_name):

    Q3 = df[column_name].quantile(0.75)

    Q1 = df[column_name].quantile(0.25)

    IQR = Q3 - Q1

    return Q1, Q3, IQR



# This function detects the outliers.

def detect_outliers(df, n, features):

        outlier_indices = []

        for col in features:

            Q1, Q3, IQR = get_IQR(df, col)

            outlier_step = 1.5 * IQR

            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

            outlier_indices.extend(outlier_list_col)



        outlier_indices = Counter(outlier_indices)        

        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )



        return multiple_outliers
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



test_df['Survived'] = 3



train_df = train_df[train_df.columns]

test_df = test_df[train_df.columns]



all_set = pd.concat([train_df, test_df], ignore_index=True)
all_set.profile_report()
def get_title(name):

    return name.split(',')[1].split('.')[0].strip()

    

all_set['Title'] = all_set['Name'].apply(get_title)
compare(all_set, 'Title', True, False)
all_set['Surname'] = all_set['Name'].apply(lambda x: x.split(',')[0].strip())
all_set['FamilySurvival'] = 0.5



for surname in all_set['Surname'].unique():

    df = all_set[all_set['Surname'] == surname]

    if df.shape[0] > 1:

        smin = df['Survived'].min()

        smax = df['Survived'].max()

        for idx, row in df.iterrows():

            passengerid = row['PassengerId']

            if smax == 1.0:

                all_set.loc[all_set['PassengerId'] == passengerid, 'FamilySurvival'] = 1.0

            elif smin == 0.0:

                all_set.loc[all_set['PassengerId'] == passengerid, 'FamilySurvival'] = 0.0
all_set['Embarked'].value_counts()
all_set['Embarked'] = all_set['Embarked'].fillna('S') 
compare(all_set, 'Embarked', True, False)
all_set[all_set['Fare'].isna()]
all_set['Fare'].fillna(all_set[all_set['Embarked'] == 'S']['Fare'].mean(), inplace=True)
compare(all_set, 'Fare', False, True)
all_set['Deck'] = all_set['Cabin'].apply(lambda x: x[0] if type(x) == str else '')
all_set[all_set['Deck'] == 'G']
compare(all_set, 'Deck', True, False)
deck_fare = {}

for deck in all_set['Deck'].unique():

    if len(deck) == 1 and deck in 'ABCDEF': # The passengers were distributed between decks A and F

        deck_fare[deck] = all_set[

            (all_set['Cabin'].apply(lambda x: True if type(x) == str else False)) &

            (all_set['Deck'] == deck)

        ]['Fare'].mean()

deck_fare
def find_deck(fare):

    dist = 1000

    res = 'F'

    for key in deck_fare.keys():

        new_dist = np.abs(fare - deck_fare[key])

        if new_dist < dist:

            dist = new_dist

            res = key

    return res



all_set.loc[all_set['Cabin'].isna(), 'Deck'] = all_set['Fare'].apply(find_deck)
compare(all_set, 'Deck', True, False)
all_set['Family'] = 1 + all_set['SibSp'] + all_set['Parch']

all_set['Alone'] = all_set['Family'].apply(lambda x: 1 if x == 1 else 0)
compare(all_set, 'Family')
age_by_title = {}

for title in all_set['Title'].unique():

    age_by_title[title] = all_set[

        (all_set['Age'].apply(lambda x: True if type(x) == float else False)) &

        (all_set['Title'] == title)

    ]['Age'].mean()

age_by_title
compare(all_set[all_set['Age'].isna()], 'Family', False, True, 'Title')
all_set.loc[all_set['Age'].isna(), 'Age'] = all_set['Title'].apply(lambda x: age_by_title[x])
compare(all_set, 'Age')
show_correlation(all_set)
all_set.dtypes
all_set['AgeBin'] = pd.qcut(all_set['Age'], 5)

all_set['AgeCode'] = all_set['AgeBin']

all_set = label_encode(all_set, 'AgeCode')
all_set['AgeDist'] = 1/ np.exp(np.abs (all_set['Age'] - all_set[all_set['Survived'] == 1]['Age'].median()))
compare(all_set, 'AgeDist')
all_set = label_encode(all_set, 'Sex')

compare(all_set, 'Sex')
all_set = label_encode(all_set, 'Embarked')

compare(all_set, 'Embarked')
all_set = label_encode(all_set, 'Deck')

compare(all_set, 'Deck')
all_set['Title'].value_counts()
dict(zip(all_set['Title'].unique(),['Rare' for x in range(len(all_set['Title'].unique()))]))
all_set['Title'] = all_set['Title'].map({

    'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',

    'Don': 'Rare', 'Rev': 'Rare', 'Dr': 'Rare', 'Mme': 'Rare',

    'Ms': 'Rare', 'Major': 'Rare', 'Lady': 'Rare', 'Sir': 'Rare',

    'Mlle': 'Rare', 'Col': 'Rare', 'Capt': 'Rare', 'the Countess': 'Rare',

    'Jonkheer': 'Rare', 'Dona': 'Rare'})
all_set = label_encode(all_set, 'Title')

compare(all_set, 'Title')
all_set['FarePerFamilyMember'] = all_set['Fare'] / all_set['Family']

all_set['FareBin'] = pd.qcut(all_set['Fare'], 5)

all_set['FareCode'] = all_set['FareBin']

all_set = label_encode(all_set, 'FareCode')

all_set['Fare'] = all_set['Fare'].apply(lambda x: np.log(x) if x > 0 else np.log(3.0))
fig, ax = plt.subplots(figsize=(14,14))

g = sns.heatmap(all_set[

    all_set.dtypes.reset_index()[ all_set.dtypes.reset_index()[0] != 'object']['index'].unique()

].corr(), annot=True, cmap='coolwarm')
all_set.profile_report()
# 0.76555

# all_tmp = all_set.drop(columns=['Age', 'Cabin', 'Fare', 'Name', 'Surname', 'Ticket', 'PassengerId', 'Embarked', 'SibSp', 'Parch'])

# all_tmp = all_set[corr[corr['Survived'] > 0.02]['index'].unique()]



# The following features will be dropped based on previous tests.

all_tmp = all_set.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'AgeBin', 'AgeCode', 'AgeDist', 'FareBin', 'FareCode'])
all_set.columns
corrmat(all_tmp.corr(), inflate=False)

plt.show()
all_dum = pd.get_dummies(all_tmp, drop_first=True)

print(all_dum.shape)



train_set = all_dum[all_dum['Survived'] < 3].copy()

test_set = all_dum[all_dum['Survived'] == 3].copy()
X = train_set.copy()

try:

    X.drop(columns=['Survived'], inplace=True)

except:

    pass

y = train_set['Survived'].copy()



scaler = StandardScaler().fit(X)

scl_X = scaler.transform(X)



tX = test_set.copy()

try:

    tX.drop(columns=['Survived'], inplace=True)

except:

    pass



scl_tX = scaler.transform(tX)
train_X, val_X, train_y, val_y = train_test_split(scl_X, y, test_size=0.33, random_state=42, stratify=y)
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(train_X, train_y)

score = rfc.score(val_X, val_y)

print('RandomForestClassifier =', score)

pred_y = rfc.predict(val_X)

target_names = ['DEAD', 'SURVIVED']

print(classification_report(val_y, pred_y,target_names=target_names))
best_model = 0

best_score = score
clfs = []

clfs.append(('ada', AdaBoostClassifier))

clfs.append(('bag', BaggingClassifier))

clfs.append(('rnd', RandomForestClassifier))

clfs.append(('knn', KNeighborsClassifier))

clfs.append(('mlp', MLPClassifier))

clfs.append(('ext', ExtraTreesClassifier))

clfs.append(('log', LogisticRegression))

clfs.append(('gbm', GradientBoostingClassifier))

    

params = []

params.append({'n_estimators': np.arange(10,500,10), 'learning_rate':[float(x/100.) for x in np.arange(1,10)]})

params.append({'n_estimators': np.arange(10,500,10)})

params.append({'n_estimators': np.arange(10,500,10)})

params.append({'n_neighbors': np.arange(3,15)})

params.append({'hidden_layer_sizes': [(100,),(200,),(300,),(400,),(500,)]})

params.append({'n_estimators': np.arange(10,200,10)})

params.append({'max_iter': np.arange(10,500,10)})

params.append({'n_estimators': np.arange(10,500,10), 'learning_rate':[float(x/100.) for x in np.arange(1,10)], 'max_depth':np.arange(3,10)})
best_estimators = []

best_params = []

estimators_weights = []

k = len(params)

for idx in range(len(clfs)):

    gs = RandomizedSearchCV(clfs[idx][1](), params[idx], cv=5)

    gs.fit(train_X, train_y)

    

    estimators_weights.append(gs.score(val_X, val_y))

    best_estimators.append(gs.best_estimator_)

    best_params.append(gs.best_params_)

    

    print(k, clfs[idx][0], gs.best_params_)

    k -= 1
estimator_list = list(zip([name for (name, model) in clfs], best_estimators))
vclf = VotingClassifier(estimator_list, voting='hard', weights=estimators_weights, n_jobs=-1)

vclf.fit(train_X, train_y)

score = vclf.score(val_X, val_y)



print('VotingClassifier =', score)



if score > best_score:

    best_model = 1

    best_score = score
pred_y = vclf.predict(val_X)

target_names = ['DEAD', 'SURVIVED']

print(classification_report(val_y, pred_y,target_names=target_names))
fit_X = np.zeros((train_y.shape[0], len(best_estimators)))

fit_X = pd.DataFrame(fit_X)



pred_X = np.zeros((val_y.shape[0], len(best_estimators)))

pred_X = pd.DataFrame(pred_X)



test_X = np.zeros((scl_tX.shape[0], len(best_estimators)))

test_X = pd.DataFrame(test_X)



print("Fitting models.")

cols = list()

for i, (name, m) in enumerate(estimator_list):

    print("%s..." % name, end=" ", flush=False)

    

    fit_X.iloc[:, i] = m.predict_proba(train_X)[:, 1]

    pred_X.iloc[:, i] = m.predict_proba(val_X)[:, 1]

    test_X.iloc[:, i] = m.predict_proba(scl_tX)[:, 1]

    

    cols.append(name)

    print("done")



fit_X.columns = cols

pred_X.columns = cols

test_X.columns = cols
corrmat(pred_X.corr(), inflate=False)

plt.show()
corrmat(test_X.corr(), inflate=False)

plt.show()
meta_estimator = GradientBoostingClassifier()

meta_params = {'n_estimators': np.arange(10,500,10), 'learning_rate':[float(x/100.) for x in np.arange(1,10)], 'max_depth':np.arange(3,10)}



meta_estimator = RandomizedSearchCV(GradientBoostingClassifier(), meta_params, cv=5)

meta_estimator.fit(fit_X, train_y)



score = meta_estimator.score(pred_X, val_y)



print('MetaEstimator =', score)



if score > best_score:

    best_model = 2

    best_score = score
pred_y = meta_estimator.predict(pred_X)

target_names = ['DEAD', 'SURVIVED']

print(classification_report(val_y, pred_y,target_names=target_names))
models = ['RandomForestClassifier', 'VotingClassifier', 'MetaEstimator']

print('Best model = ', models[best_model])
if best_model == 0:

    predictions = rfc.predict(scl_tX)

elif best_model == 1:

    predictions = vclf.predict(scl_tX)

else:

    predictions = meta_estimator.predict(test_X)



PassengerId = test_df['PassengerId'].values



results = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': predictions })

results.to_csv('results.csv', index=False)
!head -n 10 results.csv