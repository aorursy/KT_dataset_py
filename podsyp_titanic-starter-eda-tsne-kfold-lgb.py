import numpy as np 

import pandas as pd 

import pandas_summary as ps

from category_encoders import WOEEncoder
# Lgbm

import lightgbm as lgb



# Sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression



# Hyper_opt

from hyperopt import hp

from hyperopt import fmin, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

import hyperopt.pyll

from hyperopt.pyll import scope



# Suppr warning

import warnings

warnings.filterwarnings("ignore")



# Plots

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import rcParams



# Others

import shap
folder = '../input/titanic/'

train_df = pd.read_csv(folder + 'train.csv')

test_df = pd.read_csv(folder + 'test.csv')

sub_df = pd.read_csv(folder + 'gender_submission.csv')
print('train')

print('All: ', train_df.shape)

print('test')

print('All: ', test_df.shape)

print('sub')

print('sub ', sub_df.shape)
train_df.head()
test_df.head()
sub_df.head()
train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if str(x) == 'male' else 0)

test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if str(x) == 'male' else 0)
train_df.drop('PassengerId', axis=1, inplace=True)

test_df.drop('PassengerId', axis=1, inplace=True)
# Check correlation
train_corr = train_df.corr()

# plot the heatmap and annotation on it

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(train_corr, xticklabels=train_corr.columns, yticklabels=train_corr.columns, annot=True, ax=ax);
test_corr = test_df.corr()

# plot the heatmap and annotation on it

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(test_corr, xticklabels=test_corr.columns, yticklabels=test_corr.columns, annot=True, ax=ax);
y = train_df['Survived']

train_df.drop('Survived', axis=1, inplace=True)
y.value_counts()
y.value_counts(normalize=True)
y.hist(bins=2);
dfs = ps.DataFrameSummary(train_df)

print('categoricals: ', dfs.categoricals.tolist())

print('numerics: ', dfs.numerics.tolist())

dfs.summary()
dfs = ps.DataFrameSummary(test_df)

print('categoricals: ', dfs.categoricals.tolist())

print('numerics: ', dfs.numerics.tolist())

dfs.summary()
train_df.hist(figsize=(25, 20));
sns.pairplot(data=train_df);
test_df.hist(figsize=(25, 20));
sns.pairplot(data=test_df);
sns.barplot(data=train_df, x='Pclass', y=y);
sns.barplot(data=train_df, x='Sex', y=y);
sns.barplot(data=train_df, x='SibSp', y=y);
sns.barplot(data=train_df, x='Parch', y=y);
for data in (train_df, test_df):

    data['FamilySize'] = data['Parch'] + data['SibSp']
sns.barplot(data=train_df, x='FamilySize', y=y);
for data in (train_df, test_df):

    data['Parch'] = data['Parch'] / data['FamilySize']

    data['Parch'].fillna(-1, inplace=True)

    data['SibSp'] = data['SibSp'] / data['FamilySize']

    data['SibSp'].fillna(-1, inplace=True)
for data in (train_df, test_df):

    data['Fare'].fillna(train_df.groupby(['Embarked', 'Pclass'])['Fare'].transform('median'), inplace=True)



for data in (train_df, test_df):

    data['Embarked'].fillna(train_df['Embarked'].mode(), inplace=True)
class AgeFeature(BaseEstimator, TransformerMixin):

    '''AgeFeature - works with df only'''

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        # sex, name

        X['Initial'] = 0

        for i in X:

            X['Initial'] = X.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

    

        X['Initial'].replace(

            ['Dona','Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],

            ['Miss','Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],

            inplace=True

        )

        print(X.groupby('Initial')['Age'].median()) # lets check the average age by Initials



        ## Assigning the NaN Values with the Ceil values of the median ages

        X.loc[(X.Age.isnull()) & (X.Initial=='Mr'), 'Age'] = 30

        X.loc[(X.Age.isnull()) & (X.Initial=='Mrs'), 'Age'] = 35

        X.loc[(X.Age.isnull()) & (X.Initial=='Master'), 'Age'] = 3.5

        X.loc[(X.Age.isnull()) & (X.Initial=='Miss'), 'Age'] = 21.5

        X.loc[(X.Age.isnull()) & (X.Initial=='Other'), 'Age'] = 51

        return X['Age'].as_matrix().reshape(-1, 1)



    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)



ager = AgeFeature().fit(train_df)

train_df['Age'] = ager.transform(train_df)

test_df['Age'] = ager.transform(test_df)
for data in (train_df, test_df):

    data['Is_Married'] = 0

    data['Is_Married'].loc[data['Initial'] == 'Mrs'] = 1
train_df['Cabin'] = train_df['Cabin'].apply(lambda x: str(x)[0]).apply(lambda x: 'n' if x == 'T' else x)

test_df['Cabin'] = test_df['Cabin'].apply(lambda x: str(x)[0]).apply(lambda x: 'n' if x == 'T' else x)

train_df['Cabin'].unique(), test_df['Cabin'].unique()
woe_encoder = WOEEncoder(cols=['Cabin', 'Embarked', 'Initial'])

woe_encoder.fit(train_df, y)

train_df = woe_encoder.transform(train_df)

test_df = woe_encoder.transform(test_df)
train_df.drop(['Name', 'Ticket'], axis=1, inplace=True)

test_df.drop(['Name', 'Ticket'], axis=1, inplace=True)
train_df.columns.to_list()
train_df.columns.to_list()
dfs = ps.DataFrameSummary(train_df)

dfs.summary()
dfs = ps.DataFrameSummary(test_df)

dfs.summary()
scaler = StandardScaler().fit(train_df)

scaled_train_df = scaler.transform(train_df)

scaled_test_df = scaler.transform(test_df)
tsne = TSNE(perplexity=30)



train_tsne_transformed = tsne.fit_transform(scaled_train_df)

test_tsne_transformed = tsne.fit_transform(scaled_test_df)
plt.figure(figsize=(10, 10))

plt.scatter(train_tsne_transformed[:, 0], train_tsne_transformed[:, 1], c=y);
plt.figure(figsize=(10, 10))

plt.scatter(test_tsne_transformed[:, 0], test_tsne_transformed[:, 1]);
# different between train and test penalty power:

p = 0.9

# number of hyperopt iterarions

k = 350

cat=[]

models = dict()

trains = []

preds = []

folds = []

valid_scores =dict()

skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=7)



for num, spl in enumerate(skf.split(train_df, y)):

    train_index, val_index = spl[0], spl[1]

    x_train_1, x_valid_1 = train_df.iloc[train_index, :], train_df.iloc[val_index, :]

    y_train_1, y_valid_1 = y.iloc[train_index], y.iloc[val_index]

    

    print('fold: ', num+1)

    

    def score(params):

        w=[]

        best_iter = []

        skf_1 = StratifiedKFold(n_splits=2, shuffle=True, random_state=77)



        for train_index_2, val_index_2 in skf_1.split(x_train_1, y_train_1):

            x_train_2, x_valid_2 = x_train_1.iloc[train_index_2, :], x_train_1.iloc[val_index_2, :]

            y_train_2, y_valid_2 = y_train_1.iloc[train_index_2], y_train_1.iloc[val_index_2]

            train_data = lgb.Dataset(x_train_2, label=y_train_2,)

            val_data = lgb.Dataset(x_valid_2, label=y_valid_2, reference=train_data)

            gbm = lgb.train(params, train_data, valid_sets = [train_data, val_data], valid_names=['train', 'val'],

                            num_boost_round = 5800, verbose_eval = False,)

            w.append([gbm.best_score['train']['auc'], gbm.best_score['val']['auc']])

            best_iter.append(gbm.best_iteration)

        nrounds = np.mean(best_iter)

        res = list(np.mean(w, axis=0))

        return {'loss': -res[1]+np.power(np.square(res[0]-res[1]), p), 'status': STATUS_OK, 

                'mean_auc_train': res[0], 'mean_auc_test': res[1], 'best_iter': int(nrounds)}



    def optimize(trials):

        space = {

        #'max_depth': hp.choice('max_depth', [-1, 6, 7]),

        'max_depth': -1,

        'max_bin': scope.int(hp.uniform('max_bin', 2, 200)),

        'num_leaves': scope.int(hp.uniform('num_leaves', 2, 16)),

        'min_data_in_leaf': scope.int(hp.uniform('min_data_in_leaf', 2, 75)),

        'lambda_l1': hp.quniform('lambda_l1', 0, 6.5, 0.25),

        'lambda_l2': hp.quniform('lambda_l2', 0, 10.5, 0.25),

        'learning_rate': hp.quniform('learning_rate', 0.01 , 0.05, 0.005),

        'min_child_weight': hp.quniform('min_child_weight', 0.05, 25.95, 0.05),

        'feature_fraction': hp.quniform('feature_fraction', 0.4, 0.8, 0.1),

        'bagging_fraction': hp.quniform('bagging_fraction', 0.45, 0.95, 0.05),

        'metric': ('auc',),

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'nthread': 8,

        'early_stopping_rounds': 10,

        'silent':1,

        }

        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=k)

        #print(best)

        

    trials = Trials()

    optimize(trials)

    params_for = trials.best_trial['misc']['vals']

    params_for['num_leaves'] = int(params_for['num_leaves'][0])

    params_for['max_bin'] = int(params_for['max_bin'][0])

    params_for['min_data_in_leaf'] = int(params_for['min_data_in_leaf'][0])

    params_for['objective'] = 'binary'

    params_for['metric'] = {'auc'}

    

    train_data = lgb.Dataset(x_train_1, label=y_train_1)

    val_data = lgb.Dataset(x_valid_1, label=y_valid_1, reference=train_data)

    gbm_main = lgb.train(params_for, train_data, valid_sets = [train_data, val_data], valid_names=['train', 'val'],

                         num_boost_round = 10000, verbose_eval = 100, early_stopping_rounds = 20)

    trains.append(gbm_main.predict(train_df))

    preds.append(gbm_main.predict(test_df))

    folds.append(gbm_main.best_score['val']['auc'])

    print('best score: ', gbm_main.best_score['val']['auc'])

    print()

    valid_scores[num] = gbm_main.best_score['val']['auc']

    models['lgb_fold'+str(num+1)] = gbm_main
model = LogisticRegression(penalty='l1', solver='saga')

#pd.Series(np.mean(np.array(preds), axis=0))

model.fit(pd.DataFrame(np.array(trains)).T, y)

target = model.predict(pd.DataFrame(np.array(preds)).T)
sub_df['Survived'] = target
sub_df.to_csv("submission.csv", index=False)