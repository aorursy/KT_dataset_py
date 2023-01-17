import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split, GridSearchCV



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

import xgboost as xgb



from sklearn.metrics import accuracy_score

from sklearn.metrics import average_precision_score

from sklearn.metrics import auc

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import hamming_loss

from sklearn.metrics import hinge_loss

from sklearn.metrics import jaccard_score

from sklearn.metrics import make_scorer

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import plot_precision_recall_curve

from sklearn.metrics import plot_roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('precision', 2)
T = pd.read_csv('/kaggle/input/titanic/train.csv', index_col=0)
S = pd.read_csv('/kaggle/input/titanic/test.csv', index_col=0)
print('train.csv has', T.shape[0],'rows and', T.shape[1], 'columns')

print('test.csv has', S.shape[0],'rows and', S.shape[1], 'columns')
T.columns.tolist()
S.columns.tolist()
T.head()
T.tail()
S.head()
T.info()
S.info()
T.isnull().sum()
S.isnull().sum()
np.round(T.isnull().sum() * 100 / T.shape[0], 2)
np.round(S.isnull().sum() * 100 / S.shape[0], 2)
T.nunique()
S.nunique()
np.sort(T['Parch'].unique()).tolist()
np.sort(S['Parch'].unique()).tolist()
sns.set_style('white')

g = sns.jointplot(x='Age', y='Fare', data=T, color='black', alpha=0.1, height=8, marginal_kws={'kde':False})

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Scatterplot of Fare and Age')

plt.show()
pd.set_option('precision', 3)
T['Age'].corr(T['Fare']).round(3)
pd.set_option('precision', 2)
T['Survived'].value_counts(normalize=True, dropna=False)
plt.figure(figsize=(6,4))

g = sns.countplot(T['Survived'], color='grey').set_title('Count of passengers by Survived')
T['Pclass'].value_counts(normalize=True, dropna=False).sort_index()
g = T.groupby('Pclass')['Survived'].value_counts(dropna=False).unstack().fillna(0).plot(kind='bar', rot=0, stacked=True, legend=True, figsize=(6, 4), title='Count of passengers by Pclass')
pd.crosstab(index=T['Pclass'], columns=T['Survived'], margins=True, normalize='index')
T['Sex'].value_counts(normalize=True, dropna=False).sort_index()
g = T.groupby('Sex')['Survived'].value_counts(dropna=False).unstack().fillna(0).plot(kind='bar', rot=0, stacked=True, legend=True, figsize=(6, 4), title='Count of passengers by sex')
pd.crosstab(index=T['Sex'], columns=T['Survived'], margins=True, normalize='index')
pd.set_option('precision', 3)
T['Embarked'].value_counts(normalize=True, dropna=False).sort_index()
pd.set_option('precision', 2)
g = T.groupby('Embarked')['Survived'].value_counts(dropna=False).unstack().fillna(0).plot(kind='bar', rot=0, stacked=True, figsize=(6, 4), legend=True, title='Count of passengers by Embarked')
pd.crosstab(index=T['Embarked'], columns=T['Survived'], margins=True, normalize='index', dropna=False)
T[['Age', 'Fare', 'SibSp', 'Parch']].describe()
T.groupby('Survived')[['Age', 'Fare', 'SibSp', 'Parch']].mean()
plt.figure(figsize=(6,4))

g = sns.distplot(T['Age'], kde=False, color='black').set_title('Histogram of Age')
g = sns.FacetGrid(T, col='Survived', col_order=[0, 1], height=5, aspect=1)

g.map(plt.hist, 'Age', color='grey')

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Histogram of Age by Survived')

plt.show()
plt.figure(figsize=(6,4))

g = sns.boxplot(x='Survived', y='Age', data=T, color='grey').set_title('Boxplot of Age by Survived')
plt.figure(figsize=(6,4))

g = sns.distplot(T['Fare'], kde=False, color='black').set_title('Histogram of Fare')
plt.figure(figsize=(6,4))

g = sns.distplot(T['Fare']**(1/3), kde=False, color='black').set_title('Histogram of cube root of Fare')
g = sns.FacetGrid(T, col='Survived', col_order=[0, 1], height=5, aspect=1)

g.map(plt.hist, 'Fare', color='grey')

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Histogram of Fare by Survived')

plt.show()
plt.figure(figsize=(6,4))

g = sns.boxplot(x='Survived', y='Fare', data=T, color='grey').set_title('Boxplot of Fare by Survived')

plt.yscale('symlog')

plt.ylabel('Fare (symmetrical log scale)')

plt.show()
T['SibSp'].value_counts(dropna=False).sort_index()
plt.figure(figsize=(6,4))

g = sns.distplot(T['SibSp'], color='black', kde=False).set_title('Histogram of SibSp')
g = sns.FacetGrid(T, col='Survived', col_order=[0, 1], height=5, aspect=1)

g.map(plt.hist, 'SibSp', color='grey')

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Histogram of SibSp by Survived')

plt.show()
T['Parch'].value_counts(dropna=False).sort_index()
plt.figure(figsize=(6,4))

g = sns.distplot(T['Parch'], color='black', kde=False).set_title('Histogram of Parch')
g = sns.FacetGrid(T, col='Survived', col_order=[0, 1], height=5, aspect=1)

g.map(plt.hist, 'Parch', color='grey')

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Histogram of Parch by Survived')

plt.show()
pd.crosstab(columns=T['Parch'], index=T['SibSp'], margins=True)
def derive_ParchSibSp(x):

    if (x['Parch'] == 0 & x['SibSp'] == 0):

        return 0

    elif (x['Parch'] >= 1 & x['SibSp'] == 0):

        return 1

    elif (x['Parch'] == 0 & x['SibSp'] >= 1):

        return 2

    elif (x['Parch'] >= 1 & x['SibSp'] >= 1):

        return 3

    else:

        return None
T['ParchSibSp'] = T.apply(derive_ParchSibSp, axis=1)
S['ParchSibSp'] = S.apply(derive_ParchSibSp, axis=1)
T['ParchSibSp'].value_counts(normalize=True, dropna=False).sort_index()
S['ParchSibSp'].value_counts(normalize=True, dropna=False).sort_index()
T['Deck'] = T['Cabin'].str.slice(0,1,1)
S['Deck'] = S['Cabin'].str.slice(0,1,1)
pd.set_option('precision', 4)
T['Deck'].value_counts(normalize=True, dropna=False).sort_index()
S['Deck'].value_counts(normalize=True, dropna=False).sort_index()
pd.set_option('precision', 2)
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'
def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Master']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady', 'Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title == 'Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
T['Title'] = T['Name'].map(lambda x: get_title(x))
S['Title'] = S['Name'].map(lambda x: get_title(x))
T['Title'] = T.apply(replace_titles, axis=1)
S['Title'] = S.apply(replace_titles, axis=1)
T['Title'].value_counts(normalize=True, dropna=False).sort_index()
S['Title'].value_counts(normalize=True, dropna=False).sort_index()
pd.crosstab(columns=T['Sex'], index=T['Title'], margins=True)
pd.crosstab(columns=S['Sex'], index=S['Title'], margins=True)
T.head()
S.head()
age_mean = T['Age'].mean()
def ImputeMissingAgeWithMean(x):

    Age = x['Age']

    if np.isnan(Age):

        return age_mean

    else:

        return Age
T['Age_imp'] = T.apply(ImputeMissingAgeWithMean, axis=1)
T[['Age', 'Age_imp']].describe()
S['Age_imp'] = S.apply(ImputeMissingAgeWithMean, axis=1)
S[['Age', 'Age_imp']].describe()
pd.set_option('precision', 1)
T.loc[6:6, ['Age', 'Age_imp']]
S.loc[902:902, ['Age', 'Age_imp']]
# age_mean = T['Age'].mean()

# age_std = T['Age'].std()

# age_min = T['Age'].min()

# age_max = T['Age'].max()
# def ImputeMissingAgeWithDist(x):

#     age = x['Age']

#     if np.isnan(age):

#         imputed_age = np.random.normal(age_mean, age_std)

#         if imputed_age < age_min:

#             return age_min

#         elif imputed_age > age_max:

#             return age_max

#         else:

#             return imputed_age

#     else:

#         return age
# T['Age_imp'] = T.apply(ImputeMissingAgeWithDist, axis=1)
# S['Age_imp'] = S.apply(ImputeMissingAgeWithDist, axis=1)
fare_mean = T['Fare'].mean()
def ImputeMissingFare(x):

    Fare = x['Fare']

    if np.isnan(Fare):

        return fare_mean

    else:

        return Fare
S['Fare_imp'] = S.apply(ImputeMissingFare, axis=1)
pd.set_option('precision', 2)
S.loc[1044:1044, ['Fare', 'Fare_imp']]
T.loc[62:62, ['Embarked']]
Embarked_mode = T['Embarked'].mode()[0]
Embarked_mode
T['Embarked'].fillna(Embarked_mode, inplace = True)
T.loc[62:62, ['Embarked']]
y = T['Survived']
y.head()
categorical_inputs = ['Pclass', 'Sex', 'Embarked', 'Title', 'ParchSibSp']

continuous_inputs = ['Age_imp', 'SibSp', 'Parch', 'Fare']
T_categorical = T[categorical_inputs]

T_continuous = T[continuous_inputs]
S_categorical = S[categorical_inputs]

S_continuous = S[continuous_inputs]
T_cat_1hot = pd.get_dummies(data=T_categorical, drop_first=False)
T_cat_1hot.head(10)
T[categorical_inputs].head(10)
S_cat_1hot = pd.get_dummies(data=S_categorical, drop_first=False)
S_cat_1hot.head(10)
S[categorical_inputs].head(10)
X = T_cat_1hot.join(T_continuous)
X.head()
X_score = S_cat_1hot.join(S_continuous)
X_score.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
print(X_train.shape)

print(y_train.shape)



print(X_test.shape)

print(y_test.shape)



print(X_score.shape)
pd.Series(y_train).value_counts(normalize=True)
pd.Series(y_test).value_counts(normalize=True)
prior_proba = pd.Series(y_train).value_counts(normalize=True)[1]
print('Prior probability = {0:.4f}'.format(prior_proba))
class EstimatorSelectionHelper:



    def __init__(self, models, params):

        if not set(models.keys()).issubset(set(params.keys())):

            missing_params = list(set(models.keys()) - set(params.keys()))

            raise ValueError("Some estimators are missing parameters: %s" % missing_params)

        self.models = models

        self.params = params

        self.keys = models.keys()

        self.grid_searches = {}



    def fit(self, X, y, cv=5, n_jobs=-1, verbose=1, scoring=None, refit=False, return_train_score=False):

        for key in self.keys:

            print("Running GridSearchCV for %s." % key)

            model = self.models[key]

            params = self.params[key]

            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,

                              verbose=verbose, scoring=scoring, refit=refit,

                              return_train_score=return_train_score)

            gs.fit(X,y)

            self.grid_searches[key] = gs    



    def score_summary(self, sort_by='mean_score'):

        def row(key, scores, params):

            d = {

                 'estimator': key,

                 'min_score': min(scores),

                 'max_score': max(scores),

                 'mean_score': np.mean(scores),

                 'std_score': np.std(scores),

            }

            return pd.Series({**params,**d})



        rows = []

        for k in self.grid_searches:

            params = self.grid_searches[k].cv_results_['params']

            scores = []

            for i in range(self.grid_searches[k].cv):

                key = "split{}_test_score".format(i)

                r = self.grid_searches[k].cv_results_[key]        

                scores.append(r.reshape(len(params),1))



            all_scores = np.hstack(scores)

            for p, s in zip(params,all_scores):

                rows.append((row(k, s, p)))



        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)



        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']

        columns = columns + [c for c in df.columns if c not in columns]



        return df[columns]
models = {'Extra Trees': ExtraTreesClassifier(random_state=1),

          'Random Forest': RandomForestClassifier(random_state=1),

          'AdaBoost': AdaBoostClassifier(random_state=1, algorithm='SAMME.R'),

          'Gradient Boosting': GradientBoostingClassifier(random_state=1, loss='deviance', criterion='friedman_mse'),

          'SVC': SVC(random_state=1, max_iter=-1),

          'XGBoost': xgb.XGBClassifier(base_score=prior_proba, 

                                       booster='gbtree',

                                       colsample_bylevel=1,

                                       colsample_bynode=1,

                                       importance_type='gain', 

                                       max_delta_step=0, 

                                       missing=None,  

                                       n_estimators=30,  

                                       n_jobs=-1, 

                                       nthread=None,  

                                       objective='binary:logistic', 

                                       random_state=0, 

                                       reg_alpha=0,  

                                       reg_lambda=1,

                                       scale_pos_weight=1,

                                       seed=None,

                                       silent=None, 

                                       subsample=1, 

                                       tree_method='auto')}



params = {'Extra Trees': [{'criterion': ['gini', 'entropy'],

                           'n_estimators': [10, 20, 30],

                           'max_depth' : [3, 4, 5, 6]}],

          'Random Forest': [{'criterion': ['gini', 'entropy'],

                             'n_estimators': [20, 30, 50, 100],

                             'max_depth' : [3, 4, 5, 6]}],

          'AdaBoost': {'n_estimators': [20, 30, 50, 100],

                       'learning_rate':[0.5, 1, 1.5]},

          'Gradient Boosting': {'n_estimators': [50, 100, 150],

                                'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],

                                'max_depth' : [3, 4, 5, 6]},

          'SVC': [{'kernel': ['rbf', 'linear', 'sigmoid'],

                   'C': [0.5, 1, 10],

                   'gamma': ['scale', 'auto']}],

          'XGBoost': {'learning_rate' : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],

                      'max_depth' : [3, 4, 5, 6], 

                      'min_child_weight' : [1, 3, 5, 7],

                      'gamma' : [0.0, 0.1, 0.2, 0.3, 0.4],

                      'colsample_bytree' : [0.3, 0.4, 0.5, 0.7, 1]}}
helper = EstimatorSelectionHelper(models, params)

helper.fit(X_train, y_train, verbose=0, cv=5, refit=True, scoring='roc_auc', n_jobs=-1)
GridSearchSummary = helper.score_summary()
pd.set_option('precision', 4)
GridSearchSummary.head(20)
sns.set_style('whitegrid')

plt.figure(figsize=(10,10))

sns.scatterplot(data=GridSearchSummary, x='mean_score', y='std_score', hue='estimator', s=50, alpha=0.3)

plt.legend(loc='best', fontsize=16, title=None)

plt.title('Grid search summary', fontsize=16)

plt.xlim(left=0.5, right=1)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel('Mean AUC', fontsize=16)

plt.ylabel('Standard deviation of AUC', fontsize=16)

plt.show()
GridSearchSummary.iloc[0:2][['estimator', 'min_score', 'mean_score', 'max_score', 'std_score', 'learning_rate', 'gamma', 'colsample_bytree', 'max_depth', 'min_child_weight']]
model = xgb.XGBClassifier(base_score=prior_proba,

                          booster='gbtree', 

                          colsample_bylevel=1,

                          colsample_bynode=1, 

                          colsample_bytree=0.5, 

                          gamma=0,

                          importance_type='gain',

                          learning_rate=0.25,

                          max_delta_step=0, 

                          max_depth=6,

                          min_child_weight=5, 

                          missing=None, 

                          n_estimators=30, 

                          n_jobs=-1,

                          nthread=None, 

                          objective='binary:logistic', 

                          random_state=0,

                          reg_alpha=0, 

                          reg_lambda=1, 

                          scale_pos_weight=1,

                          seed=None,

                          silent=None, 

                          subsample=1, 

                          tree_method='auto',

                          verbosity=1)
model.fit(X=X_train, y=y_train, eval_metric="auc", eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=5, verbose=True)
model.get_params()
print('Number of boosting rounds = {}'.format(model.get_num_boosting_rounds()))
model_last_iter = len(model.evals_result()['validation_0']['auc'])

print('Last iteration = {}'.format(model_last_iter))
print('Best iteration = {}'.format(model.best_ntree_limit - 1))
pd.set_option('precision', 4)
AUC = pd.concat([pd.DataFrame.from_dict(data=model.evals_result()['validation_0'], orient='columns'),

                 pd.DataFrame.from_dict(data=model.evals_result()['validation_1'], orient='columns')], axis=1)



AUC.columns = ['Training AUC', 'Validation AUC']

AUC.index.name = 'Iteration'
AUC
AUC.iloc[model.best_ntree_limit - 1]
plt.figure(figsize=(10, 8))

plt.title("AUC scores by data partition\nDotted vertical line shows best iteration", fontsize='x-large')

plt.plot(np.arange(1,model_last_iter+1), model.evals_result()['validation_0']['auc'], color="b", linestyle="-", label="Training")

plt.plot(np.arange(1,model_last_iter+1), model.evals_result()['validation_1']['auc'], color="r", linestyle="-", label="Validation")

plt.ylabel("AUC score", fontsize='x-large')

plt.xlabel("Iteration", fontsize='x-large')

plt.xticks((np.arange(model_last_iter)+1), fontsize='x-large')

plt.yticks(fontsize='x-large')

plt.legend(loc='best', frameon=True, fontsize='x-large')

plt.grid(True)

plt.axvline(x=model.best_ntree_limit, color='black', linestyle='--')

plt.show()
auc_train = plot_roc_curve(model, X_train, y_train, color='blue', label='Training AUC = '+str(AUC.iloc[model.best_ntree_limit - 1][0].round(2)))

auc_test = plot_roc_curve(model, X_test, y_test, color='red', label='Validation AUC = '+str(AUC.iloc[model.best_ntree_limit - 1][1].round(2)), ax=auc_train.ax_)

auc_test.figure_.suptitle("ROC curve comparison", fontsize='x-large')

plt.xticks(fontsize='x-large')

plt.yticks(fontsize='x-large')

plt.xlabel('False Positive Rate', fontsize='x-large')

plt.ylabel('True Positive Rate', fontsize='x-large')

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.legend(loc='lower right', frameon=True, fontsize='x-large')

plt.show()
xgb.to_graphviz(model, num_trees=model.best_ntree_limit, yes_color='#00cc00', no_color='#FF0000', condition_node_params={'shape': 'box', 'style': 'solid'})
plt.rcParams["figure.figsize"] = (10, 8)

g = xgb.plot_importance(model.get_booster(), grid=False, height=0.8, color='grey', importance_type='weight')

plt.ylabel('Features', fontsize='x-large')

plt.xlabel('F score', fontsize='x-large')

plt.title('Variable importance\n(Number of times a feature appears in a tree)', fontsize='x-large')

plt.yticks(fontsize='x-large')

plt.xticks(fontsize='x-large')

plt.show()
y_train_scores_0_1 = model.predict_proba(X_train, ntree_limit=model.best_ntree_limit)

y_test_scores_0_1 = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)
type(y_train_scores_0_1)
print(y_train_scores_0_1.shape)

print(y_test_scores_0_1.shape)
y_train_scores_0_1[0:5]
y_test_scores_0_1[0:5]
y_train_scores = y_train_scores_0_1[:,1]

y_test_scores = y_test_scores_0_1[:,1]
y_train_scores_df = pd.DataFrame(y_train_scores, index=X_train.index, columns=['Probability of Survived = 1'])

y_train_scores_df['Partition'] = 'Training'



y_test_scores_df = pd.DataFrame(y_test_scores, index=X_test.index, columns=['Probability of Survived = 1'])

y_test_scores_df['Partition'] = 'Validation'



y_scores = pd.concat([y_train_scores_df, y_test_scores_df]).sort_index()
y_scores.groupby(['Partition']).describe()
sns.set_style("white")

g = sns.FacetGrid(y_scores, col='Partition', col_order=['Training', 'Validation'], height=5, aspect=1, margin_titles=True)

g.map(sns.distplot, 'Probability of Survived = 1', kde=False, bins=10, color='black')

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Histogram of predicted probability of Survived = 1 by partition', fontsize=16)

plt.show()
sns.set_style("whitegrid")

g = sns.FacetGrid(y_scores, col='Partition', col_order=['Training', 'Validation'], height=5, aspect=1, margin_titles=True, xlim=(y_scores['Probability of Survived = 1'].min(), y_scores['Probability of Survived = 1'].max()-0.008))

g.map(plt.hist, 'Probability of Survived = 1', cumulative=True, histtype='step', density=1, alpha=0.5, linewidth=2, bins=200, color='black')

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Cumulative distribution function of predicted probabilities', fontsize=16)

plt.show()
def adjusted_classes(y_train_scores, t):

    """

    Adjusts class predictions based on the prediction threshold (t).

    Will only work for binary classification problems.

    """

    return [1 if y >= t else 0 for y in y_train_scores]
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.DataFrame(columns=['Threshold', 'TN', 'FP', 'FN', 'TP', 'Precision', 'Recall', 'Accuracy', 'F1', 'Jaccard score','Hamming loss', 'Hinge loss'], dtype='float')
for i in np.arange(0, 1000, 1):

    threshold = float('0.'+str(i).zfill(3))

    pred = adjusted_classes(y_train_scores, threshold)

    df.loc[i] = {'Threshold' : threshold,

                 'TN' : confusion_matrix(y_train, pred).ravel()[0],

                 'FP' : confusion_matrix(y_train, pred).ravel()[1],

                 'FN' : confusion_matrix(y_train, pred).ravel()[2],

                 'TP' : confusion_matrix(y_train, pred).ravel()[3],

                 'Precision' : precision_score(y_train, pred, pos_label=1, average='binary', zero_division=0),

                 'Recall' : recall_score(y_train, pred, pos_label=1, average='binary'),

                 'Accuracy' : accuracy_score(y_train, pred),

                 'F1' : f1_score(y_train, pred, pos_label=1, average='binary'),

                 'Jaccard score' : jaccard_score(y_train, pred, pos_label=1, average='binary'),

                 'Hamming loss': hamming_loss(y_train, pred),

                 'Hinge loss': hinge_loss(y_train, pred)

                 }    
df['Threshold'] = df['Threshold'].astype(str).apply('{:0<5}'.format)

df['TN'] = df['TN'].astype(int)

df['FP'] = df['FP'].astype(int)

df['FN'] = df['FN'].astype(int)

df['TP'] = df['TP'].astype(int)

df['Absolute difference between precision and recall'] = abs(df['Precision'] - df['Recall'])
y_train_scores_min = str(np.round(y_train_scores.min(), 3))

y_train_scores_max = str(np.round(y_train_scores.max(), 3))
df = df.loc[(df['Threshold'] >= y_train_scores_min) & (df['Threshold'] < y_train_scores_max)]
df.tail()
df.loc[df['Threshold'] == '0.500']
df.loc[df['Threshold'] == str(np.round(prior_proba, 3))]
pd.DataFrame(df.sort_values(['Accuracy', 'Threshold'], ascending=(False, True)).iloc[0]).T
pd.DataFrame(df.sort_values(['Hamming loss', 'Threshold'], ascending=(True, True)).iloc[0]).T
pd.DataFrame(df.sort_values(['Hinge loss', 'Threshold'], ascending=(True, True)).iloc[0]).T
pd.DataFrame(df.sort_values(['F1', 'Threshold'], ascending=(False, True)).iloc[0]).T
pd.DataFrame(df.sort_values(['Jaccard score', 'Threshold'], ascending=(False, True)).iloc[0]).T
pd.DataFrame(df.sort_values(['Absolute difference between precision and recall', 'Threshold'], ascending=(True, True)).iloc[0]).T
plt.figure(figsize=(8, 8))

plt.plot(df['Threshold'].astype(float), df['TN'], label='True Negatives', linewidth=2)

plt.plot(df['Threshold'].astype(float), df['TP'], label='True Positives', linewidth=2)

plt.plot(df['Threshold'].astype(float), df['FN'], label='False Negatives', linewidth=2)

plt.plot(df['Threshold'].astype(float), df['FP'], label='False Positives', linewidth=2)

plt.title('Classification results as a function of the decision threshold', fontsize=16)

plt.ylabel("Count of passengers", fontsize=16)

plt.xlabel("Decision threshold", fontsize=16)

plt.ylim(top=500)

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.grid(True)

plt.legend(loc='upper center', fontsize=12, ncol=2, mode=None)

plt.show()
plt.figure(figsize=(8, 8))

plt.plot(df['Threshold'].astype(float), df['Precision'], color='Blue', label='Precision', linewidth=2)

plt.plot(df['Threshold'].astype(float), df['Recall'], color='Green', label='Recall', linewidth=2)

plt.title('Precision and recall as a function of the decision threshold', fontsize=16)

plt.ylabel("Score", fontsize=16)

plt.xlabel("Decision threshold", fontsize=16)

plt.xlim(0.1, 0.8)

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)  

plt.grid(True)

plt.legend(loc='best', fontsize=12, ncol=1, mode=None)

plt.show()
plt.figure(figsize=(8, 8))

plt.plot(df['Threshold'].astype(float), df['Accuracy'], color='black')

plt.title('Accuracy as a function of the decision threshold', fontsize=16)

plt.ylabel("Accuracy score", fontsize=16)

plt.xlabel("Decision threshold", fontsize=16)

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)  

plt.grid(True)

plt.show()
plt.figure(figsize=(8, 8))

plt.plot(df['Threshold'].astype(float), df['F1'], color='black')

plt.title('F1 as a function of the decision threshold', fontsize=16)

plt.ylabel("F1 score", fontsize=16)

plt.xlabel("Decision threshold", fontsize=16)

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)  

plt.grid(True)

plt.show()
plt.figure(figsize=(8, 8))

plt.plot(df['Threshold'].astype(float), df['F1'], color='black')

plt.title('Jaccard score as a function of the decision threshold', fontsize=16)

plt.ylabel("Jaccard score", fontsize=16)

plt.xlabel("Decision threshold", fontsize=16)

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)  

plt.grid(True)

plt.show()
def precision_recall_threshold(y_true, y_scores, precisions, recalls, thresholds, t):

    """

    Plots the precision recall curve and shows the current value for each

    at the classifier's threshold (t).

    """

    # Generate new class predictions based on the adjusted_classes

    # function above and view the resulting confusion matrix.

    

    y_pred = adjusted_classes(y_scores, t)

    

    print('\n')

    print('Decision threshold = {0:.4f}'.format(t))

    

    print('\n')

    print(pd.DataFrame(confusion_matrix(y_true, y_pred), 

                       columns=['Predicted Survived = 0', 'Predicted Survived = 1'],

                       index=['Actual Survived = 0', 'Actual Survived = 1']))

    

    print('\n')

    print(classification_report(y_true=y_true, y_pred=y_pred, digits=4))

    

    print('\n')

    print("Precision = {0:.4f}".format(precision_score(y_true, y_pred, pos_label=1, average='binary')))

    print("Recall = {0:.4f}".format(recall_score(y_true, y_pred, pos_label=1, average='binary')))

    print("Accuracy = {0:.4f}".format(accuracy_score(y_true, y_pred)))

    print("F1 score = {0:.4f}".format(f1_score(y_true, y_pred, pos_label=1, average='binary')))

    

    print('\n')

    

    recall_at_threshold = recalls[np.argmin(thresholds <= t)]

    precision_at_threshold = precisions[np.argmin(thresholds <= t)]

    

    print('Threshold', np.round(t, 4), 'achieves', np.round(recall_at_threshold, 4), 'recall and', np.round(precision_at_threshold, 4), 'precision')

    print('\n')

    

    # Plot the curve

    plt.figure(figsize=(8,8))

    plt.title('Precision and recall curve\nCurrent threshold set at '+str(np.round(t, 3)), fontsize=16)

    plt.plot(recalls, precisions, color='black', linestyle='-', linewidth=2)

    plt.grid(True)

    plt.ylim(top=1.05)

    plt.yticks(fontsize=14)

    plt.xticks(fontsize=14)

    plt.xlim(0, 1)

    plt.xlabel('Recall', fontsize=16)

    plt.ylabel('Precision', fontsize=16)

    

    # Plot the current threshold on the line

    plt.plot([recall_at_threshold, 0], [precision_at_threshold, precision_at_threshold], "r:")                            

    plt.plot([recall_at_threshold, recall_at_threshold], [0, precision_at_threshold], "r:")

    plt.plot([recall_at_threshold], [precision_at_threshold], "ro")                                         

    

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, t):

    """

    Plots precision and recall scores as a function of the decision threshold

    """

    recall_at_threshold = recalls[np.argmin(thresholds <= t)]

    precision_at_threshold = precisions[np.argmin(thresholds <= t)]

    

    print('Threshold', np.round(t, 4), 'achieves', np.round(recall_at_threshold, 4), 'recall and', np.round(precision_at_threshold, 4), 'precision')

    print('\n')

    

    plt.figure(figsize=(8, 8))

    plt.title("Precision and recall scores as a function of the decision threshold\nCurrent threshold set at "+str(np.round(t, 3)), fontsize=16)

    plt.plot(thresholds, precisions[:-1], color="blue", linestyle="-", linewidth=2, label="Precision")

    plt.plot(thresholds, recalls[:-1], color="green", linestyle="-", linewidth=2, label="Recall")

    plt.ylabel("Score", fontsize=16)

    plt.xlabel("Decision Threshold", fontsize=16)

    plt.ylim(top=1.05)

    plt.xlim(left=0.1)

    plt.yticks(fontsize=14)

    plt.xticks(fontsize=14)  

    plt.grid(True)

    plt.legend(loc='best', fontsize=16)

    plt.plot([t, t], [0., recall_at_threshold], "r:")

    plt.plot([0, t], [precision_at_threshold, precision_at_threshold], "r:")                            

    plt.plot([0, t], [recall_at_threshold, recall_at_threshold], "r:")

    plt.plot([t], [precision_at_threshold], "ro")                                         

    plt.plot([t], [recall_at_threshold], "ro")                             

    
p_train, r_train, thresholds_train = precision_recall_curve(y_train, y_train_scores)
precision_recall_threshold(y_train, y_train_scores, p_train, r_train, thresholds_train, t=0.419)
plot_precision_recall_vs_threshold(p_train, r_train, thresholds_train, t=0.419)
p_test, r_test, thresholds_test = precision_recall_curve(y_test, y_test_scores)
precision_recall_threshold(y_test, y_test_scores, p_test, r_test, thresholds_test, t=0.419)
plot_precision_recall_vs_threshold(p_test, r_test, thresholds_test, t=0.419)
y_score_0_1 = model.predict_proba(X_score, ntree_limit=model.best_ntree_limit)
y_score = y_score_0_1[:,1]
y_score_df = pd.DataFrame(y_score, index=X_score.index, columns=['Probability of Survived = 1'])
y_score_df.describe()
sns.set_style('white')

plt.figure(figsize=(6,4))

g = sns.distplot(y_score_df['Probability of Survived = 1'], bins=20, kde=False, color='black').set_title('Histogram of scores')
plt.hist(y_score_df['Probability of Survived = 1'], cumulative=True, histtype='step', density=1, alpha=0.5, linewidth=2, bins=200, color='black')

plt.title('Cumulative distribution function of predicted probabilities', fontsize=16)

plt.xlim(y_score_df['Probability of Survived = 1'].min(), y_score_df['Probability of Survived = 1'].max()-0.001)

plt.grid(True)

plt.show()
y_score_df['Survived'] = adjusted_classes(y_score, t=0.419)
y_score_df.head()
y_score_df['Survived'].to_csv('my_submission.csv', index=True, sep=',', header=True)