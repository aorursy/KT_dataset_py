import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import sklearn.ensemble
import sklearn.model_selection
import sklearn.tree
pd.set_option('mode.chained_assignment', 'raise')
train_data = pd.read_csv('../input/train.csv')
train_data = train_data.set_index('PassengerId')
train_data.head()
survival_rate=train_data.Survived.mean()
print('There were {0} passengers aboard the Titanic.'.format(2344))
print('We have data on {0}.'.format(len(train_data)))
print('Their general survival rate was {rate:.1f}%.'.format(rate=survival_rate * 100))
print('If were to assign verdict "Dead" to all passengers, '
      'we would be right {rate:.1f}% of times.'.format(rate=(1 - survival_rate) * 100))
print("Let's see if we can do better!")
## Show columns with NaNs
for c in train_data.columns:
    total = len(train_data)
    if train_data[c].hasnans:
        non_nans = train_data[c].count()
        print('Column {0} has NaNs. It has {1} ({2:.1f}%) non-NaN entries.'.format(c, non_nans,
                                                                                  100 * non_nans / total))
train_data.loc[:, ['Survived']] = train_data.Survived.astype(bool)

def fill_nas(data):
    data = data.assign(HasCabin=data.Cabin.notna(),
                       HasAge=data.Age.notna(),
                       HasEmbarked=data.Embarked.notna())
    for str_column in ['Cabin', 'Embarked']:
        data[str_column].fillna('', inplace=True)
    data.Fare.fillna(0.0, inplace=True)
    data.Age.fillna(0.0, inplace=True)
    data.Cabin.fillna('', inplace=True)
    return data

train_data = fill_nas(train_data)
train_data.head()
# Show cabin data
np.unique(train_data.Cabin[train_data.Cabin.notna()].values)
def calculate_cabin_count(data):
    cabin_count = data.Cabin[data.HasCabin].str.count(' ') + 1
    no_cabin_index = data[~data.HasCabin].index
    return data.assign(CabinCount=pd.concat([pd.Series(data=0, index=no_cabin_index), cabin_count]))

train_data = calculate_cabin_count(train_data)
train_data.CabinCount.head()
def extract_cabin_info(data):
    cabin_data = data.Cabin.str.extract("(?P<CabinLetter>[A-Z])(?P<CabinNumber>[0-9]*)", expand=True)
    cabin_data.CabinLetter.fillna('', inplace=True)
    cabin_data.CabinNumber.fillna(0.0, inplace=True)
    cabin_data.loc[cabin_data.CabinNumber == '', 'CabinNumber'] = 0.0
    cabin_data.CabinNumber = pd.to_numeric(cabin_data.CabinNumber)
    return pd.concat([data, cabin_data], axis=1)

train_data = extract_cabin_info(train_data)
train_data.loc[:, ['CabinLetter', 'CabinNumber']].head()
def preprocess_input(data):
    data = fill_nas(data)
    data = calculate_cabin_count(data)
    return extract_cabin_info(data)
sns.set_palette('Set1')
plt.figure(figsize=(8, 8))
colors = sns.color_palette('Set1')
plt.pie(train_data.Survived.value_counts(), labels=['Deceased', 'Survived'],
       colors=colors[:2], autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(12, 8))
plot_data = train_data[['Sex', 'Survived']].copy()
plot_data.loc[:, 'Survived'] = plot_data.Survived.apply(lambda v: 'Survived' if v else 'Deceased')
sns.countplot(x="Sex", hue="Survived", data=plot_data)
plt.legend()
plt.figure(figsize=(12, 8))
sns.distplot(train_data.Age[~train_data.Survived & train_data.HasAge].dropna(),
             label='Deceased', kde=False, norm_hist=True, hist_kws={"alpha": 0.7})
sns.distplot(train_data.Age[train_data.Survived & train_data.HasAge].dropna(),
             label='Survived', kde=False, norm_hist=True, hist_kws={"alpha": 0.7})
plt.legend()
plt.ylabel('Occurrence rate')
plt.show()

print('Only {0:.2f}% of entries have age data.'.format(train_data.HasAge.astype(float).mean() * 100))
print('The survival rate among people with age data was {0:.2f}%.'.format(
    train_data[train_data.HasAge].Survived.mean() * 100.0))
print('Only {0:.1f}% of items have related cabin data.'.format(train_data.HasCabin.mean() * 100))
survival_rate_per_deck = train_data.Survived[train_data.HasCabin].groupby(train_data.CabinLetter).mean()
survival_rate_per_deck.name = 'Survival rate per deck'
survival_rate_per_deck
plt.figure(figsize=(12, 8))
sns.pointplot(x='CabinLetter', y='Survived',
              data=train_data[train_data.HasCabin],
              order=list(
                  sorted(train_data[train_data.HasCabin].CabinLetter.value_counts().index)))
plt.ylabel('Survival rate')
plt.ylim((0.0, 1.0))
plt.figure(figsize=(12, 8))
plot_data = train_data[['Pclass', 'Survived']].copy()
plot_data.loc[:, 'Survived'] = plot_data.Survived.apply(lambda v: 'Survived' if v else 'Deceased')
sns.countplot(x="Pclass", hue="Survived", data=plot_data)
plt.legend()
import graphviz

def print_tree(clf, **kwargs):
  dd = sklearn.tree.export_graphviz(
      clf, out_file=None, filled=True, impurity=False,
      proportion=True, **kwargs)
  return graphviz.Source(dd)

from sklearn.model_selection import cross_val_score
# Preprocessing function and definition of X and y
from IPython.display import display, HTML

display(train_data.head())

def is_male(sex_column):
    return sex_column.apply(lambda v: 1.0 if v == 'male' else 0.0)

def fit_embark_encoder(embarked):
    embarked_encoder = sklearn.preprocessing.LabelEncoder()
    embarked_encoder.fit(embarked)
    def embarked_value_to_column_name(e):
        if e == '':
            return 'unknown_embarcation_point'
        else:
            return 'embark_' + e
    columns =  [embarked_value_to_column_name(c) for c in embarked_encoder.classes_]
    embarked_oh_encoder = sklearn.preprocessing.OneHotEncoder(n_values=len(columns))
    embarked_oh_encoder.fit(np.c_[np.arange(0, len(columns))])
    def transform(data):
        oh = embarked_oh_encoder.transform(np.c_[embarked_encoder.transform(data)]).toarray()
        return pd.DataFrame(data=oh, index=data.index, columns=columns)
    return transform

def deck_letter_to_float(L):
    if L == '':
        return 0.0
    else:
        return ord(L) - ord('A') + 1.0

embark_encoder = fit_embark_encoder(train_data.Embarked)

def preprocess_train_data(X):
    X_output = pd.DataFrame.from_dict({
    'class': X.Pclass,
    'is_male': is_male(X.Sex),
    'age': X.Age,
    'has_age': X.HasAge.astype(float),
    'sib_sp': X.SibSp,
    'parch': X.Parch,
    'fare': X.Fare,
    'cabin_count': X.CabinCount,
    'cabin_deck': X.CabinLetter.apply(deck_letter_to_float),
    'cabin_no': X.CabinNumber,
    })
    X_output = pd.concat([X_output, embark_encoder(X.Embarked)], axis=1)
    return X_output

X = preprocess_train_data(train_data)
y = train_data.Survived.astype(float)
# {'model_name': score}
decision_tree_scores = []
always_deceased_accuracy = np.mean(1 - y)
decision_tree_scores.append(
    ('Always deceased', always_deceased_accuracy))
print('A model that always prints "Death" would have {0:.2f}% accuracy.'.format(
    always_deceased_accuracy * 100))
clf_sex = sklearn.tree.DecisionTreeClassifier()
sex_cvs = cross_val_score(clf_sex, X[['is_male']], y, cv=10, scoring='accuracy')
decision_tree_scores.extend([
    ('Just gender', score) for score 
    in sex_cvs])
print('If we were to take only the person\'s gender into account' +
     ', we would be {0:.2f}% of times right.'.format(np.mean(sex_cvs) * 100))
clf_sex.fit(X[['is_male']], y)
print_tree(clf_sex, feature_names=['is_male'])
def prune_tree(tree):
    def verdict(i):
        v = tree.value[i, 0]
        return v[1] / sum(v) >= 0.5
    def leaf(i):
        return tree.children_left[i] == -1
    def one_above_leaf_and_same_verdict(i):
        cl = tree.children_left[i]
        cr = tree.children_right[i]
        return cl != -1 and leaf(cl) and leaf(cr) and verdict(cl) == verdict(cr)
    def leafify(i):
        tree.children_left[i] = -1.0
        tree.children_right[i] = -1.0
    visited = [False] * len(tree.value)
    visit_stack = [0]
    
    while visit_stack:
        top = visit_stack[-1]
        if visited[top]:
            del visit_stack[-1]
            if one_above_leaf_and_same_verdict(top):
                leafify(top)
        else:
            visited[top] = True
            if not leaf(top):
                visit_stack.append(tree.children_left[top])
                visit_stack.append(tree.children_right[top])
    return tree
param_grid = [
  {'min_samples_leaf': [1, 2, 4, 8, 16, 32, 40],
   'max_depth': range(2, 12),
   'max_leaf_nodes': [8, 16, 32, 64, 128]},
]
gscv = sklearn.model_selection.GridSearchCV(
    sklearn.tree.DecisionTreeClassifier(), param_grid=param_grid, cv=10)
gscv.fit(X, y)
print('Best score: {0}\nBest params: {1}'.format(gscv.best_score_, gscv.best_params_))
decision_tree_scores.append(
    ('Tree with all data', gscv.best_score_))
be = copy.deepcopy(gscv.best_estimator_)
new_tree = prune_tree(be.tree_)
print_tree(be, feature_names=X.columns, rotate=True)
param_grid = [
  {'n_estimators': [128],
   'learning_rate': [0.1, 0.15, 0.2],
   'min_samples_leaf': [4, 8, 16, 32, 40],
   'max_depth': range(4, 7),
   'max_leaf_nodes': [8, 16, 32]},
]
ggscv = sklearn.model_selection.GridSearchCV(
    sklearn.ensemble.GradientBoostingClassifier(), param_grid=param_grid,
    n_jobs=3, cv=10)
ggscv.fit(X, y)
print('Best score: {0}\nBest params: {1}'.format(ggscv.best_score_, ggscv.best_params_))
decision_tree_scores.append(
    ('Gradient Tree with all data', ggscv.best_score_))
decision_tree_scores_df = pd.DataFrame.from_records(
    decision_tree_scores, columns=['model', 'validation accuracy'])
plt.figure(figsize=(12, 8))
sns.barplot(x='model', y='validation accuracy', data=decision_tree_scores_df, ci=0)
plt.ylim((0.6, 0.85))
test_data = pd.read_csv('../input/test.csv')
test_data = test_data.set_index('PassengerId')
test_data = preprocess_input(test_data)
X_test = preprocess_train_data(test_data)
X_test.head()
survived_gender = clf_sex.predict(X_test[['is_male']])
pd.DataFrame.from_dict({'PassengerId': test_data.index, 'Survived': survived_gender.astype(int)}).to_csv('titanic_gender.csv', index=False)
survived_tree = gscv.best_estimator_.predict(X_test)
pd.DataFrame.from_dict({'PassengerId': test_data.index, 'Survived': survived_tree.astype(int)}).to_csv('titanic_dt.csv', index=False)
survived_gradient = ggscv.best_estimator_.predict(X_test)
pd.DataFrame.from_dict({'PassengerId': test_data.index, 'Survived': survived_gradient.astype(int)}).to_csv('titanic_gradient.csv', index=False)
kaggle_results = pd.DataFrame.from_records(
  [('Always deceased', 0.62679),
      ('Just gender', 0.76555),
   ('Tree with all data', 0.76555),
   ('Gradient Tree with all data', 0.75119),], columns=['model', 'test accuracy'])
plt.figure(figsize=(12, 8))
sns.barplot(x='model', y='test accuracy', data=kaggle_results, ci=0)
plt.ylim((0.6, 0.85))
plt.show()
forest = sklearn.ensemble.ExtraTreesClassifier(n_estimators=100, random_state=0)
forest.fit(X, y)
feature_importances = pd.DataFrame(
    np.concatenate([np.vstack([X.columns, tree.feature_importances_]).T
                    for tree in forest.estimators_]),
    columns=['Feature', 'Importance'])
feature_importances.Importance = feature_importances.Importance.astype(float)
order = (feature_importances.groupby('Feature').agg('mean')
         .sort_values(by='Importance').iloc[::-1].index)
plt.figure(figsize=(12, 8))
sns.barplot(x='Feature', y='Importance', data=feature_importances, order=order)
plt.xticks(rotation=45)
plt.show()