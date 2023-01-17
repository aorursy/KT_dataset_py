import graphviz 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score, GridSearchCV
train = pd.read_csv('../input/train.csv').set_index('PassengerId')
test = pd.read_csv('../input/test.csv').set_index('PassengerId')
df = pd.concat([train, test], axis=0, sort=False)
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['IsWomanOrChild'] = ((df.Title == 'Master') | (df.Sex == 'female'))
df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived
df['FamilyTotalCount'] = family.transform(lambda s: s[df.IsWomanOrChild].fillna(0).count())
df['FamilyTotalCount'] = df.mask(df.IsWomanOrChild, df.FamilyTotalCount - 1, axis=0)
df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrChild].fillna(0).sum())
df['FamilySurvivedCount'] = df.mask(df.IsWomanOrChild, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)
df['FamilySurvivalRate'] = (df.FamilySurvivedCount / df.FamilyTotalCount.replace(0, np.nan))
df['IsSingleTraveler'] = df.FamilyTotalCount == 0
x = pd.concat([
    df.FamilySurvivalRate.fillna(0),
    df.IsSingleTraveler,
    df.Sex.replace({'male': 0, 'female': 1}),
], axis=1)
train_x, test_x = x.loc[train.index], x.loc[test.index]
train_y = df.Survived.loc[train.index]
clf = tree.DecisionTreeClassifier()
grid = GridSearchCV(clf, cv=5, param_grid={
    'criterion': ['gini', 'entropy'], 
    'max_depth': [2, 3, 4, 5]})
grid.fit(train_x, train_y)
grid.best_params_
model = grid.best_estimator_
graphviz.Source(tree.export_graphviz(model, feature_names=x.columns)) 
test_y = model.predict(test_x).astype(int)
pd.DataFrame({'Survived': test_y}, index=test.index) \
.reset_index() \
.to_csv(f'survived.csv', index=False)