import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from xgboost import XGBClassifier
test = pd.read_csv('../input/poker-hand-testing.data', header=None)
train = pd.read_csv('../input/poker-hand-training-true.data', header=None)
train.columns = ['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label']
test.columns = ['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label']
train.head()
train.shape
test.shape
X_train = train.loc[:,train.columns != 'Label']
X_test = test.loc[:,test.columns != 'Label']
Y_train = train['Label']
Y_test = test['Label']
Y_train.groupby(Y_train).size()
Y_test.groupby(Y_test).size()
def preprocess_data(data:pd.DataFrame):
    df = data.copy()
    dfc = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    dfc.values.sort()
    df[['C1', 'C2', 'C3', 'C4', 'C5']] = dfc
    df = df[['C1', 'C2', 'C3', 'C4', 'C5', 'S1', 'S2', 'S3', 'S4', 'S5', 'Label']]
    return df
def add_counts(df:pd.DataFrame):
    tmp = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    df['Cnt_C1'] = tmp.apply(lambda x: sum(x==x[0]) ,axis=1)
    df['Cnt_C2'] = tmp.apply(lambda x: sum(x==x[1]) ,axis=1)
    df['Cnt_C3'] = tmp.apply(lambda x: sum(x==x[2]) ,axis=1)
    df['Cnt_C4'] = tmp.apply(lambda x: sum(x==x[3]) ,axis=1)
    df['Cnt_C5'] = tmp.apply(lambda x: sum(x==x[4]) ,axis=1)
    
    tmp = df[['S1', 'S2', 'S3', 'S4', 'S5']]
    df['Cnt_S1'] = tmp.apply(lambda x: sum(x==x[0]) ,axis=1)
    df['Cnt_S2'] = tmp.apply(lambda x: sum(x==x[1]) ,axis=1)
    df['Cnt_S3'] = tmp.apply(lambda x: sum(x==x[2]) ,axis=1)
    df['Cnt_S4'] = tmp.apply(lambda x: sum(x==x[3]) ,axis=1)    
    df['Cnt_S5'] = tmp.apply(lambda x: sum(x==x[4]) ,axis=1)
def add_diffs(df:pd.DataFrame):
    df['Diff1'] = df['C5'] - df['C4']
    df['Diff2'] = df['C4'] - df['C3']
    df['Diff3'] = df['C3'] - df['C2']
    df['Diff4'] = df['C2'] - df['C1']
def add_unique_count(df:pd.DataFrame):
    tmp = df[['S1', 'S2', 'S3', 'S4', 'S5']]
    df['UniqueS'] = tmp.apply(lambda x: len(np.unique(x)) , axis=1)
def cross_validation(alg, X_train, Y_train, folds=10):
    kf = KFold(n_splits = folds, shuffle=True)

    acc = []
    matrix = None
    first = True

    i = 1
    for train_index, test_index in kf.split(X_train, Y_train):
        print('{}-Fold'.format(i))
        fX_train, fX_test = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
        fy_train, fy_test = Y_train[train_index], Y_train[test_index]
        alg.fit(fX_train, fy_train)
        fy_pred = alg.predict(fX_test)
        curr = accuracy_score(fy_test, fy_pred, normalize=True)
        acc.append(curr)
        i = i+1

    acc = pd.Series(acc)
    return acc.mean()
alg = DecisionTreeClassifier(random_state=1)
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)
accuracy_score(Y_test, y_pred, normalize=True)
X_train_pre = preprocess_data(train)
X_test_pre = preprocess_data(test)
X_train = X_train_pre.loc[:,X_train_pre.columns != 'Label']
X_test = X_test_pre.loc[:,X_test_pre.columns != 'Label']
alg = DecisionTreeClassifier(random_state=1, criterion='gini')
cross_validation(alg, X_train, Y_train)
alg = DecisionTreeClassifier(random_state=1, criterion='gini')
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)
accuracy_score(Y_test, y_pred, normalize=True)
pd.crosstab(y_pred, Y_test, rownames=['Predicted'], colnames=['True'], margins=True)
pred_series = pd.Series(y_pred).groupby(y_pred).size()
true_series = pd.Series(Y_test.values).groupby(Y_test).size()
pred_res = pd.DataFrame()
pred_res['TrueLabel'] = true_series
pred_res['PredictedLabel'] = pred_series
pred_res
f, ax = plt.subplots()
ax.set(yscale="log")
sns.barplot(data=pred_res.stack().reset_index().rename(columns={0: 'Count', 'level_1': 'Variable'}), x='Label', y='Count', hue='Variable')
add_unique_count(X_test)
add_unique_count(X_train)
alg = DecisionTreeClassifier(random_state=1, criterion='gini')
cross_validation(alg, X_train, Y_train)
alg = DecisionTreeClassifier(random_state=1, criterion='gini')
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)
accuracy_score(Y_test, y_pred, normalize=True)
pred_series = pd.Series(y_pred).groupby(y_pred).size()
true_series = pd.Series(Y_test.values).groupby(Y_test).size()
pred_res = pd.DataFrame()
pred_res['TrueLabel'] = true_series
pred_res['PredictedLabel'] = pred_series
f, ax = plt.subplots()
ax.set(yscale="log")
sns.barplot(data=pred_res.stack().reset_index().rename(columns={0: 'Count', 'level_1': 'Variable'}), x='Label', y='Count', hue='Variable')
add_diffs(X_train)
add_diffs(X_test)
alg = DecisionTreeClassifier(random_state=1, criterion='gini')
cross_validation(alg, X_train, Y_train)
alg = DecisionTreeClassifier(random_state=1, criterion='gini')
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)
accuracy_score(Y_test, y_pred, normalize=True)
pred_series = pd.Series(y_pred).groupby(y_pred).size()
true_series = pd.Series(Y_test.values).groupby(Y_test).size()
pred_res = pd.DataFrame()
pred_res['TrueLabel'] = true_series
pred_res['PredictedLabel'] = pred_series
f, ax = plt.subplots()
ax.set(yscale="log")
sns.barplot(data=pred_res.stack().reset_index().rename(columns={0: 'Count', 'level_1': 'Variable'}), x='Label', y='Count', hue='Variable')
pd.crosstab(y_pred, Y_test, rownames=['Predicted'], colnames=['True'], margins=True)
alg = RandomForestClassifier(criterion='gini', n_estimators=10, random_state=111, n_jobs=4)
cross_validation(alg, X_train, Y_train)
alg = RandomForestClassifier(criterion='entropy', n_estimators=51, random_state=111, n_jobs=4)
cross_validation(alg, X_train, Y_train)
alg = GradientBoostingClassifier(n_estimators=10, random_state=111)
cross_validation(alg, X_train, Y_train)
alg = XGBClassifier(n_estimators=10, random_state=111)
cross_validation(alg, X_train, Y_train)
alg = DecisionTreeClassifier(criterion='gini', random_state=111)
alg.fit(X_train, Y_train)
y_pred = alg.predict(X_test)
accuracy_score(y_pred=y_pred, y_true=Y_test, normalize=True)
feature_imp = pd.DataFrame(sorted(zip(X_train.columns, alg.feature_importances_), key=lambda k: k[1], reverse=True))
feature_imp.columns = ['Feature', 'Importance']
f, ax = plt.subplots(figsize=(10, 7))
# ax.set(yscale="log")
plt.xticks(rotation=45)
sns.barplot(data=feature_imp, x='Feature', y='Importance')