# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# disable copy warning in pandas
pd.options.mode.chained_assignment = None  # default='warn'
# disable sklearn deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv').set_index('PassengerId')
df_test = pd.read_csv('../input/test.csv').set_index('PassengerId')
dataset = pd.concat([df_train, df_test], axis=0)
id_test = df_test.index
dataset.tail()
dataset.info()
dataset.describe(include='all').T
dataset.columns
fig = plt.figure(figsize=(9, 6))
sns.heatmap(df_train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), cmap='cool', annot=True, ax=fig.gca())
df_train.groupby('Pclass').Survived.agg(['count', 'sum', 'mean', 'std'])
# plt.bar([1, 2, 3], (df_train.groupby('Pclass').Survived.mean()), color='rgb', tick_label = ['1', '2', '3'])
# sns.despine()

sns.factorplot(x='Pclass', y='Survived', data=df_train, kind='bar', size=5, palette='cool')
dataset.Name.head(10)
dataset['title'] = dataset.Name.apply(lambda x: x.split(', ')[1].split('. ')[0])
# Major, 少校；Lady，贵妇；Sir，子爵; Capt, 上尉；the Countess，伯爵夫人；Col，上校。Dr,医生？
dataset['title'].replace(['Mme', 'Ms', 'Mlle'], ['Mrs', 'Miss', 'Miss'], inplace = True)
dataset['title'].value_counts()
dataset['title'] = dataset.title.apply(lambda x: 'rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)
age_na_miss_rate = len(dataset[(dataset.title == 'Miss') & (dataset.Age.isnull()) ]) / (dataset.title == 'Miss').sum()
age_nna_not_mister_rate = len(dataset[(dataset.title == 'Miss') & (dataset.Age.notnull()) & (dataset.Age >= 18)]) / (dataset.title == 'Miss').sum()
print(age_na_miss_rate, age_nna_not_mister_rate)

len(dataset[(dataset.title == 'Miss') & (dataset.Age.isnull())]) / len(dataset)
dataset.title[(dataset.title == 'Miss') & (dataset.Age < 18)] = 'Mister'
dataset.groupby('title').Age.agg([('number', 'count'), 'min', 'median',  'max'])
dataset.groupby('title').Survived.agg(['count', 'sum', 'mean', 'std'])
g= sns.factorplot(x='title', y = 'Survived', data=dataset, kind='bar', palette='cool', size=5)
g.ax.set_title('title for survived');
dataset['title_level'] = dataset.title.map({"Miss": 3, "Mrs": 3,  "Master": 2, 'Mister': 2,  "rare":2, "Mr": 1})
dataset.groupby('Sex').Survived.agg(['count', 'sum', 'mean', 'std'])
dataset.Sex = dataset.Sex.map({'female':0, 'male':1})
dataset.SibSp.value_counts().plot(kind='bar')
dataset.Parch.value_counts().plot(kind='bar')
# dataset.groupby('SibSp').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='SibSp', y='Survived', size=5, data=dataset, kind='bar', palette='cool')
# df.groupby('Parch').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='Parch', y='Survived', size=5, data=dataset, kind='bar', palette='cool')
dataset['family_size'] = dataset.SibSp + dataset.Parch
# df.groupby('family_size').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='family_size', y='Survived', size=5, data=dataset, kind='bar', palette='cool')
dataset['family_size_level'] = pd.cut(dataset.family_size, bins=[-1,0, 3.5, 12], labels=['alone', 'middle', 'large'])
# df.groupby('family_size_level').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='family_size_level', y='Survived', size=5, data=dataset, kind='bar', palette='cool')
dataset['family_size_level'] = dataset['family_size_level'].map({'alone':1, 'middle':2, 'large':0})
# dataset.groupby(['family_size_level', 'Sex']).Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='family_size_level', y='Survived', hue='Sex', size=5, data=dataset, kind='bar', palette='cool')
dataset[dataset.Fare.isna()]
dataset.Fare.fillna(dataset[dataset.Pclass == 3].Fare.median(), inplace=True)
# dataset.Fare.hist(bins=20)
sns.distplot(dataset.Fare)
dataset.Fare = dataset.Fare.apply(lambda x: np.log(x) if x > 0 else 0)
sns.distplot(dataset.Fare)
dataset.groupby('Pclass').Fare.agg(['count', 'sum', 'mean'])
# def doInverse(x):
#     if x == 3:
#         return 1
#     elif x == 1:
#         return 3
#     else:
#         return x

# df.Pclass.head(3), df.Pclass.apply(lambda x: doInverse(int(x))).head(3), df.Fare.head(3)

# fares = df.Fare.multiply(df.Pclass.apply(lambda x: doInverse(int(x))))
# plt.figure(figsize=(9, 6))
# fares.hist(bins=40)
# # 过滤出20%的人。
# fares.quantile(0.80)
# a = list(range(0, 401, 40))
# a.append(2500)


# df_temp = df.copy()
# df_temp['fare_level']= pd.cut(fares, bins=a)
# df_temp.groupby('fare_level').Survived.agg(['count', 'sum', 'mean'])
# 上限尽量设的大点，因为我们这里没把test data也一起拿出来看，所以不知道范围。
# df['upper_class'] = pd.cut(fares, bins=[0, 40, 160, 2500], labels=['low', 'middle', 'upper'])
# df.groupby('upper_class').Survived.agg(['count', 'sum', 'mean'])
# plt.figure(figsize=(9, 6))
# sns.heatmap(df[['Pclass', '']], cmap='cool', annot=True)
dataset.Ticket.head(5)
dataset.Cabin.isna().sum() / len(dataset.Cabin)
dataset.Cabin = dataset.Cabin.apply(lambda x : 0 if pd.isna(x) else 1)
sns.factorplot(x='Cabin', y='Survived', data=dataset, kind='bar')
dataset.Embarked.isna().sum(), '--'*12, dataset.Embarked.value_counts()
dataset.Embarked.fillna('S', inplace=True)
dataset.groupby('Embarked').Survived.agg(['count', 'sum', 'mean', 'std'])
dataset.groupby(['Pclass', 'Embarked']).Survived.agg(['count', 'sum', 'mean'])
sns.factorplot(x='Embarked', y = 'Survived', hue='Pclass', data=dataset, size=5, kind='bar', palette='cool')
# a = df.groupby(['Pclass', 'Embarked', 'Sex']).Survived.agg(['count', 'sum', 'mean', 'std'])
dataset[(dataset.Embarked == 'S') & (dataset.Pclass == 3)].groupby('Sex').Survived.agg(['count', 'sum', 'mean', 'std'])
# df.Age.hist(bins=20)
sns.distplot(dataset.Age[dataset.Age.notna()])
fig = plt.figure(figsize=(9, 7))
g = sns.kdeplot(dataset.Age[(dataset.Survived == 1) & (dataset.Age.notna())], color='r', ax=fig.gca())
g = sns.kdeplot(dataset.Age[(dataset.Survived == 0) & (dataset.Age.notna())], color='b', ax=fig.gca())
g.set_xlabel('Age')
g.set_ylabel('Survived')
g.legend(['Survived', 'Not'])
dataset['age_level'] = pd.cut(dataset.Age, bins=[0, 18, 60, 100], labels=[3, 2, 1])
# dataset.groupby('age_level').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot('age_level', 'Survived', hue='Sex', data=dataset, kind='bar', size=5, palette='cool')
dataset.head(5)
# dataset[dataset.age_level.notna()].age_level = dataset.age_level[dataset.age_level.notna()].astype('int')
fig = plt.figure(figsize=(9, 6))
sns.heatmap(dataset[['Age', 'Survived', 'Pclass', 'Sex', 'family_size_level', 'Parch', 'SibSp', 'title_level', 'age_level', 'Fare']].corr(), cmap='cool', annot=True, ax=fig.gca())
dataset.groupby('title_level').Age.agg(['mean', 'median', 'std'])
dataset.groupby(['title']).Age.agg(['mean', 'median', 'std', 'max'])
dataset['title_age_level'] = dataset.title.map({"Master": 1, 'Mister': 1, "Miss": 2, "Mrs": 3,  "Mr": 3, "rare": 4})
# dataset['title_age_level'] 
fig = plt.figure(figsize=(9, 6))
sns.heatmap(dataset[['Age', 'Survived', 'Pclass', 'Sex', 'family_size_level', 'Parch', 'SibSp', 'title_level', 'title_age_level', 'age_level', 'Fare']].corr(), cmap='cool', annot=True, ax=fig.gca())
['Survived', 'Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Fare']
df_age_train = dataset[dataset.Age.notnull()]
df_age_test = dataset[dataset.Age.isnull()]

df_age_train.shape, df_age_test.shape
df_age = df_age_train[['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level', 'Fare', 'age_level']]
X_age_dummied = pd.get_dummies(df_age.drop(columns='age_level'), columns=['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level'])
X_age_dummied['SibSp_8'] = np.zeros(len(X_age_dummied))
X_age_dummied['Parch_9'] = np.zeros(len(X_age_dummied))


Y_age = df_age['age_level']
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

params = {'n_estimators':range(10, 40, 5), 'max_depth':[3, 4, 5], 'max_features':range(3, 9)}
clf = RandomForestClassifier(random_state=0)

gscv = GridSearchCV(estimator=clf, param_grid=params, scoring='f1_micro', n_jobs=1, cv=5, verbose=1)
gscv.fit(X_age_dummied, Y_age)
gscv.best_score_, gscv.best_params_, gscv.best_estimator_.feature_importances_
pd.Series(gscv.best_estimator_.feature_importances_, index=X_age_dummied.columns).sort_values(ascending=True).plot.barh(figsize=(9, 6))
sns.despine(bottom=True)
X_age_dummied = pd.get_dummies(df_age.drop(columns='age_level'), columns=['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level'])

ab = df_age_test[['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level', 'Fare']]
X_age_dummied_test = pd.get_dummies(ab, columns=['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level'])
X_age_dummied.shape, X_age_dummied_test.shape
X_age_dummied.columns, X_age_dummied_test.columns
X_age_dummied_test['Parch_3'] = np.zeros(len(X_age_dummied_test))
X_age_dummied_test['Parch_5'] = np.zeros(len(X_age_dummied_test))
X_age_dummied_test['Parch_6'] = np.zeros(len(X_age_dummied_test))

X_age_dummied_test['SibSp_4'] = np.zeros(len(X_age_dummied_test))
X_age_dummied_test['SibSp_5'] = np.zeros(len(X_age_dummied_test))

df_age_test.age_level = gscv.predict(X_age_dummied_test)
df_age_test.shape
df_final = pd.concat([df_age_test, df_age_train]).sort_index()
df_final.info()
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

names = ['LR', 'KNN', 'SVM', 'Bayes', 'NW', 'DT', 'RF', 'GBDT']
models = [LogisticRegression(random_state=0), KNeighborsClassifier(n_neighbors=3), SVC(gamma='auto', random_state=0), GaussianNB(), MLPClassifier(solver='lbfgs', random_state=0),
         DecisionTreeClassifier(), RandomForestClassifier(random_state=0), GradientBoostingClassifier(random_state=0)]
selection = ['Survived', 'Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Fare', 'Embarked', 'Cabin']
train_set = df_final[df_final.Survived.notna()][selection]
test_set = df_final[df_final.Survived.isna()][selection]
X_train = train_set.drop(columns='Survived')
Y_train = train_set['Survived']
X_test = test_set.drop(columns='Survived')

X_train= pd.get_dummies(X_train, columns=['Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Embarked', 'Cabin'])
X_test = pd.get_dummies(X_test, columns=['Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Embarked', 'Cabin'])
X_train.shape, X_test.shape
X_train.columns, X_test.columns
for name, model in zip(names, models):
    score = cross_val_score(model, X_train, Y_train, cv=5, scoring='roc_auc')
    print('score on {}, mean:{:.4f}, from {}'.format(name, score.mean(), score))
lr = LogisticRegression(random_state=0).fit(X_train, Y_train)
lr.coef_
from matplotlib import cm
color = cm.inferno_r(np.linspace(.4,.8, len(X_train.columns)))

pd.DataFrame({'weights': lr.coef_[0]}, index=X_train.columns)\
.sort_values(by='weights',ascending=True)\
.plot.barh(figsize=(7, 7),fontsize=12, legend=False, title='Feature weights from Logistic Regression', color=color);
sns.despine(bottom=True);
def plot_decision_tree(clf, feature_names, class_names):
    from sklearn.tree import export_graphviz
    import graphviz
    export_graphviz(clf, out_file="adspy_temp.dot", feature_names=feature_names, class_names=class_names, filled = True, impurity = False)
    with open("adspy_temp.dot") as f:
        dot_graph = f.read()
    return graphviz.Source(dot_graph)

dtc = DecisionTreeClassifier().fit(X_train, Y_train)

pd.DataFrame({'importance': dtc.feature_importances_}, index=X_train.columns)\
.sort_values(by='importance',ascending=True)\
.plot.barh(figsize=(7, 7),fontsize=12, legend=False, title='Feature importance from Decision Tree', color=color);
sns.despine(bottom=True, left=True);

# 这个DecisionTree太大了啊，真实天性容易过拟合，这层数也太多了，所以不画了。
# plot_decision_tree(dtc, X_train.columns, 'Survived')
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=4, random_state=0)

# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# miss classifications
mcs = []

for train_index, test_index in skf.split(X_train, Y_train):
    x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
    y_pred =lr.fit(x_train, y_train).predict(x_test)
    mcs.append(x_test[y_pred != y_test].index.tolist())
mcs_index = np.concatenate((mcs))
len(mcs_index), len(Y_train), 'miss classification rate:', len(mcs_index)/len(Y_train)
# mcs_df = pd.concat([X_train.iloc[mcs_index], Y_train.iloc[mcs_index]], axis=1)
mcs_df = train_set.iloc[mcs_index]
mcs_df.head()
mcs_df.describe(include='all').T
mcs_df[mcs_df.family_size_level == 1].shape, mcs_df[mcs_df.title_level == 3].shape, mcs_df[mcs_df.age_level == 3].shape
mcs_df.groupby('family_size_level').Survived.mean()
# df_final['MPSE'] = np.ones(len(df_final))
# df_final.MPSE[(df_final.title_level == 1) & (df_final.Pclass == 3) & (df_final.Sex == 0) & (df_final.Embarked == 'S') & (df_final.family_size_level.isin([1, 2]))] = 4
# df_final.MPSE[(df_final.title_level == 1) & (df_final.Pclass == 2) & (df_final.Sex == 0) & (df_final.Embarked == 'S') & (df_final.family_size_level.isin([1, 2]))] = 3
# df_final.MPSE[(df_final.title_level == 1) & (df_final.Pclass == 1) & (df_final.Sex == 0) & (df_final.Embarked == 'S') & (df_final.family_size_level.isin([1, 2]))] = 2
# df_final.MPSE.value_counts()

df_final['Alone'] = np.zeros(len(df_final))
df_final['Alone'][df_final.family_size_level == 1] = 10
df_final['EmbkS'] = np.zeros(len(df_final))
df_final['EmbkS'][df_final.Embarked == 'S'] = 10
df_final['MrMale'] = np.zeros(len(df_final))
df_final['MrMale'][df_final.title_level == 1] = 10
selection = ['Survived', 'Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Fare', 'Embarked', 'EmbkS', 'MrMale', 'Cabin', 'Alone']
train_set = df_final[df_final.Survived.notna()][selection]
test_set = df_final[df_final.Survived.isna()][selection]
X_train = train_set.drop(columns='Survived')
Y_train = train_set['Survived']
X_test = test_set.drop(columns='Survived')

X_train= pd.get_dummies(X_train, columns=['Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Embarked', 'Cabin'])
X_test = pd.get_dummies(X_test, columns=['Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Embarked', 'Cabin'])

val_lr = LogisticRegression(random_state=0)
a_score = cross_val_score(val_lr, X_train, Y_train, cv=5, scoring='roc_auc')
a_score.mean()
from sklearn.ensemble import GradientBoostingClassifier

def tune_estimator(name, estimator, params):
    gscv_training = GridSearchCV(estimator=estimator, param_grid=params, scoring='roc_auc', n_jobs=1, cv=5, verbose=False)
    gscv_training.fit(X_train, Y_train)
    return name, gscv_training.best_score_, gscv_training.best_params_
from sklearn.linear_model import LogisticRegression
params = {'C':[0.03, 0.1, 0.3, 1, 3, 10, 20, 30, 50]}
clf = LogisticRegression(random_state=0)
tune_estimator('LR', clf, params)
# Second fine tuning.
params = {'C':[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
clf = LogisticRegression(random_state=0)

tune_estimator('LR', clf, params)
params = {'n_neighbors': range(3, 15, 3)}
clf = KNeighborsClassifier()
tune_estimator('KNN', clf, params)
params = {'n_neighbors': [10, 11, 12, 13, 14]}
clf = KNeighborsClassifier()
tune_estimator('KNN', clf, params)
params = {'C':[0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1], 'kernel':['rbf']}
clf = SVC(random_state=0)
tune_estimator('SVM', clf, params)
params = {'C':range(8, 14), 'gamma':[0.05, 0.08, 0.01, 0.03, 0.05], 'kernel':['rbf']}
clf = SVC(random_state=0)
tune_estimator('SVM', clf, params)
params = {'hidden_layer_sizes':[x for x in zip(range(20, 100, 10), range(20, 100, 20))],
          'solver':['lbfgs'], 'alpha': [0.0001, 0.001, 0.01]}
clf = MLPClassifier(random_state = 0)
tune_estimator('NW', clf, params)
params = {'n_estimators':range(10, 40, 5), 'max_depth':[3, 5], 'max_features':range(3, 7)}
clf = RandomForestClassifier(random_state=0)
tune_estimator('RF', clf, params)
# second round
params = {'n_estimators':range(31, 40, 2), 'max_depth':range(4, 7), 'max_features':range(3, 6)}
clf = RandomForestClassifier(random_state=0)
tune_estimator('RF', clf, params)
params = {'learning_rate':[0.001, 0.01, 0.1, 1], 'n_estimators':range(100, 250, 20), 'max_depth':range(2, 5), 'max_features':range(3, 6)}
clf = GradientBoostingClassifier(random_state=0)
tune_estimator('GBDT', clf, params)
# second tune
params = {'learning_rate':[0.3, 0.5, 0.7, 0.9, 1.2, 1.4, 1.7, 2], 'n_estimators':range(130, 200, 20), 'max_depth':[2, 3, 4], 'max_features':[3, 4, 5]}
clf = GradientBoostingClassifier(random_state=0)
tune_estimator('GBDT', clf, params)
from sklearn.ensemble import BaggingClassifier
params = {'n_estimators': range(50, 150, 10)}
bagging = BaggingClassifier(LogisticRegression(C=0.3))
tune_estimator('bagging', bagging, params)
from sklearn.ensemble import AdaBoostClassifier
params = {'n_estimators':range(80, 200, 20), 'learning_rate':[0.5, 1, 3, 5]}
adc = AdaBoostClassifier(random_state=0)
tune_estimator('AdaBoosting', adc, params)
# secound round
params = {'n_estimators':range(150, 180, 10), 'learning_rate':[ 0.7, 0.9, 1.3, 2]}
adc = AdaBoostClassifier(random_state=0)
tune_estimator('AdaBoosting', adc, params)
lr = LogisticRegression(C=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
svc = SVC(C=9, gamma=0.1, random_state=0)
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(20, 20), solver='lbfgs', random_state=0)
rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=31, random_state=0)
gbdt = GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=3, n_estimators=190, random_state=0)
bagging = BaggingClassifier(lr, n_estimators=140, random_state=0)
abc = AdaBoostClassifier(learning_rate=1.3, n_estimators=150, random_state=0)

names = ['LR', 'KNN', 'SVC', 'MLP', 'RF', 'GBDT', 'Bagging', 'AdaB']
models = [lr, knn, svc, mlp, rf, gbdt, bagging, abc]

result_scores = []
for name, model in zip(names, models):
    scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='roc_auc')
    result_scores.append(scores.mean())
    print('{} has a mean score {:.4f} based on {}'.format(name, scores.mean(), scores))    
sorted_score = pd.Series(data=result_scores, index = names).sort_values(ascending=False)
ax = plt.subplot(111)
sorted_score.plot(kind='line', ax=ax, title='score order', figsize=(9, 6), colormap='cool')
ax.set_xticks(range(0, 9))
ax.set_xticklabels(sorted_score.index);
sns.despine()
from sklearn.ensemble import VotingClassifier
names = ['LR', 'KNN', 'RF', 'GBDT', 'Bagging', 'AdaB']

lr = LogisticRegression(C=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=31, random_state=0)
gbdt = GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=3, n_estimators=190, random_state=0)
bagging = BaggingClassifier(lr, n_estimators=140, random_state=0)
abc = AdaBoostClassifier(learning_rate=1.3, n_estimators=150, random_state=0)

# 直接投票，票数多的获胜。
vc_hard = VotingClassifier(estimators=[('LR', lr), ('KNN', knn), ('GBDT', gbdt), ('Bagging', bagging), ('AdaB', abc)], voting='hard')
# 参数里说，soft更加适用于已经调制好的base learners，基于每个learner输出的概率。知乎文章里讲，Soft一般表现的更好。
vc_soft = VotingClassifier(estimators=[('LR', lr), ('KNN', knn), ('RF', rf), ('GBDT', gbdt), ('Bagging', bagging), ('AdaB', abc)], voting='soft')

# 'vc hard:', cross_val_score(vc_hard, X_dummied, Y, cv=5, scoring='roc_auc').mean(),\
'vc soft:', cross_val_score(vc_soft, X_train, Y_train, cv=5, scoring='roc_auc').mean()
n_train=X_train.shape[0]
n_test=X_test.shape[0]
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
def get_oof(clf, X, y, test_X):
    oof_train = np.zeros((n_train, ))
    oof_test_mean = np.zeros((n_test, ))
    # 5 is kf.split
    oof_test_single = np.empty((kf.get_n_splits(), n_test))
    for i, (train_index, val_index) in enumerate(kf.split(X,y)):
        kf_X_train = X.iloc[train_index]
        kf_y_train = y.iloc[train_index]
        kf_X_val = X.iloc[val_index]
        
        clf.fit(kf_X_train, kf_y_train)
        
        oof_train[val_index] = clf.predict(kf_X_val)
        oof_test_single[i,:] = clf.predict(test_X)
    # oof_test_single, 将生成一个5行*n_test列的predict value。那么mean(axis=0), 将对5行，每列的值进行求mean。然后reshape返回   
    oof_test_mean = oof_test_single.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test_mean.reshape(-1,1)
lr = LogisticRegression(C=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=31, random_state=0)
gbdt = GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=3, n_estimators=190, random_state=0)
bagging = BaggingClassifier(lr, n_estimators=140, random_state=0)
abc = AdaBoostClassifier(learning_rate=1.3, n_estimators=150, random_state=0)

lr_train, lr_test = get_oof(lr, X_train, Y_train, X_test)
knn_train, knn_test = get_oof(knn, X_train, Y_train, X_test)
rf_train, rf_test = get_oof(rf, X_train, Y_train, X_test)
gbdt_train, gbdt_test=get_oof(gbdt, X_train, Y_train, X_test)
bagging_train, bagging_test = get_oof(bagging,X_train, Y_train, X_test)
abc_train, abc_test = get_oof(abc,X_train, Y_train, X_test)
y_train_pred_stack = np.concatenate([lr_train, knn_train, rf_train, gbdt_train, bagging_train, abc_train], axis=1)
y_train_stack = Y_train.reset_index(drop=True)
y_test_pred_stack = np.concatenate([lr_test, knn_test, rf_test, gbdt_test, bagging_test, abc_test], axis=1)

y_train_pred_stack.shape, y_train_stack.shape, y_test_pred_stack.shape
# params = {'learning_rate':[0.001, 0.01, 0.1, 1], 'n_estimators':range(100, 250, 20), 'max_depth':[2, 5], 'max_features':range(1, 3)}
# clf = GradientBoostingClassifier(random_state=0)

# params = {'C':[0.05, 0.08, 0.1, 0.2, 0.3]}
# clf = LogisticRegression(random_state=0)

params = {'n_estimators':range(90, 150, 10)}
clf = RandomForestClassifier(random_state=0)

gscv_test= GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=5, verbose=False)
gscv_test.fit( y_train_pred_stack, y_train_stack)
gscv_test.best_score_, gscv_test.best_params_
y_pred = RandomForestClassifier(random_state=0, n_estimators=100).fit(y_train_pred_stack, y_train_stack).predict(y_test_pred_stack)
c = gscv_test.predict(y_train_pred_stack)

a = pd.DataFrame(y_train_pred_stack)
b = pd.concat([a, pd.Series(c), y_train_stack], axis=1)
b.columns = ['lr', 'knn', 'rf', 'gbdt', 'bagging', 'abc', 'predicted', 'Survived']
b[b.predicted != b.Survived]
y_pred = vc_soft.fit(X_train, Y_train).predict(X_test)
result_df = pd.DataFrame({'PassengerId': X_test.index, 'Survived':y_pred}).set_index('PassengerId')
result_df.Survived = result_df.Survived.astype('int')
result_df.to_csv('predicted_survived.csv')
pd.read_csv('predicted_survived.csv').head()
pd.read_csv('../input/gender_submission.csv').head()