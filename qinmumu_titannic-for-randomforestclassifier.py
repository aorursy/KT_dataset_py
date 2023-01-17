import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.warn('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape)
print(test.shape)

data = pd.concat([train, test], axis=0)
print(data.shape)
data.info()
data.describe()
# 特征值取值数量，少的可以可视化分析
data.apply(lambda x: x.value_counts().shape[0])
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('bmh')

f, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(20,5))
sns.countplot(data=train, x='Embarked',hue='Survived', ax=ax1)
sns.countplot(data=train, x='Pclass', hue='Survived', ax=ax2)
sns.countplot(data=train, x='Sex', hue='Survived', ax=ax3)
f.suptitle('feature_values VS Suivived_values')

f, [ax1, ax2] = plt.subplots(1,2, figsize=(20,5))
sns.countplot(data=train, x='Parch', hue='Survived', ax=ax1)
sns.countplot(data=train, x='SibSp', hue='Survived', ax=ax2)
grid = sns.FacetGrid(train, col='Pclass', hue='Sex',size=4)
grid.map(sns.countplot, 'Embarked')
grid = sns.FacetGrid(train, row='Sex',col='Pclass', hue='Survived', palette='seismic',height=4)
grid.map(sns.countplot, 'Embarked', alpha=0.8)
grid.add_legend()
f, ax = plt.subplots(figsize=(10,5))
sns.kdeplot(train.loc[train['Survived']==0, 'Age'], shade=True, label='not survived')
sns.kdeplot(train.loc[train['Survived']==1, 'Age'], shade=True, label='survived')
plt.title('Age_feature_distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
# 箱线图特征分析
f, [ax1, ax2] = plt.subplots(1,2, figsize=(20,6))
sns.boxplot(data=train, x='Pclass', y='Age', ax=ax1)
sns.swarmplot(data=train, x='Pclass', y='Age',ax=ax1)
sns.kdeplot(train.loc[train['Pclass']==1, 'Age'], shade=True, label='Pcalss=1', ax=ax2)
sns.kdeplot(train.loc[train['Pclass']==2, 'Age'], shade=True, label='Pclass=2', ax=ax2)
sns.kdeplot(train.loc[train['Pclass']==3, 'Age'], shade=True, label='Pcalss=3',ax=ax2)
ax1.set_title('Box_distribution Age_feature for Plass_feature')
ax2.set_title('Kde_distribution Age_feature for Plass_feature')
grid = sns.FacetGrid(data=train, row='Sex', col='Pclass', hue='Survived')
grid.map(plt.scatter, 'PassengerId', 'Age')
grid.add_legend()
grid = sns.FacetGrid(data=train, row='Sex', col='SibSp', hue='Survived')
grid.map(plt.scatter, 'PassengerId', 'Age')
grid.add_legend()
grid = sns.FacetGrid(data=train, row='Sex', col='Parch', hue='Survived')
grid.map(plt.scatter, 'PassengerId', 'Age')
grid.add_legend()
f, ax = plt.subplots(figsize=(10,5))
sns.kdeplot(train.loc[train['Survived']==0, 'Fare'], shade=True, label='not survived')
sns.kdeplot(train.loc[train['Survived']==1, 'Fare'], shade=True, label='survived')
plt.title('Fare_feature_distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
f, [ax1,ax2] = plt.subplots(1, 2, figsize=(26,10))
sns.boxplot(data=train, x='Pclass', y='Fare', ax=ax1)
sns.swarmplot(data=train, x='Pclass', y='Fare', ax=ax1)
ax1.set_title('Box_distribution Fare_feature for Plass_feature')

sns.kdeplot(train.loc[train['Pclass']==1, 'Fare'], shade=True, label='Pclass=1', ax=ax2)
sns.kdeplot(train.loc[train['Pclass']==2, 'Fare'], shade=True, label='Pclass=2', ax=ax2)
sns.kdeplot(train.loc[train['Pclass']==3, 'Fare'], shade=True, label='Pclass=3', ax=ax2)
ax2.set_title('Kde_distribution Fare_feature for Plass_feature')               
grid = sns.FacetGrid(data=train, row='Sex', col='Pclass', hue='Survived')
grid.map(plt.scatter, 'Age', 'Fare')
grid.add_legend()
sns.pairplot(train, hue='Survived')
data[data['Fare'].isnull()]
fare_fillna = data[(data['Age']>60) & (data['Pclass']==3) & (data['Sex']=='male')]['Fare'].mean()
data['Fare'].fillna(fare_fillna, inplace=True)
data[data['Embarked'].isnull()]
data['Embarked'].fillna('C', inplace=True)
data['CabinCate'] = pd.Categorical(data['Cabin'].fillna('0').apply(lambda x: x[0])).codes
sns.countplot(data=data, x='CabinCate', hue='Survived')
data.isnull().sum()
data['FamilySize'] = data['SibSp']+data['Parch']+1
data['FamilySize'] = pd.cut(data['FamilySize'], bins=[0,1,4,20], labels=[0,1,2])
import re
data['Title'] = data['Name'].apply(lambda x : re.search('(\w+)\.', x).group(1))
data['Title'] = data['Title'].apply(lambda x : [x, 'Mrse'][x not in ['Mr', 'Miss', 'Mrs', 'Master']])
data = pd.concat([data, pd.get_dummies(data[['Embarked', 'Title','Sex']])], axis=1)
from sklearn.ensemble import ExtraTreesRegressor
columns = ['Fare', 'Parch', 'Pclass', 'SibSp', 'CabinCate', 'FamilySize', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
           'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Mrse', 'Sex_female', 'Sex_male']
X_train = data[columns][data['Age'].notnull()]
y_train = data['Age'][data['Age'].notnull()]
X_test = data[columns][data['Age'].isnull()]
extra = ExtraTreesRegressor(n_estimators=200, random_state=0)

extra.fit(X_train, y_train)
data['Age'][data['Age'].isnull()] = extra.predict(X_test)

X_test['Age'] = extra.predict(X_test)
sns.swarmplot(data=X_test, x='Pclass', y='Age')
data = data.drop(['Cabin', 'Embarked','Name','PassengerId','Sex', 'Survived', 'Ticket','Title'], axis=1)
from sklearn.feature_selection import SelectKBest, f_classif, chi2
target = train['Survived']
train_feature = data[:len(train)]
test_feature = data[len(train):]
feat_list = data.columns.tolist()

selector = SelectKBest(f_classif, k=len(feat_list))
selector.fit(train_feature, target)
scores = -np.log10(selector.pvalues_,)
indices = np.argsort(scores)[::-1]

print('Features importance:')
for i in range(len(scores)):
    print('{:.2f}==>{}'.format(scores[indices[i]], feat_list[indices[i]]))
plt.figure(figsize=(25,25))
sns.heatmap(data.corr(), annot=True, cmap=plt.cm.RdBu)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(train_feature, target, test_size=0.2, random_state=2)
params = {'n_estimators':[120,200,300,500,800,1200]}
clf = GridSearchCV(RandomForestClassifier(random_state=2), params, cv=6, scoring='roc_auc')
clf.fit(X_train, y_train)

print('训练集得分：', clf.score(X_train, y_train))
print('测试集得分：', clf.score(X_test, y_test))
print('最佳参数：', clf.best_params_)

feature_importances = clf.best_estimator_.feature_importances_
importance_df = pd.DataFrame(feature_importances, X_train.columns, columns=['importance score'])
importance_df.sort_values('importance score', ascending=False, inplace=True)
importance_df.plot.barh()

result_df = pd.concat([test['PassengerId'], pd.Series(clf.predict(test_feature), name='Survived')], axis=1)
result_df.set_index('PassengerId')
result_df.to_csv('result_df.csv', index=False)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

train_data = train_feature.drop(['Sex_male', 'Sex_female'], axis=1)
test_data = test_feature.drop(['Sex_male', 'Sex_female'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(train_data, target, test_size=0.2, random_state=2)
params = {'n_estimators':[120,200,300,500,800,1200]}
clf = GridSearchCV(RandomForestClassifier(random_state=2), params, cv=6, scoring='roc_auc')
clf.fit(X_train, y_train)

print('训练集得分：', clf.score(X_train, y_train))
print('测试集得分：', clf.score(X_test, y_test))
print('最佳参数：', clf.best_params_)

feature_importances = clf.best_estimator_.feature_importances_
importance_df = pd.DataFrame(feature_importances, X_train.columns, columns=['importance score'])
importance_df.sort_values('importance score', ascending=False, inplace=True)
importance_df.plot.barh()

result_df = pd.concat([test['PassengerId'], pd.Series(clf.predict(test_data), name='Survived')], axis=1)
result_df.set_index('PassengerId')
result_df.to_csv('result_df.csv', index=False)
feat_list = importance_df[importance_df['importance score'] > 0.02].index.tolist()
train_df = train_data[feat_list]
test_df = test_data[feat_list]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.2, random_state=2)
params = {'n_estimators':[120,200,300,500,800,1200]}
clf = GridSearchCV(RandomForestClassifier(random_state=2), params, cv=6, scoring='roc_auc')
clf.fit(X_train, y_train)

print('训练集得分：', clf.score(X_train, y_train))
print('测试集得分：', clf.score(X_test, y_test))
print('最佳参数：', clf.best_params_)

feature_importances = clf.best_estimator_.feature_importances_
importance_df = pd.DataFrame(feature_importances, X_train.columns, columns=['importance score'])
importance_df.sort_values('importance score', ascending=False, inplace=True)
importance_df.plot.barh()

result_df = pd.concat([test['PassengerId'], pd.Series(clf.predict(test_df), name='Survived')], axis=1)
result_df.set_index('PassengerId')
result_df.to_csv('result_df.csv', index=False)
feat_list = importance_df[importance_df['importance score'] > 0.05].index.tolist()
train_df = train_data[feat_list]
test_df = test_data[feat_list]

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test = train_test_split(train_df, target, test_size=0.2, random_state=2)
params = {'n_estimators':[120,200,300,500,800,1200], 'max_depth':[3,5,9,11]}
clf = GridSearchCV(RandomForestClassifier(random_state=2), params, cv=6, scoring='roc_auc')
clf.fit(X_train, y_train)

print('训练集得分：', clf.score(X_train, y_train))
print('测试集得分：', clf.score(X_test, y_test))
print('最佳参数：', clf.best_params_)

feature_importances = clf.best_estimator_.feature_importances_
importance_df = pd.DataFrame(feature_importances, X_train.columns, columns=['importance score'])
importance_df.sort_values('importance score', ascending=False, inplace=True)
importance_df.plot.barh()

result_df = pd.concat([test['PassengerId'], pd.Series(clf.predict(test_df), name='Survived')], axis=1)
result_df.set_index('PassengerId')
result_df.to_csv('result_df.csv', index=False)
