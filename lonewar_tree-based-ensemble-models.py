import pandas as pd
df = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv', sep=',')

%matplotlib inline
df.head(5)
df.dtypes
list(df.columns)
df.shape
print(df['phone number'].nunique())
print(df['state'].nunique()) 
df.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(3, 4, figsize=(20, 10))

df.state.value_counts().plot(kind='bar', ax=axs[0,0]); axs[0,0].set_title('state')
df.hist(column='account length', ax=axs[0,1]); axs[0,1].set_title('account length')
df['area code'].value_counts().plot(kind='bar', ax=axs[0,2]); axs[0,2].set_title('area code')
df['international plan'].value_counts().plot(kind='bar', ax=axs[0,3]); axs[0,3].set_title('international plan')

df['voice mail plan'].value_counts().plot(kind='bar', ax=axs[1,0]); axs[1,0].set_title('voice mail plan')
df.hist(column='number vmail messages', ax=axs[1,1]); axs[1,1].set_title('number vmail messages')

df.hist(column='customer service calls', ax=axs[2,0]); axs[2,0].set_title('customer service calls')
df.churn.value_counts().plot(kind='bar', ax=axs[2,1]); axs[2,1].set_title('churn');
churn_true = len(df[df['churn'] == True].index)
churn_false = len(df[df['churn'] == False].index)
print('Churn rate is: {}. \nchurn_false/churn_true = {}. churn_false - churn_true = {}. \nThe data is imbalanced.'
      .format(churn_true / (churn_true + churn_false), churn_false / churn_true, churn_false - churn_true))

cols = list(df.columns)
cols.remove('state')
cols.remove('area code')
cols.remove('phone number')
cols.remove('international plan')
cols.remove('voice mail plan')
cols.remove('churn')

# Define a set of columns to be removed. They are not to be used as features.
cols_to_remove = {'phone number', } # 'churn' not included

print(cols)
print()
print(cols_to_remove)

sns.pairplot(df[cols], size=2.7)
plt.tight_layout()
plt.show()
cols
import numpy as np

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.0)
plt.figure(figsize=(10,10))
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 10},
                 xticklabels=cols,
                 yticklabels=cols)

plt.show()
cols_to_remove.update(['total day charge', 'total eve charge', 'total night charge', 'total intl charge'])
cols_to_remove
df.groupby(['churn']).mean()
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
df.groupby(['churn'])['account length'].plot(kind='kde', legend=True, ax=axs[0,0]); axs[0,0].set_title('account length')
df.groupby(['churn'])['number vmail messages'].plot(kind='kde', legend=True, ax=axs[0,1]); axs[0,1].set_title('number vmail messages')
df.groupby(['churn'])['total day minutes'].plot(kind='kde', legend=True, ax=axs[0,2]); axs[0,2].set_title('total day minutes')
df.groupby(['churn'])['total day calls'].plot(kind='kde', legend=True, ax=axs[0,3]); axs[0,3].set_title('total day calls')
df.groupby(['churn'])['total eve minutes'].plot(kind='kde', legend=True, ax=axs[1,0]); axs[1,0].set_title('total eve minutes')
df.groupby(['churn'])['total eve calls'].plot(kind='kde', legend=True, ax=axs[1,1]); axs[1,1].set_title('total eve calls')
df.groupby(['churn'])['total night minutes'].plot(kind='kde', legend=True, ax=axs[1,2]); axs[1,2].set_title('total night minutes')
df.groupby(['churn'])['total night calls'].plot(kind='kde', legend=True, ax=axs[1,3]); axs[1,3].set_title('total night calls')
df.groupby(['churn'])['total intl minutes'].plot(kind='kde', legend=True, ax=axs[2,0]); axs[2,0].set_title('total intl minutes')
df.groupby(['churn'])['total intl calls'].plot(kind='kde', legend=True, ax=axs[2,1]); axs[2,1].set_title('total intl calls')
df.groupby(['churn'])['customer service calls'].plot(kind='kde', legend=True, ax=axs[2,2]); axs[2,2].set_title('customer service calls');
# Because the classes are imbalanced, I think 'kde' is more preferred than 'hist' here.
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(x='international plan', hue='churn', data=df, ax=axs[0])
# axs[0].set_title('international plan')
sns.countplot(x='voice mail plan', hue='churn', data=df, ax=axs[1])
# axs[1].set_title('voice mail plan')
sns.countplot(x='customer service calls', hue='churn', data=df, ax=axs[2])
cols_to_remove
df2 = df.drop(list(cols_to_remove), axis=1)
df2.head()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# The following two are of multiple categories
# ohencoder = OneHotEncoder()

# TODO
# [Observations] 'state' and 'area code' columns are categorical and should be applied with one-hot encoding. 
# However due to 'state' is of high cardinality (51), this would negatively affect the prediction performance (low split gain)
# RF may be affected more than GBDT.
# 'area code' is of 3 categories. I applied one-hot encoding on it.


# The following three are of binary categories
label_encoder = LabelEncoder()
df2['international plan'] = label_encoder.fit_transform(df2['international plan'])
df2['voice mail plan'] = label_encoder.fit_transform(df2['voice mail plan'])
df2['churn'] = label_encoder.fit_transform(df2['churn'])


df2.head()
# one-hot encoding 'area code'
df2 = pd.get_dummies(df2, columns=['area code'], prefix='areacode', drop_first=True)
df2.head(10)
from sklearn.model_selection import train_test_split

X = df2.loc[:, [c for c in list(df2.columns) if c not in cols_to_remove | {'churn', 'state', 'area code'}]].values
y = df2.loc[:, 'churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0, ratio=1.0)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
X_train_balanced.shape, y_train_balanced.shape
import collections
collections.Counter(y_train_balanced)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=10, criterion='entropy')
rf.fit(X_train_balanced, y_train_balanced)
y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report, f1_score, roc_auc_score
print(classification_report(y_test, y_pred))

metric_result = pd.DataFrame(data=[['RandomForestClassifier', f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)]], 
                             columns=['algorithm', 'f1_score', 'roc_auc_score'])

del y_pred
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=3)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1)
ada.fit(X_train_balanced, y_train_balanced)
y_pred = ada.predict(X_test)

print(classification_report(y_test, y_pred))
metric_result.loc[1] = ['AdaBoostClassifier', f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)]

del y_pred
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(loss='deviance', n_estimators=100, max_depth=4)
gbc.fit(X_train_balanced, y_train_balanced)
y_pred = gbc.predict(X_test)

print(classification_report(y_test, y_pred))
metric_result.loc[2] = ['GradientBoostingClassifier', f1_score(y_test, y_pred), roc_auc_score(y_test, y_pred)]

del y_pred
metric_result
metric_result.plot(x='algorithm', kind='barh')
