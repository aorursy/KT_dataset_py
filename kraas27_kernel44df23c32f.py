import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns
data = pd.read_csv("../input/train.csv")

data.head(5)
df_test = pd.read_csv("../input/test.csv")

df_test.head(3)
sns.heatmap(data.corr()) # проверим признаки на корреляцию
y_train = data['Survived'].copy()   # TRAIN Y

df_train = data.drop('Survived', axis=1)
df_train['is_train'] = 1

df_test['is_train'] = 0

df_all = pd.concat([df_train, df_test])
df_all.head(3)
df_all.describe()
df_all['is_male'] = df_all['Sex'].replace({'male': 1, 'female': 0})

df_all = df_all.drop('Sex', axis=1)
df_all.info()
# Заполним пропуски "Cabin", так как скорее всего это это что то может значить

df_all['Cabin'] = df_all['Cabin'].fillna('NaN')

df_all['Cabin'] = df_all['Cabin'].apply(lambda x: x[0])
df_all['Cabin'].value_counts(normalize=True)
df_all = df_all.drop('Ticket', axis=1)
df_all.head()
df_all['Name'] = df_all['Name'].apply(lambda x: x.strip().split()[1])
df_all['Name'].unique()
# Все аббревиатуры не заканчивающиеся точкой отправим в другое



df_all['Name'] = df_all['Name'].apply(lambda x: x if x[-1] == '.' else 'other')
df_all['Name'].unique()
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 5)
from collections import Counter
one_big_text = " ".join(df_all['Name'])

words = one_big_text.split()

most_common = Counter(words).most_common()

most_common
abbr = pd.DataFrame()

for col, num in most_common:

    abbr[col] = df_all[df_all['is_train'] == 1]['Name'].str.contains(col).astype(int)
clf.fit(abbr, y_train)
df_all['Name'].unique()
clf.feature_importances_
df_all['Name'].value_counts()
top_names = df_all['Name'].value_counts().head()

df_all['Name'] = df_all['Name'].apply(lambda x: x if x in top_names else 'other')
df_all['Name'].value_counts(normalize=True)
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean')

df_all['Age'] = imp.fit_transform(df_all[['Age']]).ravel()

df_all['Fare'] = imp.fit_transform(df_all[['Fare']]).ravel()
df_all = pd.get_dummies(df_all, ['Embarked', 'Name', 'Cabin'], drop_first=False, dummy_na=False)
df_all.head(5)
df_all.isnull().sum()
df_all[df_all['is_train']==1].shape, df_all[df_all['is_train']==0].shape
df_train = df_all[df_all['is_train']==1]

df_test = df_all[df_all['is_train']==0]
df_train = df_train.drop('is_train', axis=1)

df_test = df_test.drop('is_train', axis=1)
df_train.shape, y_train.shape
from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier()
max_depth = [3, 4, 5, 7, 9, 11, 15] 

min_samples_leaf = np.arange(3,13)

param_grid = { 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
tree_grid = GridSearchCV(tree, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
tree_grid.fit(df_train, y_train)
print (tree_grid.best_params_)

print (tree_grid.best_score_)
tree_clf = tree_grid.best_estimator_
for i, j in zip(tree_clf.feature_importances_, df_train.columns):

    print("{:.2f}   {:10}".format(i, j))
y_pred = tree_clf.predict(df_test)
from sklearn.tree import export_graphviz
tree_clf.classes_
export_graphviz(tree_clf, out_file='titanic_tree_.dot', filled=True, feature_names=df_train.columns, \

                class_names=list(['Death', 'Survived']))
!dot -Tpng titanic_tree_.dot -o titanic_tree_.png