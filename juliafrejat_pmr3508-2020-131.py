import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

from sklearn import preprocessing
train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")
train_data.head()
train_data.shape
train_data.describe()
train_data.info()
train_data.isnull().sum().sort_values(ascending = False).head()
print(train_data['occupation'].describe(), '\n')

print(train_data['workclass'].describe(), '\n')

print(train_data['native.country'].describe(), '\n')
train_data['occupation'] = train_data['occupation'].fillna(train_data['occupation'].describe().top)

train_data['workclass'] = train_data['workclass'].fillna(train_data['workclass'].describe().top)

train_data['native.country'] = train_data['native.country'].fillna(train_data['native.country'].describe().top)
train_data.isnull().sum().sort_values(ascending = False).head()
%%capture

encoder = preprocessing.LabelEncoder()

graph_data  = train_data.copy()

graph_data['income'] = encoder.fit_transform(graph_data['income'])

graph_data = graph_data.drop(['Id'], axis = 1)
sns.pairplot(graph_data, diag_kws={'bw':"1.0"}, hue = 'income', palette='rocket')
plt.figure(figsize=(10, 8))

sns.heatmap(graph_data.corr(), annot=True, vmin=-1, vmax=1, cmap = 'rocket')
sns.lineplot(x="income", y='age', hue="sex", data=train_data, palette='rocket')
sns.countplot(x="income", hue='workclass', data=train_data, palette='rocket')
sns.lineplot(x="income", y='fnlwgt', data=train_data, palette='rocket')
sns.barplot(x="income", y="education.num", data=train_data, palette='rocket')
plt.figure(figsize=(20,4))

sns.countplot(x="education", hue='income', data=train_data, palette='rocket')
plt.figure(figsize=(16,4))

sns.countplot(x="marital.status", hue='income', data=train_data, palette='rocket')
plt.figure(figsize=(10,4))

sns.countplot(x="relationship", hue='income', data=train_data, palette='rocket')
plt.figure(figsize=(25,4))

sns.countplot(x="occupation", hue='income', data=train_data, palette='rocket')
plt.figure(figsize=(10,4))

sns.countplot(x="race", hue='income', data=train_data, palette='rocket')

train_data["race"].value_counts()
sns.countplot(x="sex", hue='income', data=train_data, palette='rocket')
sns.lineplot(x="income", y='capital.gain', data=train_data, palette='rocket')
sns.lineplot(x="income", y='capital.loss', data=train_data, palette='rocket')
sns.lineplot(x="income", y='hours.per.week', data=train_data, palette='rocket')
plt.figure(figsize=(10,30))

sns.boxplot(x="hours.per.week", y='age', orient='h', data=train_data, palette='rocket')
train_data["native.country"].value_counts()
train_data.drop_duplicates(keep='first', inplace=True)
train_data = train_data.drop(['fnlwgt', 'education', 'relationship', 'race', 'native.country'], axis=1)
train_data.head()
num_columns = ['Id','age','education.num', 'hours.per.week']

spa_columns = ['capital.gain', 'capital.loss']

cat_columns = ['workclass','marital.status','occupation','sex']
for column in cat_columns:

    train_data[column] = encoder.fit_transform(train_data[column])
train_data.head()
X = train_data.drop(['income'], axis = 1)

Y = train_data['income']
from sklearn.compose import ColumnTransformer

s_scaler = preprocessing.StandardScaler()

r_scaler = preprocessing.RobustScaler()



transformer = ColumnTransformer(transformers= [('num', s_scaler, num_columns),('spr', r_scaler, spa_columns),('cat', s_scaler, cat_columns)])

X = transformer.fit_transform(X)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
#primeira busca

neighbors =[5, 10, 15, 20, 25, 30]



i = 0

scores = [0]*6

for number in neighbors:

    score = cross_val_score(KNeighborsClassifier(n_neighbors=number), X, Y, cv = 10)

    scores[i] = score.mean()

    i += 1

best_k = neighbors[scores.index(max(scores))]



#busca refinada

refined_neighbors =[0]*5

for i in range(5):

    refined_neighbors[i] = best_k + (i-2)



i = 0

refined_scores = [0]*5

for number in refined_neighbors:

    score = cross_val_score(KNeighborsClassifier(n_neighbors=number), X, Y, cv = 10)

    refined_scores[i] = score.mean()

    i += 1



refined_best_k = refined_neighbors[refined_scores.index(max(refined_scores))]

print('Melhor k: ', refined_best_k)

print('Melhor acurácia: ', max(scores))
knn = KNeighborsClassifier(n_neighbors=refined_best_k)

knn.fit(X, Y)
test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")
X_test = test_data.drop(['fnlwgt', 'education', 'relationship', 'race', 'native.country'], axis=1)
X_test.head()
X_test.isnull().sum().sort_values(ascending = False).head()
X_test['occupation'] = X_test['occupation'].fillna(X_test['occupation'].describe().top)

X_test['workclass'] = X_test['workclass'].fillna(X_test['workclass'].describe().top)
X_test.isnull().sum().sort_values(ascending = False).head()
for column in cat_columns:

    X_test[column] = encoder.fit_transform(X_test[column])

X_test.head()
transformer = ColumnTransformer(transformers= [('num', s_scaler, num_columns),('spr', r_scaler, spa_columns),('cat', s_scaler, cat_columns)])

X_test = transformer.fit_transform(X_test)
predictions = knn.predict(X_test)
submission = pd.DataFrame()

submission[0] = test_data.index

submission[1] = predictions

submission.columns = ['Id','income']

submission.head()
submission.to_csv('submission.csv', index = False)