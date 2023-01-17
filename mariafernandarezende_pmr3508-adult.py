import numpy as np

import pandas as pd

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

data_train = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        index_col=['Id'],engine='python', na_values="?")
data_train.shape
data_train.head()
data_train.info()
data_train.describe()
data_test = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        index_col=['Id'], na_values="?")
data_test.shape
data_test.head()
data_test.info()
data_test.columns

data_train['income']
data_train['income'] = LabelEncoder().fit_transform(data_train['income'])
data_train['income']
sns.pairplot(data_train, vars=['age','education.num', 'hours.per.week', 'fnlwgt'], hue="income", palette="rocket")
plt.figure(figsize=(12, 6))

sns.heatmap(data=data_train.corr(), cmap='rocket', linewidths=0.3, annot=True)
sns.catplot(x="income", y="age", kind="boxen", data=data_train)
sns.catplot(x="income", y="education.num", kind="boxen", data=data_train)
sns.catplot(x="income", y="capital.gain", kind="boxen", data=data_train)

sns.catplot(x="income", y="capital.gain", data=data_train)
sns.catplot(x="income", y="capital.loss", kind="boxen", data=data_train)

sns.catplot(x="income", y="capital.loss", data=data_train)
sns.catplot(y="workclass", x="income", kind="bar", data=data_train);
sns.catplot(y="occupation", x="income", kind="bar", data=data_train);
sns.catplot(y="relationship", x="income", kind="bar", data=data_train);
sns.catplot(y="race", x="income", kind="bar", data=data_train);
sns.catplot(y="sex", x="income", kind="bar", data=data_train);
sns.catplot(y="native.country", x="income", kind="bar", data=data_train);
data_train["native.country"].value_counts()
def native_country_col(value):

    if value == 'United-States':

      return 1

    else:

      return 0



data_train['native.country'] = data_train['native.country'].apply(native_country_col)
data_train["native.country"].value_counts()
sns.catplot(y="native.country", x="income", kind="violin", data=data_train);
data_train['native.country'].describe()
data_train.drop_duplicates(keep='first', inplace=True)
data_train[['education', 'education.num']].drop_duplicates(subset=['education', 'education.num']).sort_values(by='education.num', ascending=True)
data_train = data_train.drop(['fnlwgt', 'education'], axis=1)
plt.figure(figsize=(12, 6))

sns.heatmap(data_train.isna(), yticklabels=False, cbar=False, cmap='rocket')
data_train.isna().sum()
data_train = data_train.dropna()

data_train.shape
data_train.columns
Y_train = data_train.pop('income')

num = data_train[['age', 'education.num', 'hours.per.week']].copy()

cat = data_train[['workclass','marital.status', 'occupation', 'relationship', 'race', 'sex']].copy()

sparse = data_train[['capital.gain', 'capital.loss', 'native.country']].copy()



num.reset_index(drop=True, inplace=True)

cat.reset_index(drop=True, inplace=True)

sparse.reset_index(drop=True, inplace=True)
cat = pd.get_dummies(cat)
cat.head()
cat.drop('sex_Female', axis=1, inplace=True)
def newtable(table, meia_tabela):

    for i in meia_tabela.columns:

        table[i] = meia_tabela[i]

    return table
cat.reset_index(drop=True, inplace=True)



X_train = pd.DataFrame()



X_train = newtable(X_train, cat)

X_train = newtable(X_train, num)

X_train = newtable(X_train, sparse)
X_train.columns
bestScore = 0

for i in range (10,36):

  knn = KNeighborsClassifier(n_neighbors= i)

  score = cross_val_score(knn, X_train, Y_train, cv = 5, scoring="accuracy")

  if score.mean() > bestScore:

    bestScore = score.mean()

    knnKey = i



print("Acurácia com cross validation:", bestScore.mean())

print("Melhor K:",knnKey)

  

data_test.columns
data_test['native.country'] = data_test['native.country'].apply(native_country_col)

data_test.drop_duplicates(keep='first', inplace=True)

data_test = data_test.drop(['fnlwgt', 'education'], axis=1)

data_test = data_test.dropna()





num_test = data_test[['age', 'education.num', 'hours.per.week']].copy()

cat_test = data_test[['workclass','marital.status', 'occupation', 'relationship', 'race', 'sex']].copy()

sparse_test = data_test[['capital.gain', 'capital.loss', 'native.country']].copy()



num_test.reset_index(drop=True, inplace=True)

cat_test.reset_index(drop=True, inplace=True)

sparse_test.reset_index(drop=True, inplace=True)



cat_test = pd.get_dummies(cat_test)



cat_test.drop('sex_Female', axis=1, inplace=True)



cat_test.reset_index(drop=True, inplace=True)



X_test = pd.DataFrame()



X_test = newtable(X_test, cat_test)

X_test = newtable(X_test, num_test)

X_test = newtable(X_test, sparse_test)
X_test.head()
knn = KNeighborsClassifier(n_neighbors=14)

knn.fit(X_train, Y_train)

predicao = knn.predict(X_test)
predict_df = pd.DataFrame()

predict_df[0] = X_test.index

predict_df[1] = predicao

predict_df.columns = ['Id','Income']

predict_df.set_index('Id', inplace=True)
predict_df[predict_df['Income'] == 0] = '<=50K'

predict_df[predict_df['Income'] == 1] = '>50K'
predict_df.head()
predict_df.to_csv('predict_df.csv')