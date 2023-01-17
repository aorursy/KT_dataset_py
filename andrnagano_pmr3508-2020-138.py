import pandas as pd

import numpy as np



import sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import KNNImputer

from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



import matplotlib.pyplot as plt

import seaborn as sns
adult_Train = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", index_col= ['Id'], na_values="?")
adult_Train.head()
adult_Train.isna().sum()
adult_Train["income"] = LabelEncoder().fit_transform(adult_Train["income"])
plt.figure(figsize=(10,10))

sns.heatmap(adult_Train.corr(), annot=True, vmin=-1, vmax=1)

plt.show()
sns.boxplot(x='income', y='age', data=adult_Train)
sns.boxplot(x='income', y='education.num', data=adult_Train)
sns.boxplot(x='income', y='hours.per.week', data=adult_Train)
sns.boxplot(x='income', y='capital.gain', data=adult_Train)
sns.boxplot(x='income', y='capital.loss', data=adult_Train)
sns.barplot(x='income', y='workclass', data=adult_Train)
sns.barplot(x='income', y='education', data=adult_Train)
sns.barplot(x='income', y='marital.status', data=adult_Train)
sns.barplot(x='income', y='occupation', data=adult_Train)
sns.barplot(x='income', y='relationship', data=adult_Train)
sns.barplot(x='income', y='race', data=adult_Train)
sns.barplot(x='income', y='sex', data=adult_Train)
sns.barplot(x='income', y='native.country', data=adult_Train)
adult_Train['native.country'].value_counts()
adult_Train['US'] = (adult_Train['native.country'] == 'United-States').astype(int)
adult_Train = adult_Train.drop(['native.country', 'fnlwgt', 'education'], axis=1)
adult_Train.head()
Yadult_Train = adult_Train.pop('income')

Xadult_Train = adult_Train
Xadult_Train.head()
Yadult_Train.head()
pipeline_categorico = Pipeline(steps = 

                 [('imputer', SimpleImputer(strategy = 'most_frequent')),

                  ('onehot', OneHotEncoder(drop='first', sparse=False))])
pipeline_numerico = Pipeline(steps = 

                [('imputer', KNNImputer(n_neighbors=10, weights="uniform")),

                 ('scaler', StandardScaler())])
pipeline_esparso = Pipeline(steps = 

                [('imputer', KNNImputer(n_neighbors=10, weights="uniform")),

                 ('scaler', RobustScaler())])
preprocessor = ColumnTransformer(transformers = [

    ('spr', pipeline_esparso, ['capital.gain', 'capital.loss']),

    ('cat', pipeline_categorico, ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex']),

    ('num', pipeline_numerico, ['age', 'education.num', 'hours.per.week'])

])
Xadult_Train = preprocessor.fit_transform(Xadult_Train)
knn = KNeighborsClassifier(n_neighbors=19)
Xadult_Train.head()
score = cross_val_score(knn, Xadult_Train, Yadult_Train, cv = 5, scoring="accuracy")

print(score.mean())
knn.fit(Xadult_Train, Yadult_Train)
adult_Teste = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", index_col=["Id"], na_values="?")
adult_Teste['US'] = (adult_Teste['native.country'] == 'United-States').astype(int)
Xadult_Teste = adult_Teste.drop(['native.country', 'fnlwgt', 'education'], axis=1)
Xadult_Teste = preprocessor.transform(Xadult_Teste)
pred = knn.predict(Xadult_Teste)
inc = []

for i in pred:

    if i == 0:

        inc.append("<=50K")

    else:

        inc.append(">50K")

        

sub = pd.DataFrame()

sub[0] = adult_Teste.index

sub[1] = inc

sub.columns = ['Id', 'income']
sub.to_csv('submission.csv', index = False)