import numpy as np

import pandas as pd
col_names = ["Age", "Workclass", "Final Weight", "Education", "Education-Num",

             "Marital Status", "Occupation", "Relationship", "Race", "Sex",

             "Capital Gain", "Capital Loss", "Hours per week", "Country", "Income"]



df = pd.read_csv('../input/adult-pmr3508/train_data.csv', index_col='Id', na_values='?')

df.columns = col_names
df.head(10)
df.info()
df.describe()
df.describe(include='object')
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
with_na = ['Workclass', 'Occupation', 'Country']



for col in with_na:

    graf = sns.countplot(x=col, data=df, hue='Income')

    graf.set_xticklabels(graf.get_xticklabels(), rotation=90)

    plt.show()
for col in with_na:

    print('-'*15 + ' ' + col + ' ' + '-'*15)

    print()

    print(df[col].value_counts())

    print()
df.drop('Country', axis=1, inplace=True)
sns.pairplot(df, hue='Income', diag_kws={'bw':"1.0"})

plt.tight_layout()

plt.show()
cols = ['Age', 'Final Weight', 'Education-Num',

        'Capital Gain', 'Capital Loss', 'Hours per week']



plt.figure(figsize=(10,8))

for i in range(len(cols)):

    plt.subplot(3, 2, i+1)

    sns.distplot(df[cols[i]], kde_kws={'bw':"1.0"})

plt.tight_layout()

plt.show()
corr = df.corr()

sns.heatmap(corr, square=True, cmap="coolwarm", center=0, annot=True)

plt.show()
other_cat = ['Education', 'Marital Status', 'Relationship', 'Race', 'Sex']



plt.figure(figsize=(10,10))

for i in range(len(other_cat)):

    plt.subplot(3, 2, i+1)

    graf = sns.countplot(x=other_cat[i], data=df, hue='Income')

    graf.set_xticklabels(graf.get_xticklabels(), rotation=90)

plt.tight_layout()

plt.show()
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Normalizer
df.drop(['Final Weight', 'Education'], axis=1, inplace=True)
df.drop_duplicates(keep='first', inplace=True)
df.isna().sum()
df[df['Workclass'].isnull() | df['Occupation'].isnull()]
df.dropna(inplace=True)
categorical = df[['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex']].copy()

numerical_std = df[['Age']].copy()

numerical_MaxMin = df[['Education-Num', 'Hours per week']].copy()

sparse = df[['Capital Gain', 'Capital Loss']].copy()

target = df[['Income']].copy()



categorical.reset_index(drop=True, inplace=True)

numerical_std.reset_index(drop=True, inplace=True)

numerical_MaxMin.reset_index(drop=True, inplace=True)

sparse.reset_index(drop=True, inplace=True)

target.reset_index(drop=True, inplace=True)
categorical = pd.get_dummies(categorical)

categorical.drop('Sex_Female', axis=1, inplace=True)
def montar(clean, df):

    for col in df.columns:

        clean[col] = df[col]

    return clean
std = StandardScaler()

numerical_std = std.fit_transform(numerical_std)

numerical_std = pd.DataFrame({'Age' : numerical_std[:, 0]})
MM = MinMaxScaler()

numerical_MaxMin = MM.fit_transform(numerical_MaxMin)

numerical_MaxMin = pd.DataFrame({'Education-Num' : numerical_MaxMin[:, 0],

                                 'Hours per week' : numerical_MaxMin[:, 1]})
RS = RobustScaler()

sparse = RS.fit_transform(sparse)

sparse = pd.DataFrame({'Capital Gain' : sparse[:, 0],

                       'Capital Loss' : sparse[:, 1]})
target = pd.get_dummies(target)

target.drop('Income_<=50K', axis=1, inplace=True)

target.columns = ['Income']

y = target

y = y['Income'].to_numpy()
categorical.reset_index(drop=True, inplace=True)



X = pd.DataFrame()



X = montar(X, categorical)

X = montar(X, numerical_std)

X = montar(X, numerical_MaxMin)

X = montar(X, sparse)
val = pd.read_csv('../input/adult-pmr3508/test_data.csv', na_values='?')

ids = val['Id']

val.columns = ['Id'] + col_names[:-1]

val.drop(['Id', 'Country', 'Final Weight', 'Education'], axis=1, inplace=True)



val_cat = val[['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex']].copy()

val_std = val[['Age']].copy()

val_MM = val[['Education-Num', 'Hours per week']].copy()

val_spr = val[['Capital Gain', 'Capital Loss']].copy()



val_cat.reset_index(drop=True, inplace=True)

val_std.reset_index(drop=True, inplace=True)

val_MM.reset_index(drop=True, inplace=True)

val_spr.reset_index(drop=True, inplace=True)



val_cat = pd.get_dummies(val_cat)

val_cat.drop('Sex_Female', axis=1, inplace=True)



val_std = std.transform(val_std)

val_std = pd.DataFrame({'Age' : val_std[:, 0]})



val_MM = MM.transform(val_MM)

val_MM = pd.DataFrame({'Education-Num' : val_MM[:, 0],

                       'Hours per week' : val_MM[:, 1]})



val_spr = RS.transform(val_spr)

val_spr = pd.DataFrame({'Capital Gain' : val_spr[:, 0],

                        'Capital Loss' : val_spr[:, 1]})



validation = pd.DataFrame()

validation = montar(validation, val_cat)

validation = montar(validation, val_std)

validation = montar(validation, val_MM)

validation = montar(validation, val_spr)



validation['Id'] = ids

validation.set_index('Id', inplace=True)

validation.drop('Workclass_Never-worked', axis=1, inplace=True)
import sklearn

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
K = list(range(15, 36))

scores = []



for k in K:

    knn = KNeighborsClassifier(n_neighbors=k)

    score = cross_val_score(knn, X, y, cv = 5, scoring="accuracy").mean()

    scores.append(score)
best_score = max(scores)

best_K = K[scores.index(best_score)]



print('A melhor acurácia foi de: ', best_score)

print('O melhor K foi: ', best_K)
knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(X, y)

predictions = knn.predict(validation)
submission = pd.DataFrame()

submission[0] = validation.index

submission[1] = predictions

submission.columns = ['Id','Income']

submission.set_index('Id', inplace=True)
submission[submission['Income'] == 0] = '<=50K'

submission[submission['Income'] == 1] = '>50K'
submission.to_csv('submission.csv')