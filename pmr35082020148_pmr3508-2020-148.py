import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import CategoricalNB

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import cross_val_predict, cross_val_score, ShuffleSplit

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
df_adults = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "age", "workclass", "fnlwgt", "education", "education-num", "martial status",

        "occupation", "relationship", "race", "sex", "capital gain", "capital loss",

        "hours per week", "country", "target"],

        sep=r'\s*,\s*',

        skiprows=1,

        index_col=0,

        engine='python',

        na_values="?")
df_adults.info()
df_adults.head()
df_adults.describe()
X_test = pd.read_csv("../input/adult-pmr3508/test_data.csv",

            names=[

            "id", "age", "workclass", "fnlwgt", "education", "education-num", "martial status",

            "occupation", "relationship", "race", "sex", "capital gain", "capital loss",

            "hours per week", "country"],

            sep=r'\s*,\s*',

            skiprows=1,

            index_col=0,

            engine='python',

            na_values="?")
X_test.info()
X_test.head()
X_test.describe()
df_adults_backup = df_adults.copy()

X_test_backup = X_test.copy()
df_adults.columns
plt.figure(figsize=(12, 6))

sns.heatmap(df_adults.isna(), yticklabels=False, cbar=False, cmap='cividis')
df_adults.isna().sum()
sns.pairplot(df_adults)
df_adults.groupby(by=['country', 'target'])['target'].count()
df_adults[df_adults['country'].isna()]['target'].value_counts() / df_adults['target'].value_counts() * 100

# Porcentagem dos dados ausentes da variável 'country' em relação ao total (em %)
df_adults[df_adults['country'] == 'United-States']['target'].value_counts()
df_adults['country'].unique()
NA = ['Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'Columbia', 'United-States', 'Haiti', 'Jamaica', 'Mexico', 'Guatemala', 'El-Salvador', 'Cuba', 'Puerto-Rico', 'Canada', 'Honduras', 'Nicaragua', 'Dominican-Republic']

EU = ['England', 'France', 'Portugal', 'Yugoslavia', 'Poland', 'Scotland', 'Germany', 'Italy', 'Holand-Netherlands', 'Greece', 'Hungary', 'Ireland']

AS = ['Philippines', 'Iran', 'South', 'China', 'Hong', 'Japan', 'Cambodia', 'Thailand', 'Laos', 'Taiwan', 'Vietnam', 'India']

SA = ['Peru', 'Ecuador']



def country_to_continent(country):

    if country in NA:

        return 'NA'

    

    if country in EU:

        return 'EU'

    

    if country in AS:

        return 'AS'

    

    if country in SA:

        return 'SA'

    

    return None
df_adults['country'] = pd.DataFrame(list(map(country_to_continent, df_adults['country'])))
df_adults.groupby(by=['country', 'target'])['target'].count()
names = (df_adults['country'].value_counts().index)

size = list(df_adults['country'].value_counts())

 

my_circle=plt.Circle((0,0), 0.7, color='white')



plt.figure(figsize=(5, 5))

plt.pie(size, labels=names, colors=['tomato', 'skyblue', 'orange', 'pink'])

p = plt.gcf()

p.gca().add_artist(my_circle)
df_adults[df_adults['workclass'].isna()]['target'].value_counts() / df_adults['target'].value_counts() * 100

# Porcentagem dos dados ausentes da variável 'workclass' em relação ao total (em %)
plt.figure(figsize=(10, 6))

sns.countplot(y='workclass', data=df_adults, hue='target', palette='rainbow')
df_adults['workclass'].value_counts()
df_adults[df_adults['occupation'].isna()]['target'].value_counts() / df_adults['target'].value_counts() * 100

# Porcentagem dos dados ausentes da variável 'occupation' em relação ao total (em %)
plt.figure(figsize=(20, 6))

sns.countplot(y='occupation', data=df_adults, hue='target', palette='rainbow')
plt.figure(figsize=(12, 6))

sns.heatmap(data=df_adults.corr(), cmap='YlGnBu', linewidths=0.3, annot=True)
df_adults[['education', 'education-num']].drop_duplicates(subset=['education', 'education-num']).sort_values(by='education-num', ascending=True)
sns.countplot(x='target', data=df_adults, hue='sex', palette='coolwarm')
df_adults.groupby(by=['sex', 'target'])['target'].count()
plt.figure(figsize=(10, 6))

sns.countplot(x='target', data=df_adults, hue='race', palette='mako')
fg_rel = sns.FacetGrid(data=df_adults, col='relationship')

fg_rel.map(plt.hist, 'target').set(xlim=(0, 1), xticks=[0, 1]).add_legend()
names = (df_adults['relationship'].value_counts().index)

size = list(df_adults['relationship'].value_counts())



my_circle=plt.Circle((0,0), 0.7, color='white')



plt.figure(figsize=(6, 6))

plt.pie(size, labels=names, colors=['tomato', 'skyblue', 'orange', 'pink', 'yellow', 'lime'])

p = plt.gcf()

p.gca().add_artist(my_circle)
df_adults.groupby(by=['relationship', 'target'])['target'].count()
fg_ms = sns.FacetGrid(data=df_adults, col='martial status')

fg_ms.map(plt.hist, 'target').set(xlim=(0, 1), xticks=[0, 1]).add_legend()
names = (df_adults['martial status'].value_counts().index)

size = list(df_adults['martial status'].value_counts())



my_circle=plt.Circle((0,0), 0.7, color='white')



plt.figure(figsize=(6, 6))

plt.pie(size, labels=names, colors=['tomato', 'skyblue', 'orange', 'pink', 'yellow', 'lime', 'cyan'])

p = plt.gcf()

p.gca().add_artist(my_circle)
df_adults.groupby(by=['martial status', 'target'])['target'].count()
plt.figure(figsize=(8, 8))

sns.stripplot(x='target', y='age', hue='sex', data=df_adults, palette='nipy_spectral')
sns.violinplot(x='target', y='age', hue='sex', data=df_adults, split=True)
sns.boxplot(x='age', y='target', data=df_adults, palette='plasma')
plt.figure(figsize=(6, 6))

sns.stripplot(x='target', y='hours per week', data=df_adults, palette='plasma')
sns.boxplot(x='hours per week', y='target', data=df_adults, palette='magma')
sns.boxplot(x='capital gain', y='target', data=df_adults)
(df_adults.groupby(by='target')['capital gain'].sum()).plot.pie(y='capital gain', figsize=(5, 5))
sns.boxplot(x='capital loss', y='target', data=df_adults)
(df_adults.groupby(by='target')['capital loss'].sum()).plot.pie(y='capital loss', figsize=(5, 5))
df_adults.drop(['fnlwgt', 'education', 'martial status', 'race', 'country'], axis=1, inplace=True)

df_adults = pd.get_dummies(data=df_adults, columns=['workclass', 'occupation', 'relationship', 'sex', 'target'], drop_first=True)
plt.figure(figsize=(15, 15))

sns.heatmap(data=df_adults.corr(), cmap='inferno', linewidths=0.1)
df_adults = df_adults_backup.copy()

X_test = X_test_backup.copy()
def fillna_label(df, label, drop_columns, dummies_columns, naive_bayes, inplace, df_test=None):

    df_fill = df.copy() 

    df_fill.drop(drop_columns, axis=1, inplace=True)

    

    if df_test is not None: 

        df_fill.drop('target', axis=1, inplace=True)

        if 'target' in dummies_columns:

            dummies_columns.remove('target')

        

    df_fill_final = pd.get_dummies(data=df_fill, columns=dummies_columns, drop_first=True)

    

    X = df_fill_final[df_fill_final[label].notna()].drop(label, axis=1)

    y = df_fill_final[df_fill_final[label].notna()][label]

    

    if df_test is not None:

        df_test_fill = df_test.copy()

        df_test_fill.drop(drop_columns, axis=1, inplace=True)

        df_test_fill_final = pd.get_dummies(data=df_test_fill, columns=dummies_columns, drop_first=True)

    

    if df_test is not None:

        X_t = df_test_fill_final[df_test_fill_final[label].isna()].drop(label, axis=1)        

    else:

        X_t = df_fill_final[df_fill_final[label].isna()].drop(label, axis=1)

    

    y_pred = naive_bayes.fit(X, y).predict(X_t)

    fill = pd.Series(data=y_pred, index=X_t.index, name=label)

    

    if df_test is not None:

        df_test[label].fillna(fill, inplace=inplace)

    else:

        df[label].fillna(fill, inplace=inplace)    
drop_columns_workclass = ['fnlwgt', 'occupation', 'education', 'country']

drop_columns_occupation = ['workclass', 'fnlwgt', 'education', 'country']



dummies_columns = ['martial status', 'relationship', 'race', 'sex', 'target']
fillna_label(df_adults, 'workclass', drop_columns_workclass, dummies_columns, CategoricalNB(), inplace=True)

fillna_label(df_adults, 'occupation', drop_columns_occupation, dummies_columns, CategoricalNB(), inplace=True)
fillna_label(df_adults, 'workclass', drop_columns_workclass, dummies_columns, CategoricalNB(), inplace=True, df_test=X_test)

fillna_label(df_adults, 'occupation', drop_columns_occupation, dummies_columns, CategoricalNB(), inplace=True, df_test=X_test)
df_adults.dropna(axis=1, how='any', inplace=True)

df_adults.drop(['fnlwgt', 'education', 'martial status', 'race'], axis=1, inplace=True)

df_adults_final = pd.get_dummies(data=df_adults, columns=['workclass', 'occupation', 'relationship', 'sex'], drop_first=True)
adults_scaler = RobustScaler(quantile_range=(10, 90)).fit_transform(df_adults_final.drop('target', axis=1))

df_adults_scaler = pd.DataFrame(adults_scaler, columns=df_adults_final.drop('target', axis=1).columns)
X = df_adults_scaler

y = df_adults_final['target']
X_test.dropna(axis=1, how='any', inplace=True)

X_test.drop(['fnlwgt', 'education', 'martial status', 'race'], axis=1, inplace=True)

X_test = pd.get_dummies(data=X_test, columns=['workclass', 'occupation', 'relationship', 'sex'], drop_first=True)
X_test_scaler = RobustScaler(quantile_range=(21, 79)).fit_transform(X_test)

X_test_final = pd.DataFrame(X_test_scaler, columns=X_test.columns)
def predict_training_KNN(X, y, K, X_test=None, score=True, n_splits=10, test_size=0.2, random_state=39):

    knn = KNeighborsClassifier(n_neighbors=K)

    knn.fit(X, y)

    

    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    

    if score:

        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')

        

        if X_test is not None:

            y_pred = knn.predict(X_test)

            return y_pred, scores

        

        return scores

    else:

        y_pred = cross_val_predict(knn, X, y, cv=n_splits)

        return y_pred
y_pred = predict_training_KNN(X, y, K=3, score=False)
print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))
accuracy = []

for i in range(1, 30):

    scores = predict_training_KNN(X, y, K=i)

    accuracy.append(scores.mean())
plt.figure(figsize=(14, 8))

plt.plot(range(1, 30), accuracy, color='blue', linestyle='dashed', marker='o')

plt.xlabel('K')

plt.ylabel('Accuracy')



value_K = accuracy.index(max(accuracy)) + 1



print(f'K minimo = {accuracy.index(max(accuracy)) + 1}')
Y_pred, scores = predict_training_KNN(X, y, K=value_K, X_test=X_test_final)
scores.mean()
file_name = 'sample_submission'

pd.DataFrame(data=Y_pred, columns=['income']).to_csv(path_or_buf=f'./{file_name}.csv' , sep=',', index_label='Id', encoding='utf-8')