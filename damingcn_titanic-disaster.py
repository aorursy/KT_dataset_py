import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('ggplot')
# read file

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



df_all = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)
# Overview: data size, column names, sample data

print('The train data size is: {}'.format(df_train.shape))

print('The train data size is: {}'.format(df_test.shape))

print('The all data size is: {}\n'.format(df_all.shape))





print('Train: {}'.format(df_train.columns))

print('Test: {}\n'.format(df_test.columns))

print('all: {}'.format(df_all.columns))
df_train.tail(3)
df_test.head(3)
# 1. EDA

# 1.1 overview, description of data

print(df_train.info(), '\n')

print(df_test.info())
# 1.2 missing data

print(df_train.isna().sum(), '\n')

print(df_test.isna().sum())
# 1.2.1 Age missing, usually median value

# df_all['Age'] = df_all.fillna(df_all['Age'].median())

# we take median age in group Pclass and Sex, as corr coef is 0.408106, and Pclass to survived corr coef is 0.338481

print(df_all.corr()[['Age', 'Survived']].abs().sort_values(by='Age', ascending=False), '\n')



print(df_all['Age'].median(), '\n')

print(df_all.groupby(['Pclass', 'Sex']).median()['Age'])



df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
# 1.2.2 Embarked missing, google it and know missing is 'S'

df_all['Embarked'] = df_all['Embarked'].fillna('S')
# 1.2.3 Fare missing, median value of SibSp, Parch and Pclass

df_all[df_all['Fare'].isna()]



med_fare = df_all.groupby(['SibSp', 'Parch', 'Pclass'])['Fare'].median()[0][0][3]

df_all['Fare'] = df_all['Fare'].fillna(med_fare)
df_all = df_all[df_all.columns[df_all.isnull().mean() < 0.7]]
# regain the train and test set

df_train = df_all[:891]

df_test = df_all[891:].drop('Survived', axis=1)
# 1.3 Target distribution

# survived %, non-survived %

sur_rate = df_train['Survived'].mean() * 100

non_sur_rate = (1 - sur_rate/100.) * 100



print('survived rate is: {:.2f}%'.format(sur_rate))

print('Non-survived rate is: {:.2f}%'.format(non_sur_rate))
fig, ax = plt.subplots(figsize=(10, 8))

sns.countplot(df_train['Survived'], ax=ax)

ax.set_xticklabels(['Non-survived({:.2f}%)'.format(non_sur_rate), 'Survived({:.2f}%)'.format(sur_rate)])
# 1.4 correlation

df_all_corr = df_all.corr().unstack().abs().sort_values(ascending=False).reset_index()

df_all_corr.columns= ['feature1', 'feature2', 'corr_coef']

df_all_corr = df_all_corr.drop(df_all_corr[df_all_corr['corr_coef'] == 1.0].index, axis=0)

df_all_corr = df_all_corr[0::2]

df_all_corr[df_all_corr['corr_coef'] > 0.1]
# correlation heatmap

fig, axs = plt.subplots(figsize=(10, 10))

sns.heatmap(df_train.corr(), annot=True, square=True, cmap='coolwarm')
fig, axs = plt.subplots(figsize=(10, 10))

sns.heatmap(df_test.corr(), annot=True, square=True, cmap='coolwarm')
# 1.5 target features

# 1.5.1 continous features, Age, Fare

cont_features = ['Age', 'Fare']

surv = df_train['Survived'] == 1.0



fig, axs = plt.subplots(1, 2, figsize=(20, 8))

for i, feature in enumerate(cont_features):

    sns.distplot(df_train[feature][surv], hist=True, label='Survived', ax=axs[i])

    sns.distplot(df_train[feature][~surv], hist=True, label='Non-Survived', ax=axs[i])

    axs[i].legend()
# 1.5.2 category features

cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']



fig, axs = plt.subplots(2, 3, figsize=(18, 12))

for i, feature in enumerate(cat_features, 1):

    plt.subplot(2, 3, i)

    sns.countplot(x=feature, hue='Survived', data=df_train)

    plt.legend(['Non-survived', 'Survived'])
# 2. Feature Engineering

# skewed data need to be binning

# 2.1 Binning continous features, Age

df_all['Age'] = pd.qcut(df_all['Age'], 12)
fig, ax = plt.subplots(figsize=(20, 8))

sns.countplot(x='Age', hue='Survived', data=df_all, ax=ax)

ax.legend(['Non-Survived', 'Survived'])
# 2.1.2 bining continous featetures, Fare

df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
fig, ax = plt.subplots(figsize=(20, 10))

sns.countplot(x='Fare', hue='Survived', data=df_all, ax=ax)

ax.legend(['Non-Survived', 'Survived'])
# 2.2 category feature frequency, family size

df_all['family_size'] = df_all['SibSp'] + df_all['Parch'] + 1

df_all['family_size'].value_counts()



family_map = {1:'Along', 2:'Small', 3:'Small', 4:'Small', 5:'Medium', 6:'Medium', 7:'Large', 8:'Large', 11:'Large'}

df_all['family_size_group'] = df_all['family_size'].map(family_map)



fig, axs = plt.subplots(1, 2, figsize=(18, 6))

sns.countplot(x='family_size', hue='Survived', data=df_all, ax=axs[0])

sns.countplot(x='family_size_group', hue='Survived', data=df_all, ax=axs[1])

axs[0].legend(['Non-Survived', 'Survived'])

axs[1].legend(['Non-Survived', 'Survived'])
# 2.2.2 cat feature on name, title and is married

df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

df_all['is_married'] = 0 

df_all['is_married'].loc[df_all['Title'] == 'Mrs'] = 1



df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')



fig, axs = plt.subplots(1, 2, figsize=(18, 6))

sns.countplot(x='Title', hue='Survived', data=df_all, ax=axs[0])

sns.countplot(x='is_married', hue='Survived', data=df_all, ax=axs[1])
df_train = df_all[:891]

df_test = df_all[891:]

dfs = [df_train, df_test]
# 2.3 feature transform

# 2.3.1 label encoding non numerical features

from sklearn.preprocessing import LabelEncoder



non_num_features = ['Age', 'Embarked', 'Fare', 'Sex', 'family_size_group', 'Title']



for df in dfs:

    for feature in non_num_features:

        df[feature] = LabelEncoder().fit_transform(df[feature])

# 2.3.2 one-hot encoding for categorical features

# Age and Fare features are not converted because they are ordinal unlike the previous ones.

from sklearn.preprocessing import OneHotEncoder



cat_features = ['Pclass', 'Embarked', 'Sex', 'family_size_group', 'Title']



encoded_features = []

for df in dfs:

    for feature in cat_features:

        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

        

        n = df[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df.index

        encoded_features.append(encoded_df)
df_train = pd.concat([df_train, *encoded_features[:5]], axis=1)

df_test = pd.concat([df_test, *encoded_features[5:]], axis=1)
df_train.columns
# 2.4 drop columns

drop_cols = ['Embarked', 'family_size', 'family_size_group', 'Survived',

             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title']
# 3. Model

# 3.1 train test split

from sklearn.preprocessing import StandardScaler



X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))

y_train = df_train['Survived']

X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))



print('X_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))

print('X_test shape: {}'.format(X_test.shape))
# 3.2 select the algorithms and training

# 3.2.1 KNN

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_predict_knn = knn.predict(X_test)
# 3.2.2 linear Logistic regression

from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)



print(log_reg.score(X_train, y_train))

y_predict_log = log_reg.predict(X_test)
algorithms = ['knn', 'log', 'svm']

submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])



submission_df['PassengerId'] = df_test['PassengerId']

submission_df['Survived'] = y_predict_log



submission_df['Survived'] = submission_df['Survived'].astype(int)
submission_df.to_csv('submissions_log.csv', header=True, index=False)

submission_df.head(10)
# 3.2.3 random forest

from sklearn.ensemble import RandomForestClassifier



rf_clf = RandomForestClassifier(n_estimators=500, oob_score=True, min_samples_split=10, min_samples_leaf=3, n_jobs=-1)

rf_clf.fit(X_train, y_train)
rf_clf.oob_score_
y_predict_rf = rf_clf.predict(X_test)
submission_df['PassengerId'] = df_test['PassengerId']

submission_df['Survived'] = y_predict_rf



submission_df['Survived'] = submission_df['Survived'].astype(int)



submission_df.to_csv('submissions_rf.csv', header=True, index=False)

submission_df.head(10)