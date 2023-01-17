import numpy as np 

import pandas as pd

import math

import random

import seaborn as sns  

import matplotlib.pyplot as plt

import plotly.express as px



from sklearn import tree

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

test = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')

train = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')
df = pd.concat([train, test], axis=0, sort=False) 
df.head(2)
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female')) # !

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived



df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)

df['Alone'] = (df.WomanOrBoyCount == 0) # !



# Title !

df['Title'] = df['Title'].replace('Ms','Miss')

df['Title'] = df['Title'].replace('Mlle','Miss')

df['Title'] = df['Title'].replace('Mme','Mrs')



# Embarked, Fare !

df['Embarked'] = df['Embarked'].fillna('S')

med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

df['Fare'] = df['Fare'].fillna(med_fare)



# Cabin, Deck, famous_cabin !

df['famous_cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'



# Family_Size !

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1



# Name_length !

df['Name_length'] = df['Name'].apply(len)



df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)

df.WomanOrBoyCount = df.WomanOrBoyCount.fillna(0)

df.FamilySurvivedCount = df.FamilySurvivedCount.fillna(0)

df.Alone = df.Alone.fillna(0)
Y = df.Survived.loc[train.index].astype(int)

X_train, X_test = df.loc[train.index], df.loc[test.index]
print(X_train.isnull().sum())

print(X_test.isnull().sum())
cols_to_drop_train = ['Name','Ticket','Cabin']   

cols_to_drop_test = ['Name','Ticket','Cabin', 'Survived']

X_train = X_train.drop(cols_to_drop_train, axis=1)

X_test = X_test.drop(cols_to_drop_test, axis=1)
# Determination categorical features

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

features = X_train.columns.values.tolist()

for col in features:

    if X_train[col].dtype in numerics: continue

    categorical_columns.append(col)

categorical_columns
# Encoding categorical features

for col in categorical_columns:

    if col in X_train.columns:

        le = LabelEncoder()

        le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))

        X_train[col] = le.transform(list(X_train[col].astype(str).values))

        X_test[col] = le.transform(list(X_test[col].astype(str).values))   
X_train = X_train.reset_index()

X_test = X_test.reset_index()
X_dropna_categor = X_train.dropna().astype(int)

Xtest_dropna_categor = X_test.dropna().astype(int)
# Surviving girls:

Sex_female_Survived = X_dropna_categor.loc[(X_dropna_categor.Sex == 0) & (X_dropna_categor.Survived == 1)]

# Dead girls:

Sex_female_NoSurvived = X_dropna_categor.loc[(X_dropna_categor.Sex == 0) & (X_dropna_categor.Survived == 0)]

# Surviving guys:

X_Sex_male_Survived = X_dropna_categor.loc[(X_dropna_categor.Sex == 1) & (X_dropna_categor.Survived == 1)] 

# Dead guys:

X_Sex_male_NoSurvived = X_dropna_categor.loc[(X_dropna_categor.Sex == 1) & (X_dropna_categor.Survived == 0)]



X_test_male = Xtest_dropna_categor.loc[Xtest_dropna_categor.Sex == 1]

X_test_female = Xtest_dropna_categor.loc[Xtest_dropna_categor.Sex == 0]
# age distribution of survivors and non-survivors:

sns.set(rc={'figure.figsize': (15, 9)})

plt.subplot (221)

sns.distplot(Sex_female_Survived['Age'] , kde_kws = {'color': 'g', 'lw':1, 'label': 'Sex_female_Survived' })

plt.subplot (222)

sns.distplot(Sex_female_NoSurvived['Age'] , kde_kws = {'color': 'r', 'lw':1, 'label': 'Sex_female_NoSurvived' })

plt.subplot (223)

sns.distplot(X_Sex_male_Survived['Age'] , kde_kws = {'color': 'blue', 'lw':1, 'label': 'X_Sex_male_Survived' })

plt.subplot (224)

sns.distplot(X_Sex_male_NoSurvived['Age'] , kde_kws = {'color': 'gray', 'lw':1, 'label': 'X_Sex_male_NoSurvived' })
female_Survived_mean, female_NoSurvived_mean = Sex_female_Survived['Age'].mean(), Sex_female_NoSurvived['Age'].mean()

male_Survived_mean, male_NoSurvived_mean = X_Sex_male_Survived['Age'].mean(), X_Sex_male_NoSurvived['Age'].mean()



female_Survived_std, female_NoSurvived_std = Sex_female_Survived['Age'].std(), Sex_female_NoSurvived['Age'].std()

male_Survived_std, male_NoSurvived_std = X_Sex_male_Survived['Age'].std(), X_Sex_male_NoSurvived['Age'].std()



female_std, female_mean = X_test_female['Age'].std(), X_test_female['Age'].mean()

male_std, male_mean = X_test_male['Age'].std(), X_test_male['Age'].mean()
# Confidence interval calculation function: 

def derf(sample, mean, std):

    age_shape = sample['Age'].shape[0] # sample size

    standard_error_ofthe_mean = std / math.sqrt(age_shape)

    random_mean = random.uniform(mean-(1.96*standard_error_ofthe_mean), mean+(1.96*standard_error_ofthe_mean))

    return round(random_mean, 2)    
X_train['Survived'] = X_train['Survived'].astype(int)
X_train.head(2) 
for i in X_train.loc[(X_train['Sex']==0) & (X_train['Survived']==1) & (X_train['Age'].isnull())].index:

    X_train.at[i, 'Age'] = derf(Sex_female_Survived, female_Survived_mean, female_Survived_std)



for h in X_train.loc[(X_train['Sex']==0) & (X_train['Survived']==0) & (X_train['Age'].isnull())].index:

    X_train.at[h, 'Age'] = derf(Sex_female_NoSurvived, female_NoSurvived_mean, female_NoSurvived_std)

    

for l in X_train.loc[(X_train['Sex']==1) & (X_train['Survived']==1) & (X_train['Age'].isnull())].index:

    X_train.at[l, 'Age'] = derf(X_Sex_male_Survived, male_Survived_mean, male_Survived_std)

    

for b in X_train.loc[(X_train['Sex']==1) & (X_train['Survived']==0) & (X_train['Age'].isnull())].index:

    X_train.at[b, 'Age'] = derf(X_Sex_male_NoSurvived, male_NoSurvived_mean, male_NoSurvived_std)

    

for p in X_test.loc[(X_test['Sex']==1) & (X_test['Age'].isnull())].index:

    X_test.at[p, 'Age'] = derf(X_test_male, male_mean, male_std)



for y in X_test.loc[(X_test['Sex']==0) & (X_test['Age'].isnull())].index:

    X_test.at[y, 'Age'] = derf(X_test_female, female_mean, female_std)

print(X_train.isnull().sum())

print(X_test.isnull().sum())
cor_map = plt.cm.RdBu

plt.figure(figsize=(15,17))

plt.title('Pearson Correlation', y=1.05, size=15)

sns.heatmap(X_train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=cor_map, linecolor='white', annot=True)  
X_train = X_train.drop(['Survived'], axis = 1)
rf = RandomForestClassifier()
random_grid = {'criterion': ['gini', 'entropy'],

               'bootstrap': [True, False],

               'max_depth': [3, 5, 7, 9, 11, 13, 16, 19, 20],

               'max_features': ['auto', 'sqrt'],

               'min_samples_leaf': [5, 10, 15, 20, 25],

               'min_samples_split': [40, 50, 60, 62, 64, 66, 68],

               'n_estimators': [300, 600, 900, 1200, 1500, 1800]}
rf_search_one = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv=5, n_iter = 100)
print(X_train.shape)

print(Y.shape)

print(X_test.shape)
rf_search_one.fit(X_train, Y)

#rf_search_one.fit(X_train_rf_1, y_train_rf_1)

best_rf_s1 = rf_search_one.best_estimator_

#best_rf_s1.fit(X_train_rf_1, y_train_rf_1)

best_rf_s1.fit(X_train, Y)

y_predicted_prob_1 = best_rf_s1.predict(X_test) 

print(best_rf_s1.classes_)

y_predicted_prob_1 = list(y_predicted_prob_1)

print('finish')
print(rf_search_one.best_params_)
finall_F = pd.DataFrame.from_dict({'PassengerId': list(X_test.PassengerId), 'Survived': y_predicted_prob_1})
finall_F.head(6)
finall_F.to_csv("Submission_test8.csv", index=False)