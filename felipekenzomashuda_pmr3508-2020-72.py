import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import sklearn
training = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values="?")

training.head()
del training['Id']

training.head()
training = training.rename(columns = {'education.num':'education-num',

                          'marital.status':'marital-status',

                          'capital.gain':'capital-gain',

                          'capital.loss':'capital-loss',

                          'hours.per.week':'hours-per-week',

                          'native.country':'native-country'})
training.info()
workclassMode = training['workclass'].mode()[0]

occupationMode = training['occupation'].mode()[0]

nativeCountryMode = training['native-country'].mode()[0]





training['workclass'] = training['workclass'].fillna(workclassMode)

training['occupation'] = training['occupation'].fillna(occupationMode)

training['native-country'] = training['native-country'].fillna(nativeCountryMode)



training.info()
training['workclass'].value_counts().plot(kind="bar");
(training.loc[training['income']=='>50K']['workclass'].value_counts()/training['workclass'].value_counts()).sort_values(ascending=False).plot(kind='bar')
sns.displot(training['fnlwgt'])
training['education'].unique()
training['education'].describe()
training['education-num'].unique()
training['education-num'].value_counts().plot(kind="bar")
(training.loc[training['income']=='>50K']['education-num'].value_counts()/training['education-num'].value_counts()).plot(kind='bar')
training['marital-status'].unique()
training['marital-status'].describe()
training['marital-status'].value_counts().plot(kind="bar")
(training.loc[training['income']=='>50K']['marital-status'].value_counts()/training['marital-status'].value_counts()).plot(kind='bar')
training.loc[training['marital-status']=='Married-AF-spouse'].shape
training['occupation'].describe()
training['occupation'].unique()
training['occupation'].value_counts().plot(kind="bar")
(training.loc[training['income']=='>50K']['occupation'].value_counts()/training['occupation'].value_counts()).sort_values(ascending=False).plot(kind='bar')
training['relationship'].describe()
training['relationship'].unique()
training['relationship'].value_counts().plot(kind="bar")
(training.loc[training['income']=='>50K']['relationship'].value_counts()/training['relationship'].value_counts()).sort_values(ascending=False).plot(kind='bar')
training['race'].describe()
training['race'].unique()
training['race'].value_counts().plot(kind="bar")
(training.loc[training['income']=='>50K']['race'].value_counts()/training['race'].value_counts()).sort_values(ascending=False).plot(kind='bar')
training['sex'].describe()
training['sex'].value_counts().plot(kind="bar")
(training.loc[training['income']=='>50K']['sex'].value_counts()/training['sex'].value_counts()).plot(kind='bar')
sns.displot(training['capital-gain'], bins=30)
sns.displot(training['capital-loss'], bins=30)#
# Porcentagem de com ganhos

(training.loc[training['capital-gain'] > 0]).loc[training['income'] == '>50K'].shape[0]/(training.loc[training['capital-gain'] > 0]).shape[0]
# Porcentagem sem nada

(training.loc[training['capital-gain'] == 0]).loc[training['income'] == '>50K'].shape[0]/(training.loc[training['capital-gain'] == 0]).shape[0]
# Porcentagem com perdas

(training.loc[training['capital-loss'] > 0]).loc[training['income'] == '>50K'].shape[0]/(training.loc[training['capital-loss'] > 0]).shape[0]
training['hours-per-week'].describe()
sns.displot(training['hours-per-week'], bins=30, kde=True)
sns.displot(training.loc[training['income']=='<=50K']['hours-per-week'], bins=30, kde=True)
sns.displot(training.loc[training['income']=='>50K']['hours-per-week'], bins=30, kde=True)
training['native-country'].describe()
training['native-country'].value_counts().plot(kind="bar")
(training.loc[training['income']=='>50K']['native-country'].value_counts()/training['native-country'].value_counts()).sort_values(ascending=False).plot(kind='bar')
training.info()
def removeFeatures(df):

    df = df.drop(columns=['fnlwgt', 'education', 'native-country'])

    return df



training = removeFeatures(training)
def encodeFeatures(df):

    # workclass

    df = pd.concat([df,pd.get_dummies(df['workclass'])], axis=1)

    df = df.drop(columns=['workclass'])



    # marital-status

    df = pd.concat([df,pd.get_dummies(df['marital-status'])], axis=1)

    df = df.drop(columns=['marital-status'])

    

    # occupation

    df = pd.concat([df,pd.get_dummies(df['occupation'])], axis=1)

    df = df.drop(columns=['occupation'])

    

    # relationship

    df = pd.concat([df,pd.get_dummies(df['relationship'])], axis=1)

    df = df.drop(columns=['relationship'])



    # race

    df = pd.concat([df,pd.get_dummies(df.race)], axis=1)

    df = df.drop(columns=['race'])



    # sex

    df = pd.concat([df,pd.get_dummies(df['sex'])], axis=1)

    df = df.drop(columns=['sex'])

    

    

    return df

    

training = encodeFeatures(training)

training.head()
from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler



robScaler = RobustScaler()

stdScaler = StandardScaler()



training[['hours-per-week']] = robScaler.fit_transform(training[['hours-per-week']])

training[['age','education-num']] = stdScaler.fit_transform(training[['age','education-num']])



training.head()
y = training['income']

X = training.drop(columns=['income'])
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



accuracies = []

bestScore = 0

bestStd = 0

bestK = 0



for k in range(12,36, 2):

    knn = KNeighborsClassifier(n_neighbors=k, metric="manhattan")

    scores = cross_val_score(knn, X, y, cv=5)

    score = scores.mean()

    accuracies.append(score)

    print(f'K = {k}, accuracy = {score*100} +/- {scores.std()*100}')

    if score > bestScore:

        bestScore = score

        bestK = k

        bestStd = scores.std()



print(f'Melhor K = {bestK}, com acurácia = {bestScore*100} +/- {bestStd*100}')
plt.plot(range(16,40, 2), accuracies)

plt.xlabel('K')

plt.ylabel('score')
test = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values="?")



test = test.rename(columns = {'education.num':'education-num',

                          'marital.status':'marital-status',

                          'capital.gain':'capital-gain',

                          'capital.loss':'capital-loss',

                          'hours.per.week':'hours-per-week',

                          'native.country':'native-country'})



# Imputação de dados faltantes

test['workclass'] = test['workclass'].fillna(workclassMode)

test['occupation'] = test['occupation'].fillna(occupationMode)

test['native-country'] = test['native-country'].fillna(nativeCountryMode)



# Remoção de Features

test = removeFeatures(test)

del test['Id']



# "Encodação" das features

test = encodeFeatures(test)



# Normalização

test[['hours-per-week']] = robScaler.transform(test[['hours-per-week']])

test[['age','education-num']] = stdScaler.transform(test[['age','education-num']])



test.head()
knn = KNeighborsClassifier(n_neighbors=26, metric="manhattan")

knn.fit(X,y)



predictions = knn.predict(test)



output = pd.DataFrame({'income':predictions})



output.to_csv("submission.csv", index=True, index_label='Id')