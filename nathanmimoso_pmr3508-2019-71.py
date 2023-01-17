#Importando as bibliotecas

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns
#Importando os dados

adult = pd.read_csv("../input/adult-pmr3508/train_data.csv", skipinitialspace = True, na_values = "?")

adult.set_index('Id',inplace=True)

adult.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education_num', 'Marital_status', 'Occupation', 

                 'Relationship', 'Race', 'Sex', 'Capital_gain', 'Capital_loss', 'Hours_per_week', 'Native_country', 'Income']
adult.shape
adult.info()
adult.head()
adult.describe(include='all')
adult.corr()
sns.pairplot(adult)
plt.hist(adult.Age, bins=int((adult.Age.max()-adult.Age.min())/5), orientation='horizontal', rwidth=.6)

plt.xlabel("Age")

plt.ylabel("Frequency")
adult.Workclass.value_counts().plot(kind='bar')
adult.Education.value_counts().plot(kind='bar')
plt.hist(adult.Education_num, bins=int((adult.Education_num.max()-adult.Education_num.min())/3), rwidth=.8)

plt.xlabel("Years of Education")

plt.ylabel("Frequency")
adult.Marital_status.value_counts().plot(kind='bar')
adult.Occupation.value_counts().plot(kind='bar')
adult.Relationship.value_counts().plot(kind='bar')
adult.Race.value_counts().plot(kind='bar')
adult.Sex.value_counts().plot(kind='pie')
plt.hist(adult.Hours_per_week, bins=int((adult.Hours_per_week.max()-adult.Hours_per_week.min())/5), rwidth=.6)

plt.xlabel("Hours per week")

plt.ylabel("Frequency")
adult.Native_country.value_counts()
adult.Income.value_counts().plot(kind='pie')
sns.boxplot(x='Income', y='Age', data=adult)
sns.boxplot(x='Income', y='Education_num', data=adult)
sns.boxplot(x='Income', y='Hours_per_week', data=adult)
def freq(column):

    return column*100//float(column[-1])
Income_Sex = pd.crosstab(adult.Income, adult.Occupation, margins=True)

Income_Sex.apply(freq, axis=0)
Income_Sex = pd.crosstab(adult.Income, adult.Sex, margins=True)

Income_Sex.apply(freq, axis=0)
Income_Sex.apply(freq, axis=0).plot(kind="bar")
Income_Race = pd.crosstab(adult.Income, adult.Race, margins=True)

Income_Race.apply(freq, axis=0)
Income_Country = pd.crosstab(adult.Income, adult.Native_country, margins=True)

Income_Country.apply(freq, axis=0)
Income_Education = pd.crosstab(adult.Education_num, adult.Income, margins=True)

Income_Education.apply(freq, axis=1).plot()
Income_Age = pd.crosstab(adult.Age, adult.Income, margins=True)

Income_Age.apply(freq, axis=1).plot()
adult.isna()
adult.isnull().sum()
train = adult.dropna()

train.shape
df = pd.read_csv('../input/adult-pmr3508/test_data.csv')

df.set_index('Id',inplace=True)

test = df.dropna()

test.columns = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education_num', 'Marital_status', 'Occupation', 

                 'Relationship', 'Race', 'Sex', 'Capital_gain', 'Capital_loss', 'Hours_per_week', 'Native_country']

test.shape
test.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
Xtrain = train[["Age","Education_num","Capital_gain", "Capital_loss", "Hours_per_week"]]

Ytrain = train.Income

Xtest = test[["Age","Education_num","Capital_gain", "Capital_loss", "Hours_per_week"]]
# O código a seguir foi usado para avaliar qual hiperparâmetro k fornecia a maior acurácia média

'''for i in range(1, 31):

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)

    knn.fit(Xtrain, Ytrain)

    Ypred = knn.predict(Xtest)

    print(i, 'Cross validation: ', scores, 'Média: ', cross_val_score(knn, Xtrain, Ytrain, cv=10).mean(), 'Previsão: ', Ypred, '\n')

'''
from sklearn import preprocessing
train_num = train.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

train_num = train_num.join(train.Income)
train_num.head()
test_num = test.iloc[:,0:14].apply(preprocessing.LabelEncoder().fit_transform)

test_num.head()
train_num2 = train.apply(preprocessing.LabelEncoder().fit_transform)

train_num2.corr()
Xtrain = train_num[['Age', 'Education_num', 'Marital_status',

                 'Relationship', 'Sex', 'Capital_gain', 'Capital_loss', 'Hours_per_week']]

Ytrain = train_num.Income

Xtest = test_num[['Age', 'Education_num', 'Marital_status',

                 'Relationship', 'Sex', 'Capital_gain', 'Capital_loss', 'Hours_per_week']]
knn = KNeighborsClassifier(n_neighbors=26)

scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)

scores, scores.mean()
knn.fit(Xtrain, Ytrain)

Ypred = knn.predict(Xtest)

Ypred
id_index = pd.DataFrame({'Id' : list(range(len(Ypred)))})

income = pd.DataFrame({'income' : Ypred})

result = id_index.join(income)

result.head()
#Salvando os resultados para submissão

result.to_csv("submission.csv", index = False)
result.income.value_counts().plot(kind='pie')
sns.boxplot(x=result.income, y=test.Age)
sns.boxplot(x=result.income, y=test.Education_num)
sns.boxplot(x=result.income, y=test.Hours_per_week)