import pandas as pd 

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import numpy as np

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer



df=pd.read_csv('train.csv')

dftest=pd.read_csv('test.csv')



# categories having missing values



df.columns[df.isnull().any()]

miss=df.isnull().sum()/len(df)

miss = miss[miss > 0]

miss.sort_values(inplace=True)

print(miss)



#visualising missing stuff



miss=miss.to_frame()

miss.columns=['count']

miss.index.names=['Name']

miss['Name']=miss.index



#plotting



sns.set(style="whitegrid", color_codes=True)

sns.barplot(x = 'Name', y = 'count', data=miss)

plt.xticks(rotation = 90)

plt.show()



# Dropping unique columns

df=df.drop('PassengerId',axis=1)

df_=dftest['PassengerId']

dftest=dftest.drop('PassengerId',axis=1)





df=df.drop('Name',axis=1)

dftest=dftest.drop('Name',axis=1)



df=df.drop('Ticket',axis=1)

dftest=dftest.drop('Ticket',axis=1)



df=df.drop('Cabin',axis=1)

dftest=dftest.drop('Cabin',axis=1)



#Differentiating between numberic and categoric data



numeric_data=df.select_dtypes(include=[np.number])

cat_data=df.select_dtypes(exclude=[np.number])



# Converting categorial to values using Label Encoding



discrete=['Sex','Embarked']

LE=LabelEncoder()

df = df.replace(np.nan, 'NaN')

dftest = dftest.replace(np.nan,'NaN')

for col in discrete:

	LE.fit(df[col].astype(str))

	df[col]=LE.transform(df[col])

	dftest[col]=LE.transform(dftest[col])



#data imputation



df['Age']=df['Age'].replace('NaN',df['Age'].median())

dftest['Age']=dftest['Age'].replace('NaN',dftest['Age'].median())

X,y=df.drop('Survived',axis=1),df.Survived

dftest=dftest.astype(float)

# Data cleaning

def clean_dataset(df):

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)

clean_dataset(dftest)



#Random Forest Classifier



RFC = RandomForestClassifier(n_estimators=530, random_state = 2017 , oob_score = 'TRUE',criterion='entropy',max_features='sqrt') 

RFC.fit(X,y)

preds = RFC.predict(dftest)

print(len(preds))

preds = pd.DataFrame(preds).reset_index()

preds.columns = ['PassengerId','Survived']



preds['PassengerId']=df_

preds.to_csv('solution.csv', index = False)

print(train_df.columns.values)
train_df.head()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();



grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()