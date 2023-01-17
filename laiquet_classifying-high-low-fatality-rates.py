import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab as pl

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import linear_model

import warnings
#Disabling warnings

warnings.simplefilter("ignore")
#Importing data

data = pd.read_csv('../input/master.csv')
#Finding empty cells

data.isna().sum()
#Filling empty cells

data['HDI for year'] = data['HDI for year'].fillna(0)
#Renaming columns

data.rename(columns={'suicides/100k pop':'suicides_K','HDI for year':'HDI','country-year':'country_year',' gdp_for_year ($) ':'gdp_for_year','gdp_per_capita ($)':'gdp_per_capita'}, inplace=True)
#Shape and Description

print(data.shape)

print(data.describe())
#Suicides w.r.t countries

sns.set(context='notebook', style='whitegrid')

pl.figure(figsize =(20,20))

data.groupby(['country']).suicides_no.count().plot('barh')

plt.xlabel('Total No. of Suicides', fontsize=12)

plt.ylabel('Country', fontsize=12)

plt.title('Suicides by country', fontsize=15)

plt.show()
#Suicides w.r.t gender

pl.figure(figsize =(15,3))

data.groupby(['sex']).suicides_no.sum().plot('barh')

plt.xlabel('Total No. of Suicides', fontsize=12)

plt.ylabel('Gender', fontsize=12)

plt.title('Suicides by gender', fontsize=15)

plt.show()
#Suicides w.r.t age buckets

pl.figure(figsize =(15,3))

data.groupby(['age']).suicides_no.sum().plot('barh')

plt.xlabel('Total No. of Suicides', fontsize=12)

plt.ylabel('Age', fontsize=12)

plt.title('Suicides by age', fontsize=15)

plt.show()
#Suicides w.r.t age buckets

pl.figure(figsize =(15,3))

data.groupby(['generation']).suicides_no.count().plot('barh')

plt.xlabel('Total No. of Suicides', fontsize=12)

plt.ylabel('Generation', fontsize=12)

plt.title('Suicides by generation', fontsize=15)

plt.show()
#Suicides w.r.t Year

pl.figure(figsize =(20,12))

data.groupby(['year']).suicides_no.count().plot('barh')

plt.xlabel('Total No. of Suicides', fontsize=12)

plt.ylabel('Year', fontsize=12)

plt.title('Suicides by year', fontsize=15)

plt.show()
#Suicides/100k pop w.r.t gender

pl.figure(figsize =(10,5))

plt.xlabel('Suicides/100k pop', fontsize=12)

plt.ylabel('Gender', fontsize=12)

plt.title('Suicides/100k Population by Gender', fontsize=15)

sns.boxplot(x="suicides_K", y="sex", data=data, whis="range", palette="vlag")

plt.show()
#Data Transformations

data['generation']=data['generation'].str.replace('Boomers','0')

data['generation']=data['generation'].str.replace('G.I. Generation','3')

data['generation']=data['generation'].str.replace('Generation X','1')

data['generation']=data['generation'].str.replace('Generation Z','2')

data['generation']=data['generation'].str.replace('Millenials','4')

data['generation']=data['generation'].str.replace('Silent','5')

data['gdp_for_year']=data['gdp_for_year'].str.replace(',','')

data['sex']=data['sex'].str.replace('female', '1')

data['sex']=data['sex'].str.replace('male', '0')

pd.to_numeric(data['generation'])

pd.to_numeric(data['sex'])

pd.to_numeric(data['gdp_for_year'])

print(data['generation'][:5])

print(data['sex'][:5])

print(data['gdp_for_year'][:5])
#Dropping columns

data=data.drop(columns=['country', 'age', 'country_year'])
#Correlation matrix & Heatmap

pl.figure(figsize =(10,10))

corrmat = data.corr()

sns.heatmap(corrmat, annot=True, fmt='.3f', vmin=0, vmax=1, square=True);
#Classifying the fatality rate for each entry

#If suicides/100k value for each entry is greater than its mean value then

#1: High else 0:low

data['fatality_rate']=np.where(data['suicides_K']>data['suicides_K'].mean(), 1, 0)
#Separating Labels and featureSet columns

columns = data.columns.tolist()

columns = [c for c in columns if c not in ['fatality_rate']]

target = 'fatality_rate'



X = data[columns]

y = data[target]
#Splitting data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)



print("Training FeatureSet:", X_train.shape)

print("Training Labels:", y_train.shape)

print("Testing FeatureSet:", X_test.shape)

print("Testing Labels:", y_test.shape)
#Initializing the model with some parameters.

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=1)

#Fitting the model to the data.

model.fit(X_train, y_train)

#Generating predictions for the test set.

predictions = model.predict(X_test)

#Computing the Model Accuracy

print("Random Forrest Accuracy:",round(metrics.accuracy_score(y_test, predictions), 2)*100)

#Computing the error.

print("Mean Absoulte Error:", round(mean_absolute_error(predictions, y_test), 2)*100)

#Computing classification Report

print("Classification Report:\n", classification_report(y_test, predictions))

#Plotting confusion matrix

print("Confusion Matrix:")

df = pd.DataFrame(

    confusion_matrix(y_test, predictions),

    index = [['actual', 'actual'], ['0','1']],

    columns = [['predicted', 'predicted'], ['0','1']])

print(df)