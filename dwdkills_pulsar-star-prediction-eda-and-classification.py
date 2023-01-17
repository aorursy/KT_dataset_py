# data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style='ticks', rc={'figure.figsize':(15, 10)})



# machine learning

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
data = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")
data.head()
def detailed_analysis(df):

  obs = df.shape[0]

  types = df.dtypes

  counts = df.apply(lambda x: x.count())

  nulls = df.apply(lambda x: x.isnull().sum())

  distincts = df.apply(lambda x: x.unique().shape[0])

  missing_ratio = (df.isnull().sum() / obs) * 100

  uniques = df.apply(lambda x: [x.unique()])

  skewness = df.skew()

  kurtosis = df.kurt()

  print('Data shape:', df.shape)



  cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']

  details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1)



  details.columns = cols 

  dtypes = details.types.value_counts()

  print('________________________\nData types:\n', dtypes)

  print('________________________')



  return details
details = detailed_analysis(data)

details
data.describe()
data.columns.values
data.rename(columns={ ' Mean of the integrated profile':'mean_profile', ' Standard deviation of the integrated profile':'std_profile', 

                     ' Excess kurtosis of the integrated profile':'kurt_profile', ' Skewness of the integrated profile':'skew_profile',

                     ' Mean of the DM-SNR curve':'mean_dmsnr_curve', ' Standard deviation of the DM-SNR curve':'std_dmsnr_curve',

                     ' Excess kurtosis of the DM-SNR curve':'kurt_dmsnr_curve', ' Skewness of the DM-SNR curve': 'skew_dmsnr_curve'}, inplace=True)



data.head()
values = data.target_class.value_counts()

indexes = values.index



sns.barplot(indexes, values)



plt.xlabel('target_class')

plt.ylabel('Number of values')
sns.pairplot(data=data, vars=data.columns.values[:-1], hue='target_class')
sns.jointplot(x='mean_profile', y='std_profile', data=data, kind='kde', height=12)
fig = plt.figure(figsize=(25, 25))



fig1 = fig.add_subplot(421)

sns.distplot(data.mean_profile, color='r')



fig2 = fig.add_subplot(422)

sns.distplot(data.std_profile, color='g')



fig3 = fig.add_subplot(423)

sns.distplot(data.kurt_profile, color='b')



fig4 = fig.add_subplot(424)

sns.distplot(data.skew_profile, color='y')



fig5 = fig.add_subplot(425)

sns.distplot(data.mean_dmsnr_curve, color='purple')



fig6 = fig.add_subplot(426)

sns.distplot(data.std_dmsnr_curve, color='grey')



fig6 = fig.add_subplot(427)

sns.distplot(data.kurt_dmsnr_curve, color='black')



fig6 = fig.add_subplot(428)

sns.distplot(data.skew_dmsnr_curve, color='orange')
fig = plt.figure(figsize=(12, 10))



correlation = data.corr()

sns.heatmap(correlation, annot=True, cmap='GnBu_r', center=1)
fig = plt.figure(figsize=(20, 20))



values = data.groupby('target_class')[data.columns.values[:-1]].mean()

for item in range(len(values.iloc[0])):

  plot = fig.add_subplot(4, 2, item + 1)

  sns.barplot(x=data.target_class.value_counts().index, y=values[values.columns.values[item]], palette='Greens')
X = data.drop('target_class', axis=1)

y = data.target_class



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Input train shape', X_train.shape)

print('Output train shape', y_train.shape)

print('Input test shape', X_test.shape)

print('Output test shape', y_test.shape)
model = DecisionTreeClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



print('Accuracy:', accuracy * 100)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)



y_pred = gbc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



print('Accuracy:', accuracy * 100)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
model = RandomForestClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



print('Accuracy:', accuracy * 100)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
model = GaussianNB()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



print('Accuracy:', accuracy * 100)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
xgb = XGBClassifier()

xgb.fit(X_train, y_train)



y_pred = xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



print('Accuracy:', accuracy * 100)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)