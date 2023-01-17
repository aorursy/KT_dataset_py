# data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style='dark')



# machine learning

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import SelectKBest, f_classif
data = pd.read_csv("../input/top50spotify2019/top50.csv", encoding='ISO-8859-1')
data.head()
data.drop('Unnamed: 0', inplace=True, axis=1)
data.shape
data.rename(columns={'Track.Name':'Track_Name', 'Artist.Name':'Artist_Name',

                      'Beats.Per.Minute':'Beats_Per_Minute', 'Loudness..dB..':'Loudness',

                      'Valence.':'Valence', 'Length.':'Length', 'Acousticness..':'Acousticness', 'Speechiness.':'Speechiness'}, inplace=True)

data.drop('Track_Name', axis=1, inplace=True)

data.head()
def detailed_analysis(df, pred=None):

  obs = df.shape[0]

  types = df.dtypes

  counts = df.apply(lambda x: x.count())

  uniques = df.apply(lambda x: [x.unique()])

  nulls = df.apply(lambda x: x.isnull().sum())

  distincts = df.apply(lambda x: x.unique().shape[0])

  missing_ratio = (df.isnull().sum() / obs) * 100

  skewness = df.skew()

  kurtosis = df.kurt()

  print('Data shape:', df.shape)



  if pred is None:

    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']

    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1)

  else:

    corr = df.corr()[pred]

    details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis, corr], axis=1, sort=False)

    corr_col = 'corr ' + pred

    cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis', corr_col]



  details.columns = cols

  dtypes = details.types.value_counts()

  print('____________________________\nData types:\n', dtypes)

  print('____________________________')

  return details
details = detailed_analysis(data, 'Energy')

display(details.sort_values(by='corr Energy', ascending=False))
data.describe()
data.Genre.value_counts()
values = data.Genre.value_counts()

indexes = values.index



fig = plt.figure(figsize=(35, 15))

sns.barplot(indexes, values)



plt.ylabel('Number of values')

plt.xlabel('Genre')
data.Artist_Name.value_counts()
values = data.Liveness.value_counts()

indexes = values.index



fig = plt.figure(figsize=(15, 10))

sns.barplot(indexes, values)



plt.ylabel('Number of values')

plt.xlabel('Liveness')
data.Beats_Per_Minute.describe()
def transform(x):

  if x <= 100:

    return '85-100'

  elif x <= 120:

    return '101-120'

  elif x <= 140:

    return '121-140'

  else:

    return '141-190'



groups_of_bpm = data.Beats_Per_Minute.apply(transform)



values = groups_of_bpm.value_counts()

labels = values.index

colors = ['red', 'blue', 'green', 'brown']



fig = plt.figure(figsize=(15, 10))

plt.pie(values, colors=colors, autopct='%1.1f%%', startangle=90, textprops={ 'color': 'w' })



plt.title('BPM distribution', fontsize=15)

plt.legend(labels)

plt.show()
correlations = data.corr()



fig = plt.figure(figsize=(12, 10))

sns.heatmap(correlations, annot=True, cmap='GnBu_r', center=1)
fig = plt.figure(figsize=(15, 10))

sns.regplot(x='Energy', y='Loudness', data=data)
le = LabelEncoder()



for col in data.columns.values:

  if data[col].dtypes == 'object':

    le.fit(data[col].values)

    data[col] = le.transform(data[col])



data.head()
X = data.drop('Loudness', axis=1)

y = data.Loudness



selector = SelectKBest(score_func=f_classif, k=5)

fitted = selector.fit(X, y)

features_scores = pd.DataFrame(fitted.scores_)

features_columns = pd.DataFrame(X.columns)



best_features = pd.concat([features_columns, features_scores], axis=1)

best_features.columns = ['Feature', 'Score']

best_features.sort_values(by='Score', ascending=False, inplace=True)

best_features
X.drop('Artist_Name', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train
sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



type(X_train), type(X_test)
regressor = LinearRegression()

regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)



print(regressor.intercept_, regressor.coef_)

print(mean_squared_error(y_test, y_pred))
regressor = SVR(C=0.5)

regressor.fit(X_train, y_train)



y_pred = regressor.predict(X_test)

print(mean_squared_error(y_test, y_pred))
clustering = KMeans(n_clusters=2)

clustering.fit(X_train, y_train)



y_pred = clustering.predict(X_test)

print(mean_squared_error(y_test, y_pred))