

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for data visualisation

import seaborn as sns # for data visualisation

import statsmodels.api as sm

from sklearn import linear_model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr, ttest_ind



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing data to pandas dataframe

filename='/kaggle/input/top50spotify2019/top50.csv'

df=pd.read_csv(filename,encoding='ISO-8859-1', index_col = 0)

df.head()
print(df.shape)
print(df.dtypes)
df.rename(columns={'Track.Name':'track_name','Artist.Name':'artist_name','Genre':'genre','Beats.Per.Minute':'bpm','Energy':'energy','Danceability':'danceability','Loudness..dB..':'loudness','Liveness':'liveness','Valence.':'valence','Length.':'length', 'Acousticness..':'acousticness','Speechiness.':'speechiness','Popularity':'popularity'},inplace=True)

df.head()
df.isnull().sum()

df.fillna(0)
pd.set_option('precision', 2)

df.describe()
fig = plt.figure(figsize = (16,9))

df.groupby('genre')['track_name'].agg(len).sort_values(ascending = False).plot(kind = 'bar')

plt.xlabel('Genre', fontsize = 20)

plt.ylabel('Count of songs', fontsize = 20)

plt.title('Genre vs Songs', fontsize = 30)
fig = plt.figure(figsize = (16,9))

df.groupby('artist_name')['track_name'].agg(len).sort_values(ascending = False).plot(kind = 'bar')

plt.xlabel('Artist Name', fontsize = 20)

plt.ylabel('Count of songs', fontsize = 20)

plt.title('Artist vs Songs', fontsize = 30)
plt.figure(figsize=(10,10))

plt.title('Correlation between variables')

sns.heatmap(df.corr(),linewidth=3.1,annot=True,center=1)
sns.pairplot(df)
sns.set_style("whitegrid")

intensity = sum(df.energy)/len(df.energy)

df['energy_level'] = ['energized' if i > intensity else 'without energy' for i in df.energy]



sns.relplot(x='loudness', y='energy',data=df, kind='line', style='energy_level', hue='energy_level', markers=True, dashes=False, ci='sd')

plt.xlabel('Loudness (dB)', fontsize = 20)

plt.ylabel('Energy', fontsize = 20)

plt.title('Connection between the Loudness (dB) and Energy', fontsize = 25)

# from the plot the appropriate interpretation is loudness and energy were signifanlly correlated, because the more Loud the song the more Energetic the song.
sns.catplot(x='loudness', y='energy',data=df, kind='point', hue='energy_level')

plt.xlabel('Loudness (dB)', fontsize = 20)

plt.ylabel('Energy', fontsize = 20)

plt.title('Connection between the Loudness (dB) and Energy', fontsize = 25)
# trainning dataset

independent_var = df[['bpm','energy','danceability','loudness','liveness','valence','length','acousticness','speechiness']]

dependent_var = df['popularity']
result = linear_model.LinearRegression()

result.fit(independent_var, dependent_var)



intercept = result.intercept_

reg_coef = result.coef_

print('Label: bpm(x1), energy(x2), danceability(x3), loudness(x4), liveness(x5), valence(x6), length(x7), acousticness(x8), speechiness(x9)')

print('\nIntercept value (a): %0.3f' % intercept)

print('\nRegression Equation: ŷ = %0.3f + %0.3f*X1 + %0.3f*X2 + %0.3f*X3 + %0.3f*X4, + %0.3f*X5, + %0.3f*X6, + %0.3f*X7, + %0.3f*X8, + %0.3f*X9' % (intercept, reg_coef[0], reg_coef[1], reg_coef[2], reg_coef[3], reg_coef[4], reg_coef[5], reg_coef[6], reg_coef[7], reg_coef[8]))
x_var = sm.add_constant(independent_var)

model = sm.OLS(dependent_var, x_var).fit()

predictions = model.predict(x_var)

print(model.summary())
X = df[['bpm','energy','danceability','loudness','liveness','valence','length','acousticness','speechiness']]

y = df['genre']

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)
X.shape
y.shape
predict = knn.predict(X)

pd.Series(predict).value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
knn.score(X_test, y_test)