import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from statsmodels.stats import weightstats as stests



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



data = pd.read_csv('/kaggle/input/google-playstore-apps/Google-Playstore-Full.csv')

data.head()
data = data.drop(columns = ['App Name', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Last Updated', 'Minimum Version', 'Latest Version'])

data = data.rename(columns={'Content Rating':'CR'})
data = data[data.Size.str.contains('\d')]

data.Size[data.Size.str.contains('k')] = "0."+data.Size[data.Size.str.contains('k')].str.replace('.','')

data.Size = data.Size.str.replace('k','')

data.Size = data.Size.str.replace('M','')

data.Size = data.Size.str.replace(',','')

data.Size = data.Size.str.replace('+','')

data.Size = data.Size.astype(float)



data = data[data.Installs.str.contains('\+')]

data.Installs = data.Installs.str.replace('+','')

data.Installs = data.Installs.str.replace(',','')

data.Installs.astype(int)



data.Price = data.Price.str.contains('1|2|3|4|5|7|8|9').replace(False, 0)



data = data[data.applymap(np.isreal).Reviews]

data.Reviews = data.Reviews.astype(float)



data = data[data.Rating.str.contains('\d') == True]

data.Rating = data.Rating.astype(float)
data.Category.unique()
data.Category = data.Category.fillna('Unknown')

games = data[data.Category.str.contains('GAME', regex=False)]

other = data[~data.Category.str.contains('GAME', regex=False)]
z_Rating = np.abs(stats.zscore(games.Rating))

games = games[z_Rating < 3]

z_Reviews = np.abs(stats.zscore(games.Reviews))

games = games[z_Reviews < 3]



z_Rating2 = np.abs(stats.zscore(other.Rating))

other = other[z_Rating2 < 3]

z_Reviews2 = np.abs(stats.zscore(other.Reviews))

other = other[z_Reviews2 < 3]
games_mean = np.mean(games.Rating)

games_std = np.std(games.Rating)



other_mean = np.mean(other.Rating)

other_std = np.std(games.Rating)



print('Games mean and std: ', games_mean, games_std)

print('Other categories mean and std: ', other_mean, other_std)



ztest, pval = stests.ztest(games.Rating, other.Rating, usevar='pooled', value=0, alternative='smaller')

print('p-value: ', pval)
f, ax = plt.subplots(3,2,figsize=(10,15))



games.Category.value_counts().plot(kind='bar', ax=ax[0,0])

ax[0,0].set_title('Frequency of Games per Category')



ax[0,1].scatter(games.Reviews[games.Reviews < 100000], games.Rating[games.Reviews < 100000])

ax[0,1].set_title('Reviews vs Rating')

ax[0,1].set_xlabel('# of Reviews')

ax[0,1].set_ylabel('Rating')



ax[1,0].hist(games.Rating, range=(3,5))

ax[1,0].set_title('Ratings Histogram')

ax[1,0].set_xlabel('Ratings')



d = games.groupby('Category')['Rating'].mean().reset_index()

ax[1,1].scatter(d.Category, d.Rating)

ax[1,1].set_xticklabels(d.Category.unique(),rotation=90)

ax[1,1].set_title('Mean Rating per Category')



ax[2,0].hist(games.Size, range=(0,100),bins=10, label='Size')

ax[2,0].set_title('Size Histogram')

ax[2,0].set_xlabel('Size')



games.CR.value_counts().plot(kind='bar', ax=ax[2,1])

ax[2,1].set_title('Frequency of Games per Content Rating')



f.tight_layout()
games_dum = pd.get_dummies(games, columns=['Category','CR','Price'])
corrmat = games_dum.corr() 

  

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
other = other[other.Category.map(other.Category.value_counts() > 3500)]
f, ax = plt.subplots(3,2,figsize=(10,15))



other.Category.value_counts().plot(kind='bar', ax=ax[0,0])

ax[0,0].set_title('Frequency of Others per Category')



ax[0,1].scatter(other.Reviews[other.Reviews < 100000], other.Rating[other.Reviews < 100000])

ax[0,1].set_title('Reviews vs Rating')

ax[0,1].set_xlabel('# of Reviews')

ax[0,1].set_ylabel('Rating')



ax[1,0].hist(other.Rating, range=(3,5))

ax[1,0].set_title('Ratings Histogram')

ax[1,0].set_xlabel('Ratings')



d = other.groupby('Category')['Rating'].mean().reset_index()

ax[1,1].scatter(d.Category, d.Rating)

ax[1,1].set_xticklabels(d.Category.unique(),rotation=90)

ax[1,1].set_title('Mean Rating per Category')



ax[2,0].hist(other.Size, range=(0,100),bins=10, label='Size')

ax[2,0].set_title('Size Histogram')

ax[2,0].set_xlabel('Size')



other.CR.value_counts().plot(kind='bar', ax=ax[2,1])

ax[2,1].set_title('Frequency of Others per Content Rating')



f.tight_layout()
other_dum = pd.get_dummies(other, columns=['Category','CR','Price'])
corrmat = other_dum.corr() 

  

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
y = games_dum.Rating

X = games_dum.drop(columns=['Rating'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print('Train', X_train.shape, y_train.shape)

print('Test', X_test.shape, y_test.shape)
reg = LinearRegression()

reg.fit(X_train, y_train)

pred = reg.predict(X_test)



mae = mean_absolute_error(y_test, pred)

mse = mean_squared_error(y_test, pred)

r2 = r2_score(y_test,pred)



print('MAE: ', mae)

print('RMSE: ', np.sqrt(mse))

print('R2: ', r2)
d = range(4)

for degree in d:

    poly = PolynomialFeatures(degree=degree)

    Xpoly = poly.fit_transform(X)

    Xpoly_test = poly.fit_transform(X_test)



    polyreg = LinearRegression()

    polyreg.fit(Xpoly, y)

    predpoly = polyreg.predict(Xpoly_test)



    mae2 = mean_absolute_error(y_test, predpoly)

    mse2 = mean_squared_error(y_test, predpoly)

    r2poly = r2_score(y_test,pred)

    

    print('Degree: ', degree)

    print('MAE: ', mae2)

    print('RMSE: ', np.sqrt(mse2))

    print('R2: ', r2poly)