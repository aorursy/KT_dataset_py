#Import required libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr

from scipy.stats import f_oneway

from statsmodels.stats.multicomp import MultiComparison

from statsmodels.stats.multicomp import pairwise_tukeyhsd

import math
df_players = pd.read_csv("../input/social-power-nba/nba_2017_players_with_salary_wiki_twitter.csv", index_col=0)

df_players.head()
df_players.info()
df_players['TWITTER_FAVORITE_COUNT'] = df_players['TWITTER_FAVORITE_COUNT'].fillna(0)

df_players['TWITTER_RETWEET_COUNT'] = df_players['TWITTER_RETWEET_COUNT'].fillna(0)
# Sorted by salary in descending order

df_sorted = df_players.sort_values(by='SALARY_MILLIONS', ascending=False)

df_sorted.head(5)
# Sorted by salary in ascending order

df_sorted = df_players.sort_values(by='SALARY_MILLIONS', ascending=True)

df_sorted.head(5)
print('Mean salary is %.3f' % df_players['SALARY_MILLIONS'].mean())

print('Standard deviation of the salary is %.3f' % df_players['SALARY_MILLIONS'].std())
# Sort by the real plus minus

df_sorted = df_players.sort_values(by='RPM', ascending=False)

df_sorted.head(5)
df_players.columns
df_players = df_players[['PLAYER', 'POSITION', 'AGE', 'MPG', 'POINTS', 'ORPM', 'DRPM', 'RPM', 

                         'SALARY_MILLIONS', 'PAGEVIEWS', 'TWITTER_FAVORITE_COUNT', 'TWITTER_RETWEET_COUNT']]



# Temporary remove the player name for this plot

df_temp = df_players.drop(columns=['PLAYER'])

sns.set(style="ticks")

sns.pairplot(df_temp, hue="POSITION")

plt.show()
sns.lmplot(x="MPG", y="SALARY_MILLIONS", data=df_players)

plt.show()
X = df_players[['MPG']]

Y = df_players['SALARY_MILLIONS']



# Build linear regression model

lr_model = LinearRegression(fit_intercept=True, normalize=False)

lr_model.fit(X, Y)

sc = lr_model.score(X, Y)

print('R2 score: %.3f' % sc)
y_pred = lr_model.predict(X)

rmse = math.sqrt(mean_squared_error(Y, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Calculate Pearson correlation coefficient between the two variables

corr, _ = pearsonr(df_players['MPG'], df_players['SALARY_MILLIONS'])

print('Pearson correlation coefficient: %.3f' % corr)
sns.lmplot(x="POINTS", y="SALARY_MILLIONS", data=df_players)

plt.show()
X = df_players[['POINTS']]

Y = df_players['SALARY_MILLIONS']



# Build linear regression model

lr_model = LinearRegression(fit_intercept=True, normalize=False)

lr_model.fit(X, Y)

sc = lr_model.score(X, Y)

print('R2 score: %.3f' % sc)
y_pred = lr_model.predict(X)

rmse = math.sqrt(mean_squared_error(Y, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Calculate Pearson correlation coefficient between the two variables

corr, _ = pearsonr(df_players['POINTS'], df_players['SALARY_MILLIONS'])

print('Pearson correlation coefficient: %.3f' % corr)
sns.lmplot(x="DRPM", y="SALARY_MILLIONS", data=df_players)

plt.show()
X = df_players[['DRPM']]

Y = df_players['SALARY_MILLIONS']



# Build linear regression model

lr_model = LinearRegression(fit_intercept=True, normalize=False)

lr_model.fit(X, Y)

sc = lr_model.score(X, Y)

print('R2 score: %.3f' % sc)
y_pred = lr_model.predict(X)

rmse = math.sqrt(mean_squared_error(Y, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Calculate Pearson correlation coefficient between the two variables

corr, _ = pearsonr(df_players['DRPM'], df_players['SALARY_MILLIONS'])

print('Pearson correlation coefficient: %.3f' % corr)
# Plot sepal width as a function of sepal_length across days

sns.lmplot(x="ORPM", y="SALARY_MILLIONS", data=df_players)

plt.show()
X = df_players[['ORPM']]

Y = df_players['SALARY_MILLIONS']



# Build linear regression model

lr_model = LinearRegression(fit_intercept=True, normalize=False)

lr_model.fit(X, Y)

sc = lr_model.score(X, Y)

print('R2 score: %.3f' % sc)
y_pred = lr_model.predict(X)

rmse = math.sqrt(mean_squared_error(Y, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Calculate Pearson correlation coefficient between the two variables

corr, _ = pearsonr(df_players['ORPM'], df_players['SALARY_MILLIONS'])

print('Pearson correlation coefficient: %.3f' % corr)
# Show each observation with a scatterplot

sns.set(style="whitegrid")

sns.stripplot(x="SALARY_MILLIONS", y="POSITION", order=['PG','SG','SF','PF','C'], data=df_players, alpha=.50)

sns.pointplot(x="SALARY_MILLIONS", y="POSITION", order=['PG','SG','SF','PF','C'], data=df_players, palette="dark", markers="d")

plt.show()
# One-way ANOVA analysis



# Get each position independently

salary_pg = df_players['SALARY_MILLIONS'][df_players['POSITION']=='PG']

salary_sg = df_players['SALARY_MILLIONS'][df_players['POSITION']=='SG']

salary_sf = df_players['SALARY_MILLIONS'][df_players['POSITION']=='SF']

salary_pf = df_players['SALARY_MILLIONS'][df_players['POSITION']=='PF']

salary_c = df_players['SALARY_MILLIONS'][df_players['POSITION']=='C']



# Example of how it would be done for only two groups

fstat, pval = f_oneway(salary_sg, salary_c)

print('P Value: %.4f' % pval)
# Compare all five positions with post-hoc correction

salary_pos = df_players[['SALARY_MILLIONS', 'POSITION']]



multi_comp = MultiComparison(salary_pos['SALARY_MILLIONS'], salary_pos['POSITION'])

# Print all the possible pairwise comparisons

print(multi_comp.tukeyhsd().summary())
sns.set(style="whitegrid")

sns.stripplot(x="DRPM", y="POSITION", order=['PG','SG','SF','PF','C'], data=df_players, alpha=.50)

sns.pointplot(x="DRPM", y="POSITION", order=['PG','SG','SF','PF','C'], data=df_players, palette="dark", markers="d")

plt.show()
# One-way ANOVA analysis



# Get each position independently

drpm_pg = df_players['DRPM'][df_players['POSITION']=='PG']

drpm_sg = df_players['DRPM'][df_players['POSITION']=='SG']

drpm_sf = df_players['DRPM'][df_players['POSITION']=='SF']

drpm_pf = df_players['DRPM'][df_players['POSITION']=='PF']

drpm_c = df_players['DRPM'][df_players['POSITION']=='C']



# Example of how it would be done for only two groups

fstat, pval = f_oneway(drpm_pg, drpm_sf)

print('P Value: %.4f' % pval)
# Compare all five positions with post-hoc correction

drpm_pos = df_players[['DRPM', 'POSITION']]



multi_comp = MultiComparison(drpm_pos['DRPM'], drpm_pos['POSITION'])

# Print all the possible pairwise comparisons

print(multi_comp.tukeyhsd().summary())
sns.lmplot(x="TWITTER_RETWEET_COUNT", y="SALARY_MILLIONS", order=2, data=df_players)

plt.show()
# Remove outliers in the social impact

len_players = len(df_players)

q95 = df_players['TWITTER_RETWEET_COUNT'].quantile(0.95)

data = df_players[df_players['TWITTER_RETWEET_COUNT'] <= q95]

print('Removed %d players' % (len_players-len(data)))
sns.lmplot(x="TWITTER_RETWEET_COUNT", y="SALARY_MILLIONS", order=2, data=data)

plt.show()
X = df_players[['TWITTER_RETWEET_COUNT']]

Y = df_players['SALARY_MILLIONS']



# Relationship is not linear, we use a polynomial regression

poly = PolynomialFeatures(degree=2)

X_ = poly.fit_transform(X)
# Build linear regression model

lr_model = LinearRegression(fit_intercept=True, normalize=False)

lr_model.fit(X_, Y)

sc = lr_model.score(X_, Y)

print('R2 score: %.3f' % sc)
y_pred = lr_model.predict(X_)

rmse = math.sqrt(mean_squared_error(Y, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Calculate Pearson correlation coefficient between the two variables

corr, _ = pearsonr(df_players['TWITTER_RETWEET_COUNT'], df_players['SALARY_MILLIONS'])

print('Pearson correlation coefficient: %.3f' % corr)
sns.lmplot(x="PAGEVIEWS", y="SALARY_MILLIONS", order=2, data=df_players)

plt.show()
sns.lmplot(x="PAGEVIEWS", y="SALARY_MILLIONS", order=2, data=data)

plt.show()
X = df_players[['PAGEVIEWS']]

Y = df_players['SALARY_MILLIONS']



# Relationship is not linear, we use a polynomial regression

poly = PolynomialFeatures(degree=2)

X_ = poly.fit_transform(X)
# Build linear regression model

lr_model = LinearRegression(fit_intercept=True, normalize=False)

lr_model.fit(X_, Y)

sc = lr_model.score(X_, Y)

print('R2 score: %.3f' % sc)
y_pred = lr_model.predict(X_)

rmse = math.sqrt(mean_squared_error(Y, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Calculate Pearson correlation coefficient between the two variables

corr, _ = pearsonr(df_players['PAGEVIEWS'], df_players['SALARY_MILLIONS'])

print('Pearson correlation coefficient: %.3f' % corr)
sns.lmplot(x="AGE", y="SALARY_MILLIONS", data=df_players)

plt.show()
X = df_players[['AGE']]

Y = df_players['SALARY_MILLIONS']



# Build linear regression model

lr_model = LinearRegression(fit_intercept=True, normalize=False)

lr_model.fit(X, Y)

sc = lr_model.score(X, Y)

print('R2 score: %.3f' % sc)
y_pred = lr_model.predict(X)

rmse = math.sqrt(mean_squared_error(Y, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Calculate Pearson correlation coefficient between the two variables

corr, _ = pearsonr(df_players['AGE'], df_players['SALARY_MILLIONS'])

print('Pearson correlation coefficient: %.3f' % corr)
mapping = { "PG":1, "SG":2, "SF":3, "PF":4, "C":5}

df_players['POSITION_NUM'] = df_players['POSITION'].map(mapping).copy()

df_players
X = df_players[['PAGEVIEWS', 'TWITTER_FAVORITE_COUNT', 'TWITTER_RETWEET_COUNT', 'MPG', 

                'POINTS', 'DRPM', 'ORPM', 'POSITION_NUM', 'AGE']]

Y = df_players['SALARY_MILLIONS']
# Separate train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
lr_model = LinearRegression(fit_intercept=True, normalize=True)

lr_model.fit(X_train, y_train)

sc = lr_model.score(X_train, y_train)

print('R2 score: %.3f' % sc)
# Make predictions over train set

y_pred = lr_model.predict(X_train)

rmse = math.sqrt(mean_squared_error(y_train, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Make predictions over test set

y_pred = lr_model.predict(X_test)

rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
print(lr_model.coef_)
import matplotlib.pyplot as plt



plt.figure(figsize=(12, 5))

plt.bar(X.columns, lr_model.coef_, color=(0.8, 0.2, 0.6, 0.8))

plt.xticks(rotation=90)

plt.title('Standardized regression coefs.')

plt.show()
# Relationship is not linear, we use a polynomial regression

poly = PolynomialFeatures(degree=2)

X_ = poly.fit_transform(X)
# Separate train and test

X_train, X_test, y_train, y_test = train_test_split(X_, Y, test_size=0.20, random_state=42)
lr_model = LinearRegression(fit_intercept=True, normalize=True)

lr_model.fit(X_train, y_train)

sc = lr_model.score(X_train, y_train)

print('R2 score: %.3f' % sc)
# Make predictions over train set

y_pred = lr_model.predict(X_train)

rmse = math.sqrt(mean_squared_error(y_train, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
# Make predictions over test set

y_pred = lr_model.predict(X_test)

rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print('RMSE is %.3f. Data STD is %.3f' % (rmse, Y.std()))
print(lr_model.coef_)