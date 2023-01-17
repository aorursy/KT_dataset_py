import numpy as np

import pandas as pd

import matplotlib as mpl

from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



sns.set()

plt.rcParams['figure.figsize'] = (20, 10)
df = pd.read_csv('/kaggle/input/board-game-data/bgg_db_1806.csv', encoding='latin1')

df['weight'].replace(0.0, 1.0, inplace=True) # There are just 9 games with 0.0 weight, missing values?

df.head()
df.describe()
fig, ax = plt.subplots(2, 3)

# Outliers for the below variables cause visualizations to be useless, remove them

sns.countplot(data=df, x='min_players', ax=ax[0][0])

sns.countplot(x=df['max_players'].map(lambda x: min(x, 16)), ax=ax[0][1]) # 16 means 16 or more

df['weight'].plot.hist(ax=ax[0][2])

ax[0][2].set_xlabel('weight')

df['min_time'].map(lambda x: min(x, 300)).plot.hist(ax=ax[1][0])

ax[1][0].set_xlabel('min_time')

df['avg_time'].map(lambda x: min(x, 300)).plot.hist(ax=ax[1][1])

ax[1][1].set_xlabel('avg_time')

df['max_time'].map(lambda x: min(x, 300)).plot.hist(ax=ax[1][2])

ax[1][2].set_xlabel('max_time');
fig, ax = plt.subplots(2, 3)

df['avg_rating'].plot.hist(ax=ax[0][0], title='avg_rating')

df['geek_rating'].plot.hist(ax=ax[0][1], title='geek_rating')

df[df['num_votes'] < df['num_votes'].quantile(0.99)]['num_votes'].plot.hist(ax=ax[0][2], title='num_votes')

df['age'].plot.hist(ax=ax[1][0], title='age')

df[df['owned'] < df['owned'].quantile(0.99)]['owned'].plot.hist(ax=ax[1][1], title='owned');
(df['max_time'] - df['avg_time']).describe()
dfsort = df.sort_values(by='min_time').reset_index()

plt.figure()

plt.fill_between(dfsort.index, dfsort['min_time'], dfsort['avg_time'], color='purple')

plt.ylim(-10, 610)

plt.xlabel('Game number')

plt.ylabel('Duration')

plt.title('Range of durations for each game');
sns.scatterplot(x=df['avg_time'].map(lambda x: min(x, 600)), y=df['weight']);
_, ax = plt.subplots(1, 2)

plt.sca(ax[0])

plt.scatter(df['num_votes'], df['geek_rating'], alpha=0.5, c='black', edgecolors='none')

plt.xlabel('num_votes')

plt.ylabel('geek_rating')

plt.title('num_votes vs. geek_rating')

plt.sca(ax[1])

plt.scatter(df['num_votes'], df['geek_rating'], alpha=0.5, c='black', edgecolors='none')

plt.xlabel('num_votes')

plt.ylabel('geek_rating')

plt.title('num_votes vs. geek_rating (zoomed)')

plt.xlim((-100, 10000));
sns.scatterplot(x=df['avg_rating'], y=df['geek_rating'], hue=df['num_votes'].map(np.log), palette=plt.cm.cool, legend=False);
plt.figure()

dfsort = df.reset_index().sort_values(by='rank')

x = dfsort['rank']

plt.plot(x, dfsort['geek_rating'], label='geek_rating')

plt.plot(x, dfsort['avg_rating'], label='avg_rating')

plt.legend()

plt.xlabel('rank')

plt.ylabel('rating');
authors = set()

pd.Series(df.designer.unique()).map(lambda x: x.split(', ')).apply(lambda x: authors.update(x))

authors = pd.Series(index=sorted(authors), name='num_games', data=0)

def incrcount(x):

    for auth in x:

        authors[auth] += 1

df.designer.map(lambda x: x.split(', ')).apply(incrcount)

authors.drop('(Uncredited)', inplace=True)

authors.sort_values(ascending=False, inplace=True)

sns.countplot(x=authors.map(lambda x: min(x, 12)))
def popular_auth(auths):

    return float(any([auth for auth in auths.split(', ') if auth != '(Uncredited)' and authors[auth] > 2]))

df['popular_designer'] = df.designer.map(popular_auth)

sns.countplot(data=df, x='popular_designer');
df['top'] = (df['rank'] < 1000).astype('float64')
features = [

    'min_players',

    'max_players',

    'min_time',

    'avg_time',

    'age',

    'owned',

    'weight',

    'popular_designer'

]

corr = df[features + ['geek_rating']].corr()

idx = corr['geek_rating'].abs().sort_values(ascending=False).index

sns.heatmap(corr.loc[idx, idx], cmap=plt.cm.BrBG, annot=True)
sns.pairplot(df[['geek_rating', 'owned', 'weight', 'popular_designer', 'age']]);
x_min, x_max = -1000, 45000

y_min, y_max = 0.8, 5.

plt.figure(figsize=(16, 8))

sns.scatterplot(data=df, x='owned', y='weight', hue='top', palette={0.0: 'red', 1.0: 'green'})

plt.xlim((x_min, x_max))

plt.ylim((y_min, y_max));
X, y = df[['owned', 'weight']], df['top']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LogisticRegression()

model.fit(X_train, y_train);
model.score(X_test, y_test)
theta0 = model.intercept_[0]

theta1 = model.coef_[0][0]

theta2 = model.coef_[0][1]

print(theta0)

print(theta1)

print(theta2)
xx, yy = np.meshgrid(np.arange(x_min, x_max, 10), np.arange(y_min, y_max+0.05, 0.05))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

xboundary = np.array([0, 10000])

yboundary = (-theta0 - theta1 * xboundary) / theta2



plt.figure(figsize=(16, 8))

plt.pcolormesh(xx, yy, Z, cmap=ListedColormap([(1.0, 0.7, 0.7), (0.7, 1.0, 0.7)]))

sns.scatterplot(data=df, x='owned', y='weight', hue='top', palette={0.0: 'red', 1.0: 'green'})

plt.plot(xboundary, yboundary, color='black', linewidth=4, linestyle='--')

plt.xlim((x_min, x_max))

plt.ylim((y_min, y_max));
probas = model.predict_proba(df[['owned', 'weight']])[:, 1]

plt.figure(figsize=(16, 8))

sns.scatterplot(data=df, x='owned', y='weight', hue=probas, palette=plt.cm.RdYlGn, legend=False)

plt.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(), plt.cm.RdYlGn), label='Predicted probability of y=1')

plt.plot(xboundary, yboundary, color='black', linewidth=4, linestyle='--')

plt.xlim((x_min, x_max))

plt.ylim((y_min, y_max));
probas = model.predict_proba(df[['owned', 'weight']])[:, 1]

costs = -y * np.log(probas) - (1-y) * np.log(1-probas)

sizes = costs * 10

plt.figure(figsize=(16, 8))

plt.scatter(x=df['owned'], y=df['weight'], c=y, s=sizes, cmap=ListedColormap(['red', 'green']))

plt.plot(xboundary, yboundary, color='black', linewidth=4, linestyle='--')

plt.xlim((x_min, x_max))

plt.ylim((y_min, y_max))

plt.xlabel('owned')

plt.ylabel('weight')

plt.title('Cost of each example');