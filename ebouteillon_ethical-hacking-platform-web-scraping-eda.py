import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import requests

import seaborn as sns



from joblib import Parallel, delayed

from lxml import html

from sklearn.linear_model import Lasso

from tqdm import tqdm
sns.set(style="whitegrid")

sns.set_color_codes("pastel")
r = requests.get('https://ctflearn.com/challenge/1/browse')

tree = html.fromstring(r.content)



titles = tree.xpath('//div[contains(@class, "uk-card")]/h3/text()')

points = tree.xpath('//div[contains(@class, "uk-card")]/p[contains(text(), "points")]/text()')

solves = tree.xpath('//div[contains(@class, "uk-card")]/p[contains(text(), "solve")]/text()')

authors = tree.xpath('//div[contains(@class, "uk-card")]/a[contains(@href, "user")]/text()')

links = tree.xpath('//div[contains(@class, "uk-card")]/a[contains(text(), "View")]/@href')

categories = tree.xpath('//div[contains(@class, "uk-card")]/p[contains(@class, "uk-margin-remove-top")]/text()')
def scrap_challenge(link):

    r = requests.get(f'https://ctflearn.com/{link}')

    tree = html.fromstring(r.content)

    date = str(tree.xpath('//span/@data-timestamp')[0])

    rating = str(tree.xpath('//p[contains(text(), "Community Rating")]/b/text()')[0])

    rating = -1 if rating == 'Unrated' else float(rating.split()[0])

    return date, rating
challenge_data = Parallel(n_jobs=8)(

    delayed(scrap_challenge)(link)

    for link in tqdm(links)

)
creation_dates, ratings = zip(*challenge_data)
df = pd.DataFrame.from_dict({

    'title': titles,

    'point': [int(p.split()[0]) for p in points],

    'solve': [int(p.split()[0]) for p in solves],

    'author': authors,

    'link': links,

    'category': categories,

    'creation': pd.to_datetime(creation_dates, format='%Y-%m-%dT%H:%M:%S'),

    'rating': ratings,

})
df
categories = df.groupby('category').category.count().sort_values(ascending=False)

sns.barplot(x=categories.values, y=categories.index, color="b")

sns.despine(left=True, bottom=True)
categories = df.groupby('category').rating.mean().sort_values(ascending=False)

sns.barplot(x=categories.values, y=categories.index, color="b")

sns.despine(left=True, bottom=True)
categories = df.groupby('category').solve.mean().sort_values(ascending=False)

sns.barplot(x=categories.values, y=categories.index, color="b")

sns.despine(left=True, bottom=True)
df[df.rating >= 0].rating.mean()
sns.distplot(df[df.rating >= 0].rating);
df[df.rating >= 0].sort_values(['rating', 'solve'], ascending=False)[:20]
df[df.rating >= 0].sort_values('rating')[:20]
df.groupby('author').title.count().sort_values(ascending=False)[:10]
df.groupby('author').rating.mean().sort_values(ascending=False)[:10]
for c in df.category.unique():

    df[f'is_{c}'] = df.category == c
df.corr()[['rating']].sort_values('rating')
df['lifetime'] = (df.creation.max() - df.creation).dt.days

df['log_lifetime'] = np.log1p(df['lifetime'])
df['solve_per_day'] = df['solve'] / (df['lifetime']+1)

df['log_solve'] = np.log1p(df['solve'])
df.corr()[['point']].sort_values('point')
X = df[[

    'rating',

    'log_solve',

    'log_lifetime',

    'solve_per_day',

    'is_Web',

    'is_Programming',

    'is_Forensics',

    'is_Miscellaneous',

    'is_Cryptography',

    'is_Reverse Engineering',

    'is_Binary',

]]

m = Lasso()

m.fit(X, df.point)

m.score(X, df.point)
pred = m.predict(X)

df['predicted_points'] = pred

df['delta'] = np.abs(df.point - pred)

df.sort_values('delta', ascending=False)[:20][['title', 'point', 'predicted_points']]