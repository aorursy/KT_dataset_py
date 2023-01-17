import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')
beer = pd.read_csv('../input/beers.csv')

del beer['Unnamed: 0'] # Remove index column from original .csv data
beer.head(5)
beer.columns
print(beer.shape)
beer.isnull().sum(0)/len(beer)*100
del beer['ibu']
style = beer.groupby('style')['id'].count()

style = style.sort_values(ascending = False)

top25_style = style[:25]
fig, ax = plt.subplots(figsize = (15,12))

plt.xticks(rotation='90')

sns.barplot(top25_style.index, y=top25_style)

plt.xlabel('Styles', fontsize=15)

plt.ylabel('Count', fontsize=15)

plt.title('Number of Craft Beers by Style', fontsize=15)
names_by_style = beer.groupby('style')['name'].apply(lambda x: list(x)).reset_index()
len(names_by_style)
names_by_style.style
names_by_style['num_names'] = names_by_style.name.apply(lambda x: len(x))
names = names_by_style.sort_values(by='num_names', ascending = False)
names.head(10)
names.num_names.median()
names = names[names.num_names > 50]
len(names)
from sklearn.feature_extraction.text import CountVectorizer
def get_names(x):

    stop_words = ['american', 'pale', 'ipa', 'ale', 'apa','red', 'amber', 'blonde', 'double', 'imperial', 'beer',

                 'wheat', 'brown', 'porter', 'saison', 'farmhouse', 'witbier', 'the', 'a', 'an', 'is', 'am', 'be',

                 'rye', 'india', 'session', 'extra', 'of', 'wit', 'style','2010', '2011', '2012', '2013', '2004',

                 '2014', 'on', '1881', '2009', '2006', '12', '16', '1335', '413', '88', '2006', '2007','2015','805',

                 'hop', 'hopped','full', 'farm', 'white','belgian']

    cv = CountVectorizer(stop_words = stop_words)

    cv_fit = cv.fit_transform(x)

    names = cv.get_feature_names()

    counts = list(cv_fit.toarray().sum(axis = 0))

    top10 = sorted(list(zip(names, counts)), key = lambda x: x[1], reverse = True)[:10]

    return str([feature[0] for feature in top10])

names['feat_counts'] = names.name.apply(lambda x: get_names(x))
pd.set_option('display.max_colwidth', -1)

names[['style', 'feat_counts']]