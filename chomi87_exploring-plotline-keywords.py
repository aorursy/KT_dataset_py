%matplotlib inline



import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv('../input/movie_metadata.csv')

df.head()
a = df['plot_keywords'].copy().str.split('|').apply(pd.Series, 1).stack()

a.index.droplevel(-1)
df = df[df['num_voted_users'] > 100][['plot_keywords', 'gross', 'budget','duration','imdb_score', 'movie_title']]

# Split the plot_keyword string based on '|' and add an row for each string that is returned

tags = df['plot_keywords'].copy().str.split('|').apply(pd.Series, 1).stack()

tags.index = tags.index.droplevel(-1)

tags.name = 'tags'

df = df.join(tags)

df.head()
count_vect = CountVectorizer(ngram_range=(1,2), stop_words ={'and', 'in', 'of', 'the', 'on','to', 'title','reference',\

                                                             'female','male','by'})

# Calculated only term frequency.

X_train_counts = count_vect.fit_transform(df.tags.dropna())

X_train_counts
X_train_counts.todense().sum(axis = 0).tolist()[0]
MaxFeatureValues = X_train_counts.todense().sum(axis = 0).tolist()[0]

phrase_scores = [pair for pair in zip(range(0, len(MaxFeatureValues)), MaxFeatureValues) if pair[1] > 0]

TopTags = sorted(phrase_scores, key=lambda t: t[1] * -1)[:10]

ind = [x[0] for x in TopTags]

topValues = [x[1] for x in TopTags]

featurelist = count_vect.get_feature_names()

TopTagsNames = [featurelist[j] for j in ind]

TopTagsDf = pd.DataFrame({'Tag':TopTagsNames,'ValueCount':topValues})

ax = sns.factorplot(x="Tag", y="ValueCount", data = TopTagsDf, kind="bar", size=2, aspect=2)
# These are top 30 tags in all the plot keywords

TopTagsNames
Top250 = df[['movie_title','imdb_score']].drop_duplicates().sort_values('imdb_score').tail(250)

Top250IMDBScore = Top250.join(df[['tags']])

X_train_counts = count_vect.fit_transform(Top250IMDBScore.tags.dropna())

X_train_counts
MaxFeatureValues = X_train_counts.todense().sum(axis = 0).tolist()[0]

phrase_scores = [pair for pair in zip(range(0, len(MaxFeatureValues)), MaxFeatureValues) if pair[1] > 0]

TopTags = sorted(phrase_scores, key=lambda t: t[1] * -1)[:30]

ind = [x[0] for x in TopTags]

featurelist = count_vect.get_feature_names()

topValuesIMDB250 = [x[1] for x in TopTags]

TopTagsNamesIMDB250 = [featurelist[j] for j in ind]

ay = sns.factorplot(x="Tag", y="ValueCount", data = pd.DataFrame({'Tag':TopTagsNamesIMDB250,

                                                                  'ValueCount':topValuesIMDB250}),kind="bar",

                    size=6, aspect=4)
#Tag associated with the top 250 movies on IMDB

TopTagsNamesIMDB250
Top250 = df[['movie_title','gross']].drop_duplicates().sort_values('gross').tail(250)

Top250Grossing = Top250.join(df[['tags']])

X_train_counts = count_vect.fit_transform(Top250Grossing.tags.dropna())

X_train_counts
MaxFeatureValues = X_train_counts.todense().sum(axis = 0).tolist()[0]

phrase_scores = [pair for pair in zip(range(0, len(MaxFeatureValues)), MaxFeatureValues) if pair[1] > 0]

TopTags = sorted(phrase_scores, key=lambda t: t[1] * -1)[:30]

ind = [x[0] for x in TopTags]

featurelist = count_vect.get_feature_names()

topValues250Grossing = [x[1] for x in TopTags]

TopTagsNames250Grossing = [featurelist[j] for j in ind]

ay = sns.factorplot(x="Tag", y="ValueCount", data = pd.DataFrame({'Tag':TopTagsNames250Grossing,

                                                                  'ValueCount':topValues250Grossing}),kind="bar",

                    size=6, aspect=4)
# Tags associated with the highest grossing movies

TopTagsNames250Grossing
Top250 = df[['movie_title','budget']].drop_duplicates().sort_values('budget').tail(250)

Top250Budget = Top250.join(df[['tags']])

X_train_counts = count_vect.fit_transform(Top250Budget.tags.dropna())

X_train_counts
MaxFeatureValues = X_train_counts.todense().sum(axis = 0).tolist()[0]

phrase_scores = [pair for pair in zip(range(0, len(MaxFeatureValues)), MaxFeatureValues) if pair[1] > 0]

TopTags = sorted(phrase_scores, key=lambda t: t[1] * -1)[:30]

ind = [x[0] for x in TopTags]

featurelist = count_vect.get_feature_names()

topValues250Budget = [x[1] for x in TopTags]

TopTagsNames250Budget = [featurelist[j] for j in ind]

ay = sns.factorplot(x="Tag", y="ValueCount", data = pd.DataFrame({'Tag':TopTagsNames250Budget,

                                                                  'ValueCount':topValues250Budget}),kind="bar",

                    size=6, aspect=4)
#Tags associated with the highest budgeted movies

TopTagsNames250Budget
# AS suggested by PeterWendel here is an analysis of the word count for the common terms 

commonTags = list(set(TopTagsNames250Budget).intersection(TopTagsNames250Grossing).intersection(TopTagsNamesIMDB250))



Top250Budget = pd.DataFrame({'Tag':TopTagsNames250Budget,'ValueCount':topValues250Budget})

Top250IMDB = pd.DataFrame({'Tag':TopTagsNamesIMDB250,'ValueCount':topValuesIMDB250})

Top250Gross = pd.DataFrame({'Tag':TopTagsNames250Grossing,'ValueCount':topValues250Grossing})



Top250BudgetFiltered = Top250Budget.copy()[Top250Budget['Tag'].isin(commonTags)]

Top250BudgetFiltered['type'] = 'Budget'

Top250IMDBFiltered = Top250IMDB.copy()[Top250IMDB['Tag'].isin(commonTags)]

Top250IMDBFiltered['type'] = 'IMDBScore'

Top250GrossFiltered = Top250Gross.copy()[Top250Gross['Tag'].isin(commonTags)]

Top250GrossFiltered['type'] = 'Grossing'

Top250BudgetFiltered = Top250BudgetFiltered.append(Top250IMDBFiltered).append(Top250GrossFiltered)

sns.pointplot(x="Tag", y="ValueCount", hue="type", data=Top250BudgetFiltered)
commonTags