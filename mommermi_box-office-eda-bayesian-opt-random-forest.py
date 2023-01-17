import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.set_index('id', inplace=True)

test.set_index('id', inplace=True)
train.revenue = np.log1p(train.revenue)
train.describe(include='all')
test.describe(include='all')
train_completeness = pd.DataFrame({'filled': [train.loc[:, col].dropna().count() for col in train.columns],

                                   'total': [len(train)]*len(train.columns)},

                                  index=train.columns)

train_completeness.drop('revenue')

test_completeness = pd.DataFrame({'filled': [test.loc[:, col].dropna().count() for col in test.columns],

                                   'total': [len(test)]*len(test.columns)},

                                 index=test.columns)



f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(15,5))

train_completeness.filled.plot.bar(title='train sample', ax=ax1)

test_completeness.filled.plot.bar(title='test sample', color='orange', ax=ax2)
featurelist = []
train['in_collection'] = train.belongs_to_collection.agg(lambda x: 1)

train.loc[train.belongs_to_collection.isna(), 'in_collection'] = 0
train.groupby('in_collection').revenue.plot.hist(alpha=0.3, bins=np.linspace(0, 10, 20), legend=True)
test['in_collection'] = test.belongs_to_collection.agg(lambda x: 1)

test.loc[test.belongs_to_collection.isna(), 'in_collection'] = 0



featurelist += ['in_collection']
train['nmovies_in_collection'] = train.belongs_to_collection.apply(

    lambda x: train.loc[train.belongs_to_collection == x].belongs_to_collection.count())

train.loc[train.belongs_to_collection.isna(), 'nmovies_in_collection'] = 1

train.boxplot(by='nmovies_in_collection', column='revenue', figsize=(8,5))
train.groupby('belongs_to_collection').revenue.std().plot.hist(alpha=0.3)
train_collections = set(train.belongs_to_collection.unique())

test_collections = set(test.belongs_to_collection.unique())

print('number of unique collections:  train: {}, test: {}'.format(len(train_collections), len(test_collections)))
print('number of collections present in both samples: {}'.format(len(train_collections.intersection(test_collections))))
train['median_revenue_collection'] = train.belongs_to_collection.apply(

    lambda x: train.loc[train.belongs_to_collection == x].revenue.median())

train.median_revenue_collection.fillna(train.loc[train.in_collection == 0].revenue.median(), inplace=True)



test['median_revenue_collection'] = test.belongs_to_collection.apply(

    lambda x: train.loc[train.belongs_to_collection == x].revenue.median())

test.median_revenue_collection.fillna(train.loc[train.in_collection == 0].revenue.median(), inplace=True)
train.plot.scatter('median_revenue_collection', 'revenue', alpha=0.2)

print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.median_revenue_collection, train.revenue)))
featurelist += ['median_revenue_collection']
train.budget = np.log1p(train.budget.replace(0, pd.concat([train, test], sort=False).budget.median()))

test.budget = np.log1p(test.budget.replace(0, pd.concat([train, test], sort=False).budget.median()))
train.plot.scatter(x='budget', y='revenue', alpha=0.1)

print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.budget, train.revenue)))
featurelist += ['budget']
genres = set([genre['name'] for genre in eval("+".join(

    [g for g in pd.concat([train, test], sort=False).genres.fillna(

    "[{'id':999, 'name':'NoGenre'}]")]))])

print(len(genres), 'different genres')
# extract only those genres members that occur in train

genre_train = set([genre['name'] for genre in eval("+".join(

    [c for c in train.genres.fillna(

    "[{'id':999, 'name':'NoGenre'}]")]))])



# calculate cumulative revenue per genre

cum_rev_genre = [train.loc[train.genres.fillna(

    "[{'id':999, 'name':'NoGenre'}]").str.contains("'name': '{}'".format(c)), 'revenue'].agg('exp').sum()

                  for c in genre_train]



# sort genres based on revenue

cum_rev_genre = pd.Series(cum_rev_genre, index=genre_train).sort_values(ascending=False)



# scale the cumulative revenue

cumulative_revenue = cum_rev_genre.cumsum()/cum_rev_genre.cumsum()[-1]
f, ax = plt.subplots(figsize=(12,5))



cumulative_revenue.plot.line(rot=90, ax=ax)



ax.set_xticks(range(len(cumulative_revenue)))

ax.set_xticklabels(list(cumulative_revenue.index))

ax.set_ylabel('cumulative revenue')
genre_weights = []



# define tiers and weighting scheme; high-revenue = high weight

weight = 10

threshold = 0.1

for i in range(len(cumulative_revenue)): 

    if cumulative_revenue[i] > threshold:

        weight -= 1

        threshold += 0.1

    genre_weights.append(weight)



genre_weights = pd.Series(genre_weights + [0], 

                          index=list(cum_rev_genre.index) + ["NoGenre"])
train['genre_weight'] = train.genres.fillna(

    "[{'id':999, 'name':'NoGenre'}]").apply(

    lambda x: np.sqrt(np.sum([genre_weights[c['name']]**2 for c in eval(x)])))



# genre members that occur in test but not in train get zero weights assigned (no revenue information available)

test['genre_weight'] = test.genres.fillna(

    "[{'id':999, 'name':'NoGenre'}]").apply(

    lambda x: np.sqrt(np.sum([genre_weights[c['name']]**2 

                              if c['name'] in genre_train else 0 

                              for c in eval(x)])))
train.loc[[15, 24], ['original_title', 'genre_weight', 'revenue']]
f, ax = plt.subplots()



ax.scatter(train.genre_weight, train.revenue, alpha=0.1)

ax.set_xlabel('genre_weight')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.genre_weight, train.revenue)))
featurelist += ['genre_weight']
train.homepage.fillna(0, inplace=True)

train.loc[train.homepage != 0, 'homepage'] = 1
train.groupby('homepage').revenue.plot.hist(alpha=0.3, bins=np.linspace(0, 10, 20), legend=True)
train.loc[train.homepage == 1, 'revenue'].agg(['mean', 'std'])
train.loc[train.homepage == 0, 'revenue'].agg(['mean', 'std'])
languages = pd.concat([train, test], sort=False).original_language.unique()

print(len(languages), 'different languages')
language_counts = pd.concat([train, test], sort=False).original_language.value_counts()



# rename languages with less than 10 occurences to 'ot' for 'other'

for lang in language_counts[language_counts < 10].index:

    train.loc[train.original_language == lang, 'original_language'] = 'ot'

    test.loc[test.original_language == lang, 'original_language'] = 'ot'

    

languages = pd.concat([train, test], sort=False).original_language.unique()

print('languages used in more than 10 movies:', languages)
# extract only those languages that occur in train

olang_train = train.original_language.unique()



# calculate cumulative revenue per olang member

cum_rev_olang = [train.loc[train.original_language == l, 'revenue'].agg('exp').sum()

                  for l in olang_train]

cum_rev_olang = pd.Series(cum_rev_olang, index=olang_train).sort_values(ascending=False)



cumulative_revenue = cum_rev_olang.cumsum()/cum_rev_olang.cumsum()[-1]



olang_weights = []



# define tiers and weighting scheme; high-revenue = high weight

weight = 10

threshold = 0.1

for i in range(len(cumulative_revenue)): 

    if cumulative_revenue[i] > threshold:

        weight -= 1

        threshold += 0.1

    olang_weights.append(weight)



olang_weights = pd.Series(olang_weights + [0], 

                          index=list(cum_rev_olang.index) + ["xx"])



train['olang_weight'] = train.original_language.map(olang_weights)



# olang members that occur in test but not in train get zero weights assigned (no revenue information available)

test['olang_weight'] = test.original_language.map(olang_weights).fillna(0)
f, ax = plt.subplots()



ax.scatter(train.olang_weight, train.revenue, alpha=0.1)

ax.set_xlabel('olang_weight')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.olang_weight, train.revenue)))
featurelist += ['olang_weight']
titlelen = train.original_title.apply(lambda x: len(x))



f, ax = plt.subplots()

ax.scatter(titlelen, train.revenue, alpha=0.1)

ax.set_xlabel('title length in characters')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(titlelen, train.revenue)))
overviewlen = train.overview.fillna('notext').apply(lambda x: len(x))



f, ax = plt.subplots()

ax.scatter(overviewlen, train.revenue, alpha=0.1)

ax.set_xlabel('overview length in characters')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(overviewlen, train.revenue)))
f, ax = plt.subplots()



ax.scatter(np.log(train.popularity), train.revenue, alpha=0.1)

ax.set_xlabel('log(popularity)')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(np.log(train.popularity), train.revenue)))
train.popularity = np.log1p(train.popularity)

test.popularity = np.log1p(test.popularity)

featurelist += ['popularity']
companies = set([company['name'] for company in eval("+".join(

    [c for c in pd.concat([train, test], sort=False).production_companies.dropna()]))])

print(len(companies), 'different production companies')
# calculate cumulative revenue per company

cum_rev_companies = [train.loc[train.production_companies.fillna('').str.contains(c), 'revenue'].agg('exp').sum()

                     for c in companies]

cum_rev_companies = pd.Series(cum_rev_companies, index=companies).sort_values(ascending=False)



cumulative_revenue = cum_rev_companies.cumsum()/cum_rev_companies.cumsum()[-1]



company_weights = []



# define tiers and weighting scheme; high-revenue = high weight

weight = 10

threshold = 0.1

for i in range(len(cumulative_revenue)): 

    if cumulative_revenue[i] > threshold:

        weight -= 1

        threshold += 0.1

    company_weights.append(weight)



company_weights = pd.Series(company_weights + [0], index=list(cum_rev_companies.index) + ["nocompany"])



train['production_companies_weight'] = train.production_companies.fillna("[{'name': 'nocompany'}]").apply(

    lambda x: np.sqrt(np.sum([company_weights[company['name']]**2 for company in eval(x)])))



test['production_companies_weight'] = test.production_companies.fillna("[{'name': 'nocompany'}]").apply(

    lambda x: np.sqrt(np.sum([company_weights[company['name']]**2 for company in eval(x)])))
f, ax = plt.subplots()



ax.scatter(train.production_companies_weight, train.revenue, alpha=0.1)

ax.set_xlabel('production_companies_weight')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.production_companies_weight, train.revenue)))
featurelist += ['production_companies_weight']
countries = set([company['name'] for company in eval("+".join(

    [c for c in pd.concat([train, test], sort=False).production_countries.dropna()]))])

print(len(countries), 'different production countries')
# calculate cumulative revenue per country

cum_rev_countries = [train.loc[train.production_countries.fillna('').str.contains(c), 'revenue'].agg('exp').sum()

                     for c in countries]

cum_rev_countries = pd.Series(cum_rev_countries, index=countries).sort_values(ascending=False)



cumulative_revenue = cum_rev_countries.cumsum()/cum_rev_countries.cumsum()[-1]



country_weights = []



# define tiers and weighting scheme; high-revenue = high weight

weight = 10

threshold = 0.1

for i in range(len(cumulative_revenue)): 

    if cumulative_revenue[i] > threshold:

        weight -= 1

        threshold += 0.1

    country_weights.append(weight)



country_weights = pd.Series(country_weights + [0], index=list(cum_rev_countries.index) + ["nocountry"])



train['production_countries_weight'] = train.production_countries.fillna("[{'name': 'nocountry'}]").apply(

    lambda x: np.sqrt(np.sum([country_weights[country['name']]**2 for country in eval(x)])))



test['production_countries_weight'] = test.production_countries.fillna("[{'name': 'nocountry'}]").apply(

    lambda x: np.sqrt(np.sum([country_weights[country['name']]**2 for country in eval(x)])))
f, ax = plt.subplots()



ax.scatter(train.production_countries_weight, train.revenue, alpha=0.1)

ax.set_xlabel('production_countries_weight')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.production_countries_weight, train.revenue)))
featurelist += ['production_countries_weight']
train.release_date = train.release_date.apply(

        lambda x: "{}/{}/{}".format(

            x.split('/')[0],

            x.split('/')[1],

            (str(20)+x.split('/')[2] if (float(x.split('/')[2]) < 20) 

                                     else str(19)+x.split('/')[2])))



test.release_date = test.release_date.fillna(test.release_date.iloc[1000]).apply(

        lambda x: "{}/{}/{}".format(

            x.split('/')[0],

            x.split('/')[1],

            (str(20)+x.split('/')[2] if (float(x.split('/')[2]) < 20) 

                                     else str(19)+x.split('/')[2])))
train.release_date = pd.to_datetime(train.release_date)

test.release_date = pd.to_datetime(test.release_date)
release_year = train.release_date.apply(lambda x: x.year)

f, ax = plt.subplots()



ax.scatter(release_year, train.revenue, alpha=0.1)

ax.set_xlabel('release_year')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(release_year, train.revenue)))
train['release_year'] = pd.cut(train.release_date.apply(lambda x: x.year), 

                               bins=np.arange(1920, 2030, 10), right=True, labels=False)

test['release_year'] = pd.cut(test.release_date.apply(lambda x: x.year), 

                               bins=np.arange(1920, 2030, 10), right=True, labels=False)
release_month = train.release_date.apply(lambda x: x.month)

f, ax = plt.subplots()



ax.scatter(release_month, train.revenue, alpha=0.1)

ax.set_xlabel('release_month')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(release_month, train.revenue)))
release_season = train.release_date.apply(lambda x: np.sin(x.month/12*np.pi))

# winter is 0, summer is 1 (northern hemisphere)



f, ax = plt.subplots()



ax.scatter(release_season, train.revenue, alpha=0.1)

ax.set_xlabel('release_month')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(release_season, train.revenue)))
release_day = train.release_date.apply(lambda x: x.day)



f, ax = plt.subplots()



ax.scatter(release_day, train.revenue, alpha=0.1)

ax.set_xlabel('release_day')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(release_day, train.revenue)))
pd.concat([train, test], sort=False).runtime.replace(0, np.nan).fillna(train.runtime.median()).plot.hist(

    alpha=0.3, bins=np.arange(0, 400, 30), logy=True)
f, ax = plt.subplots()



ax.scatter(train.runtime.replace(0, np.nan).fillna(train.runtime.median()), train.revenue, alpha=0.1)

ax.set_xlabel('runtime (min)')

ax.set_ylabel('log(revenue)')

ax.set_xticks([0, 60, 90, 120, 180, 360])

ax.grid()
train['runtime_bin'] = pd.cut(train.runtime.replace(0, np.nan).fillna(train.runtime.median()), 

                              bins=[0, 60, 90, 120, 180, 360], right=True, include_lowest=True, labels=False)

test['runtime_bin'] = pd.cut(test.runtime.replace(0, np.nan).fillna(train.runtime.median()), 

                             bins=[0, 60, 90, 120, 180, 360], right=True, include_lowest=True, labels=False)
featurelist += ['runtime_bin']
languages = set([lang['name'] for lang in eval("+".join(

    [l for l in pd.concat([train, test], sort=False).spoken_languages.dropna()]))])

print(len(languages), 'different spoken languages')
# extract only those languages that occur in train

languages_train = set([lang['name'] for lang in eval("+".join(

    [l for l in train.spoken_languages.dropna()]))])



# calculate cumulative revenue per language

cum_rev_languages = [train.loc[train.spoken_languages.fillna('').str.contains(l), 'revenue'].agg('exp').sum()

                     for l in languages_train]

cum_rev_languages = pd.Series(cum_rev_languages, index=languages_train).sort_values(ascending=False)



cumulative_revenue = cum_rev_languages.cumsum()/cum_rev_languages.cumsum()[-1]



language_weights = []



# define tiers and weighting scheme; high-revenue = high weight

weight = 10

threshold = 0.1

for i in range(len(cumulative_revenue)): 

    if cumulative_revenue[i] > threshold:

        weight -= 1

        threshold += 0.1

    language_weights.append(weight)



language_weights = pd.Series(language_weights + [0], 

                             index=list(cum_rev_languages.index) + ["nolang"])



train['spoken_languages_weight'] = train.spoken_languages.fillna("[{'name': 'nolang'}]").apply(

    lambda x: np.sqrt(np.sum([language_weights[lang['name']]**2 for lang in eval(x)])))



# languages that occur in test but not in train get zero weights assigned (no revenue information available)

test['spoken_languages_weight'] = test.spoken_languages.fillna("[{'name': 'nolang'}]").apply(

    lambda x: np.sqrt(np.sum([language_weights[lang['name']]**2 

                              if lang['name'] in languages_train else 0 

                              for lang in eval(x)])))
f, ax = plt.subplots()



ax.scatter(train.spoken_languages_weight, train.revenue, alpha=0.1)

ax.set_xlabel('spoken_languages_weight')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.spoken_languages_weight, train.revenue)))
featurelist += ['spoken_languages_weight']
pd.concat([train, test], sort=False).status.unique()
train = train.loc[train.status == 'Released']

len(train)
train.tagline.fillna(0, inplace=True)

train.loc[train.tagline != 0, 'tagline'] = 1



train.groupby('tagline').revenue.plot.hist(alpha=0.3, bins=np.linspace(0, 10, 20), legend=True)
titlelen = train.title.apply(lambda x: len(x))



f, ax = plt.subplots()

ax.scatter(titlelen, train.revenue, alpha=0.1)

ax.set_xlabel('title length in characters')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(titlelen, train.revenue)))
keywords_train = set([keyword['name'] for keyword in eval("+".join([k for k in train.Keywords.dropna()]))])

print('{} different keywords like {}...'.format(len(keywords_train), ", ".join(list(keywords_train)[:10])))
cum_rev_keywords = [train.loc[train.Keywords.fillna('').str.contains(l), 'revenue'].agg('exp').sum()

                    for l in keywords_train]

cum_rev_keywords = pd.Series(cum_rev_keywords, index=keywords_train).sort_values(ascending=False)

cumulative_revenue = cum_rev_keywords.cumsum()/cum_rev_keywords.cumsum()[-1]
cumulative_revenue.plot.line()
len(cumulative_revenue[cumulative_revenue < 0.5])
train['hot_keyword'] = [1 if any([keyword['name'] in cumulative_revenue[cumulative_revenue < 0.5] 

                                  for keyword in eval(x)]) else 0

                        for x in train.Keywords.fillna("''")]



test['hot_keyword'] = [1 if any([keyword['name'] in cumulative_revenue[cumulative_revenue < 0.5] 

                                  for keyword in eval(x)]) else 0

                        for x in test.Keywords.fillna("''")]
train.groupby('hot_keyword').revenue.plot.hist(alpha=0.3, bins=np.linspace(0, 10, 20), legend=True)
actors = set([actor['name'] for actor in eval("+".join(

    [a for a in pd.concat([train, test], sort=False).cast.dropna()]))])

print(len(actors), 'different actors')
# extract only those actors that occur in train

actors_train = set([actor['name'] for actor in eval("+".join(

    [a for a in train.cast.dropna()]))])



# calculate cumulative revenue per actor

cum_rev_actors = [train.loc[train.cast.fillna('').str.contains("'name': '{}'".format(a)), 'revenue'].agg('exp').sum()

                  for a in actors_train]

cum_rev_actors = pd.Series(cum_rev_actors, index=actors_train).sort_values(ascending=False)



cumulative_revenue = cum_rev_actors.cumsum()/cum_rev_actors.cumsum()[-1]



actor_weights = []



# define tiers and weighting scheme; high-revenue = high weight

weight = 10

threshold = 0.1

for i in range(len(cumulative_revenue)): 

    if cumulative_revenue[i] > threshold:

        weight -= 1

        threshold += 0.1

    actor_weights.append(weight)



actor_weights = pd.Series(actor_weights + [0], 

                          index=list(cum_rev_actors.index) + ["noactor"])



train['cast_weight'] = train.cast.fillna("[{'name': 'noactor'}]").apply(

    lambda x: np.sqrt(np.sum([actor_weights[a['name']]**2 for a in eval(x)])))



# actors that occur in test but not in train get zero weights assigned (no revenue information available)

test['cast_weight'] = test.cast.fillna("[{'name': 'noactor'}]").apply(

    lambda x: np.sqrt(np.sum([actor_weights[a['name']]**2 

                              if a['name'] in actors_train else 0 

                              for a in eval(x)])))
cum_rev_actors.iloc[:10]
f, ax = plt.subplots()



ax.scatter(np.log1p(train.cast_weight), train.revenue, alpha=0.1)

ax.set_xlabel('cast_weight')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.cast_weight, train.revenue)))
featurelist += ['cast_weight']
crew = set([crew['name'] for crew in eval("+".join(

    [c for c in pd.concat([train, test], sort=False).crew.dropna()]))])

print(len(crew), 'different crew members')
# extract only those crew members that occur in train

crew_train = set([crew['name'] for crew in eval("+".join(

    [c for c in train.crew.dropna()]))])



# calculate cumulative revenue per crew member

cum_rev_crew = [train.loc[train.crew.fillna('').str.contains("'name': '{}'".format(c)), 'revenue'].agg('exp').sum()

                  for c in crew_train]

cum_rev_crew = pd.Series(cum_rev_crew, index=crew_train).sort_values(ascending=False)



cumulative_revenue = cum_rev_crew.cumsum()/cum_rev_crew.cumsum()[-1]



crew_weights = []



# define tiers and weighting scheme; high-revenue = high weight

weight = 10

threshold = 0.1

for i in range(len(cumulative_revenue)): 

    if cumulative_revenue[i] > threshold:

        weight -= 1

        threshold += 0.1

    crew_weights.append(weight)



crew_weights = pd.Series(crew_weights + [0], 

                          index=list(cum_rev_crew.index) + ["nocrew"])



train['crew_weight'] = train.crew.fillna("[{'name': 'nocrew'}]").apply(

    lambda x: np.sqrt(np.sum([crew_weights[c['name']]**2 for c in eval(x)])))



# crew members that occur in test but not in train get zero weights assigned (no revenue information available)

test['crew_weight'] = test.crew.fillna("[{'name': 'noactor'}]").apply(

    lambda x: np.sqrt(np.sum([crew_weights[c['name']]**2 

                              if c['name'] in crew_train else 0 

                              for c in eval(x)])))
f, ax = plt.subplots()



ax.scatter(np.log1p(train.crew_weight), train.revenue, alpha=0.1)

ax.set_xlabel('crew_weight')

ax.set_ylabel('log(revenue)')



print('Pearson correlation test: r={} p={}'.format(*pearsonr(train.crew_weight, train.revenue)))
featurelist += ['crew_weight']
random_state = 1

n_folds = 5



def prepare_params(n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,

                  max_leaf_nodes, min_impurity_decrease):

    params = {'n_jobs': -1, 'random_state': 42}

    params['n_estimators'] = max(min(int(n_estimators), 10000), 0)

    params['max_depth'] = max(min(int(max_depth), 100), 1)

    params['min_samples_split'] = max(min(int(min_samples_split), 10), 2)

    params['min_samples_leaf'] = max(min(int(min_samples_leaf), 20), 1)

    params['min_weight_fraction_leaf'] = max(min(min_weight_fraction_leaf, 1), 0)

    params['max_features'] = max(min(int(max_features), len(featurelist)), 1)

    params['max_leaf_nodes'] = max(min(max_leaf_nodes, 1), 10000)

    params['min_impurity_decrease'] = max(min(min_impurity_decrease, 1), 0)

    return params

    

def eval_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,

               max_features, max_leaf_nodes, min_impurity_decrease):

    params = prepare_params(n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf,

               max_features, max_leaf_nodes, min_impurity_decrease)



    model = RandomForestRegressor(**params)

    

    cv_result = np.mean(cross_val_score(model, train.loc[:, featurelist], train.revenue, cv=n_folds, 

                                    scoring='neg_mean_squared_error'))

    

    return -np.sqrt(-cv_result)
rf_optimizer = BayesianOptimization(f=eval_model, 

                                    pbounds={'n_estimators': (800, 1200), 

                                             'max_depth': (7, 10), 

                                             'min_samples_split': (2, 5),

                                             'min_samples_leaf': (1, 2), 

                                             'min_weight_fraction_leaf': (0.01, 0.03),

                                             'max_features': (3, 6),

                                             'max_leaf_nodes': (8, 12),

                                             'min_impurity_decrease': (0.003, 0.008)}, #len(featurelist))},

                                    random_state=random_state)
# rf_optimizer.maximize(init_points=100, n_iter=20, alpha=1e-4)

# rf_optimizer.max
bestfit = {'params': {'max_depth': 9.873668590451505,

  'max_features': 4.599495854919051,

  'max_leaf_nodes': 10.767508455801893,

  'min_impurity_decrease': 0.004577578155030315,

  'min_samples_leaf': 1.6865009276815837,

  'min_samples_split': 4.503877015692119,

  'min_weight_fraction_leaf': 0.010365765546883836,

  'n_estimators': 950.0288629889935},

 'target': -2.1252538162396446}
params = bestfit['params']

params = prepare_params(params['n_estimators'], params['max_depth'], params['min_samples_split'], 

                        params['min_samples_leaf'], params['min_weight_fraction_leaf'], 

                        params['max_features'], params['max_leaf_nodes'], params['min_impurity_decrease'])



# train a model using the best-fit parameters

model = RandomForestRegressor(**params).fit(train.loc[:, featurelist], train.revenue)



# make the prediction

pred = model.predict(test.loc[:, featurelist])

pred = pd.DataFrame({'revenue': np.expm1(pred).astype(np.int64)}, index=test.index)

pred.to_csv('submission.csv')
fimp = pd.Series(model.feature_importances_, index=featurelist)

fimp.plot.bar()