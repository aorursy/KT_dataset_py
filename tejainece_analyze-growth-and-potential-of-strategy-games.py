import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read the data and take a look at its columns



ds = pd.read_csv('../input/ign.csv')

ds.head(2)
# We don't need some columns for this analysis



ds.drop(['Unnamed: 0', 'url', 'release_month', 'release_day'], axis=1, inplace=True)

ds.head(2)
# Plot number of strategy game releases by year



ds_strategy = ds[(ds['genre']=='Strategy')]



ds_strategy.release_year.value_counts().sort_index().plot.bar()



# Conclusion: 2008 to 2010 was golden period for strategy games
# Plot ratio of number strategy games released to number of total games released by year



g = ds_strategy.groupby(by='release_year')

df = g.agg({

    'editors_choice': {

        'sum': lambda x: x.value_counts().get('Y', 0)

    }

})



df['editors_choice'].plot.line(figsize=(10, 6))
# Plot average score of strategy games by year



ds_strategy.groupby(by='release_year').mean().plot.line(figsize=(10, 6))
# Plot number of editor's choice for strategy games by year



g = ds_strategy.groupby(by='release_year')

df = g.agg({

    'editors_choice': {

        'sum': lambda x: x.value_counts().get('Y', 0)

    }

})



df['editors_choice'].plot.line(figsize=(10, 6))
strategy_scores = ds_strategy['score_phrase'].value_counts()

strategy_scores.plot.pie(figsize=(6,6))
# Pie chart of great and good strategy games by platform



ds_strategy_great = ds_strategy[(ds_strategy['score_phrase'] == 'Great') | (ds_strategy['score_phrase'] == 'Good')]

f = ds_strategy_great.platform.value_counts().where(lambda x: x > 3).dropna()

f.plot.pie(figsize=(10, 10))