import pandas as pd

flavors_of_cacao = pd.read_csv('../input/flavors_of_cacao.csv')
perc_and_rating = flavors_of_cacao[['Cocoa\nPercent', 'Rating']]
perc_and_rating.head()
mean_rating_by_perc = perc_and_rating.groupby(['Cocoa\nPercent'])['Rating'].mean()
mean_rating_by_perc.head()
sorted_mean_ratings = mean_rating_by_perc.sort_values(ascending=False)
sorted_mean_ratings.head()