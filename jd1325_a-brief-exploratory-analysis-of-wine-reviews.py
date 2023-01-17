
import numpy as np 
import pandas as pd 
import seaborn as sns
import plotnine as pn
import matplotlib.pyplot as plt
reviews = pd.read_csv('../input/winemag-data-130k-v2.csv', index_col = 0)



reviews.head()
years = reviews.title.str.extract('([1-2][0-9]{3})').astype('float64')

years[years < 1990] = None
reviews = reviews.assign(year = years)


good_countries = reviews.loc[reviews.country.isin(['US','Italy','Portugal','Spain','France','Germany','Australia']),:]
plt.subplots(figsize=(10,10))

sns.violinplot(x = good_countries.country,y = np.log(good_countries.price), 
                          figure_size = [2,2]).set_title('Price by Country')


plt.xlabel("Country")
plt.ylabel("Log of Price")

(pn.ggplot(good_countries,pn.aes(x = 'points', y = 'price', color = 'country')) 
 + pn.facet_wrap('~country', scales = 'free')+ pn.stat_smooth(method = 'lowess', span = .5)
)
yearly_price_mean = reviews.groupby('year').price.agg(['mean'])
yearly_price_max = reviews.groupby('year').price.agg(['max'])
yearly_point_mean = reviews.groupby('year').points.agg(['mean'])
yearly_point_max = reviews.groupby('year').points.agg(['max'])
reviews.year.value_counts().sort_index()
fig, axarr = plt.subplots(2, 2, figsize=(16, 10))

(yearly_price_mean[yearly_point_mean.index >= 1994]
 .plot
 .line(title = 'Mean Price by Vintage',ax = axarr[0][0])
 .set(xlabel = 'Year',ylabel = 'Average Price')
)

(yearly_price_max[yearly_point_max.index >= 1994]
.plot
.line(title = 'Max Price by Vintage',ax = axarr[0][1])
.set(xlabel = 'Year',ylabel = 'Max Price'))

(yearly_point_mean[yearly_point_mean.index >= 1994]
.plot
.line(title = 'Mean Rating by Vintage',ax = axarr[1][0])
.set(xlabel = 'Year',ylabel = 'Average Rating'))

(yearly_point_max[yearly_point_max.index >= 1994]
.plot
.line(title = 'Max Rating by Vintage',ax = axarr[1][1])
.set(xlabel = 'Year',ylabel = 'Max Rating'))
is_word_used = reviews.description.str.contains(pat = 'aroma|taste|color|grape|age')

sum(is_word_used)/len(is_word_used)


is_word_used = reviews.description.str.contains(
    pat = 'fruit|crisp|clean|sweet|tart|red|white|wood|apple|pear|pineapple|lemon|pomegranate|wood|oak')

sum(is_word_used)/len(is_word_used)