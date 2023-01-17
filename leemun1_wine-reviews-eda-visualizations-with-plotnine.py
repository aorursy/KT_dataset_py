import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 100
# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import re

%matplotlib inline
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
reviews.head(2)
df = pd.DataFrame(reviews['country'].value_counts().head(10)).reset_index()
df.columns = ['country', 'count']
(ggplot(df)
 + aes(x='country', y='count')
 + geom_col(fill='darkslateblue')
 + ggtitle("Countries with Most Reviews")
 + xlab('Country')
 + ylab('Reviews')
 + coord_flip()
 + theme(figure_size=(8, 6))
)
# check for missing values in taster_name column
reviews.loc[reviews.taster_name.isnull()].head(2)
# fill missing values with "Unknown"
reviews.taster_name.fillna('Unknown', inplace=True)
df = pd.DataFrame(reviews['taster_name'].value_counts()).reset_index()
df.columns = ['Taster', 'Reviews']

(ggplot(df)
 + aes(x='Taster', y='Reviews')
 + geom_col(fill='gold', width=0.85)
 + ggtitle("Reviews by Each Taster")
 + geom_text(aes(label='Reviews'), size=9.5, ha='left')
 + xlab('Taster')
 + ylab('Reviews')
 + coord_flip(ylim=(0, 30000))
 + theme(figure_size=(8, 10))
)
(ggplot(reviews)
 + aes(x='points')
 + geom_bar(fill='maroon')
 + ggtitle("Review Scores")
 + xlab('Score')
 + ylab('Count')
 + theme(figure_size=(8, 6))
)
df = (reviews
    .groupby('variety').variety.agg([len])
    .sort_values(by='len', ascending=False)
    .reset_index()
    .head(10)
)

(ggplot(df)
 + aes(x='variety', y='len', fill='len')
 + geom_col()
 + ggtitle("Most Frequently Reviewed Varieties")
 + geom_text(aes(label='len'), ha='right', nudge_y=-100)
 + xlab('Variety')
 + ylab('Count')
 + coord_flip(ylim=(0, 20000)) # rotate axis
 + scale_fill_cmap('Paired') # set custom colormap
 + guides(fill=False) # remove legend
 + theme(figure_size=(8, 6))
)
# subset reviews dataframe for frequent varieties
common_wines = reviews.loc[reviews.variety.isin(df.variety.values)]
(ggplot(common_wines)
    + aes('points', 'variety')
    + geom_bin2d(bins=20)
    + coord_fixed(ratio=1)
    + ggtitle("Review Scores for Frequently Reviewed Varieties")
    + xlab('Score')
    + ylab('Variety')
    + scale_fill_cmap('RdPu')
    + theme(figure_size=(8, 4))
)
df = (common_wines
    .groupby('variety').price.mean()
    .reset_index()
    .round(2)
)

(ggplot(df)
 + aes(x='variety', y='price', fill='variety')
 + geom_col(fill='skyblue')
 + ggtitle("Average Prices of Frequently Reviewed Wines")
 + geom_text(aes(label='price'), ha='right', nudge_y=-0.8)
 + xlab('Variety')
 + ylab('Price')
 + coord_flip()
 + guides(fill=False)
 + theme(figure_size=(8, 6))
)
(ggplot(common_wines.loc[common_wines.variety.isin(['Rosé', 'Cabernet Sauvignon'])])
 + aes(x='points', y='price', color='variety')
 + geom_point()
 + ggtitle("Review Scores Vs. Price")
 + xlab('Score')
 + ylab('Price')
 + theme(figure_size=(8, 6))
)
df = (reviews
    .groupby('winery').points.agg(['mean'])
    .sort_values(by='mean', ascending=False)
    .reset_index()
    .rename(columns={'mean': 'price'})
    .round(2)
    .head(10)
)

(ggplot(df)
 + aes(x='winery', y='price')
 + geom_col(fill='#DB6058')
 + ggtitle("Top 10 Wineries")
 + geom_text(aes(label='price'), color='white', ha='right', nudge_y=-0.15)
 + xlab('Winery')
 + ylab('Average Price')
 + coord_flip(ylim=(90, 100))
 + theme(figure_size=(8, 6))
)
# extract year from title column
reviews['year'] = reviews.title.str.extract('((19|20)\d{2})')[0]

# check result
reviews.head(2)
df = (reviews
    .loc[reviews.year.notnull()]
    .assign(year=reviews.year.astype('float64'))
)

(ggplot(df)
 + aes('year')
 + geom_bar(width=0.7)
 + ggtitle("Years of Wines")
 + xlab('Year')
 + ylab('Count')
 + theme(axis_text_x=element_text(rotation=90),
         figure_size=(10, 4))
 + scale_x_continuous(breaks=range(1900, 2020, 5))
)
(ggplot(df)
 + aes('year')
 + geom_bar(width=0.7)
 + ggtitle("Years of Wines (1900 ~ 2000)")
 + xlab('Year')
 + ylab('Count')
 + theme(axis_text_x=element_text(rotation=90),
         figure_size=(10, 4))
 + scale_x_continuous(breaks=range(1900, 2020, 5))
 + scale_y_continuous(breaks=range(0, 21, 1))
 + coord_cartesian(xlim=(1900, 2000), ylim=(0, 20))
)
(ggplot(df)
 + aes(x='year', y='price')
 + geom_point(size=0.5)
 + ggtitle("Price of Wines Vs. Year")
 + xlab('Year')
 + ylab('Price')
 + theme(axis_text_x=element_text(rotation=90),
         figure_size=(10, 4))
 + scale_x_continuous(breaks=range(1900, 2020, 5))
)
(ggplot(df)
 + aes(x='year', y='points')
 + geom_point(size=0.5)
 + ggtitle("Review Scores Vs. Year")
 + xlab('Year')
 + ylab('Score')
 + theme(axis_text_x=element_text(rotation=90),
         figure_size=(10, 4))
 + scale_x_continuous(breaks=range(1900, 2020, 5))
)
(ggplot(df)
 + aes(x='year', y='price', size='points')
 + geom_point(fill='#DB6058', color='lightgray', alpha=0.25)
 + ggtitle("Price - Year - Review Score")
 + xlab('Year')
 + ylab('Price')
 + theme(axis_text_x=element_text(rotation=90),
         figure_size=(10, 4))
 + scale_x_continuous(breaks=range(1900, 2020, 5))
 + scale_size_radius(range=(3, 12))
)