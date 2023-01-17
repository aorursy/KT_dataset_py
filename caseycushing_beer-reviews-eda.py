# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
beer_data = pd.read_csv('/kaggle/input/beerreviews/beer_reviews.csv')



beer_data.head(20)



beer_data
beer_data.dtypes
beer_data.describe()
beer_data.isna().sum()
beer_data = beer_data.dropna()



beer_data.isna().sum()
beer_data
import matplotlib.pyplot as plt



beer_data.hist(figsize=(12,12))



plt.show()

beers_abv = beer_data.loc[:,['brewery_name','beer_name','beer_abv']]



beers_abv = beers_abv.groupby(['brewery_name','beer_name'])['beer_abv'].mean()



beers_abv = pd.DataFrame(data=beers_abv).reset_index()



beers_abv
beer_style_taste_abv = beer_data.loc[:,['beer_style','review_taste','review_overall','beer_abv']]



beer_style_taste_abv = beer_style_taste_abv.groupby('beer_style')['review_taste','review_overall','beer_abv'].mean()



beer_style_taste_abv = pd.DataFrame(data=beer_style_taste_abv)



beer_style_taste_abv = beer_style_taste_abv.sort_values(by=['review_taste'],ascending=False).reset_index()



beer_style_taste_abv
import plotly.express as px



import statsmodels



fig = px.scatter(beer_style_taste_abv,x="review_overall",y="review_taste",trendline="ols")



fig.show()
fig = px.scatter(beer_style_taste_abv,x="review_overall",y="beer_abv",trendline="ols")



fig.show()
fig = px.scatter(beer_style_taste_abv,x="review_overall",y="review_taste",color='beer_style',size='beer_abv')



fig.show()
brewery_style = beer_data.loc[:,['brewery_name','beer_style','beer_abv']]





brewery_style
flanders_style = brewery_style.query('beer_style == "Flanders Red Ale"').reset_index(drop=True)



flanders_style = flanders_style.groupby(['brewery_name'])['beer_abv'].mean()



flanders_style = flanders_style.reset_index()



flanders_style
fig = px.scatter(flanders_style,x="brewery_name",y="beer_abv")



fig.show()
gose_style = brewery_style.query('beer_style == "Gose"').reset_index()



gose_style = brewery_style.groupby(['brewery_name'])['beer_abv'].mean()



gose_style = gose_style.reset_index()



gose_style
fig = px.scatter(gose_style,x="brewery_name",y="beer_abv")



fig.show()
beer_style_taste_abv
fig = px.scatter(beer_style_taste_abv,x="beer_style",y="beer_abv")



fig.show()
style_count = brewery_style['beer_style'].value_counts()



style_count = pd.DataFrame(data=style_count).reset_index()



style_count = style_count.rename(columns={"index":"style","beer_style":"count"})



style_count
fig = px.bar(style_count,x="style",y="count")



fig.show()
reviewers = beer_data.loc[:,['review_profilename','beer_name','review_overall']]



reviewers
reviewers = beer_data.loc[:,['review_profilename','review_overall']]



reviewers
#reviewers_count = reviewers['review_profilename']



reviewers_count = reviewers['review_profilename'].value_counts().reset_index()



reviewers_count = reviewers_count.rename(columns={"index":"review_profilename","review_profilename":"count"})



reviewers_count
reviewers_count.dtypes
reviewers = reviewers.groupby('review_profilename')['review_overall'].mean()



reviewers = reviewers.round().astype('int64').reset_index()



reviewers
reviewers = reviewers.merge(reviewers_count,on='review_profilename')



reviewers
reviewers.describe()
reviewers.shape
#reviewers['count'] = reviewers['count'] != 0



reviewers['count'].hist(figsize=(10,10))
reviewers = reviewers.query("count > 1000")



fig = px.bar(reviewers,x="review_profilename",y="count")



fig.show()