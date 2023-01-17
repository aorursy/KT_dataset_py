import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly

import plotly.figure_factory as ff



df = pd.read_csv("../input/beerreviews/beer_reviews.csv")

df.head(5)
print("types of each columns: \n\n",df.dtypes)

print("\ninformation of the columns: \n")

print(df.info())
print("Count of unique breweries, by brewery_id: " ,df.brewery_id.nunique())

print("Count of unique breweries, by brewery_name: " ,df.brewery_name.nunique())
print("Count of unique beers, by beer_id: " ,df.beer_beerid.nunique())

print("Count of unique beers, by beer_name: " ,df.beer_name.nunique())
print("Count of unique users, by review_profilename: " ,df.review_profilename.nunique())
print("Overview of missing values in the dataset: /n",df.isnull().sum())

df=df.dropna()

print("After dropping the missing value",df.info())
print("a user review the same beer more than one time, by beer_beerid: \n",df.loc[df.duplicated(['review_profilename','beer_beerid'],keep=False)][['review_profilename','beer_name','beer_beerid','review_overall']])
print("a user review the same beer more than one time, by beer_name: \n",df.loc[df.duplicated(['review_profilename','beer_name'],keep=False)][['review_profilename','beer_beerid','beer_name',"review_overall"]])
df = df.sort_values("review_overall",ascending=False)

df = df.drop_duplicates(subset = ['review_profilename','beer_beerid'],keep='first')

df = df.drop_duplicates(subset = ['review_profilename','beer_name'],keep='first')

df.info()
round(df.describe(),2)
df = df.loc[(df.review_overall>=1) & (df.review_appearance>=1)]

round(df.describe(),2)
df.review_time = pd.to_datetime (df.review_time,unit = 's')
df.dtypes
df.hist(figsize=(15,15),color='#007399')
bar = go.Bar(x=df.brewery_name.value_counts().head(10).sort_values(ascending=True),

             y=df.brewery_name.value_counts().head(10).sort_values(ascending=True).index,

             hoverinfo = 'x',

             text=df.brewery_name.value_counts().head(10).sort_values(ascending=True).index,

             textposition = 'inside',

             orientation = 'h',

             opacity=0.75, 

             marker=dict(color='rgb(1, 77, 102)'))



layout = go.Layout(title='The Top 10 popular breweries',

                   xaxis=dict(title="Count of reviews",),

                   margin = dict(l = 220),

                   font=dict(family='Comic Sans MS',

                            color='dark gray'))



fig = go.Figure(data=bar, layout=layout)



# Plot it

plotly.offline.iplot(fig)
brewery_type = df.groupby('brewery_name')

brewery_type = brewery_type.agg({"beer_name":"nunique"})

brewery_type = brewery_type.reset_index()





bar2 = go.Bar(x=brewery_type.sort_values(by="beer_name",ascending=False).head(10).sort_values(by="beer_name",ascending=True).beer_name,

              y=brewery_type.sort_values(by="beer_name",ascending=False).head(10).sort_values(by="beer_name",ascending=True).brewery_name,

              hoverinfo = 'x',

              text=brewery_type.sort_values(by="beer_name",ascending=False).head(10).sort_values(by="beer_name",ascending=True).brewery_name,

              textposition = 'inside',

              orientation = 'h',

              opacity=0.75, 

              marker=dict(color='rgb(1, 77, 102)'))



layout = go.Layout(title='Top 10 brewery with the most beer types',

                   xaxis=dict(title="Count of beer types",),

                   margin = dict(l = 220),

                   font=dict(family='Comic Sans MS',

                            color='dark gray'))



fig = go.Figure(data=bar2, layout=layout)



# Plot it

plotly.offline.iplot(fig)
bar3 = go.Bar(x=df.beer_name.value_counts().head(10).sort_values(ascending=True),

              y=df.beer_name.value_counts().head(10).sort_values(ascending=True).index,

              hoverinfo = 'x',

              text=df.beer_name.value_counts().head(10).sort_values(ascending=True).index,

              textposition = 'inside',

              orientation = 'h',

              opacity=0.75, 

              marker=dict(color='rgb(1, 77, 102)'))



layout = go.Layout(title='Top 10 popular beers',

                   xaxis=dict(title="Count of reviews",),

                   margin = dict(l = 220),

                   font=dict(family='Comic Sans MS',

                            color='dark gray'))



fig = go.Figure(data=bar3, layout=layout)



plotly.offline.iplot(fig)
rate_beer = df[['beer_name','review_overall']].groupby('beer_name').agg('mean')



rate_beer = rate_beer.reset_index()



rate_beer

bar4 = go.Bar(x=rate_beer.sort_values(by="review_overall",ascending=False).head(10).sort_values(by="review_overall",ascending=True).review_overall,

              y=rate_beer.sort_values(by="review_overall",ascending=False).head(10).sort_values(by="review_overall",ascending=True).beer_name,

              hoverinfo = 'x',

              text=rate_beer.sort_values(by="review_overall",ascending=False).head(10).sort_values(by="review_overall",ascending=True).review_overall,

              textposition = 'inside',

              orientation = 'h',

              opacity=0.75, 

              marker=dict(color='rgb(1, 77, 102)'))



layout = go.Layout(title='Top 10 beers with highest rating',

                   xaxis=dict(title="Count of reviews",),

                   margin = dict(l = 220),

                   font=dict(family='Comic Sans MS',

                            color='dark gray'))



fig = go.Figure(data=bar4, layout=layout)



plotly.offline.iplot(fig)
aa=list(rate_beer.sort_values(by="review_overall",ascending=False).head(10).sort_values(by="review_overall",ascending=True).beer_name)

df[df['beer_name'].isin(aa)].groupby("beer_name").agg("count").reset_index().beer_name.value_counts()
bar5 = go.Bar(x=df.beer_style.value_counts().head(10).sort_values(ascending=True),

              y=df.beer_style.value_counts().head(10).sort_values(ascending=True).index,

              hoverinfo = 'x',

              text=df.beer_style.value_counts().head(10).sort_values(ascending=True).index,

              textposition = 'inside',

              orientation = 'h',

              opacity=0.75, 

              marker=dict(color='rgb(1, 77, 102)'))



layout = go.Layout(title='The Top 10 popular beers styles',

                   xaxis=dict(title="Count of reviews",),

                   margin = dict(l = 220),

                   font=dict(family='Comic Sans MS',

                            color='dark gray'))



fig = go.Figure(data=bar5, layout=layout)



# Plot it

plotly.offline.iplot(fig)
popular= df.brewery_name.value_counts().sort_index()

popular = popular.reset_index()

print("The correlation of reviews and beer types: ",brewery_type.beer_name.corr(popular.brewery_name))
### Distribution of beer_abv
plt.figure(figsize=(10,8))

plt.title("Distribution of beer abv")

sns.distplot(df.beer_abv)

plt.xlabel("beer abv %")
print("Review count of each beer \n ",df.beer_name.value_counts().describe())
sns.distplot(df.beer_beerid.value_counts(),kde=False)

plt.xlabel("beer_id")

plt.ylabel("count of reviews")

plt.title("distribution of beer's reviews")

plt.show()
reshape=df[['review_overall','beer_name']].groupby("beer_name").agg(['count','mean'])

print("Beers with review_overall more than 4: \n",reshape[reshape['review_overall',  'mean']>4])

print("Beers with review_overall more than 4, and number of review less than 30: \n",reshape[reshape['review_overall',  'mean']>4][reshape[reshape['review_overall',  'mean']>4]['review_overall',  'count']<30])
top10_breweries=df.brewery_name.value_counts().head(10).reset_index()

top10_styles=df.beer_style.value_counts().head(10).reset_index()

subset = df[df['brewery_name'].isin(top10_breweries['index'])& df['beer_style'].isin(top10_styles['index'])]

reshaped_subset = subset[['review_overall','beer_name']].groupby("beer_name").agg(['count','mean'])

reshaped_subset = reshaped_subset[reshaped_subset['review_overall',  'count']>30]

reshaped_subset.columns

reshaped_subset.sort_values(('review_overall',  'mean'),ascending=False).head(2)

categories=['review_overall','review_aroma', 'review_appearance', 'review_palate', 'review_taste']

r1=df[df.beer_name=="Founders CBS Imperial Stout"]

r2=df[df.beer_name=="Founders KBS (Kentucky Breakfast Stout)"]

r1_value=[r1.review_overall.mean(),r1.review_aroma.mean(),r1.review_appearance.mean(),r1.review_palate.mean(),r1.review_taste.mean()]

r2_value=[r2.review_overall.mean(),r2.review_aroma.mean(),r2.review_appearance.mean(),r2.review_palate.mean(),r2.review_taste.mean()]



mean_value=[df.review_overall.mean(),df.review_aroma.mean(),df.review_appearance.mean(),df.review_palate.mean(),df.review_taste.mean()]
fig = go.Figure()



fig.add_trace(go.Scatterpolar(

      r=r1_value,

      theta=categories,

      fill='toself',

      name='Founders CBS Imperial Stout'

))



fig.add_trace(go.Scatterpolar(

      r=mean_value,

      theta=categories,

      fill='toself',

      name='Overall_mean'

))



fig.update_layout(title="Radar chart of review features - Founders CBS Imperial Stout",

  polar=dict(

    radialaxis=dict(

      visible=True,

      range=[0, 5]

    )),

  showlegend=True

)



fig.show()
fig = go.Figure()

fig.add_trace(go.Scatterpolar(

      r=r2_value,

      theta=categories,

      fill='toself',

      name='Founders KBS (Kentucky Breakfast Stout)'

))

fig.add_trace(go.Scatterpolar(

      r=mean_value,

      theta=categories,

      fill='toself',

      name='Overall_mean'

))

fig.update_layout(title="Radar chart of review features - Founders KBS (Kentucky Breakfast Stout)",

  polar=dict(

    radialaxis=dict(

      visible=True,

      range=[0, 5]

    )),

  showlegend=True

)



fig.show()
print("Breweries of the beer recommendations: ",df[df.beer_name=="Founders KBS (Kentucky Breakfast Stout)"].brewery_name.unique(),df[df.beer_name=="Founders CBS Imperial Stout"].brewery_name.unique())

print("Styles of the recommendations: ",df[df.beer_name=="Founders KBS (Kentucky Breakfast Stout)"].beer_style.unique(),df[df.beer_name=="Founders CBS Imperial Stout"].beer_style.unique() )
df[df.brewery_name=='Founders Brewing Company'].beer_name.value_counts()
bar5 = go.Bar(x=df[df.brewery_name=='Founders Brewing Company'].beer_name.value_counts().head(20).sort_values(ascending=True),

              y=df[df.brewery_name=='Founders Brewing Company'].beer_name.value_counts().head(20).sort_values(ascending=True).index,

              hoverinfo = 'x',

              text=df[df.brewery_name=='Founders Brewing Company'].beer_name.value_counts().head(20).sort_values(ascending=True).index,

              textposition = 'inside',

              orientation = 'h',

              opacity=0.75, 

              marker=dict(color='rgb(1, 77, 102)'))



layout = go.Layout(title='The Top 15 bestsellers of Founders Brewing Company',

                   xaxis=dict(title="Count of reviews",),

                   margin = dict(l = 220),

                   font=dict(family='Comic Sans MS',

                            color='dark gray'))



fig = go.Figure(data=bar5, layout=layout)



# Plot it

plotly.offline.iplot(fig)
corr= df[["review_appearance","review_aroma","review_palate","review_taste", "review_overall"]].corr()

corr

x=list(corr.index)

y=list(corr.columns)



fig = ff.create_annotated_heatmap(x=x,y=y,z=corr.values.round(2),colorscale=[[0, 'navy'], [1, 'plum']],font_colors = ['white', 'black'])



fig.show()
time=df["review_time"].groupby(df.review_time.dt.date).agg('count')

fig = go.Figure(data=go.Scatter(x=time.index, y=time.values))

fig.update_layout(title='The time-series line chart of reviews',

                   xaxis_title='Date',

                   yaxis_title='Count of reviews')

fig.show()