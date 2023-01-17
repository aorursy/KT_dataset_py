%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime as dt

required_df = pd.read_csv('../input/minor-tmdb/tmdb-movies .csv')
required_df.head(1)
required_df.dtypes
print(required_df.shape)

print(required_df.duplicated().sum())

print("Display Duplicate Row :")

for i,v in required_df.duplicated().iteritems():

    if (v):

        print (required_df.loc[i])

required_df.isnull().sum()
required_df.describe()
required_df['genres'] = required_df['genres'].str.split("|",expand=True)

required_df['genres'].value_counts()
required_df['production_companies'] = required_df['production_companies'].str.split("|",expand=True)

required_df['production_companies'].value_counts()
required_df['release_date'] = pd.to_datetime(required_df['release_date'])

required_df["release_date"].head()
required_df.isna().sum()

required_df.fillna(0)
required_df.drop(['imdb_id','homepage','tagline','overview','budget_adj','revenue_adj'], axis=1, inplace=True)
required_df.head()
required_df.drop_duplicates(inplace=True)

#check

required_df.duplicated().sum()
required_df.shape

print("Rows With Zero Values In  Budget Column:",required_df[(required_df['budget']==0)].shape[0])

print("Rows With Zero Values In Revenue Column:",required_df[(required_df['revenue']==0)].shape[0])

print("Rows With Zero Values In Runtime Column:",required_df[(required_df['runtime']==0)].shape[0])
required_df['budget']=required_df['budget'].replace(0, np.NAN)

required_df['revenue'] = required_df['revenue'].replace(0,np.NAN)

required_df['runtime'] = required_df['runtime'].replace(0,np.NAN)
required_df.head()
correlation = required_df.corr()

mask = np.zeros_like(correlation)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(10, 7))

    ax = sns.heatmap(correlation, mask=mask,linewidths = 0.5, annot = True, cmap="YlGnBu")

budget = required_df["budget"]

y_label = "Count"

print("Consider Budget,")

print("\t\t Minimum Value :",budget.min())

print("\t\t Maximum Value :",budget.max())

print("\t\t Mean          :",round(budget.mean(),2))

print("\t\t Median        :",budget.median())

print("\t\t Std. Deviation:",round(budget.std(),2))

fig = sns.distplot(budget,kde=False, rug=True);

fig.set(ylabel = y_label)
revenue = required_df["revenue"]

y_label = "Count"

print("Consider Budget,")

print("\t\t Minimum Value :",revenue.min())

print("\t\t Maximum Value :",revenue.max())

print("\t\t Mean          :",round(revenue.mean(),2))

print("\t\t Median        :",revenue.median())

print("\t\t Std. Deviation:",round(revenue.std(),2))

fig = sns.distplot(revenue,kde=False, rug=True);

fig.set(ylabel = y_label)
popularity = required_df["popularity"]

y_label = "Count"

print("Consider Popularity,")

print("\t\t Minimum Value :",popularity.min())

print("\t\t Maximum Value :",popularity.max())

print("\t\t Mean          :",round(popularity.mean(),2))

print("\t\t Median        :",popularity.median())

print("\t\t Std. Deviation:",round(popularity.std(),2))

fig = sns.distplot(popularity,kde=False, rug=True);

fig.set(ylabel = y_label)
release_year = required_df["release_year"]

y_label = "Count"

print("Consider Budget,")

print("\t\t Minimum Value :",release_year.min())

print("\t\t Maximum Value :",release_year.max())

print("\t\t Mean          :",round(release_year.mean(),2))

print("\t\t Median        :",release_year.median())

print("\t\t Std. Deviation:",round(release_year.std(),2))

fig = sns.distplot(release_year,kde=False, rug=True);

fig.set(ylabel = y_label)
year_pop = required_df.groupby('release_year')['popularity'].mean()

year_pop.plot(title='Popularity rend over years')

plt.xlabel('Release Year',fontsize=15)

plt.ylabel('Popularity',fontsize=15)

def cut_level(column_name):

    min_value=required_df[column_name].min()

    first_quantile=required_df[column_name].quantile(0.25)

    second_quantile=required_df[column_name].quantile(0.5)

    third_quantile=required_df[column_name].quantile(0.75)

    max_value=required_df[column_name].max()

    bin_edges=[min_value, first_quantile, second_quantile, third_quantile, max_value]

    bin_names=['low','medium','high','sky-high']

    return pd.cut(required_df[column_name], bin_edges, labels = bin_names)    
required_df['binned_budget']= cut_level('budget')

required_df['binned_budget']
sns.scatterplot(budget,popularity,data = required_df, hue = 'binned_budget', alpha = 0.7)
required_df['binned_revenue']= cut_level('revenue')

sns.scatterplot(revenue,popularity,data = required_df, alpha = 0.75,hue = 'binned_revenue')
sns.scatterplot(budget,revenue,data = required_df, hue = 'binned_budget', alpha = 0.7)
required_df['binned_vote']= cut_level('vote_average')

vote_popu = required_df.groupby('binned_vote')['popularity'].mean()

vote_popu.plot(kind='bar',alpha=0.7)

plt.xlabel('Votes',fontsize=15)

plt.ylabel('Popularity',fontsize=15)

plt.title('Popularity at different Vote Averages')

plt.show()
inf = required_df.groupby('release_year')['runtime'].mean()

inf.plot(title='Avg Runtime trend over years',xticks = np.arange(1960,2016,5))

plt.xlabel('Release Year',fontsize=15)

plt.ylabel('Avg Runtime',fontsize=15)

inf[2006]
info = pd.DataFrame(required_df['budget'].sort_values(ascending = False))

info['original_title'] = required_df['original_title']

print('Third hightest budget :',info.iloc[2])

info2 = pd.DataFrame(required_df['budget'].sort_values(ascending = True))

info2['original_title'] = required_df['original_title']

print('Third lowest budget:',info2.iloc[2])
minb = required_df["revenue"].idxmin()

maxb = required_df["revenue"].idxmax()

print("Movie with highest revenue:", required_df["original_title"][maxb])

print("Movie with lowest revenue:", required_df["original_title"][minb])
info3 = pd.DataFrame(required_df['revenue'].sort_values(ascending = True))

info3['production_companies'] = required_df['production_companies']

info3['orignal_title'] = required_df['original_title']

info3["production_companies"]
required_df["count"] = required_df['original_title'].str.split().apply(len)

year_count = required_df.groupby('release_year')['count'].mean()

year_count.plot(title='Avg #Words in Movie Titles trend over years')

plt.xlabel('Release Year',fontsize=15)

plt.ylabel('Avg #Words',fontsize=15)

y = required_df[(required_df["release_date"]>="2000") & (required_df["release_date"]<="2005")]

y["count"].mean()