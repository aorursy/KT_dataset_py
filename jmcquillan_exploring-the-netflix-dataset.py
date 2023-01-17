#First, we must import libraries needed for analysis

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#read the dataset

netflix = pd.read_csv('../input/netflix-shows-exploratory-analysis/netflix_titles.csv')

netflix.head()

netflix['date_added'].head()
netflix['date_added'] = pd.to_datetime(netflix['date_added'])

netflix['date_added'].head()
#we can filter by all movies released in the United States since 2010, and get the counts

movies_since_2010 = netflix[(netflix.date_added >= '2010-01-01') & (netflix.type == 'Movie')

                           & (netflix.country.str.contains('United States', case=False))]

movies_by_year = movies_since_2010.groupby(movies_since_2010.date_added.dt.year).show_id.count()



fig, ax = plt.subplots()



ax.bar(movies_by_year.index, movies_by_year)

ax.spines['right'].set_visible(False) #removing spines to minimize chartjunk

ax.spines['top'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.set_title('Movies added to Netflix (Released in U.S.)')

ax.tick_params(left=True, bottom=False)

ax.grid(False)

plt.show()
netflix[['title', 'country']].head()
#from stackoverflow

def explode(df, lst_cols, fill_value='', preserve_index=False):

    # make sure `lst_cols` is list-alike

    if (lst_cols is not None

        and len(lst_cols) > 0

        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):

        lst_cols = [lst_cols]

    # all columns except `lst_cols`

    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists

    lens = df[lst_cols[0]].str.len()

    # preserve original index values    

    idx = np.repeat(df.index.values, lens)

    # create "exploded" DF

    res = (pd.DataFrame({

                col:np.repeat(df[col].values, lens)

                for col in idx_cols},

                index=idx)

             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)

                            for col in lst_cols}))

    # append those rows that have empty lists

    if (lens == 0).any():

        # at least one list in cells is empty

        res = (res.append(df.loc[lens==0, idx_cols], sort=False)

                  .fillna(fill_value))

    # revert the original index order

    res = res.sort_index()

    # reset index if requested

    if not preserve_index:        

        res = res.reset_index(drop=True)

    return res
#since countries appear as comma separated values in the country column, we need to separate the values, and create new rows for each value

new = netflix.copy() #create a copy of the original dataframe

new = new.replace(np.nan, '', regex=True)

new['country'] = new.country.str.split(',')

new = explode(new, ['country']) #new dataframe with new rows added for each country

new['country'] = new.country.str.strip()

new[['title', 'country']].head()
results = new[new.type == 'Movie'].groupby('country').show_id.count().sort_values(ascending=True)[-11:] #group by the country column, get top ten countries by movie releases

results = results.drop(labels='') #dropping rows with blank values



fig, ax = plt.subplots()



ax.barh(results.index, results)

ax.set_title('Number of Netflix movies added')

ax.spines['right'].set_visible(False) #removing spines to minimize chartjunk

ax.spines['top'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.tick_params(left=False, bottom=False) #remove ticks

ax.grid(False) #remove grid



plt.show()

netflix[netflix.type == 'Movie'].duration.head()
def to_minutes(series): #function to return only the first element

    return series.split()[0]

    

        

netflix['duration'] = netflix.duration.apply(lambda x: to_minutes(x))
netflix[netflix.type == 'Movie'].duration.head()
netflix['duration'] = netflix.duration.astype('int') #to determine the average duration, must convert to int
netflix[netflix.type == 'Movie'].duration.mean() #average duration for all movies in the dataset
sns.set(style='whitegrid')



sns.distplot(netflix[netflix.type == 'Movie'].duration).set_title('Distribution of Movie Durations')

plt.show()
sns.boxplot(netflix[netflix.type == 'Movie'].duration).set_title('Distribution of Movie Durations')

plt.show()
duration_by_year = netflix[netflix.type == 'Movie'].groupby(netflix.date_added.dt.year).mean().duration

duration_by_year
fig, ax = plt.subplots()



ax.plot(duration_by_year)

ax.grid(False)

ax.set_title('Average duration of Netflix movies')

ax.spines['right'].set_visible(False) #removing spines to minimize chartjunk

ax.spines['top'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.set_xlabel('Year')

ax.set_ylabel('Average Duration')

plt.show()
netflix[['title','cast']].head()
cast_df = netflix.copy()

cast_df = cast_df.dropna(axis=0, how='any', subset=['cast'])

cast_df['cast'] = cast_df.cast.str.split(',')

cast_df = explode(cast_df, ['cast']) #new dataframe with new rows added for each country

cast_df[['title', 'cast']].head()
cast_df.groupby('cast').count()
cast_df_filtered = cast_df[(cast_df.cast != '') & cast_df.country.str.contains('United States', case=False)]

grouped_cast = cast_df_filtered.groupby('cast').count().show_id.sort_values()[-11:] #get top ten cast members

grouped_cast
fig, ax = plt.subplots(figsize=(7,5))



ax.barh(grouped_cast.index, grouped_cast)

ax.grid(False)

ax.spines['right'].set_visible(False) #remove spines to minimize chartjunk

ax.spines['top'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.set_title('Top Actors on Netflix')

ax.set_xlabel('Number of Titles')

ax.set_xlim(left=10) #start x-axis at 10

plt.show()