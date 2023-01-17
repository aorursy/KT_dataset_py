# importing the libraries for vizualization

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as plt
df_Netflix = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df_Netflix.head()
# Let us remove the rows which has country as blank

for i in df_Netflix.columns:

    null_rate = df_Netflix[i].isna().sum() / len(df_Netflix) * 100 

    if null_rate > 0 :

        print(f"{i}'s null rate : {null_rate}%")
# Let's remove the nulls from the country column

df_Netflix.dropna(subset=['country'], inplace=True)
df_Netflix.head()
df_Netflix["country"]= df_Netflix["country"].astype(str)
df_Netflix.count()
# as there are multiple values in the column country let us clean that

# Credits to this github link - https://gist.github.com/jlln/338b4b0b55bd6984f883



def splitDataFrameList(df,target_column,separator):

    ''' df = dataframe to split,

    target_column = the column containing the values to split

    separator = the symbol used to perform the split

    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 

    The values in the other columns are duplicated across the newly divided rows.

    '''

    def splitListToRows(row,row_accumulator,target_column,separator):

        split_row = row[target_column].split(separator)

        for s in split_row:

            new_row = row.to_dict()

            new_row[target_column] = s

            row_accumulator.append(new_row)

    new_rows = []

    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))

    new_df = pd.DataFrame(new_rows)

    return new_df
df_Netflix.head(10)
df_Netflix=splitDataFrameList(df_Netflix, 'country', ',')
df_Netflix.count()
df_Netflix.head(20)
sns.countplot(x="type", data=df_Netflix)
CountryPlot=sns.countplot(x="country", data=df_Netflix, order=df_Netflix.country.value_counts().iloc[:10].index)

for item in CountryPlot.get_xticklabels():

    item.set_rotation(90)
DirectorPlot=sns.countplot(x="director", data=df_Netflix, order=df_Netflix.director.value_counts().iloc[:10].index)

for item in DirectorPlot.get_xticklabels():

    item.set_rotation(90)
ReleaseYear_Plot=sns.countplot(x="release_year", data=df_Netflix, order=df_Netflix.release_year.value_counts().iloc[:10].index)

for item in ReleaseYear_Plot.get_xticklabels():

    item.set_rotation(90)
df_Plot_1 = pd.crosstab(index=df_Netflix["release_year"], 

                          columns=df_Netflix["type"])

df_Plot_1['Total'] = df_Plot_1['Movie'] + df_Plot_1['TV Show']

df_Plot_1_Top_10 = df_Plot_1.sort_values('Total',ascending = False).head(10)

df_Plot_1_Top_10.drop(['Total'], axis=1, inplace=True)
df_Plot_1_Top_10.plot(kind="bar", 

                 figsize=(10,10),

                 stacked=True)
df_Plot_2 = pd.crosstab(index=df_Netflix["country"], 

                          columns=df_Netflix["type"])

df_Plot_2['Total'] = df_Plot_2['Movie'] + df_Plot_2['TV Show']

df_Plot_2_Top_10 = df_Plot_2.sort_values('Total',ascending = False).head(10)

df_Plot_2_Top_10.drop(['Total'], axis=1, inplace=True)

df_Plot_2_Top_10
df_Plot_2_Top_10.plot(kind="bar", 

                 figsize=(10,10),

                 stacked=True)
df_Plot_3 = pd.crosstab(index=df_Netflix["rating"], 

                          columns=df_Netflix["type"])

df_Plot_3['Total'] = df_Plot_3['Movie'] + df_Plot_3['TV Show']

df_Plot_3_Top_10 = df_Plot_3.sort_values('Total',ascending = False).head(10)

df_Plot_3_Top_10.drop(['Total'], axis=1, inplace=True)

df_Plot_3_Top_10
df_Plot_3_Top_10.plot(kind="bar", 

                 figsize=(10,10),

                 stacked=True)
# There are few other rows in country which has India in them, for now let us ignore them and filter only on India
# We shall filter only one country (India) for now and continue the analysis



df_India_Netflix = df_Netflix.loc[df_Netflix['country'] == 'India']
df_India_Netflix.head()
df_India_Netflix.dtypes
## Let us first check when did Netflix come to India, we will see the min for the date_added ()



df_India_Netflix["date_added"] = df_India_Netflix["date_added"].apply(pd.to_datetime)
df_India_Netflix.dtypes
min_df_India_Netflix=min(df_India_Netflix['date_added'])

min_df_India_Netflix
# Let us extract the row which has the minimum date and see which was the first NEtflix movie/TV shwo addition to India



df_India_Netflix[df_India_Netflix['date_added'] == min_df_India_Netflix]
sns.countplot(x="type", data=df_India_Netflix)
India_DirectorPlot=sns.countplot(x="director", data=df_India_Netflix, order=df_India_Netflix.director.value_counts().iloc[:10].index)

for item in India_DirectorPlot.get_xticklabels():

    item.set_rotation(90)
# Let us check the addition by the director

df_India_Netflix[df_India_Netflix['director'] == 'S.S. Rajamouli']
# It is clear that there are 2 movies only which are repeated in different languages, hence the count is more
df_India_Netflix[df_India_Netflix['director'] == 'Rajiv Mehra']
# It is actally Rajiv Mehra who has contributed more in terms of number of movies
India_ReleaseYear_Plot=sns.countplot(x="release_year", data=df_India_Netflix, order=df_India_Netflix.release_year.value_counts().iloc[:10].index)

for item in India_ReleaseYear_Plot.get_xticklabels():

    item.set_rotation(90)
India_Rating_Plot=sns.countplot(x="rating", data=df_India_Netflix, order=df_India_Netflix.rating.value_counts().iloc[:10].index)

for item in India_Rating_Plot.get_xticklabels():

    item.set_rotation(90)
df_Plot_India_1 = pd.crosstab(index=df_India_Netflix["release_year"], 

                          columns=df_India_Netflix["type"])

df_Plot_India_1['Total'] = df_Plot_India_1['Movie'] + df_Plot_India_1['TV Show']

df_Plot_India_1_Top_10 = df_Plot_India_1.sort_values('Total',ascending = False).head(10)

df_Plot_India_1_Top_10.drop(['Total'], axis=1, inplace=True)
df_Plot_India_1_Top_10.plot(kind="bar", 

                 figsize=(10,10),

                 stacked=True)