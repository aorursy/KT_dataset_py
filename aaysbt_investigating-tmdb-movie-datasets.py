import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()
df = pd.read_csv('../input/tmdb_movies_data.csv')

df_copy = df.copy()

df.head()
df.info()
#Let's count the null rows using isnull() and sum() function

df.isnull().sum()
#'duplicated()' function return the duplicate row as True and othter as False

# using the sum() functions we can count the duplicate elements 

sum(df.duplicated())
#Let's drop these row using 'drop_duplicates()' function

df.drop_duplicates(inplace=True)
# Let's check the dataframe shape to see just 1 row dropped.

print('Shape of Data Frame after droppping duplicated rows:\n(Rows : Cloumns):', df.shape)
#Changing Format Of Release Date Into Datetime Format

df['release_date'] = pd.to_datetime(df['release_date'])

df['release_date'].head()
#Let's handle the budget and revenue

#this will replace the value of '0' to NaN of columns given in the list

df[['budget','revenue']] = df[['budget','revenue']].replace(0,np.NAN)



df.dropna(subset=['budget', 'revenue'], inplace=True)

print('After cleaning, we have {} rows'.format(df.shape[0]))
df.columns
#Let's delete the unused columns

del_col = ['imdb_id', 'homepage','tagline', 'keywords', 'overview','vote_average', 'budget_adj','revenue_adj']

df.drop(del_col, axis=1, inplace=True)

print('We have {} rows and {} columns' .format(df.shape[0], df.shape[1]))
#Before answering the questions, lets figure out the profits of each movie

df['profit'] = df['revenue']-df['budget']

df['profit'] = df['profit'].apply(np.int64)

df['budget'] = df['budget'].apply(np.int64)

df['revenue'] = df['revenue'].apply(np.int64)
df.head()
print(df.isnull().sum())
df.dtypes
def find_min_max(col_name):

    #using idxmin()  and idxmax() functions to find min and max value of the given column.

    #idxmin to find the index of lowest in given col_name

    min_index = df[col_name].idxmin()

    #idxmax to find the index of highest in given col_name

    max_index = df[col_name].idxmax()

    #select the lowest and hisghest value from given col_name

    low  = pd.DataFrame(df.loc[min_index,:])

    high = pd.DataFrame(df.loc[max_index,:])

    #Print the results

    

    print('Movie which has highest '+col_name+' : ', df['original_title'][max_index])

    print('Movie which has lowest '+col_name+' : ', df['original_title'][min_index])

    return pd.concat([high,low], axis=1)
def top_10(col_name,size=10):

    #find the all times top 10 for a fiven column

    #sort the given column and select the top 10

    df_sorted = pd.DataFrame(df[col_name].sort_values(ascending=False))[:size]

    df_sorted['original_title'] = df['original_title']

    plt.figure(figsize=(12,6))

    #Calculate the avarage

    avg = np.mean(df[col_name])   

    sns.barplot(x=col_name, y='original_title', data=df_sorted, label=col_name)

    plt.axvline(avg, color='k', linestyle='--', label='mean')

    if (col_name == 'profit' or col_name == 'budget' or col_name == 'revenue'):

        plt.xlabel(col_name.capitalize() + ' (U.S Dolar)')

    else:

        plt.xlabel(col_name.capitalize())

    plt.ylabel('')

    plt.title('Top 10 Movies in: ' + col_name.capitalize())

    plt.legend()
from matplotlib import gridspec

def each_year_best(col_name, size=15):

        #this function plot the last size=15 years best given varible 

        release = df[['release_year',col_name,'original_title']].sort_values(['release_year',col_name],

                                                                               ascending=False)

        # group by release year and find the best profit for each year

        release = pd.DataFrame(release.groupby(['release_year']).agg({col_name:[max,sum],

                                                                      'original_title':['first'] })).tail(size)

        #select the max from given column

        x_max = release.iloc[:,0]

        #select the sum from given column

        x_sum = release.iloc[:,1]

        #select the name title

        y_title = release.iloc[:,2]

        #select the index

        r_date = release.index  

        #plot the desirible variable

        fig = plt.figure(figsize=(12, 6))

        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2]) 

        ax0 = plt.subplot(gs[0])

        ax0 = sns.barplot(x=x_max, y=y_title, palette='deep')

        for j in range(len(r_date)):

            #put the year information on the plot

            ax0.text(j,j*1.02,r_date[j], fontsize=12, color='black')

        plt.title('Last ' +str(size)+ ' years highest ' +col_name+ ' movies for each year')

        plt.xlabel(col_name.capitalize())

        plt.ylabel('')

        ax1 = plt.subplot(gs[1])

        ax1 = sns.barplot(x=r_date, y=x_sum, palette='deep')

        plt.xticks(rotation=90) 

        plt.xlabel('Release Year')

        plt.ylabel('Total '+col_name.capitalize())

        plt.title('Last ' +str(size)+ ' years total '+ col_name)

        plt.tight_layout()
find_min_max('profit')
top_10('profit')
each_year_best('profit')
find_min_max('budget')
top_10('budget')
each_year_best('budget')
find_min_max('revenue')
top_10('revenue')
each_year_best('revenue')
#Let's also check it out longes and shortes movie using find_min_max() function

find_min_max('runtime')
def split_count_data(col_name, size=15):

    ##function which will take any column as argument from which data is need to be extracted and keep track of count

    #take a given column, and separate the string by '|'

    data = df[col_name].str.cat(sep='|')

    #storing the values separately in the series

    data = pd.Series(data.split('|'))

    #Let's count the most frequenties values for given column

    count = data.value_counts(ascending=False)

    count_size = count.head(size)

    #Setting axis name for multiple names

    if (col_name == 'production_companies'):

        sp = col_name.split('_')

        axis_name = sp[0].capitalize()+' '+ sp[1].capitalize()

    else:

        axis_name = col_name.capitalize()

    fig = plt.figure(figsize=(14, 6))

    #set the subplot 

    gs = gridspec.GridSpec(1,2, width_ratios=[2,2])

    #count of given column on the bar plot

    ax0 = plt.subplot(gs[0])

    count_size.plot.barh()

    plt.xlabel('Number of Movies')

    plt.ylabel(axis_name)

    plt.title('The Most '+str(size)+' Filmed ' +axis_name+' Versus Number of Movies')

    ax = plt.subplot(gs[1])

    #setting the explode to adjust the pei chart explode variable to any given size

    explode = []

    total = 0

    for i in range(size):

         total = total + 0.015

         explode.append(total)

    #pie chart for given size and given column

    ax = count_size.plot.pie(autopct='%1.2f%%', shadow=True, startangle=0, pctdistance=0.9, explode=explode)

    plt.title('The most '+str(size)+' Filmed ' +axis_name+ ' in Pie Chart')

    plt.xlabel('')

    plt.ylabel('')

    plt.axis('equal')

    plt.legend(loc=9, bbox_to_anchor=(1.4, 1))
split_count_data("genres")
split_count_data("cast")
split_count_data("director")
df_month = df.copy()

df_month['release_month'] = df_month['release_date'].dt.strftime("%B")



fig = plt.figure(figsize=(12,6))

count_month = df_month.groupby('release_month')['profit'].count()

plt.subplot(1,2,1)

count_month.plot.bar()

plt.xlabel('Release Month')

plt.ylabel('Number of Movies')

plt.title('Number of Movies released in each month')



plt.subplot(1,2,2)

sum_month = df_month.groupby('release_month')['profit'].sum()



sum_month.plot.bar()

plt.xlabel('Release Month')

plt.ylabel('Monthly total Profit ')

plt.title('Total profit by month (1950-2015)')

top_10('popularity', size=30)
top_10('vote_count', size=30)
df_related = df[['profit','budget','revenue','runtime', 'vote_count','popularity','release_year']]

sns.pairplot(df_related, kind='reg')