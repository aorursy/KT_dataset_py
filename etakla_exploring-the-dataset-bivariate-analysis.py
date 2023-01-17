# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from IPython.display import display, HTML

# Any results you write to the current directory are saved as output.



#For plotting

from matplotlib import pyplot as plt

import matplotlib.patches as mpatches



import seaborn as sns

%matplotlib inline
vg_df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

vg_df.User_Score = vg_df.User_Score.convert_objects(convert_numeric=True)
plt.figure(figsize=(12, 8))



vg_corr = vg_df.corr()

sns.heatmap(vg_corr, 

            xticklabels = vg_corr.columns.values,

            yticklabels = vg_corr.columns.values,

            annot = True);
plt.figure(figsize=(14, 14))



sns.pairplot(vg_df, diag_kind='kde');
#Group the entries by year, then get how many entries are there; i.e. the number of releases

temp1 = vg_df.groupby(['Year_of_Release']).count()

temp1 = temp1.reset_index()



#Do the same, but sum the values to get the total values of everything by year.

temp2 = vg_df.groupby(['Year_of_Release']).sum()

temp2 = temp2.reset_index()



#Normalize the data, i.e. zero mean and unit std. I did this to be able to compare the shapes of both graphs, since 

#they have different ranges

normalised_df = pd.DataFrame()



normalised_df['release_count'] = temp1['Name']

normalised_df['global_sales'] = temp2['Global_Sales']

normalised_df = (normalised_df - normalised_df.mean()) / normalised_df.std()#(normalised_df.max() - normalised_df.min()) 

normalised_df['year'] = temp1['Year_of_Release']





#Plot

plt.figure(figsize=(15, 9))

ax = sns.pointplot(x = normalised_df.year, y = normalised_df.release_count, color = 'blue', label='Release Count')

ax = sns.pointplot(x = normalised_df.year, y = normalised_df.global_sales, color = 'red', label='Global Sales')



blue_patch = mpatches.Patch(color='blue', label='NUMBER OF RELEASES')

red_patch = mpatches.Patch(color='red', label='GLOBAL SALES')

plt.legend(handles=[blue_patch, red_patch], loc='upper left', fontsize = 16)



plt.xticks(rotation=45);
fig = plt.figure(figsize=(10, 8))



genre_sales_percentages_by_year = (vg_df.groupby(['Year_of_Release', 'Genre']).Global_Sales.sum())*(100)/vg_df.groupby(['Year_of_Release']).Global_Sales.sum()

genre_sales_percentages_by_year.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', grid=False, figsize=(13, 4))



yearlySales = vg_df.groupby(['Year_of_Release','Genre']).Global_Sales.sum()

yearlySales.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(13, 4) ) ;
x = vg_df.groupby(['Genre']).sum().copy()

ax = x.Global_Sales.sort_values(ascending=False).plot(kind='bar', figsize=(13, 5));



for p in ax.patches:

    ax.annotate(str( round( p.get_height() ) ) + "\n" + str(round( p.get_height() /89.170) )+ "%", 

                (p.get_x() * 1.007, p.get_height() * 0.75),

                color='black')
#First is the number of releases per genre, second is the sales per genre, third is the average sales per game per genre

genre_difference_metric = [vg_df.Genre.value_counts().index, vg_df.groupby(['Genre']).sum().Global_Sales.sort_values(ascending=False).index, vg_df.groupby(['Genre']).mean().Global_Sales.sort_values(ascending=False).index]



#Dataframe to be used for plotting.

genre_evolution_df = pd.DataFrame(columns=['genre', 'rank_type', 'rank'])



#Populate the dataframe

for metric in range(3):

    for genre in range(len(genre_difference_metric[metric])):

        genre_evolution_df = genre_evolution_df.append({'genre':genre_difference_metric[metric][genre], 'rank_type': metric, 'rank':genre},

                                   ignore_index=True)



        

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)



sns.pointplot(x=genre_evolution_df.rank_type,

              y=12-genre_evolution_df['rank'], 

              hue=genre_evolution_df.genre)



for i in range(len(genre_difference_metric[0])):

    ax.text(-0.75, 12-i, genre_difference_metric[0][i], fontsize=11)

    ax.text(2.1, 12-i, genre_difference_metric[2][i], fontsize=11)

    

ax.set_xlim([-2,4])



xs = [0.0, 1.0, 2.0]

x_labels = ['total releases', 'total sales', 'average sales']

plt.xticks(xs, x_labels, rotation='vertical')



ax.set_xlabel('Sales Metric')



ys = range(1,13)

y_labels = ['12th', '11th', '10th', '9th', '8th', '7th', '6th', '5th', '4th', '3rd', '2nd', '1st']

plt.yticks(ys, y_labels)

ax.set_ylabel('Genre Rank')



plt.show();
rating_sales_percentages_by_year = (vg_df.groupby(['Year_of_Release', 'Rating']).Global_Sales.sum())*(100)/vg_df.groupby(['Year_of_Release']).Global_Sales.sum()

rating_sales_percentages_by_year.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(13, 4));
g = sns.jointplot(x = 'Critic_Score', 

              y = 'User_Score',

              data = vg_df, 

              kind = 'hex', 

              cmap= 'hot', 

              size=6)



#http://stackoverflow.com/questions/33288830/how-to-plot-regression-line-on-hexbins-with-seaborn

sns.regplot(vg_df.Critic_Score, vg_df.User_Score, ax=g.ax_joint, scatter=False, color='grey');
sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

sales_normalised_df = vg_df[sales_cols].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))



sns.regplot(x = sales_normalised_df.Global_Sales, y = sales_normalised_df.NA_Sales,    marker="+")

sns.regplot(x = sales_normalised_df.Global_Sales, y = sales_normalised_df.EU_Sales,    marker=".")

sns.regplot(x = sales_normalised_df.Global_Sales, y = sales_normalised_df.JP_Sales,    marker="x")

sns.regplot(x = sales_normalised_df.Global_Sales, y = sales_normalised_df.Other_Sales, marker="o")



plt.xlim(-0.05, 1.05)

plt.ylim(-0.05, 1.05)

fig.tight_layout();
genre_geo_rankings = [vg_df.groupby('Genre').sum().unstack().NA_Sales.sort_values(ascending=False).index, 

                      vg_df.groupby('Genre').sum().unstack().EU_Sales.sort_values(ascending=False).index,

                      vg_df.groupby('Genre').sum().unstack().Other_Sales.sort_values(ascending=False).index,

                      vg_df.groupby('Genre').sum().unstack().JP_Sales.sort_values(ascending=False).index

                      ]



#First is the number of releases per genre, second is the sales per genre, third is the average sales per game per genre

genre_geo_rank_df = pd.DataFrame(columns=['genre', 'rank_type', 'rank'])



#for metric in genre_difference_metric:

for region in range(4):

    for genre in range(len(genre_geo_rankings[region])):

        genre_geo_rank_df = genre_geo_rank_df.append({'genre':genre_geo_rankings[region][genre], 'rank_type': region, 'rank':genre},

                                   ignore_index=True)



fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)



sns.pointplot(x=genre_geo_rank_df.rank_type,

              y=12-genre_geo_rank_df['rank'], 

              hue=genre_geo_rank_df.genre)



for i in range(len(genre_geo_rankings[0])):

    ax.text(-0.9, 12-i, genre_geo_rankings[0][i], fontsize=11)

    ax.text(3.2, 12-i, genre_geo_rankings[3][i], fontsize=11)

    

ax.set_xlim([-2,5])



xs = [0.0, 1.0, 2.0, 3.0]

x_labels = ['North America', 'E.U.', 'Rest of the World', 'Japan']

plt.xticks(xs, x_labels, rotation='vertical')

ax.set_xlabel('Region')



ys = range(1,13)

y_labels = ['12th', '11th', '10th', '9th', '8th', '7th', '6th', '5th', '4th', '3rd', '2nd', '1st']

plt.yticks(ys, y_labels)

ax.set_ylabel('Genre Rank')



plt.show();
#temp is the sum of all variables for each platform by year

temp = vg_df.groupby(['Year_of_Release', 'Platform']).sum().reset_index().groupby('Year_of_Release')



platform_yearly_winner_df = pd.DataFrame()



for year, group in temp:

    current_year = temp.get_group(year)

    this_year_max_sales = 0.0

    current_year_winner = ""

    row = {'year':"", 'winner':"", 'sales':""}

    for index, platform_data in current_year.iterrows():

        if platform_data.Global_Sales > this_year_max_sales:

            this_year_max_sales = platform_data.Global_Sales

            current_year_winner = platform_data.Platform

    

    row['year'] = year

    row['winner'] = current_year_winner

    row['sales'] = this_year_max_sales

    platform_yearly_winner_df = platform_yearly_winner_df.append(row, ignore_index=True)



fig = plt.figure(figsize=(13, 4))



g = sns.pointplot(x = platform_yearly_winner_df.year ,

              y = platform_yearly_winner_df.sales , 

              hue = platform_yearly_winner_df.winner);



#http://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn-factorplot

g.set_xticklabels(g.get_xticklabels(), rotation=90);
platform_yearly_winner_df.set_index('year', inplace=True)

HTML(platform_yearly_winner_df.to_html())
x = vg_df.groupby(['Publisher']).sum().Global_Sales.copy()

x.sort_values(ascending=False, inplace=True)

x.head(10)
x = vg_df.groupby(['Publisher']).count().Name.copy()

x.sort_values(ascending=False, inplace=True)

x.head(10)
#http://stackoverflow.com/questions/30328646/python-pandas-group-by-in-group-by-and-average

vg_df.groupby(['Publisher', 'Year_of_Release'], as_index=False).mean().groupby('Publisher').mean().Global_Sales.sort_values(ascending=False).head(10)
vg_df.groupby(['Publisher']).filter(lambda x: len(x) > 5).groupby(['Publisher', 'Year_of_Release'], as_index=False).mean().groupby('Publisher').mean().Global_Sales.sort_values(ascending=False).head(10)
x = vg_df.groupby(['Publisher']).mean().Global_Sales.copy()

x.sort_values(ascending=False, inplace=True)

x.head(10)
vg_df[vg_df.Publisher == 'Palcom']
vg_df.groupby(['Publisher']).filter(lambda x: len(x) > 10).groupby(['Publisher']).Global_Sales.mean().sort_values(ascending=False).head(10)
top_publishers = ['Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment', \

                  'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Take-Two Interactive',

                  'Sony Computer Entertainment Europe', 'Microsoft Game Studios', 'Enix Corporation', 'Bethesda Softworks', 'SquareSoft'\

                  'Take-Two Interactive', 'LucasArts', '989 Studios', 'Hasbro Interactive', 'Universal Interactive']

#You can use set to create the list, I just have handtyped them just to be more attentive to the names.



top_publisher_df = vg_df[ vg_df['Publisher'].isin(top_publishers) ]
print("They make", 100*top_publisher_df.shape[0]/float(vg_df.shape[0]),"% of the number of releases in this dataset")
total_games_sales = vg_df.Global_Sales.sum()

top_publisher_total_sales = top_publisher_df.Global_Sales.sum()



print("Total Video Games Sales:", total_games_sales, "Million US$")

print("Total Top Publishers Sales:", top_publisher_total_sales, "Million US$")

print("They make ", 100*top_publisher_total_sales/total_games_sales,"% of the total video games sales")
x = top_publisher_df.groupby(['Publisher', 'Genre']).count().copy()

x.unstack().Name.idxmax(axis=1)
x = top_publisher_df.groupby(['Publisher', 'Genre']).sum().copy()

x.unstack().Global_Sales.idxmax(axis=1)
x = top_publisher_df.groupby(['Genre', 'Publisher']).count().copy()

x.unstack().Name.idxmax(axis=1)
x = top_publisher_df.groupby(['Genre', 'Publisher']).sum().copy()

x.unstack().Global_Sales.idxmax(axis=1)
x = top_publisher_df.groupby(['Genre', 'Publisher']).mean().copy()

x.unstack().Global_Sales.idxmax(axis=1)
ax = vg_df.groupby('Publisher').sum().unstack().NA_Sales.sort_values(ascending=False).head(10).plot(kind='bar', figsize=(13, 5));



for p in ax.patches:

    ax.annotate(str( round( p.get_height() ) ) + "\n" + str(round( 100.0* p.get_height() /vg_df.NA_Sales.sum()) )+ "%", 

                (p.get_x() + 0.13, p.get_height()-85),

                color='white', fontsize=12, fontweight='bold')
vg_df.groupby('Publisher').sum().unstack().EU_Sales.sort_values(ascending=False).head(10).plot(kind='bar');
vg_df.groupby('Publisher').sum().unstack().JP_Sales.sort_values(ascending=False).head(10).plot(kind='bar');
vg_df.groupby('Publisher').sum().unstack().Other_Sales.sort_values(ascending=False).head(10).plot(kind='bar');
vg_df.groupby('Genre').sum().unstack().NA_Sales.sort_values(ascending=False).plot(kind='bar');
vg_df.groupby('Genre').sum().unstack().EU_Sales.sort_values(ascending=False).head(10).plot(kind='bar');
vg_df.groupby('Genre').sum().unstack().JP_Sales.sort_values(ascending=False).head(10).plot(kind='bar');
vg_df.groupby('Genre').sum().unstack().Other_Sales.sort_values(ascending=False).head(10).plot(kind='bar');
vg_df.sort_values('Global_Sales', ascending=False).head(10).Name
#There are games with duplicate names (For each platform for example), so let's deal with this

x = vg_df.groupby(['Genre', 'Name']).sum().reset_index().groupby('Genre')



#A dataframe that will hold rankings, for nice display

best_selling_titles_by_genre_df = pd.DataFrame()



for name, group in x:

    temp_col = group.sort_values('Global_Sales', ascending=False).head(10).Name.reset_index(drop=True)

    best_selling_titles_by_genre_df[name] = temp_col
best_selling_titles_by_genre_df
#First is the number of releases per genre, second is the sales per genre, third is the average sales per game per genre

genre_difference_metric = [vg_df.Genre.value_counts().index, vg_df.groupby(['Genre']).sum().Global_Sales.sort_values(ascending=False).index, vg_df.groupby(['Genre']).mean().Global_Sales.sort_values(ascending=False).index]



#Dataframe to be used for plotting.

genre_evolution_df = pd.DataFrame(columns=['genre', 'rank_type', 'rank'])



#Populate the dataframe

for metric in range(3):

    for genre in range(len(genre_difference_metric[metric])):

        genre_evolution_df = genre_evolution_df.append({'genre':genre_difference_metric[metric][genre], 'rank_type': metric, 'rank':genre},

                                   ignore_index=True)



        

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)



sns.pointplot(x=genre_evolution_df.rank_type,

              y=12-genre_evolution_df['rank'], 

              hue=genre_evolution_df.genre)



for i in range(len(genre_difference_metric[0])):

    ax.text(-0.75, 12-i, genre_difference_metric[0][i], fontsize=11)

    ax.text(2.1, 12-i, genre_difference_metric[2][i], fontsize=11)

    

ax.set_xlim([-2,4])



xs = [0.0, 1.0, 2.0]

x_labels = ['total releases', 'total sales', 'average sales']

plt.xticks(xs, x_labels, rotation='vertical')



ax.set_xlabel('Sales Metric')



ys = range(1,13)

y_labels = ['12th', '11th', '10th', '9th', '8th', '7th', '6th', '5th', '4th', '3rd', '2nd', '1st']

plt.yticks(ys, y_labels)

ax.set_ylabel('Genre Rank')



plt.show();
g = sns.jointplot(x = 'Critic_Score', 

              y = 'User_Score',

              data = vg_df, 

              kind = 'hex', 

              cmap= 'hot', 

              size=6)



#http://stackoverflow.com/questions/33288830/how-to-plot-regression-line-on-hexbins-with-seaborn

sns.regplot(vg_df.Critic_Score, vg_df.User_Score, ax=g.ax_joint, scatter=False, color='grey');
genre_geo_rankings = [vg_df.groupby('Genre').sum().unstack().NA_Sales.sort_values(ascending=False).index, 

                      vg_df.groupby('Genre').sum().unstack().EU_Sales.sort_values(ascending=False).index,

                      vg_df.groupby('Genre').sum().unstack().Other_Sales.sort_values(ascending=False).index,

                      vg_df.groupby('Genre').sum().unstack().JP_Sales.sort_values(ascending=False).index

                      ]



#First is the number of releases per genre, second is the sales per genre, third is the average sales per game per genre

genre_geo_rank_df = pd.DataFrame(columns=['genre', 'rank_type', 'rank'])



#for metric in genre_difference_metric:

for region in range(4):

    for genre in range(len(genre_geo_rankings[region])):

        genre_geo_rank_df = genre_geo_rank_df.append({'genre':genre_geo_rankings[region][genre], 'rank_type': region, 'rank':genre},

                                   ignore_index=True)



fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111)



sns.pointplot(x=genre_geo_rank_df.rank_type,

              y=12-genre_geo_rank_df['rank'], 

              hue=genre_geo_rank_df.genre)



for i in range(len(genre_geo_rankings[0])):

    ax.text(-0.9, 12-i, genre_geo_rankings[0][i], fontsize=11)

    ax.text(3.2, 12-i, genre_geo_rankings[3][i], fontsize=11)

    

ax.set_xlim([-2,5])



xs = [0.0, 1.0, 2.0, 3.0]

x_labels = ['North America', 'E.U.', 'Rest of the World', 'Japan']

plt.xticks(xs, x_labels, rotation='vertical')

ax.set_xlabel('Region')



ys = range(1,13)

y_labels = ['12th', '11th', '10th', '9th', '8th', '7th', '6th', '5th', '4th', '3rd', '2nd', '1st']

plt.yticks(ys, y_labels)

ax.set_ylabel('Genre Rank')



plt.show();