import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/EDA_data.csv', index_col = 0, header = 0)
pd.set_option('display.max_columns', None)
#Df partial view

#Df info

#Df shape

#Df Descriptive Statistics (Mean, Minimum, Maximum)

#Variables distributions

#Variable correlations
#Df partial view

#Df shape



print('DataFrame shape is {}'.format(df.shape))

print('\nDataFrame partial view:')

df.head()
#Df info about columns names, null values(no null values, beacues I did some ETL) and data types



df.info()
#Df Descriptive Statistics (Mean, Minimum, Maximum)



#Columns 9 to 28 are just categorical data, so I'll look at them later

df.describe().iloc[:,:6]
#From here, we already can extract some useful information, like average runtime of moveis, avearage budget, 

#quartiles and so on.

#But this insights better be visualized in some beautiful way
#Variables distributions



num_columns = ['budget', 'popularity', 'revenue', 'runtime']
#As we can see from desrciptive statistics even 75 procentile for 'budget' is 0, so there is a lot of 0's in this column

#Beacuse of this I'll use non 0 values for building histograms
#As there is a huge range difference between min and max values I'll draw plots using 90 procentile to set axis x limit



def draw_hist(series, name):

    '''

    Draws sns.hist with xlim(0, 95 procentile) 

    '''

    series = series.loc[series != 0]

    sorted = series.sort_values()

    procentile = sorted.iloc[int(0.9 * sorted.size)]

    

    hist = sns.distplot(series[series < procentile])

    hist.set_xlim((0, procentile))

    hist.set_title('{} distribution'.format(name), fontsize = 12)

    plt.show()

    

    return 
for i in range(4):

    draw_hist(df[num_columns[i]], num_columns[i])

    plt.show()
#Variable correlations
#Since we already have num features lets start with them
corr = df[num_columns].corr()
plt.figure(figsize = (8,6))



cmap = ['deepskyblue', 'gray', 'gray', 'gray', 'deepskyblue']

sns.heatmap(corr, linewidth = .5, annot = True, vmin = -1, vmax = 1, center = 0, cmap = cmap)



plt.show()
#So you  see from here, that there is some correlation beetwen revenue and budget, which makes sense.

#But, what is interesting, relation between popularity and revenue is not so strong, as I assumed.
#As there are a lot of 0s in this dataset, and this gives ground to doubt. So I'll drop 0 values and see what will change
test = df[num_columns].copy()

test.shape
test.describe()
test = test.loc[test['budget'] > 0 ]
test.describe()
test.shape
#From here we can conclude, that most of rows that contains 0 in budget also contains 0 in revenue, which is some kind of 

#possitive, becaues we don`t lose inforamtion by droping this rows
test = test.loc[test['revenue'] > 0 ]
test.shape
fig, (axis1, axis2)= plt.subplots(1,2, figsize = (14,6))



axis1 = sns.heatmap(corr, linewidth = .5, annot = True, vmin = -1, vmax = 1, center = 0, cmap = cmap, ax = axis1)

axis1.set_title('With 0s\n', fontsize = 12)



axis2 = sns.heatmap(test.corr(), linewidth = .5, annot = True, vmin = -1, vmax = 1, center = 0, cmap = cmap, ax = axis2)

axis2.set_title('Without 0\n', fontsize = 12)



plt.show()
#Like we can see, almost all values decreased. It can be beacuse of relatilvly small dataset(5295 vs 45107) or 

#it can be more clear interpretation of the real situation
#For more coplex analyse we need to look at some scatter plots
sns.pairplot(df[num_columns], palette = 'deepskyblue', corner = True)

plt.show()
sns.pairplot(test, palette = 'deepskyblue', diag_kind = 'kde', corner = True)

plt.show()
#Categorical anlysis
#Lest see if there is some relation between movie genres
cat = df.iloc[:,9:]
plt.figure(figsize = (16,8))

sns.heatmap(cat.corr(), cmap = cmap, linewidth = 0.5, annot = True, vmin = -1, vmax = 1, center = 0)

plt.show()
#As expected, there is no significant relation between genres
#Revenue of each genre barplot / Most profitable genre
total_rev = []

genres = df.columns[9:]



for genre in genres:

    total = df.loc[df[genre] == 1]['revenue'].sum()

    total_rev.append(int(total))
rev_barplot_df = pd.DataFrame(index = genres, data = total_rev, columns = ['total_rev'])

rev_barplot_df = rev_barplot_df.sort_values(by = 'total_rev', ascending = False)

rev_barplot_df.head()
plt.figure(figsize = (7,4))



rev_barplot = sns.barplot(data = rev_barplot_df, x = rev_barplot_df.index, y = 'total_rev', 

                          palette = sns.color_palette("ch:2.5,-.2,dark=.3"))

rev_barplot.set_title('Total revenue of each genre\n', fontsize = 12)



plt.xticks(rotation = 70)

plt.show()
#Popularity of each genre
total_pop = []



for genre in genres:

    pop = df[genre].value_counts().loc[1]

    total_pop.append(pop)
pop_barplot_df = pd.DataFrame(data = total_pop, index = genres, columns = ['total_pop'])

pop_barplot_df = pop_barplot_df.sort_values(by = 'total_pop', ascending  = False)

pop_barplot_df.head()
plt.figure(figsize = (7,4))



pop_barplot = sns.barplot(data = pop_barplot_df, x = pop_barplot_df.index, y = 'total_pop', 

                          palette = sns.color_palette("ch:2.5,-.2,dark=.3"))

pop_barplot.set_title('Popularity of each genre\n', fontsize = 14)



plt.xticks(rotation = 70)

plt.show()
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16,4))

fig.suptitle('Comparison', fontsize = 18)



ax1 = sns.barplot(data = rev_barplot_df, x = rev_barplot_df.index, y = 'total_rev', 

                                                                   palette = sns.color_palette("ch:2.5,-.2,dark=.3"), ax = ax1)

ax1.set_xticklabels(labels = rev_barplot_df.index, rotation = 70)

ax1.set_title('Total revenue of each genre', fontsize = 14)





ax2 = sns.barplot(data = pop_barplot_df, x = pop_barplot_df.index, y = 'total_pop', 

                                                                    palette = sns.color_palette("ch:2.5,-.2,dark=.3"), ax = ax2)

ax2.set_xticklabels(labels = pop_barplot_df.index, rotation = 70)

ax2.set_title('Popularity of each genre', fontsize = 14)





plt.show()
fig, (ax1, ax2) = plt.subplots(2,1, figsize = (9, 8))

plt.subplots_adjust(hspace = .6)





ax1 = sns.barplot(data = rev_barplot_df, x = rev_barplot_df.index, y = 'total_rev', 

                                                                   palette = sns.color_palette("ch:2.5,-.2,dark=.3"), ax = ax1)

ax1.set_xticklabels(labels = rev_barplot_df.index, rotation = 70)

ax1.set_title('Total revenue of each genre', fontsize = 14)





ax2 = sns.barplot(data = pop_barplot_df, x = pop_barplot_df.index, y = 'total_pop', 

                                                                    palette = sns.color_palette("ch:2.5,-.2,dark=.3"), ax = ax2)

ax2.set_xticklabels(labels = pop_barplot_df.index, rotation = 70)

ax2.set_title('Popularity of each genre', fontsize = 14)





plt.show()
scatter_df = rev_barplot_df.join(pop_barplot_df)
scatter_df.head()
plt.figure(figsize = (7,5))

sns.scatterplot(data = scatter_df, x = 'total_rev', y = 'total_pop', marker = 'o', s = 50, color = 'deepskyblue')

plt.show()
sns.jointplot(data = scatter_df, x = 'total_rev', y = 'total_pop', kind = 'kde')

plt.show()
df.head()
#Company's mean revenue of all time



comp_mean_rev = pd.DataFrame(df.groupby(['production_companies'])['revenue'].mean()).sort_values(by = 'revenue', 

                                                                                                 ascending = False)

comp_mean_rev['revenue'] = comp_mean_rev['revenue'].round(2)

comp_mean_rev
comp_mean_rev.rename(index = {'Twentieth Century Fox Film Corporation': '20th Century Fox',

                             'Metro-Goldwyn-Mayer (MGM)': 'Metro Goldwyn Mayer'}, inplace = True)
plt.figure(figsize = (8,5))





comp_mean_rev_barplot = sns.barplot(data = comp_mean_rev, x = ['Universal Pictures', '20th Century Fox', 'Paramount Pictures',

       'Warner Bros.', 'Metro Goldwyn Mayer', 'other'], y = 'revenue', palette = sns.color_palette('Blues_d'))

comp_mean_rev_barplot.set_title('Companies mean revenue', fontsize = 14)

comp_mean_rev_barplot.set_ylim((0,70))

sns.set_style('whitegrid')

sns.despine()



inter = iter(range(6))



for index, row in comp_mean_rev.iterrows():

    comp_mean_rev_barplot.text(y = row.revenue, x = next(inter), s = str(row.revenue) + ' mln'+'\n', 

                                                                                    color='black', ha="center", fontsize = 12)





plt.xticks(rotation = 20)

plt.show()
#Companiy's total revenue per year

#Just for convenience I'll plot info only for 2000s



comp_rev = df[['production_companies', 'release_year', 'revenue']].loc[df['release_year'] > 2000]
comp_rev_annual = comp_rev.loc[comp_rev['production_companies'] == 'Warner Bros.'].groupby(['release_year']).sum()

comp_rev_annual.reset_index(inplace = True)
production_companies = list(df['production_companies'].unique())
fig, axis = plt.subplots(3,2, figsize = (14, 12))

plt.subplots_adjust(hspace = 0.4, wspace = 0.15)

plt.suptitle('Company\'s total revenue per year', fontsize = 18)



axis = axis.ravel()

sns.set_style('whitegrid')

sns.despine()





for i in range(6):

    

    comp_rev_annual = comp_rev.loc[comp_rev['production_companies'] == production_companies[i]].groupby(['release_year']).sum()

    comp_rev_annual.reset_index(inplace = True)

    

    lineplot = sns.lineplot(x = 'release_year', y = 'revenue', data = comp_rev_annual, ax = axis[i], color = 'deepskyblue', 

                                                                                                                linewidth = 1.5)

    lineplot.set_title(production_companies[i], fontsize = 14)

    lineplot.set_xlim((2000, 2020))

    lineplot.set_xticks(list(range(2000, 2021, 2)))

    lineplot.set_ylim((-100, comp_rev_annual['revenue'].max() * 1.2))
#Companiy's mean revenue per year





fig, axis = plt.subplots(3,2, figsize = (14, 12))

plt.subplots_adjust(hspace = 0.4, wspace = 0.15)

plt.suptitle("Company's mean revenue per year", fontsize = 18)



axis = axis.ravel()

sns.set_style('whitegrid')

sns.despine()







for i in range(6):

    

    comp_rev_annual = comp_rev.loc[comp_rev['production_companies'] == production_companies[i]].groupby(['release_year']).mean()

    comp_rev_annual.reset_index(inplace = True)

    

    lineplot = sns.lineplot(x = 'release_year', y = 'revenue', data = comp_rev_annual, ax = axis[i], color = 'deepskyblue',

                                                                                                               linewidth = 1.5)

    lineplot.set_title(production_companies[i], fontsize = 14)

    lineplot.set_xlim((2000, 2020))

    lineplot.set_xticks(list(range(2000, 2021, 2)))

    lineplot.set_ylim((-5, comp_rev_annual['revenue'].max() * 1.2))