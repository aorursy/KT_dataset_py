import numpy as np



import pandas as pd

from pandas.plotting import scatter_matrix

import geopandas as gpd



import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'

import seaborn as sns



from IPython.display import HTML



import warnings

warnings.filterwarnings('ignore')
!pip install gif

import gif
pd.options.display.float_format = '{:,.2f}'.format

pd.set_option('precision', 2)

font_size = 17
df_m = pd.read_csv('../input/housing-in-london/housing_in_london_monthly_variables.csv', parse_dates = ['date'])



print ('This dataset contains {} rows and {} columns.'.format(df_m.shape[0], df_m.shape[1]))

df_m.head()
df_m.info()
null_df_m = df_m.isnull().sum().sort_values(ascending = False)

percent = (df_m.isnull().sum()/df_m.isnull().count()).sort_values(ascending = False)*100



null_df_m = pd.concat([null_df_m, percent], axis = 1, keys = ['Counts', '% Missing'])

print ('Missing: ')

null_df_m.head()
df_m.drop('no_of_crimes', axis = 1, inplace = True)   # drop the 'no_of_crimes column



df_m['houses_sold'].fillna(df_m.groupby('area')['houses_sold'].transform('mean'), inplace = True) # fill NaN values with the mean of that particular area
df_m['year'] = df_m['date'].dt.year

df_m.iloc[[0, -1]]
df_m = df_m[df_m['year'] < 2020]

df_m['year'].max()
lnd_boroughs = df_m[df_m['borough_flag'] == 1]['area'].unique()

len(lnd_boroughs)
df_m[df_m['borough_flag'] == 0]['area'].nunique()
df_m[df_m['borough_flag'] == 0]['area'].unique()
eng_regions = ['south west', 'south east', 'east of england', 'west midlands', 'east midlands', 'yorks and the humber', 'north west', 'north east']
lnd = df_m[df_m['area'].isin(lnd_boroughs)]

eng = df_m[df_m['area'].isin(eng_regions)]
lnd_pr = lnd.groupby('date')['average_price'].mean()

eng_pr = eng.groupby('date')['average_price'].mean()
plt.figure(figsize = (9, 5))



lnd_pr.plot(y = 'average_price', color = 'royalblue', lw = 2, label = 'London')

eng_pr.plot(y = 'average_price', color = 'firebrick', lw = 2, label = 'England')



plt.axvspan('2007-12-21', '2009-06-21', alpha = 0.5, color = '#E57715')

plt.text(x = '2008-04-01', y = 390000, s = 'Recession', rotation = 90, fontsize = font_size-2)

plt.axvline(x = '2016-06-23', lw = 2, color = '#E57715', linestyle = '--')

plt.text(x = '2015-08-01', y = 210000, s = 'Brexit Referendum', rotation = 90, fontsize = font_size-2)



plt.title('Time evolution of the average house price', size = font_size)

plt.ylabel('Average Price', size = font_size)

plt.xticks(size = font_size - 3)

plt.xlabel('Date', size = font_size)

plt.yticks(size = font_size - 3)

plt.legend(fontsize = font_size - 3);
@gif.frame

def plot(df_lnd, df_eng, date):

    

    ### select a sub-dataframe from the start until date ###

    d_ln = df_lnd.loc[df_lnd.index[0]:date]

    d_eng = df_eng.loc[df_eng.index[0]:date]

    

    fig = plt.figure(figsize = (9, 5))

    plt.xlim(pd.Timestamp('1994-12-01'), pd.Timestamp('2020-01-01'))

    plt.ylim(47000, 550000)

    

    ### for the vertical orange rectangle and the vertical dashed line ###

    if (date > pd.Timestamp('2007-12-22') and date < pd.Timestamp('2009-06-21')):

        plt.axvspan(pd.Timestamp('2007-12-21'), date, alpha = 0.5, color = '#E57715') 

    elif (date > pd.Timestamp('2009-06-21')):

        plt.axvspan(pd.Timestamp('2007-12-21'), pd.Timestamp('2009-06-21'), alpha = 0.5, color = '#E57715')

        plt.text(x = pd.Timestamp('2008-04-29'), y = 390000, s = 'Recession', rotation = 90, fontsize = font_size-2)

    if (date > pd.Timestamp('2016-06-23')):

        plt.axvline(x = pd.Timestamp('2016-06-23'), lw = 2, color = '#E57715', linestyle = '--')

        plt.text(x = pd.Timestamp('2015-08-01'), y = 210000, s = 'Brexit Referendum', rotation = 90, fontsize = font_size-2)

    ############################################################################################################

    

    plt.plot(d_ln, color = 'royalblue', lw = 2, label = 'London')

    plt.plot(d_eng, color = 'firebrick', lw = 2, label = 'England')

    

    plt.title('Time evolution of the average house price', size = font_size)

    plt.ylabel('Average Price', size = font_size)

    plt.xticks(size = font_size - 3)

    plt.xlabel('Date', size = font_size)

    plt.yticks(size = font_size - 3)

    plt.legend(loc = 2, fontsize = font_size - 3);
frames = []

for months in pd.date_range(start = lnd_pr.index[0], end = lnd_pr.index[-1], freq = '3MS'): # 3MS --> every three months

    frame = plot(lnd_pr, eng_pr, months)

    frames.append(frame)

    

gif.save(frames, 'Price-Lnd_vs_Eng.gif', duration = 1, unit = 's', between = 'startend')
HTML('<img src="./Price-Lnd_vs_Eng.gif" />')
lnd_b_prices = lnd.groupby('area')['average_price'].mean()

lnd_top10_pr = lnd_b_prices.sort_values(ascending = False).to_frame()



print ('\nThe 10 most expensive boroughs in London are:')

lnd_top10_pr.head(10)
lnd_top10_pr.head(10).sort_values(by = 'average_price', ascending = True).plot(kind = 'barh', figsize = (9, 5), 

                                                                               color = 'steelblue', edgecolor = 'firebrick',

                                                                               legend = False)



plt.title('Average price in the most expensive London boroughs (1995-2019)', size = font_size, y = 1.05)

plt.ylabel('London Borough', size = font_size)

plt.yticks(size = font_size - 3)

plt.xlabel('Average Price', size = font_size)

plt.xticks([0, 200_000, 400_000, 600_000], size = font_size - 3);
top5_indeces = lnd_top10_pr.head().index

colors = ['#e74c3c', '#3498db', '#95a5a6', '#34495e', '#2ecc71']



plt.figure(figsize = (9, 5))



for index, i in enumerate(top5_indeces):

    df_ = lnd[lnd['area'] == i]

    df_ = df_.groupby('date')['average_price'].mean()

    

    df_.plot(y = 'average_price', label = i, color = colors[index])

       

plt.title('Average price in the most expensive boroughs', y = 1.04, size = font_size)

plt.xlabel('Date', size = font_size)

plt.xticks(size = font_size - 3)

plt.ylabel('Average Price', size = font_size)

plt.yticks([0.2*1E+6, 0.6*1E+6, 1.0*1E+6, 1.4*1E+6], size = font_size - 3)

plt.legend(fontsize = font_size - 5);
eng_prices = eng.groupby('area')['average_price'].mean()

eng_top3_pr = eng_prices.sort_values(ascending = False).to_frame()



print('The top 3 most expensive regions in England are:')

eng_top3_pr.head(3)
top3_indeces = eng_top3_pr.head(3).index

colors = ['darkorange', '#8EB8E5', 'forestgreen', ]



plt.figure(figsize = (9, 5))



for index, i in enumerate(top3_indeces):

    df_ = eng[eng['area'] == i]

    df_ = df_.groupby('date')['average_price'].mean()

    df_.plot(y = 'average_price', label = i, color = colors[index])



plt.title('Average price in the most expensive English regions by date', size = font_size, y = 1.04)

plt.xlabel('Date', size = font_size)

plt.xticks(size = font_size - 3)

plt.ylabel('Average Price', size = font_size)

plt.yticks([100_000, 200_000, 300_000], size = font_size - 3)

plt.legend(fontsize = font_size - 3);
plt.figure(figsize = (9, 5))



for index, i in enumerate(top3_indeces):

    df_ = eng[eng['area'] == i]

    df_ = df_.groupby('date')['average_price'].mean()

    df_.plot(y = 'average_price', label = i, color = colors[index])



lnd_bng_pr = lnd[lnd['area'] == 'barking and dagenham'].groupby('date')['average_price'].mean()

lnd_bng_pr.plot(y = 'average_price', lw = 2, linestyle = '--', color = '#A30015', label = 'barking and dagenham')



plt.title('3 expensive English regions VS cheapest London borough', size = font_size, y = 1.06)

plt.xlabel('Date', size = font_size)

plt.xticks(size = font_size - 3)

plt.ylabel('Average Price', size = font_size)

plt.yticks([0.1*1E+6, 0.2*1E+6, 0.3*1E+6], size = font_size - 3)

plt.legend(labels = ['South East (Eng)', 'East of England (Eng)', 'South West (Eng)', 'Barking and Dagenham (L)'], 

           fontsize = font_size - 3);
lnd_houses = lnd.groupby('date')['houses_sold'].sum()

lnd_houses.plot(figsize = (9, 5), lw = 2, y = 'houses_sold', color = '#00072D')



plt.axvspan('2007-12-21', '2009-06-21', alpha = 0.5, color = '#F08700')

plt.text(x = '2008-04-01', y = 10700, s = 'Recession', rotation = 90, fontsize = font_size-2)

plt.axvspan('2016-01-1', '2016-05-01', alpha = 0.7, color = '#FFCAAF')



# plt.axvline(x = '2016-06-23', lw = 2, color = '#E57715', linestyle = '--')

plt.text(x = '2016-06-01', y = 10000, s = 'New tax legislation', rotation = 90, fontsize = font_size-2)



plt.title('Houses sold in London by date', size = font_size)

plt.xlabel('Date', size = font_size)

plt.xticks(size = font_size - 3)

plt.ylabel('Houses sold', size = font_size)

plt.yticks([4000, 8000, 12000, 16000], size = font_size - 3);
lnd_b_houses = lnd.groupby('area')['houses_sold'].sum()

lnd_top5_h = lnd_b_houses.sort_values(ascending = False).to_frame()

lnd_top5_h.head(5)
london_map = gpd.read_file('../input/london-borough-and-ward-boundaries-up-to-2014/London_Wards/Boroughs/London_Borough_Excluding_MHW.shp')

london_map.columns = london_map.columns.str.lower()

london_map.head()
london_map['name'] = london_map['name'].str.lower()

london_map.rename(columns = {'name': 'area'}, inplace = True)

london_map.rename(columns = {'gss_code': 'code'}, inplace = True)



london_map = london_map[['area', 'code', 'hectares', 'geometry']]

london_map.head()
lnd_m = lnd.groupby('area').agg({'average_price': ['mean'], 'houses_sold': 'sum'})



lnd_m.columns = ['average_price', 'houses_sold']

lnd_m.reset_index(inplace = True)

lnd_m.head()
np.intersect1d(lnd_m['area'], london_map['area']).size
lnd_m_map = pd.merge(london_map, lnd_m, how = 'inner', on = ['area'])

lnd_m_map.head()
type(lnd_m_map)
fig, ax = plt.subplots(1, 2, figsize = (12, 12))



lnd_m_map.plot(ax = ax[0], column = 'average_price', cmap = 'Reds', edgecolor = 'maroon', legend = True, legend_kwds = {'label': 'Average Price', 'orientation' : 'horizontal'})



lnd_m_map.plot(ax = ax[1], column = 'houses_sold', cmap = 'Blues', edgecolor = 'maroon', legend = True, legend_kwds = {'label': 'Houses Sold', 'orientation' : 'horizontal'})



ax[0].axis('off')

ax[0].set_title('Average House Price (All years)', size = font_size)

ax[1].axis('off')

ax[1].set_title('Houses Sold (All years)', size = font_size);
df_y = pd.read_csv('../input/housing-in-london/housing_in_london_yearly_variables.csv', parse_dates = ['date'])

df_y = df_y[df_y['area'].isin(lnd_boroughs)] # select only London boroughs



print ('This dataset contains {} rows and {} columns.'.format(df_y.shape[0], df_y.shape[1]))

df_y.head()
null_df_y = df_y.isnull().sum().sort_values(ascending = False)

percent = (df_y.isnull().sum()/df_y.isnull().count()).sort_values(ascending = False)*100



null_df = pd.concat([null_df_y, percent], axis = 1, keys = ['Counts', '%'])

null_df.head(10)
# import missingno as msno

# msno.matrix(df_y)
df_y[~df_y['mean_salary'].str.isnumeric()]['mean_salary'].value_counts()
df_y['mean_salary'] = df_y['mean_salary'].replace(['#'], np.NaN)

df_y['mean_salary'] = df_y['mean_salary'].astype(float)
df_y['year'] = df_y['date'].dt.year



print ('yearly_variables dataset')

print ('\tFirst date: ', df_y['year'].min())

print ('\tFinal date: ', df_y['year'].max())
lnd_m_group = lnd.groupby(['area', 'year']).mean().reset_index()  # group based on area and year (take mean)

lnd_m_group = lnd_m_group[lnd_m_group['year'] >= 1999]            # select all years after 1999 (included)



print ('monthly_variables dataset')

print ('\tFirst date: ', lnd_m_group['year'].min())

print ('\tFinal date: ', lnd_m_group['year'].max())
lnd_y_group = df_y.groupby(['area', 'year']).mean().reset_index() # group it based on area and year

lnd_y_group.head()
lnd_total = pd.merge(lnd_y_group, lnd_m_group, on = ['area', 'year'], how = 'left')

lnd_total.drop(['borough_flag_x', 'borough_flag_y'], axis = 1, inplace = True)



lnd_total.head()
corr_table = lnd_total.corr()

corr_table['average_price'].sort_values(ascending = False)
plt.figure(figsize = (10, 8))



mask = np.triu(np.ones_like(corr_table, dtype = np.bool))



ax = sns.heatmap(corr_table, mask = mask, annot = True, cmap = 'YlGnBu_r')

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5);
columns = ['average_price', 'median_salary', 'mean_salary', 'number_of_jobs']



scatter_matrix(lnd_total[columns], figsize = (12, 12), color = '#D52B06', alpha = 0.3, 

               hist_kwds = {'color':['bisque'], 'edgecolor': 'firebrick'});