import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

%matplotlib inline
uncleaned_data = pd.read_csv('../input/commodity_trade_statistics_data.csv')

#cleaned data

global_trade_df = uncleaned_data.dropna(how='all')
#having a first look at the dataframe

global_trade_df.head(10)
#number of lines of the datafame

print ('This dataframe has ' + str(len(global_trade_df.index)) + ' lines !')
#new column for unitary value



global_trade_df['trade_usd'].astype('float64');

global_trade_df['quantity'].astype('float64')

global_trade_df['unit_value'] = global_trade_df['trade_usd']/global_trade_df['quantity']

global_trade_df['unit_value'].head()

# the array of unique categories

global_trade_df['category'].unique()
# the array of unique countries

global_trade_df['country_or_area'].unique()
#checking data type of the column 'year'

global_trade_df['year'].unique().dtype
#We are only interested in data from a small subset of columns

trade_by_country = global_trade_df[['country_or_area','year','flow', 'category' ,'trade_usd']]



#using groupby function and building a multiIndex to make analysis easier

trade_by_country = trade_by_country.groupby(['country_or_area','year','flow', 'category'])[['trade_usd']].sum()

trade_by_country.head()
brazil_df = global_trade_df[global_trade_df['country_or_area'] == 'Brazil']

brazil_years = brazil_df['year'].unique()

brazil_years.sort()



exports_br = trade_by_country['trade_usd'].loc['Brazil', : ,'Export', 'all_commodities']

imports_br = trade_by_country['trade_usd'].loc['Brazil', : ,'Import', 'all_commodities']





fig=plt.figure(figsize=(10, 8), dpi= 80, facecolor='w', edgecolor='k')

plt.rcParams.update({'font.size':15})



#plot Brazil's Trade Balance

p1 = plt.bar(brazil_years, exports_br)

p2 = plt.bar(brazil_years, imports_br)

plt.title("Brazil's Trade Balance - Exports vs Imports")

plt.ylabel('Trade worth - in 100 billion US dollars')

plt.xlabel('year')

plt.legend((p1[0], p2[0]), ('Exports', 'Imports'))
#function that returns the n most important commodity categories- in descending order-, along with 

#their percentage on total trade worth



def n_most_important_categories(n, country, year, flow):

    list_of_categories = trade_by_country.loc[country, year, flow]

    list_of_categories =  list_of_categories['trade_usd'].sort_values(ascending=False)

    

    cont = 0

    all_commodities_value = float(list_of_categories.loc['all_commodities'])

    for index in list_of_categories.index:

        

        if cont > n+1:

            break

        cont = cont + 1

        

        list_of_categories.loc[index] = float(list_of_categories.loc[index])/all_commodities_value

        

    if n != None:

        return list_of_categories[:n+1]

    

    else:

        return list_of_categories

   

# testing the function above

list_ = n_most_important_categories(20,'Brazil', 2002, 'Export')

#adds 'other' category  

list_ = list_.append(pd.Series([1 - list_[1:].sum()], index=['other']))

print (list_)

from matplotlib import animation, rc 

from IPython.display import HTML



rc('animation', html='html5')



colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'limegreen', 

          'red', 'navy', 'blue', 'magenta', 'crimson', 'orange']



fig, ax = plt.subplots();



def update(year, country, n, flow):

    ax.clear()     

    list_1 = n_most_important_categories(n, country, year, flow);

    list_1 = list_1.append(pd.Series([1 - list_1[1:].sum()], index=['other']));

    ax.pie(list_1[1:], labels=list_1.index[1:], colors=colors, labeldistance = 1.1, autopct='%.1f%%', shadow=True);

    ax.set_title(str(year));



fig.set_size_inches(10,10);



#animation showing the composition of Brazil's exports from 1989 to 2016

anim = animation.FuncAnimation(fig, update, frames=np.arange(1989,2017,1), fargs = ('Brazil', 10, 'Export'), interval=2000, repeat=True);

#Animation is not working

#HTML(anim.to_html5_video())

#animation showing the composition of Brazil's imports from 1989 to 2016

anim = animation.FuncAnimation(fig, update, frames=np.arange(1989,2017,1), fargs = ('Brazil', 10, 'Import'), interval=2000, repeat=True);

#HTML(anim.to_html5_video())