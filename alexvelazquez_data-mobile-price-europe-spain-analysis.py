import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
prices = pd.read_excel("../input/worldwide-mobile-data-pricing/global-mobile-data-price-comparison-2020.xlsx", index_col=0)

prices.head()
usd_to_eur = 0.85 # 25 ago. 8:45 UTC

prices = prices[["Name", "Continental region", "Average price of 1GB (USD)", "Cheapest 1GB for 30 days (USD)", "Most expensive 1GB (USD)"]]

prices.rename(columns={"Average price of 1GB (USD)": "Average price", "Cheapest 1GB for 30 days (USD)": "Cheapest", "Most expensive 1GB (USD)": "Most expensive"}, inplace=True)

prices["Average price"] *=  usd_to_eur 

prices["Cheapest"] *= usd_to_eur

prices["Most expensive"] *= usd_to_eur

prices.head()
prices.groupby('Continental region')['Continental region'].count()
prices.dtypes
prices2019 = pd.read_excel("../input/worldwide-mobile-data-pricing/global-mobile-data-price-comparison-2020.xlsx", 

                           sheet_name="Previous year's data (2019)", 

                           index_col=0)

usd_to_eur = 0.85 # 25 ago. 8:45 UTC

prices2019 = prices2019[["Name", "Continental region", "Average price of 1GB (USD)", "Cheapest 1GB (USD)", "Most expensive 1GB (USD)"]]

prices2019.rename(columns={"Average price of 1GB (USD)": "Average price 2019", "Cheapest 1GB (USD)": "Cheapest", "Most expensive 1GB (USD)": "Most expensive"}, inplace=True)

prices2019["Average price 2019"] *=  usd_to_eur 

prices2019["Cheapest"] *= usd_to_eur

prices2019["Most expensive"] *= usd_to_eur

prices2019.head()
prices[prices['Average price'] == min(prices['Average price'])] 
prices[prices['Average price'] == max(prices['Average price'])] 
prices.head(10)
#Calculate top10 cheapest

top10cheap = prices.head(10)

top10cheap.set_index('Name', inplace=True)

top10cheap = top10cheap[['Average price']]

top10cheap.sort_values(['Average price'], ascending=False, axis=0, inplace=True)



#Print a barh

top10cheap.plot(kind='barh', alpha=0.7, figsize=(16, 12)) 

plt.xlabel('Average price')

plt.ylabel('Country')

plt.title('Top 10 Cheapest countries - 1GB data mobile')

for index,value in enumerate(top10cheap['Average price']): 

    label = str(round(value,2)) 

    plt.annotate(label+'€', xy=(value-0.025, index-0.05), color='white')

plt.show()
prices.head(10).groupby('Continental region')['Continental region'].count().sort_values(ascending=False)
prices.sort_values("Average price", ascending=False).head(10)
#Calculate top10 most expensive

top10exp = prices.tail(10)

top10exp.set_index('Name', inplace=True)

top10exp = top10exp[['Average price']]

top10exp.sort_values(['Average price'], ascending=True, axis=0, inplace=True)



#Print a barh

top10exp.plot(kind='barh', alpha=0.7, figsize=(16, 12)) 

plt.xlabel('Average price')

plt.ylabel('Country')

plt.title('Top 10 Most expensive countries - 1GB data mobile')

for index,value in enumerate(top10exp['Average price']): 

    label = str(round(value,2)) 

    plt.annotate(label+'€', xy=(value-2.2, index-0.05), color='white')

plt.show()
prices.tail(10).groupby('Continental region')['Continental region'].count().sort_values(ascending=False)
# Define x-axis, in 0.5€ steps

segments = int(prices['Average price'].max() // 0.5) + 1

bin_edges = []

for i in range(segments+1):

    bin_edges.append(0.5*i)



# Plots the histogram

prices['Average price'].plot(kind='hist', 

                             figsize=(18, 12), 

                             bins=len(bin_edges), 

                             alpha=0.6, 

                             xticks=bin_edges,

                             rot=90)

plt.title('Histogram: Average price of 1GB data connection - Worldwide') 

plt.ylabel('Number of Countries') 

plt.xlabel('Average Price (EUR)') 



# Annotate arrow for worldwide average

plt.annotate('Average',                      

             xy=(prices['Average price'].mean(), 12.2),             

             xytext=(3.35, 20),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for 50% countries

w50pc = prices['Average price'].iloc[prices['Name'].count() // 2]

plt.annotate('50% countries',                      

             xy=(w50pc, 17.2),             

             xytext=(1.1, 25),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



plt.show()
# Create a new dataframe for the pie chart representation

price_steps, steps_count = [1,2,3,5,10], []

steps_names = ['Less than 1€', 'Between 1 and 2€', 'Between 2 and 3€', 'Between 3 and 5€', 'Between 5 and 10€', 'More than 10€']

for i,val in enumerate(price_steps):

    count = (prices['Name'][prices['Average price'] < val]).count()

    count -= sum(steps_count)

    steps_count.append(count)

    if i == (len(price_steps)-1):

        steps_count.append((prices['Name'].count())-sum(steps_count))

pie_data = pd.DataFrame(list(zip(steps_names, steps_count)), columns=['Group',' '])

pie_data.set_index('Group', inplace=True)



# Plot data

colors_list = ['palegreen', 'yellowgreen', 'gold', 'orange', 'pink', 'lightcoral']

explode = [0, 0.1, 0, 0, 0, 0]

pie_data[' '].plot(kind='pie',

                            figsize=(15, 6),

                            autopct='%1.1f%%', 

                            startangle=90,

                            counterclock=False,

                            shadow=True,   

                            explode=explode,

                            colors=colors_list

                            )



plt.title('Average price 1GB Data mobile - Worldwide', y=1.12) 

plt.axis('equal')  

plt.show()
prices['Average price'].plot(kind='box', figsize=(10, 10))

plt.title('1GB Data mobile prices - Worldwide')

plt.ylabel('Price (EUR)')

plt.show()
prices[['Average price']].describe()
prices[prices['Name'] == 'Spain']
europe = prices[(prices['Continental region'] == 'WESTERN EUROPE') | 

               (prices['Continental region'] == 'BALTICS') | 

               (prices['Continental region'] == 'EASTERN EUROPE')]

europe
# Create the new rank

europe = europe.assign(Eur_rank=pd.Series(europe['Average price'].rank(ascending=True)))

# Calculate differential

spa_price = europe['Average price'][europe['Name'] == 'Spain'].iloc[0]

europe = europe.assign(Differential=pd.Series(europe['Average price'] - spa_price))

europe = europe.assign(Percent=pd.Series(((europe['Average price']*100)/spa_price)-100))

# Rename and correct types

europe.rename(columns={'Rank': 'World rank', 'Eur_rank': 'Eur rank', 'Diffential': 'Differential SPA', 'Percent': '%Diff SPA'}, inplace=True)

europe = europe.astype({'Eur rank': int, '%Diff SPA':int})

europe.set_index('Eur rank', inplace=True)

europe = europe[['Name', 'Continental region', 'Average price', 'Cheapest', 'Most expensive', 'Differential', '%Diff SPA']]

europe
# Define x-axis, in 0.5 steps

segments = int(europe['Average price'].max() // 0.5) + 1

bin_edges = []

for i in range(segments+1):

    bin_edges.append(0.5 * i)



# Plots the histogram

europe['Average price'].plot(kind='hist', 

                             figsize=(12, 7), 

                             bins=len(bin_edges), 

                             alpha=0.6, 

                             xticks=bin_edges)

plt.title('Histogram: Average price of 1GB data mobile - Europe') 

plt.ylabel('Number of Countries') 

plt.xlabel('Average Price (EUR)') 



# Annotate arrow for Spain position

plt.annotate('Spain',                      

             xy=(spa_price, 6.2),             

             xytext=(1.31, 8),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for EUR average

plt.annotate('EUR average',                      

             xy=(europe['Average price'].mean(), 1.25),             

             xytext=(2.15, 10),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for 50% countries

eur_50pc = europe['Average price'].iloc[europe['Name'].count() // 2]

plt.annotate('50% countries',                      

             xy=(eur_50pc, 5.2),             

             xytext=(1.3, 9),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



plt.show()
# Create a new dataframe for the pie chart representation

price_steps, steps_count_w, steps_count_eur = [1,2,3,5,10], [], []

steps_names = ['Less than 1€', 'Between 1 and 2€', 'Between 2 and 3€', 'Between 3 and 5€', 'Between 5 and 10€', 'More than 10€']

for i,val in enumerate(price_steps):

    count_w = (prices['Name'][prices['Average price'] < val]).count()

    count_w -= sum(steps_count_w)

    count_eur = (europe['Name'][europe['Average price'] < val]).count()

    count_eur -= sum(steps_count_eur)

    steps_count_w.append(count_w)

    steps_count_eur.append(count_eur)

    if i == (len(price_steps)-1):

        steps_count_w.append((prices['Name'].count())-sum(steps_count_w))

        steps_count_eur.append((europe['Name'].count())-sum(steps_count_eur))

pie_data = pd.DataFrame(list(zip(steps_names, steps_count_w, steps_count_eur)), columns=['Group','',' '])

pie_data.set_index('Group', inplace=True)



# Plot data

colors_list = ['palegreen', 'yellowgreen', 'gold', 'orange', 'pink', 'lightcoral']

explode = [0, 0.1, 0, 0, 0, 0]

pie_data[['',' ']].plot(kind='pie',

                                      figsize=(14, 8),

                                      labels=steps_names,

                                      autopct='%1.1f%%', 

                                      startangle=90,

                                      counterclock=False,

                                      shadow=True,   

                                      explode=explode,

                                      colors=colors_list,

                                      pctdistance=0.8,

                                      subplots=True,

                                      legend=False

                                      )



plt.title('Average price 1GB Data mobile - Worldwide vs. Europe', x=-0.6, y=1, fontdict={'fontsize':22}) 

plt.annotate('Worldwide', xy=(-1,0), xytext=(-4.4,-1.1), fontsize=20)

plt.annotate('Europe', xy=(-1,0), xytext=(1,-1.1), fontsize=20)

plt.show()
europe['Average price'].plot(kind='box', figsize=(14, 14))

plt.title('1GB Data mobile prices - Europe')

plt.ylabel('Price (EUR)')

plt.show()
europe[['Average price']].describe()
prices[prices['Name'] == 'United States']
# Define x-axis, in 0.5€ steps

segments = int(prices['Average price'].max() // 0.5) + 1

bin_edges = []

for i in range(segments+1):

    bin_edges.append(0.5*i)



# Plots the histogram

prices['Average price'].plot(kind='hist', 

                             figsize=(18, 12), 

                             bins=len(bin_edges), 

                             alpha=0.6, 

                             xticks=bin_edges,

                             rot=90)

plt.title('Histogram: Average price of 1GB data mobile - Worldwide') 

plt.ylabel('Number of Countries') 

plt.xlabel('Average Price (EUR)') 



# Annotate arrow for Spain position

plt.annotate('Spain',                      

             xy=(spa_price, 24),             

             xytext=(0.85, 29),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for worldwide average

plt.annotate('Average',                      

             xy=(prices['Average price'].mean(), 12.2),             

             xytext=(3.35, 20),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for 50% countries

w50pc = prices['Average price'].iloc[prices['Name'].count() // 2]

plt.annotate('50% countries',                      

             xy=(w50pc, 17.2),             

             xytext=(1.05, 22.5),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for USA

usa_price = prices['Average price'][prices['Name'] == 'United States'].iloc[0]

w50pc = prices['Average price'].iloc[prices['Name'].count() // 2]

plt.annotate('USA',                      

             xy=(usa_price, 4),             

             xytext=(6.3, 10),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



plt.show()
compare = pd.merge(left=prices2019[['Name','Average price 2019']], 

                   right=prices[['Name','Average price']], 

                   left_on='Name', 

                   right_on='Name')

compare = compare.assign(Differential=pd.Series(compare['Average price'] - compare['Average price 2019']))

compare.rename(columns={'Average price': 'Average price 2020'}, inplace=True)

compare.head()
countries_num = compare['Name'].count()

drop = compare['Name'][compare['Differential'] < 0].count()

rise = compare['Name'][compare['Differential'] > 0].count()

equal = compare['Name'][compare['Differential'] == 0].iloc[0]

mean2019 = compare['Average price 2019'].mean()

mean2020 = compare['Average price 2020'].mean()

max_drop = compare['Differential'].min()

max_drop_country = compare['Name'][compare['Differential'] == max_drop].iloc[0]

max_rise = compare['Differential'].max()

max_rise_country = compare['Name'][compare['Differential'] == max_rise].iloc[0]

spa_price2019 = compare['Average price 2019'][compare['Name'] == 'Spain'].iloc[0]

print('Prices decreased in {} countries and raised in {}'.format(drop, rise))

print('{} is the only country with no price changes'.format(equal))

print('The average price in 2019 was {:.2f}€, meanwhile in 2020 it is {:.2f}€. Prices drop by {:.2f}€'.format(mean2019, mean2020, mean2019-mean2020))

print('The country with the biggest drop in prices has been {}, where prices drop by {:.2f}€'.format(max_drop_country, max_drop))

print('The country with the highest price increase has been {}, where prices rising by {:.2f}€'.format(max_rise_country, max_rise))

print('The price in Spain dropped from {:.2f}€ to {:.2f}€, a {:.1f}%'.format(spa_price2019, spa_price, ((spa_price-spa_price2019)/spa_price2019)*100))
compare[['Average price 2019']].describe()
compare[compare['Differential'] > 0].sort_values('Differential', ascending=False)
# Define x-axis, in 0.5€ steps

segments = int(compare['Average price 2019'].max() // 0.5) + 1

bin_edges = []

for i in range(segments+1):

    bin_edges.append(0.5*i)



# Plots the histogram

compare[['Average price 2019', 'Average price 2020']].plot(kind='hist', 

                             figsize=(18, 12), 

                             bins=len(bin_edges), 

                             alpha=0.5, 

                             xticks=bin_edges,

                             rot=90)

plt.title('Histogram: Average price of 1GB data mobile - Worldwide 2019 vs. 2020') 

plt.ylabel('Number of Countries') 

plt.xlabel('Average Price (EUR)') 



# Annotate arrow for worldwide average

plt.annotate('Average 2020',                      

             xy=(compare['Average price 2020'].mean(), 16),             

             xytext=(3, 30),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for 50% countries

plt.annotate('50% countries 2020',                      

             xy=(w50pc, 16),             

             xytext=(-0.35, 35),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for worldwide average

plt.annotate('Average 2019',                      

             xy=(compare['Average price 2019'].mean(), 5.2),             

             xytext=(4.8, 13),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )



# Annotate arrow for 50% countries

w50pc2019 = compare['Average price 2019'].iloc[compare['Name'].count() // 2]

plt.annotate('50% countries 2019',                      

             xy=(w50pc2019, 16),             

             xytext=(6, 26),         

             xycoords='data',         

             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='red', alpha=0.4, lw=2)

            )

plt.show()
# Create a new dataframe for the pie chart representation

price_steps, steps_count2019, steps_count2020 = [1,2,3,5,10], [], []

steps_names = ['Less than 1€', 'Between 1 and 2€', 'Between 2 and 3€', 'Between 3 and 5€', 'Between 5 and 10€', 'More than 10€']

for i,val in enumerate(price_steps):

    count2019 = (compare['Name'][compare['Average price 2019'] < val]).count()

    count2019 -= sum(steps_count2019)

    count2020 = (compare['Name'][compare['Average price 2020'] < val]).count()

    count2020 -= sum(steps_count2020)

    steps_count2019.append(count2019)

    steps_count2020.append(count2020)

    if i == (len(price_steps)-1):

        steps_count2019.append((compare['Name'].count())-sum(steps_count2019))

        steps_count2020.append((compare['Name'].count())-sum(steps_count2020))

pie_data = pd.DataFrame(list(zip(steps_names, steps_count2019, steps_count2020)), columns=['Group','', ' '])

pie_data.set_index('Group', inplace=True)



# Plot data

colors_list = ['palegreen', 'yellowgreen', 'gold', 'orange', 'pink', 'lightcoral']

explode = [0.1, 0, 0, 0, 0, 0]

pie_data[['',' ']].plot(kind='pie',

                                figsize=(16, 8),

                                autopct='%1.1f%%', 

                                startangle=90,

                                counterclock=False,

                                shadow=True,   

                                explode=explode,

                                colors=colors_list, 

                                pctdistance=0.7,

                                subplots=True,

                                legend=False

                                )



plt.title('Average price 1GB Data - 2019 vs. 2020', x=-0.8, y=1, fontdict={'fontsize':22})  

plt.annotate('2019', xy=(-1,0), xytext=(-4.4,-1.1), fontsize=20)

plt.annotate('2020', xy=(-1,0), xytext=(1,-1.1), fontsize=20)

plt.show()