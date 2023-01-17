# Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy as sp



# Read the data

data = pd.read_csv("../input/data-yield-curve/Data Yield Curve.csv")



# Rename CHHUSD in CHFUSD

data.rename(columns = {'CHHUSD':'CHFUSD'}, inplace = True)



# Create variable year

data['Year'] = data['Date'].apply(lambda i: i.split('-')[0]).astype(int)



# Yield Curve Inverted

inverted_data = pd.read_csv("../input/data-yield-curve/Yield-curve-inverted.csv")

df = pd.melt(inverted_data, id_vars='Maturities', value_vars=['23.08.2019', '23.08.2018'], var_name='Dates')



# Visuals setup

sns.set(style = 'whitegrid')
plt.figure(figsize=(16,9))



ax1 = sns.lineplot(x = 'Maturities', y = 'value', hue = 'Dates', data = df, lw = 5, err_style=None,

                  legend = 'full', size_order=['1 Mo', '2 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr',

       '7 Yr', '10 Yr', '20 Yr', '30 Yr'], sort = False, palette = ["#DF2727", "#2750DF"],

                  markers = True, style = "Dates", markersize = 13)





plt.legend(['23.08.2019', '23.08.2018'])

plt.title('Yield Curve Maturities in 23.08 - 2018 and 2019', fontsize = 25)

plt.xlabel('Maturities', fontsize = 20)

plt.ylabel('Yield', fontsize = 20)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.legend(fontsize = 15);
plt.figure(figsize=(16,9))



ax1 = sns.lineplot(x = 'Year', y = '1 yr', data=data, err_style=None, lw = 5, estimator='mean',

                  palette = "Dark2")

ax2 = sns.lineplot(x = 'Year', y = '30YR', data=data, err_style=None, lw = 5, estimator='mean',

                  palette = "Dark2")



plt.title('Yield values for 1 and 30 years maturity - from 1977 to 2019', fontsize = 25)

plt.xlabel('Year', fontsize = 20)

plt.ylabel('Yield value(%)', fontsize = 20)

plt.legend(['1yr','30yrs'], ncol=2, loc='upper right', fontsize = 15)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)



ax1.set_xlim(1977,2019);
plt.figure(figsize=(16,9))



ax3 = sns.lineplot(x = 'Year', y = 'SPREAD', data = data, lw = 5, err_style=None, estimator='mean')

plt.plot([1977, 2019], [0, 0], color = '#839192', lw = 4)



plt.title('Spread of difference between 30Yrs Yield and 1Yr Yield (%)', fontsize = 25)

plt.xlabel('Year', fontsize = 20)

plt.ylabel('Spread', fontsize = 20)



ax3.set_xlim(1977, 2019)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15);



#the curve seems like will touch again
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex = True)

fig.set_figheight(13)

fig.set_figwidth(16)

plt.subplots_adjust(hspace = 0.1)



ax1.plot(data['Year'], data['GOLD'], lw = 4)

ax1.plot([1980, 1980], [90, 1800], '#FF4026', lw = 2)

ax1.plot([1989, 1989], [90, 1800], '#FF4026', lw = 2)

ax1.plot([2000, 2000], [90, 1800], '#FF4026', lw = 2)

ax1.plot([2006, 2006], [90, 1800], '#FF4026', lw = 2)

ax1.plot([2008, 2008], [90, 1800], '#F39C12', lw = 3)



ax2.plot(data['Year'], data['OIL'], lw = 4)

ax2.plot([1980, 1980], [0, 134], '#FF4026', lw = 2)

ax2.plot([1989, 1989], [0, 134], '#FF4026', lw = 2)

ax2.plot([2000, 2000], [0, 134], '#FF4026', lw = 2)

ax2.plot([2006, 2006], [0, 134], '#FF4026', lw = 2)

ax2.plot([2008, 2008], [0, 134], '#F39C12', lw = 3)



ax3.plot(data['Year'], data['CHFUSD'], lw = 4)

ax3.plot([1980, 1980], [0.78, 2.8], '#FF4026', lw = 2)

ax3.plot([1989, 1989], [0.78, 2.8], '#FF4026', lw = 2)

ax3.plot([2000, 2000], [0.78, 2.8], '#FF4026', lw = 2)

ax3.plot([2006, 2006], [0.78, 2.8], '#FF4026', lw = 2)

ax3.plot([2008, 2008], [0.78, 2.8], '#F39C12', lw = 3)



ax4.plot(data['Year'], data['JPYUSD'], lw = 4)

ax4.plot([1980, 1980], [76.64, 281], '#FF4026', lw = 2)

ax4.plot([1989, 1989], [76.64, 281], '#FF4026', lw = 2)

ax4.plot([2000, 2000], [76.64, 281], '#FF4026', lw = 2)

ax4.plot([2006, 2006], [76.64, 281], '#FF4026', lw = 2)

ax4.plot([2008, 2008], [76.64, 281], '#F39C12', lw = 3)



ax1.set_xlim(1977,2019) 

ax1.set_ylim(data['GOLD'].min(),data['GOLD'].max()) 

ax2.set_xlim(1977,2019)

ax2.set_ylim(data['OIL'].min(),data['OIL'].max()) 

ax3.set_xlim(1977,2019) 

ax3.set_ylim(data['CHFUSD'].min(),data['CHFUSD'].max()) 

ax4.set_xlim(1977,2019)

ax4.set_ylim(data['JPYUSD'].min(),data['JPYUSD'].max()) 



plt.xlabel('YEAR',fontsize = 20)

ax1.set_ylabel('GOLD', fontsize = 20)

ax2.set_ylabel('OIL', fontsize = 20)

ax3.set_ylabel('CHFUSD', fontsize = 20)

ax4.set_ylabel('JPYUSD', fontsize = 20)



fig.suptitle('Impact of the Yield Curve inversion on price of goods', fontsize = 25);
data['Flag is neg'] = np.where(data['SPREAD'] < 0, 1, 0)

corr_gold = sp.stats.pearsonr(x = data['SPREAD'], y = data['GOLD'])

corr_oil = sp.stats.pearsonr(x = data['SPREAD'], y = data['OIL'])

corr_sp500 = sp.stats.pearsonr(x = data['SPREAD'], y = data['SP500'])

corr_CHHUSD = sp.stats.pearsonr(x = data['SPREAD'], y = data['CHFUSD'])

corr_JPYUSD = sp.stats.pearsonr(x = data['SPREAD'], y = data['JPYUSD'])



print('Gold:', corr_gold, '\n'

     'Oil:', corr_oil, '\n'

     'SP500:', corr_sp500, '\n'

     'CHFUSD:', corr_CHHUSD, '\n'

     'JPYUSD:', corr_JPYUSD, '\n')

#All Significant
# Compute the correlation matrix

corr = data[['SP500', 'GOLD', 'OIL', 'CHFUSD', 'JPYUSD', 'Flag is neg']].corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(16, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});