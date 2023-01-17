# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import seaborn as sns

import math

from matplotlib.ticker import MaxNLocator

from fbprophet import Prophet

import squarify

from IPython.display import HTML

from subprocess import check_output





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.



# Loading the Data

State_time_series=pd.read_csv("../input/State_time_series.csv", parse_dates=True)

State_time_series.Date=pd.to_datetime(State_time_series.Date)

State_time_series['year'] = State_time_series.Date.dt.year

# GET Costliest States

states = set(State_time_series[

    ~State_time_series['ZHVI_AllHomes'].isnull() &

    ~State_time_series['Sale_Prices'].isnull()

                      ]['RegionName'].values)



State_time_series_year = State_time_series[State_time_series['RegionName'].isin(states)].copy()

highest_cost_states = State_time_series_year[['RegionName', 'ZHVI_AllHomes']].groupby('RegionName').max().sort_values(by=['ZHVI_AllHomes'], ascending=False)[:5].index.values.tolist()

State_time_series_year=State_time_series_year[State_time_series_year.RegionName.isin(highest_cost_states)]

State_time_series_year.year = State_time_series_year.Date.dt.year
# colorblind-Friendly colors



colors= [[0,0,0], [230/255, 159/255, 0], [86/255, 180/255, 233/255], [0, 158/255, 115/255],

         [213/255, 94/255, 0], [0, 144/255, 178/255]]



States_year_SalePrices = State_time_series_year.groupby([State_time_series_year.year,State_time_series_year.RegionName])['ZHVI_AllHomes'].mean().dropna().reset_index(name='SoldPrice')

fte_graph=States_year_SalePrices.pivot(index='year',columns='RegionName', values='SoldPrice').plot(figsize=(15,8), color=colors, legend=False)

fte_graph.figure.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

fte_graph.tick_params(axis = 'both', which ='major', labelsize = 18)

plt.xlabel("")

fte_graph.set_yticklabels(labels= [-10, '100000' , '200000' , '300000', '400000' , '500000 $'])



# Generate a Bolded Horizontal Line at y= 0



fte_graph.axhline (y = 100000, color = 'black', linewidth = 1.3, alpha = .7)



# Add an extra vertical line by tweaking the range of the X-axis



fte_graph.set_xlim(left = 1996, right = 2017)



# The Signature Bar



fte_graph.text(x = 1994, y = 30000,

              s = ' @Poonam Lingade/Nikunj Lata Source : Zillow ',

               fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')



# Adding a Title and a Subtitle



fte_graph.text(x = 1994, y = 570000, s = "The Growth of Real Estate Sector in US",

               fontsize = 26, weight = 'bold', alpha = .75)

fte_graph.text(x = 1994, y = 550000,

               s = 'Increase in House Median Sale Prices from 1996 to 2017 in the US states',

               fontsize = 19, alpha = .85)



# Add colored labels



fte_graph.text(x = 2002, y = 390000, s = 'California', color = colors[0], weight = 'bold', rotation = 55, backgroundcolor = '#f0f0f0')



fte_graph.text(x = 2012, y = 530000, s = 'District of Columbia', color = colors[1], weight = 'bold', rotation = 30, backgroundcolor='#f0f0f0')



fte_graph.text(x = 2015, y = 310000, s = 'Massachusettes', color = colors[4], weight = 'bold', rotation = 15, backgroundcolor = '#f0f0f0')



fte_graph.text(x = 2013, y = 220000, s = 'New Jersey', color = colors[4], weight = 'bold', rotation = 15, backgroundcolor = '#f0f0f0');

fig = plt.figure(figsize=(25,21))

regions=State_time_series.RegionName.value_counts().to_frame()

ax = fig.add_subplot(111, aspect="equal")

ax = squarify.plot(sizes=regions['RegionName'].values,label=regions.index,

                  color=sns.color_palette('viridis', 52), alpha = 1)

ax.set_xticks([])

ax.set_yticks([])

fig=plt.gcf()

fig.set_size_inches(40,25)

plt.title("Treemap of House Property counts across different States", fontsize=18)

plt.show();

plt.figure(figsize=(15,8));



#plt.figure.gca().xaxis.set_major_locator(MaxNLocator(integer=True))



State_time_series.groupby(State_time_series['year'])['ZHVI_TopTier'].median().plot(linewidth=4,c='g')

State_time_series.groupby(State_time_series['year'])['ZHVI_MiddleTier'].median().plot(linewidth=4,c='b')

State_time_series.groupby(State_time_series['year'])['ZHVI_BottomTier'].median().plot(linewidth=4,c='r')





# Generate a bolded HoriZontal line at y=0



plt.axhline(y= 50000, color='black', linewidth=1.3, alpha=.7)



# Add an extra vertical line by tweaking the range of the x-axis



plt.xlim(left=1996, right=2017)



# the Signature bar



plt.text(x = 1996,y = 25000,

        s = ' Poonam Ligade/Nikunj Lata Source:Kaggle', fontsize =14, color = '#f0f0f0', backgroundcolor = 'grey')



# Adding a title and a subtitle



plt.text(x = 1996, y = 350000, s = "Real Estate Properties ZHVI for different Tiers in US",fontsize = 25, weight = 'bold', alpha=.75)

plt.text(x = 1996, y = 330000, s = "ZHVI Values for Top, middle and Bottom Tiers from 1997 to 2017 in the US States", fontsize = 18, alpha= .85);



plt.text(x = 1999, y = 215000, s = 'Top Tier', color= 'g', weight = 'bold', rotation = 35, backgroundcolor = '#f0f0f0')



plt.text(x = 1999, y = 140000, s = 'Middle Tier', color = 'b', weight = 'bold', rotation = 20, backgroundcolor = '#f0f0f0')



plt.text(x = 1999, y = 140000, s = 'Bottom tier', color = 'r', weight = 'bold', rotation = 5, backgroundcolor = '#f0f0f0');

# ZHVI Vs Days on Zillow



plt.figure(figsize=(20, 6))

mean_group = State_time_series[['DaysOnZillow_AllHomes','ZHVI_AllHomes']].groupby(['DaysOnZillow_AllHomes'])['ZHVI_AllHomes'].mean()

plt.plot(mean_group, color=[230/255, 159/255, 0])

plt.tick_params(axis='both', which = 'major', labelsize = 18)



# The Signature Bar



plt.text(x = 27, y = -1500, s = '@Poonam Ligade/Nikunj Lata Source: Kaggle ', fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')





# adding a title and a subtitle.



plt.text(x = 30, y = 680000, s = "House Sale Value Decreases as listings ages on Zillow", fontsize = 26, weight = 'bold', alpha=.75)

plt.text(x = 30, y = 640000, s = "As number of Days after house listings is posted on Zillow increases, its Zillow Home Value Index(ZHVI) deacreases.", fontsize = 19, alpha = .85);
plt.figure(figsize=(15,8));



plt.scatter(State_time_series_year.DaysOnZillow_AllHomes,State_time_series_year.ZHVI_AllHomes, c="gold");

plt.scatter(State_time_series_year.DaysOnZillow_AllHomes,State_time_series_year.Sale_Prices, c ="green");



# Generate a Bolded Horizontal Line at y= 0



plt.axhline( y=200000, color='black', linewidth=1.3, alpha= .7)



# Add an extra vertical line by tweaking the range of x-axis



plt.xlim(left = 45, right = 255)



# The Signature bar



plt.text(x= 35, y = 160000,

        s = 'Poonam Ligade/Nikunj Lata Source:Kaggle ', fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')



plt.legend(fontsize = 12, loc="best", frameon=1, fancybox = 1)



# Adding a title and subtitle



plt.text(x = 35, y = 590000, s = "The Linear Relation between Real Estate Sector sale prices in US", fontsize = 25, weight = 'bold', alpha= .75)

plt.text(x = 35, y = 570000, s = "Linearity between ZHVI and Median Sold Prices w.r.t Days on Zillow from 1996 to 2017 in the US States", fontsize = 18, alpha= .85);
# ZRI VS Days. on Zillow.



plt.figure(figsize=(20,6))

mean_group = State_time_series[['DaysOnZillow_AllHomes', 'ZRI_AllHomes']].groupby(['DaysOnZillow_AllHomes'])['ZRI_AllHomes'].mean().dropna().reset_index(name='ZRI_AllHomes_Mean')

mean_group=mean_group[(mean_group.DaysOnZillow_AllHomes < 250)]

mean_group=mean_group[(mean_group.ZRI_AllHomes_Mean > 500)]



plt.plot(mean_group, color=[213/255, 94/255,0])



plt.tick_params(axis ='both', which = 'major', labelsize = 18)



# The Signature Bar



plt.text ( x= -40, y = -500,

           s = ' Poonam Ligade/ Nikunj Lata Source : Kaggle',

         fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')



# Adding a title and a subtitle 



plt.text (x = -40, y = 3300, s = "House Rent Value Decreases as listing ages on Zillow",

         fontsize = 26, weight = 'bold', alpha=.75)

plt.text ( x = -40, y = 3000, s = "As number of Days after house listing is posted on Zillow increases, its Zillow Rental Index(ZRI) decreases ", fontsize =19, alpha = .85 );

plt.show();
plt.figure(figsize=(15,8));



plt.scatter(State_time_series_year.DaysOnZillow_AllHomes, State_time_series_year.ZRI_AllHomes, c='orange')

plt.scatter(State_time_series_year.DaysOnZillow_AllHomes, State_time_series_year.MedianRentalPrice_AllHomes, c='navy');





# Generate a bolded Horizontal line at y= 0



plt.axhline(y = 1400, color = 'black', linewidth = 1.3, alpha= .7)



# Add an extra vertical line by tweaking the range of X axis



plt.xlim(left =45, right =255)



# The Signature bar



plt.text( x = 45, y= 1250, s = 'Poonam Ligade/ Nikunj Lata Source:Kaggle', fontsize=14, color='#f0f0f0', backgroundcolor = 'grey')



plt.legend(fontsize =12, loc = 'best', frameon = 1, fancybox =1)



# Adding a title and a subtitle



plt.text(x = 45, y = 2870, s = 'The Linear Relation between Real Estate Sector Rental Prices w.r.t Days on Zillow from 1996 to 2017 in US States', fontsize =18, alpha =.85);



plt.figure(figsize=(15,8));

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_1Bedroom'].mean().dropna().plot(linewidth=4,c='lightgreen', legend=False)

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_2Bedroom'].mean().dropna().plot(linewidth=4,c='b')

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_3Bedroom'].mean().dropna().plot(linewidth=4,c='brown')

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_4Bedroom'].mean().dropna().plot(linewidth=4,c='purple')

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_5BedroomOrMore'].mean().dropna().plot(linewidth=4,c='magenta')

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_AllHomes'].mean().dropna().plot(linewidth=4,c='black')

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_CondoCoop'].mean().dropna().plot(linewidth=4,c='orange',legend=False)

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_DuplexTriplex'].mean().dropna().plot(linewidth=4,c='darkblue',legend=False)

State_time_series.groupby(State_time_series['year'])['MedianListingPricePerSqft_SingleFamilyResidence'].mean().dropna().plot(linewidth=4,c='red')

plt.legend(fontsize=12,loc=(2009,200),frameon=1,fancybox=1,bbox_to_anchor=(0.02,1.35), ncol=2)





# Generate a bolded horizontal line at y = 0 

plt.axhline(y = 50, color = 'black', linewidth = 1.3, alpha = .7)

# Add an extra vertical line by tweaking the range of the x-axis

plt.xlim(left = 2010, right = 2017)

# The signature bar



plt.text(x = 2010, y = 30,

    s = '   Poonam Ligade/Nikunj Lata                                                                                                                                                      Source: Kaggle   ',

    fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')



# Adding a title and a subtitle

plt.text(x = 2010, y = 230, s = "Real Estate Listing Prices Per SquareFoot in US",

               fontsize = 25, weight = 'bold', alpha = .75)

plt.text(x = 2010, y = 220, 

               s = 'Median Listing Prices Per Sqft W.r.t Number of Bedrooms from 2010 to 2017 in the US states',

              fontsize = 18, alpha = .85);
plt.figure(figsize=(15,8));



State_time_series.groupby(State_time_series['year'])['PctOfHomesDecreasingInValues_AllHomes'].median().plot(linewidth=4, c='lightgreen', legend=False)

State_time_series.groupby(State_time_series['year'])['PctOfHomesIncreasingInValues_AllHomes'].median().plot(linewidth=4, c='b')



# Generate a bolded horizontal line at y = 0



plt.axhline(y=10, color='black', linewidth = 1.3, alpha=.7)



# Add an extra vertical line by tweaking the range of the x- axis



plt.xlim(left = 1996, right = 2017 )



# The Signature Bar



plt.text( x= 1996, y = 0, s = 'Poonam Ligade/Nikunj Lata Source: Kaggle', fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')



# Adding a title and a subtitle 



plt.text(x= 1996, y = 95, s = "Real Estate Properties Decreasing vs Increasing in Values in US", fontsize = 25, weight = 'bold', alpha = .75)

plt.text(x= 1996, y = 90, s = 'Percentage of Homes Increasing and Decreasing in Values from 1997 to 2017 in the US States', fontsize = 18, alpha = .85);

plt.figure(figsize=(15,8));



State_time_series.groupby(State_time_series['year'])['PctOfHomesSellingForGain_AllHomes'].median().plot(linewidth=4, c='lightgreen', legend=False)

State_time_series.groupby(State_time_series['year'])['PctOfHomesSellingForLoss_AllHomes'].median().plot(linewidth=4, c='gold')

plt.legend()



# Generate a bolded horizontal line at y = 0



plt.axhline(y=0, color = 'black', linewidth = 1.3, alpha = .7)



# Add an extra vertical line by tweaking the range of x - axis



plt.xlim(left = 2001, right= 2017)



# The Signature bar



plt.text(x = 2001, y = -15, s = 'Poonam Ligade/Nikunj Lata Source: Kaggle', fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')



# Adding a title and a subtitle.



plt.text(x = 2001, y = 110, s = 'Real Estate Properties Sailing For Gain and Loss in US', fontsize = 25, weight = 'bold', alpha = .75)

plt.text( x= 2001, y= 105, s = 'Percentage of Homes Selling for Gain and Loss from 2002 to 2017 in US States', fontsize = 18, alpha = .85);

plt.figure().gca().xaxis.set_major_locator(MaxNLocator(integer=True))

State_time_series.day=State_time_series.Date.dt.day

plt.bar(State_time_series.day.value_counts().index,State_time_series.day.value_counts().values,color = colors);

plt.tick_params(axis = 'both', which = 'major', labelsize = 18)



# the Signature Bar



plt.text(x=27, y = -1500, s = ' @Poonam Ligade/Nikunj Lata    Source:Kaggle', fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')



# Adding a title and a subtitle



plt.text(x=27, y=10000, s="Recorded Dates", fontsize = 26, weight = 'bold', alpha=.75)

plt.text(x= 27, y = 9000, s = "The days on which data is captured", fontsize = 19, alpha = .85);
