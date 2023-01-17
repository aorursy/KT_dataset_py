# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import pandas as pd

import numpy as np

import json as json

import seaborn as sbn

import matplotlib.pyplot as plt

import scipy as sp

import re

import datetime as dt



# import plotly

import plotly as py

import plotly.graph_objs as go



# these two lines are what allow your code to show up in a notebook!

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

data_dir = "../input/repository/ecodan-housing_affordability_data-60a936a"

print(os.listdir(data_dir))



# Any results you write to the current directory are saved as output.
# read in the case-schiller index data

df_schiller = pd.read_excel(

    os.path.join(data_dir, "cs_idx.xls"),

    sheet_name="Data", 

    skiprows=range(0,6)

)
# courtesy of https://stackoverflow.com/questions/20911015/decimal-years-to-datetime-in-python/20911144#20911144

from datetime import datetime, timedelta



def decimal_year_to_dt(dyear):

    year = int(dyear)

    rem = dyear - year

    base = datetime(year, 1, 1)

    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)

    return result
# drop all but the first two columns (Date and Index)

df_hp_idx = df_schiller.iloc[:,0:2]



# resample the dates to be whole years

df_hp_idx['dt'] = df_hp_idx['Date'].apply(lambda x: decimal_year_to_dt(x))

df_hp_idx_yr = df_hp_idx.resample('AS', on='dt', kind='timestamp').mean()

df_hp_idx_yr['Date'] = df_hp_idx_yr['Date'].astype(int)

# peek at the data

df_hp_idx_yr.head()
# plot the index

df_hp_idx_yr.plot.line( 

    figsize=(12,6), 

    title='Case-Schiller House Price Idx 1890-2017 (2017 dollars)',

    style='.-',

    x='Date',

    y='Index',

    ylim=(0, 225),

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)
# plot the index since 1967

df_hp_idx_yr[df_hp_idx_yr['Date'] >= 1967].plot.line( 

    figsize=(12,6), 

    title='Case-Schiller House Price Idx 1967-2017 (2017 dollars)',

    style='.-',

    x='Date',

    y='Index',

    ylim=(0, 225),

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)
z = np.polyfit(

    df_hp_idx_yr[df_hp_idx_yr['Date'] >= 1967]['Date'], 

    df_hp_idx_yr[df_hp_idx_yr['Date'] >= 1967]['Index'], 

    2

)

p = np.poly1d(z)

# plot the index with the best fit

df_hp_idx_yr['fit'] = df_hp_idx_yr['Date'].apply(lambda x: p(x)) # only valid after 1967

df_hp_idx_yr[df_hp_idx_yr['Date'] >= 1967].plot.line( 

    figsize=(12,6), 

    title='Schiller House Price Idx 1967-2017 (2017 dollars)',

    style={'Index':'.-','fit':'-'},

    x='Date',

    y=['Index', 'fit'],

    ylim=(0,225)

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)
# read US census household income data broken out by tiers (20%, 40%, 60%, 80%, 95%) 

df_inc_tiers_2017 = pd.read_excel(

    os.path.join(data_dir, "census_income_tiers_h01ar.xls"),

    sheet_name="h01ar", 

    names=['year', 'household_count', 't20', 't40', 't60', 't80', 't95'],

    skiprows=range(0,3)

)
# strip footnotes off of the years

df_inc_tiers_2017['year'] = df_inc_tiers_2017['year'].apply(lambda x: int(str(x)[0:4]))

# convert households back to millions

df_inc_tiers_2017['household_count'] = df_inc_tiers_2017['household_count'].apply(lambda x: x*1000)

# move the "year" to the index

df_inc_tiers_2017.set_index(df_inc_tiers_2017['year'].astype(int), inplace=True)

# reverse the rows (oldest at top)

df_inc_tiers_2017 = df_inc_tiers_2017.iloc[::-1]
# peek at the resulting data

df_inc_tiers_2017.head()
# graph the tiers over time

df_inc_tiers_2017.iloc[:,2::].plot.line( 

    figsize=(12,6), 

    title='Earnings by income tiers 1967-2017 (2017 dollars)',

    style='.-',

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)
df_inc_tiers_2017.iloc[:,2::].div(

    df_inc_tiers_2017['t20'], 

    axis=0, 

).plot.line( 

    figsize=(12,6), 

    title='Ratio of income tiers to 20th percentile since 1967',

    style='.-',

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)



# read in the census poverty line history

df_poverty_line = pd.read_excel(

    os.path.join(data_dir, "hstpov1.xls"),

    sheet_name="pov01", 

    skiprows=range(0,3)

)



# get only the columns we need and drop the last row which is a legend

df_poverty_line = df_poverty_line.iloc[0:-1,np.r_[0,4,7,15]]

df_poverty_line.columns = ['year', '2', '3', 'CPI']



# get the average of column 4 (2 person households) and 7 (3 person households)

df_poverty_line['2.5'] = df_poverty_line.apply(lambda x: (x['2']+x['3'])/2.0, axis=1)



df_poverty_line.drop(['2','3'], axis=1, inplace=True)
# strip footnotes off of the years

df_poverty_line['year'] = df_poverty_line['year'].apply(lambda x: int(str(x)[0:4]))



# drop the extra 2013

df_poverty_line.drop(df_poverty_line.index[df_poverty_line['year'] == 2013].tolist()[0], axis=0, inplace=True)



# drop the years prior to 1967

df_poverty_line.drop(df_poverty_line.index[df_poverty_line['year'] < 1967].tolist(), axis=0, inplace=True)



# move the "year" to the index

df_poverty_line.set_index(df_poverty_line['year'].astype(int), inplace=True)



# adjust to 2017 dollars with CPI

cpi_2017 = df_poverty_line['CPI'].iloc[-1]

df_poverty_line['2.5 adj'] = df_poverty_line.apply(lambda x: x['2.5'] * cpi_2017 / x['CPI'], axis=1)
# peek at the data

df_poverty_line.head()
df_inc_tiers_2017_mod = df_inc_tiers_2017.iloc[:,2::].sub(df_poverty_line['2.5'], axis=0) 
df_inc_tiers_2017_mod.plot.line( 

    figsize=(12,6), 

    title='"House buying power" by income quintile 1967-2017 (2017 dollars)',

    style='.-',

    ylim=(0,225000)

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)
df_inc_tiers_2017_mod.div(

    df_inc_tiers_2017_mod['t20'], 

    axis=0, 

).plot.line( 

    figsize=(12,6), 

    title='Ratio of "house buying power" tiers (1967-2017)',

    style='.-',

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)
# graph income tiers against Case-Schiller actuals and trend

ax = df_inc_tiers_2017_mod.plot.line( 

    figsize=(15,8), 

    title='"House buying power" vs Schiller Price Index by income quintile 1967-2017 (2017 dollars)',

    style='.-',

    ylim=(0,225000),

)

ax.legend(

    loc='center left', 

    bbox_to_anchor=(1.1, 0.5)

)

ax.set_ylabel("Income (USD)")



# add the schiller index to secondary y axis

ser_schiller = pd.Series(index=df_inc_tiers_2017_mod.index.tolist(), data=[p(x) for x in df_inc_tiers_2017_mod.index.tolist()])

ax2 = ax.twinx()

ax2.set_ylim((0,225))

line2, = ax2.plot(

    df_inc_tiers_2017_mod.index.tolist(), 

    ser_schiller, 

    'k-',

    label='Schiller Index Trend'

)

line3, = ax2.plot(

    df_inc_tiers_2017_mod.index.tolist(), 

    df_hp_idx_yr[df_hp_idx_yr['Date'] > 1967]['Index'], 

    color="0.75",

    linestyle="dashed",

    label='Actual Schiller Index'

)

ax2.legend(

    loc='center left', 

    bbox_to_anchor=(1.1, 0.7)

)

# run a pearson correlation between the shiller index and incomes at each tier

df_inc_tiers_2017_mod.corrwith(ser_schiller)
# assume 20% down, max 35% of take home pay, 4% mortgage interest



# this method has hardcoded thresholds for 2017 tax brackets.  Shame on me.

def calc_takehome(income):

    # take off standard deduction - 2400 for married household

    mod_income = income - 2400

    if mod_income < 0:

        return income

    

    taxable_by_tier = np.zeros(4)

#     tier_0 = mod_income if mod_income < 19051 else 19051

    taxable_by_tier[0] = (mod_income - 19051) if mod_income < 77400 else (77400-19051)

    taxable_by_tier[1] = (mod_income - 77400) if mod_income < 165000 else (165000-77400)

    taxable_by_tier[2] = (mod_income - 165000) if mod_income < 315000 else (315000-165000)

    taxable_by_tier[3] = (mod_income - 315000) if mod_income < 400000 else (400000-315000)

    

    taxable_by_tier = [x if x > 0 else 0 for x in taxable_by_tier]

    tax = (taxable_by_tier[0] * .12) + (taxable_by_tier[1] * .22) + (taxable_by_tier[2] * .24) + (taxable_by_tier[3] * .32)

    

    # 7.65% for SS and medicare up to 128,000

    ssmed = .0765 * (income if income < 128000 else 128000)

    return (income - tax - ssmed)





# test

print("  2300: {0}".format(calc_takehome(2300)))

print(" 18000: {0}".format(calc_takehome(18000)))

print(" 50000: {0}".format(calc_takehome(50000)))

print("250000: {0}".format(calc_takehome(250000)))

# create a new dataframe to hold the results of the affordability calculations

df_mortage_2017 = pd.DataFrame(

    index=df_inc_tiers_2017.columns[2::], 

    data=df_inc_tiers_2017.iloc[-1,2::].T,

)

df_mortage_2017.rename({2017:'base'}, inplace=True, axis=1)
df_mortage_2017
# compute the after tax net income

df_mortage_2017['takehome'] = df_mortage_2017['base'].apply(lambda x: calc_takehome(x))



# apply the 25% and 35% mortgage afforgability rules

df_mortage_2017['max_monthly_35'] = df_mortage_2017['takehome'].apply(lambda x: x * .35 / 12)

df_mortage_2017['max_monthly_25'] = df_mortage_2017['takehome'].apply(lambda x: x * .25 / 12)
df_mortage_2017
# how much house is that?

# c*(1-(1+r)-N)/r = P

def calc_max_price(pymt, interest, periods):

    return (pymt * (1-(1+(interest/12/100))**(0-periods))) / (interest/12/100)



# test

print("1264.14, 6.5, 360: {0}".format(calc_max_price(1264.14, 6.5, 360))) # should be 200K



# assuming 20% down

df_mortage_2017['max_house_35'] = df_mortage_2017['max_monthly_35'].apply(lambda x: calc_max_price(x, 5, 360)*1.2)

df_mortage_2017['max_house_25'] = df_mortage_2017['max_monthly_25'].apply(lambda x: calc_max_price(x, 5, 360)*1.2)



df_mortage_2017['max_house_35'].plot.barh(

    figsize=(12,6), 

    title='Max house price by tiers based on 35% mortage to income ratio (2017)',

    style='.-',

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)
df_mortage_2017['max_house_25'].plot.barh(

    figsize=(12,6), 

    title='Max house price by tiers based on 25% mortage to income ratio (2017)',

    style='.-',

).legend(

    loc='center left', bbox_to_anchor=(1, 0.5)

)
df_mortage_2017['price_inc_ratio_35'] = df_mortage_2017['max_house_35'] / df_mortage_2017['base']

df_mortage_2017['price_inc_ratio_25'] = df_mortage_2017['max_house_25'] / df_mortage_2017['base']
df_mortage_2017
# read data from zillow on price-to-income ratio (PIR) compared to the median income in major US markets

df_z_aff = pd.read_csv(os.path.join(data_dir,'Zillow_Affordability_Wide_Public.csv'))

df_z_aff.drop(['RegionID','City','State','SizeRank','HistoricAverage_1985thru1999'], axis=1, inplace=True)

df_z_aff = df_z_aff[df_z_aff['Index'] == 'Price To Income']

df_z_aff.drop('Index', axis=1, inplace=True)

df_z_aff.set_index('RegionName', inplace=True)
df_z_aff = df_z_aff.unstack().reset_index()

df_z_aff.rename({'level_0':'date',0:'pir'}, axis=1, inplace=True)
# function manually convert the dates

patt = re.compile('(\d?\d)/(\d?\d)/(\d\d)')

def to_date(dstr):

    m = patt.match(dstr)

    yr = int(m.group(3))

    yr = 2000 + yr if yr < 50 else 1900 + yr

    return dt.date(yr, int(m.group(1)), int(m.group(2)))



to_date('9/01/14')
# convert the dates

df_z_aff['date'] = df_z_aff['date'].apply(lambda x: to_date(x))

df_z_aff['date'] = pd.to_datetime(df_z_aff['date'])
# funtion to re-format the market name in the zillow data

patt = re.compile('([^-]+)(.*), (.+)')

def split_region(region_name):

    m = patt.search(region_name)

    return pd.Series((m.group(1), m.group(3)))
# extract the city and state

df_z_aff[['City', 'State']] = df_z_aff['RegionName'].apply(lambda x: split_region(x))

df_z_aff.head()
# read in a city-to-lat/lon lookup table

df_geo = pd.read_csv(os.path.join(data_dir,'us_city_lookup.csv'))
# join the lat/lon data with the PIR dataset

df_z_aff_geo = df_z_aff.merge(df_geo[['city', 'state_id', 'population', 'lat', 'lng']], how='inner', left_on=['City','State'], right_on=['city','state_id'])

df_z_aff_geo['population'] = df_z_aff_geo['population'].apply(lambda x: x**.45)

df_z_aff_geo.head()
print('max pop: {0} | min pop: {1}'.format(df_z_aff_geo['population'].max(), df_z_aff_geo['population'].min()))
# population isn't critical to this analysis, but the NaNs are problematic so just fill with mean

df_z_aff_geo['population'] = df_z_aff_geo['population'].fillna(df_z_aff_geo['population'].mean())
# show the data with plotly; red color = 5.0 PIR 

def show_map_for_year(df, year):

    data = []

    df_yr = df[(df['date'].dt.year == year)&(df['date'].dt.month == 9)]

    data = [

        dict(

            type = 'scattergeo',

            locationmode = 'USA-states',

            lon = df_yr['lng'],

            lat = df_yr['lat'],

            text = ["{1}<br>price-to-income: {0:0.1f}".format(x[1][1],x[1][0]) for x in df_yr[['RegionName','pir']].iterrows()],

            mode = "markers",

            marker = dict(

                cmin=1,

                cmax=5,

                colorscale=[[0, "rgb(0,255,50)"],[1,"Red"]],

                size = df_yr['population'],

                color = df_yr['pir'],

                line = dict(width=0.5, color='rgb(40,40,40)'),

                sizemode = 'area',

                sizeref=2.*df_z_aff_geo['population'].max()/(20.**2),

                showscale=True,

            ),

            name='blah'

        )

    ]



    layout = dict(

            title = '{0} Price to Income ratio (red = 5.0 PIR)'.format(year),

            showlegend = False,

            geo = dict(

                scope='usa',

                projection=dict( type='albers usa' ),

                showland = True,

                landcolor = 'rgb(217, 217, 217)',

                subunitwidth=1,

                countrywidth=1,

                subunitcolor="rgb(255, 255, 255)",

                countrycolor="rgb(255, 255, 255)"

            ),

        )



    fdata = dict(data=data, layout=layout)

    return fdata

fdata = show_map_for_year(df_z_aff_geo, 1980)

py.offline.iplot(fdata)

fdata = show_map_for_year(df_z_aff_geo, 2017)

py.offline.iplot(fdata)