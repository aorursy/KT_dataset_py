# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

if not os.path.exists('./plots/'):

     os.makedirs('./plots/')



print(check_output(["ls", "-laR", "../input"]).decode("utf8"))

print(check_output(["ls", "-laR", "./plots/"]).decode("utf8"))

print(check_output(["ls", "-laR", "."]).decode("utf8"))





# Any results you write to the current directory are saved as output.
import pandas

import numpy as np

import matplotlib

import matplotlib.pyplot as plt
df = pandas.read_csv('../input/wfp_market_food_prices.csv', encoding = "ISO-8859-1", index_col=['adm0_name'])

df
# Wage and skill/unskilled labour is listed as a commodity only on some Countries.

# I complete data with minimum wage picked up from external sources (PPP), expressed in local units.

# For a quick setup, I used: 

# https://en.wikipedia.org/wiki/List_of_minimum_wages_by_country



# Could be improved: some countries has minimumwage of 1 local unit : 

# that beacuse there is no local wage by law, or unreported
minimumWagePPP = {

'Afghanistan' : 5000, 

'Algeria' : 18000,

'Armenia' : 55000,

'Azerbaijan' : 130,

'Bangladesh' : 1500,

 'Benin' : 40000,

 'Bhutan' : 3750,

 'Bolivia' :  1805,

 'Burkina Faso' : 34664,

 'Burundi' : 1,

 'Cambodia' : 1,

 'Cameroon' : 36270,

 'Cape Verde' : 11000,

 'Central African Republic' : 35000,

 'Chad' : 59995,

 'Colombia' : 737717,

 'Congo' : 90000 , 

 'Costa Rica' : 286467,

 "Cote d'Ivoire" : 36607,

 'Democratic Republic of the Congo' : 50400,

 'Djibouti' : 35000,

 'Egypt' : 1200,

 'El Salvador' : 113,

 'Ethiopia' : 336,

 'Gambia' : 1500,

 'Georgia' : 20,

 'Ghana' : 240,

 'Guatemala' : 2247,

 'Guinea' : 440000,

 'Guinea-Bissau' : 19030,

 'Haiti' : 3750,

 'Honduras' : 5682,

 'India' : 160,

 'Indonesia' : 1337745,

 'Iran  (Islamic Republic of)' : 929931,

 'Iraq' : 250000,

 'Jordan' : 220,

 'Kenya' : 5437,

 'Kyrgyzstan' : 1140,

 "Lao People's Democratic Republic" : 800000,

 'Lebanon' : 675000,

 'Lesotho' : 1178,

 'Liberia' : 3600,

 'Madagascar' : 133013,

 'Malawi' : 20631,

 'Mali' : 28465,

 'Mauritania' : 30000,

 'Mozambique' : 3002,

 'Myanmar' : 108000,

 'Nepal' : 8000,

 'Niger' : 30047,

 'Nigeria' : 18000,

 'Pakistan' : 15000,

 'Panama' : 175,

 'Peru' : 850,

 'Philippines' : 7290,

 'Rwanda' : 15000,

 'Senegal' : 50184,

 'Somalia' : 1,

 'South Sudan' : 1500,

 'Sri Lanka' : 10000,

 'State of Palestine' : 1450,

 'Sudan' : 425,

 'Swaziland' : 420,

 'Syrian Arab Republic' : 9765,

 'Tajikistan' : 250,

 'Timor-Leste' : 115,

 'Turkey'  : 1778,

 'Uganda' : 6000,

 'Ukraine' : 3200,

 'United Republic of Tanzania' : 40000,

 'Yemen' : 21000,

 'Zambia' : 522,

 'Zimbabwe' : 82151

}





# Index my data by Countries

countries = df.groupby(by=['adm0_name'])

# select a country, and then the commodities that are traded within

country = countries.get_group('Algeria')

commodities = country.groupby(by=['cm_name','pt_name'])

# then group by commodities and markets (Apples in Retail market)

# and then by year:

# we want to plot price trends for a commodity for each year we have data

years = commodities.get_group(('Apples','Retail')).groupby(by=['mp_year'])



# we have years:

years.groups.keys()
def normaliseByLocalWage(Country, commodityPrice):

    '''

    Return percentage of minimumWage (PPP) that is necessary to purchase a commodity at a given commodityPrice

    '''

    return 100 * commodityPrice / minimumWagePPP[ Country ]

def averagePricePerMonth(district):

    '''

    For a Country, aggregate prices of local market provinces to their regional districts

    Collate and return a table of months and averaged price per month

    '''

    aggregate_functions = {'mp_price' : 'mean','adm1_name': 'first', 'um_name' : 'first','cur_name' : 'first'}

    return district.groupby(district['mp_month'],as_index=False).aggregate(  aggregate_functions )
def normaliseMissingMonths( district ):

    '''

    We have missing data or unreported data for months.

    If a month is missing, this function add that month to index with an empty value (Nan)

    We need this helper to have a nice plot!

    '''

    district.set_index("mp_month")

    new_index = pandas.Index(np.arange(1,13,1), name="mp_month")

    #new_index = pandas.Index(['Jan','Feb','Mar','Apr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dec'], name="mp_month")

    return district.set_index("mp_month").reindex(new_index)
def commodityRelativePrice(Country, district):

    '''

    For a commodity traded in a certain district,

    Return a dataframe object with prices listed in percentage of local minimum wage

    '''

    aggregatedDistrict = averagePricePerMonth(district)

    #aggregatedDistrict['rel_price'] = aggregatedDistrict.apply(lambda row: normaliseByLocalWage( Country , row.mp_price ), axis=1)

    aggregatedDistrict = normaliseMissingMonths( aggregatedDistrict )

    return aggregatedDistrict.apply(lambda row: normaliseByLocalWage( Country , row.mp_price ), axis=1)
# examples

# districts that reported data for a certain year

districts = years.get_group(2015).groupby(by="adm1_name")

# let's pick up one and display prices

district = districts.get_group('Alger')

averagePricePerMonth( district )
a = commodityRelativePrice('Algeria', districts.get_group('Alger') )

#a.plot(x='mp_month',y='mp_price',style='o-',kind='line', use_index=True).set_xticklabels(['1df','2','3','4','5dfd','6','7','32','3232'], rotation=0)

#a.plot(x='mp_month',y='mp_price',style='o-',kind='line')

a.plot(x='mp_month',y='mp_price',style='o-',kind='line')
def pickColor( position,  districts_list ):

    '''

    Helper to yield a color of a plot for a commodity

    '''

    cmap = matplotlib.cm.get_cmap('rainbow')

    #norm = matplotlib.colors.Normalize(vmin=1, vmax= len(commodity_list))

    # rgba = cmap( 0.5)

    return cmap( position / len(districts_list))



def getPriceTrendDistrictPlot(Country, District, Color):

    '''

    Return the plot for a commodity traded in a certain district

    '''

    c = commodityRelativePrice( Country , districts.get_group( District ) )

    return c.plot(x='mp_month', y='rel_price', color= Color, grid=True, label= str(District), style='.-',kind='line')



def getCommodityUnitMeasure(District_name):

    '''

    This is an helper to provide the unit measure of traded commodity in a given district. 

    I have not normalised them, and left as data are reported.

    Assume that if no digit in unit measure, then it is 1 unit measure

    '''

    c = districts.get_group( District_name )

    um = set( filter(lambda v: v==v, c.um_name.values) ).pop()

    # let's format the string: if no digit in unit measure, then it is 1 unit : e.g. 'Kg' > '1 Kg'

    if any(char.isdigit() for char in um):

        return um

    else:

        return '1 '.join(['',um])

    



def getDistrictPlot(Country, district_names, commodity_name, year, colors=None):

    '''

    Return a plot of a traded commodity in a certain year:

    display price trends recorded in all regional market districts of a Country. 

    '''

    plt.figure(figsize=(12,5))

    for c, district_name in enumerate( district_names ):

        #print( colors[c], district)

        #

        color = pickColor( c, district_names)

        #

        ax = getPriceTrendDistrictPlot( Country , district_name, color)

        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dec'], rotation=45)

        ax.set_xticks(np.arange(1,13,1))

        # here I assume unit_measure are all the same for all districts

        # the last processed district will assign the value to my plot title.

        unit_measure = getCommodityUnitMeasure( district_name)

        #

        h1, l1 = ax.get_legend_handles_labels()

        plt.title(  'Relative price of {0} of {1} in terms of local minimum wage, in Counties of {2} (along year {3})'.format( unit_measure, commodity_name , Country, str( year )) )

        plt.legend(h1, l1, loc=2)

        plt.xlabel( 'months' )

        plt.ylabel( 'relative price (% of minimum wage)' )

    return plt
# Example

district_names = ['Alger','Tindouf']

getDistrictPlot('Algeria', district_names,'Milk', 2016)
# helper functions to save plots

import os 



def getFileName( Country, Commodity_tuple, year):

    '''

    return filename to save a plot

    '''

    Commodity_tuple = [s.replace('/','_') for s in Commodity_tuple]

    return Country + '_' + '_'.join(Commodity_tuple) + '_' + str(year)



def setDirectory( commodity_name, subFolder=False ):

    '''

    Will create a ./plots/<commodity_name>/ folder if does not exist

    and set it as current path

    '''

    # trubles displaying results in output?

    if subFolder:

        if not os.path.exists('./plots/' + commodity_name):

            os.makedirs('./plots/' + commodity_name)

        return './plots/' + commodity_name + '/'

    else:

        return ''

# I create graphs here

if not os.path.exists('./plots/'):

     os.makedirs('./plots/')
## Given a country, get the commodities that are there traded.

# Then iterate for all years

country_name = 'Syrian Arab Republic';



country = countries.get_group( country_name )

commodities = country.groupby(by=['cm_name','pt_name'])

print( commodities.groups.keys() )

for commodity_tuple in commodities.groups.keys():

    years = commodities.get_group( commodity_tuple ).groupby(by=['mp_year'])

    for year in years.groups.keys():

        #print(year)

        districts = years.get_group( year ).groupby(by="adm1_name")

        #print( districts.groups.keys() )

        print( 'processing : ', country_name , 'year : ', year , 'commodity : ', commodity_tuple)

        district_names = list(districts.groups.keys())

        #

        fileName = getFileName( country_name , commodity_tuple, year)

        commodity_name = commodity_tuple[0] + ' (' + commodity_tuple[1] + ')'

        # output stored in root folder or can I create subfolders?

        path = setDirectory(commodity_tuple[0], subFolder=False)

        #

        fig = getDistrictPlot( country_name , district_names, commodity_name , year)

        fig.show()

        fig.savefig( path + fileName +'.png')

        #fig.close()
# Warning, this will take ~350Mb on disk for about 7500 files

# I won't show the plot here, I will save them 

# in path: ./plots/<my_commodity>/<Country_Name>_<Commodity_name>_<Market_Type>_<year>.png

# example: ./plots/Apples/Algeria_Apples_Retail_2015.png



for countryKey in countries.groups.keys():

    #

    country = countries.get_group( countryKey )

    commodities = country.groupby(by=['cm_name','pt_name'])

    print( commodities.groups.keys() )

    for commodity_tuple in commodities.groups.keys():

        years = commodities.get_group( commodity_tuple ).groupby(by=['mp_year'])

        for year in years.groups.keys():

            print( 'processing : ', countryKey , 'year : ', year , 'commodity : ', commodity_tuple)

            districts = years.get_group( year ).groupby(by="adm1_name")

            # print( districts.groups.keys() )

            district_names = list(districts.groups.keys())

            #

            fileName = getFileName( countryKey , commodity_tuple, year)

            commodity_name = commodity_tuple[0] + ' (' + commodity_tuple[1] + ')'

            path = setDirectory(commodity_tuple[0], subFolder=True)

            #

            fig = getDistrictPlot( countryKey , district_names, commodity_name , year)

            fig.savefig( path + fileName +'.png')

            fig.close()