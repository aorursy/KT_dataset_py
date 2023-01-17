import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)
import pandas as pd

import matplotlib.pyplot as plt



uncover_county = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/New_York_Times/covid-19-county-level-data.csv')

uncover_state = pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/New_York_Times/covid-19-state-level-data.csv')



#uncover_county['date'] = uncover_county['date'].map(pd.to_datetime)



uncover_county['datetime'] = pd.to_datetime(uncover_county['date'])

uncover_county = uncover_county.set_index('datetime')

uncover_county.drop(['date'], axis= 1, inplace= True)







#uncover_county.head() Prints out the table
def county(countyname):

    county_name = uncover_county[(uncover_county['county'] == countyname) & (uncover_county['state'] == 'California')]

    return county_name



sb_county = county('San Bernardino')

la_county = county('Los Angeles')

Orang_county = county('Orange')

san_diego = county('San Diego')

riverside = county('Riverside')

imperial = county('Imperial')

ventura = county('Ventura')

kern = county('Kern')

slo = county('San Luis Obispo')

santa_barbara = county('Santa Barbara')





sbla = [sb_county, la_county, Orang_county, san_diego, riverside, imperial, ventura, kern, slo, santa_barbara]

counties = ['San Bernardino', 'Los Angeles', 'Orange', 'San Diego', 'Riverside', 'Imperial', 'Ventura', 'Kern', 'SLO', 'Santa Barbara']



def tables():

    for i in sbla:

        display(i.head(5))

tables()





plt.figure(figsize=(16,14))

for county in sbla:

    plt.plot(county.index.values, county['cases'], lw=2)

    plt.xlabel('Dates',)

    plt.ylabel('Amount of Cases')



plt.legend(counties)

plt.title('Southern California (COVID-19 Amount of Cases Over Time)')

plt.show()
'''

uc = uncover_county[uncover_county['state'] == 'California']

uc_grouped = uc.groupby(['county', 'cases', 'datetime']).count()

print(uc_grouped.head(15))

'''


def county(countyname):

    county_name = uncover_county[(uncover_county['county'] == countyname) & (uncover_county['state'] =='California')]

    return county_name



alameda = county('Alameda')

contra_costa = county('Contra Costa')

marin = county('Marin')

napa = county('Napa')

san_francisco = county('San Francisco')

san_mateo = county('San Mateo')

santa_clara = county('Santa Clara')

solano = county('Solano')

sonoma = county('Sonoma')



bay_area = [alameda, contra_costa, marin, napa , san_francisco, san_mateo, santa_clara, solano, sonoma]



initial = alameda

for i in bay_area[1:]:

    merged_bay = pd.merge(initial,i, on = 'datetime' )

    initial = merged_bay

    

                                

    

#Using this to be able to see data tables from each county of Bay Area

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''def tables():

    for i in bay_area:

        display(i.head(5))

tables()

'''

# Not displaying the above code/tables so readablity of graphs is easier



merged_bay.columns =['county_x','state_x','fips_x','cases_AlA','deaths_x','county_y','state_y','fips_y','cases_CC','deaths_y','county_x','state_x','fips_x','cases_Marin','deaths_x','county_y','state_y','fips_y','cases_Napa','deaths_y','county_x','state_x','fips_x','cases_SF','deaths_x', 'county_y', 'state_y','fips_y','cases_SM','deaths_y','county_x','state_x','fips_x','cases_SC','deaths_x','county_y','state_y','fips_y','cases_solano','deaths_y','county','state','fips','cases_sonoma','deaths']



#pandas renaming columns was being problematic for columns w/ same name       



#Following code will display cases from start of March when most of these counties had cases, so if cases

#start at 2> it's because this is including cases start from March and on when cases significantly started increasing



merged_bay['cases'] = merged_bay['cases_AlA']+merged_bay['cases_CC']+merged_bay['cases_Marin']+merged_bay['cases_Napa']+merged_bay['cases_SF']+merged_bay['cases_SM']+merged_bay['cases_SC']+merged_bay['cases_solano']+merged_bay['cases_sonoma']



pd.set_option('display.max_columns', None)

merged_bay.head(10)



los_angeles = county('Los Angeles')



county_lst = [merged_bay, los_angeles]

county_names = ['Bay Area', 'Los Angeles']





plt.figure(1, figsize=(16,14))

for county in county_lst:

    plt.plot(county.index.values, county['cases'], lw=2)

    plt.xlabel('Dates',)

    plt.ylabel('Amount of Cases')



plt.legend(county_names)

plt.title('Bay Area vs LA Area (COVID-19 Amount of Cases Over Time))')

plt.show()



'''plt.figure(2, figsize=(16,14))

for county in county_lst:

    plt.plot(county.index.values, county['deaths'], lw=2)

    plt.xlabel('Dates',)

    plt.ylabel('Amount of Deaths')



plt.legend(county_names)

plt.title('Bay Area vs LA Area (COVID-19 Amount of Deaths Over Time))')

plt.show()

'''



#cannot show an accurate death graph until i rename the columns for deaths and add them up just like in the cases situation
def county_(countyname):

    county_name = uncover_county[(uncover_county['county'] == countyname) & (uncover_county['state'] =='New York')]

    return county_name



new_york_city = county_('New York City') #All Boroughs of New York City are under one county name: New York City

'''queens = county_('Queens')

#bronx = county_('Bronx')

#brooklyn = county_('Kings')

#staten_island = county_('Richmond')

'''

#new_york_counties = [manhattan, queens, bronx, brooklyn, staten_island]

new_york_list = ['New York City Cases', 'New York City Deaths']



plt.figure(1, figsize=(12,10))

plt.plot(new_york_city.index.values, new_york_city['cases'], lw=2)

plt.xlabel('Dates',)

plt.ylabel('Amount of Cases')

plt.legend(new_york_list)

plt.title('New York City (COVID-19 Amount of Cases Over Time))')





plt.show()











new_york_city_table = uncover_county[(uncover_county['county'] == 'New York City') & (uncover_county['state'] == 'New York')]

pd.set_option('display.max_columns', None)

new_york_city_table.tail(10)
plt.figure(2, figsize=(12,10))

plt.plot(new_york_city.index.values, new_york_city['deaths'], lw=2)

plt.xlabel('Dates',)

plt.ylabel('Amount of Deaths')



plt.legend(new_york_list)

plt.title('New York City (COVID-19 Amount of Deaths Over Time))')



plt.show()
new_list = ['Cases', 'Deaths']



plt.figure(3, figsize=(12,10))

plt.plot(new_york_city.index.values, new_york_city['cases'], lw=2)

plt.plot(new_york_city.index.values, new_york_city['deaths'], lw=2)

plt.xlabel('Dates',)

plt.ylabel('Amount of Cases/Deaths')



plt.legend(new_list)

plt.title('New York City (COVID-19 Amount of Cases vs Deaths)')



plt.show()
        

def fatality_rate(data_set):

    f_rate = (data_set['deaths'] / data_set['cases']) * 100

    return f_rate



#No column w/ fatilty rate, only here in this instance is it 'created'



names_lst = ['New York City', 'LA County']

county_initiation = [fatality_rate(new_york_city), fatality_rate(la_county)]



plt.figure(4, figsize=(16,14))

for county in county_initiation:

    plt.plot(county.index.values, county, lw=2)

    plt.xlabel('Dates',)

    plt.ylabel('Case Fatality Rate (%)')



plt.legend(names_lst)

plt.title('COVID-19 Case Fatality Rate')

plt.show()





#sbla will contain data that I need



plt.figure(5, figsize=(16,14))



for i in sbla:

    plt.plot(i.index.values, fatality_rate(i))

    plt.xlabel('Dates',)

    plt.ylabel('Case Fatality Rate (%)')



plt.legend(counties)

plt.title('COVID-19 Case Fatality Rate SOCAL')

plt.show()
