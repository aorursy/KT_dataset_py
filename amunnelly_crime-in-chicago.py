import pandas as pd

import matplotlib.pyplot as plt

import datetime as datetime

import json

from collections import defaultdict

%matplotlib inline
data = pd.read_csv("../input/Chicago_Crimes_2012_to_2017.csv")
# Ideally loaded from .json, but I'm not sure that's possible in this environment.

areas = {

    "1.0": "Rogers Park",

"2.0": "West Ridge",

"3.0": "Uptown",

"4.0": "Lincoln Square",

"5.0": "North Center",

"6.0": "Lakeview",

"7.0": "Lincoln Park",

"8.0": "Near North Side",

"9.0": "Edison Park",

"10.0": "Norwood Park",

"11.0": "Jefferson Park",

"12.0": "Forest Glen",

"13.0": "North Park",

"14.0": "Albany Park",

"15.0": "Portage Park",

"16.0": "Irving Park",

"17.0": "Dunning",

"18.0": "Montclare",

"19.0": "Belmont Cragin",

"20.0": "Hermosa",

"21.0": "Avondale",

"22.0": "Logan Square",

"23.0": "Humboldt Park",

"24.0": "West Town",

"25.0": "Austin",

"26.0": "West Garfield Park",

"27.0": "East Garfield Park",

"28.0": "Near West Side",

"29.0": "North Lawndale",

"30.0": "South Lawndale",

"31.0": "Lower West Side",

"32.0": "Loop",

"33.0": "Near South Side",

"34.0": "Armour Square",

"35.0": "Douglas",

"36.0": "Oakland",

"37.0": "Fuller Park",

"38.0": "Grand Boulevard",

"39.0": "Kenwood",

"40.0": "Washington Park",

"41.0": "Hyde Park",

"42.0": "Woodlawn",

"43.0": "South Shore",

"44.0": "Chatham",

"45.0": "Avalon Park",

"46.0": "South Chicago",

"47.0": "Burnside",

"48.0": "Calumet Heights",

"49.0": "Roseland",

"50.0": "Pullman",

"51.0": "South Deering",

"52.0": "East Side",

"53.0": "West Pullman",

"54.0": "Riverdale",

"55.0": "Hegewisch",

"56.0": "Garfield Ridge",

"57.0": "Archer Heights",

"58.0": "Brighton Park",

"59.0": "McKinley Park",

"60.0": "Bridgeport",

"61.0": "New City",

"62.0": "West Elsdon",

"63.0": "Gage Park",

"64.0": "Clearing",

"65.0": "West Lawn",

"66.0": "Chicago Lawn",

"67.0": "West Englewood",

"68.0": "Englewood",

"69.0": "Greater Grand Crossing",

"70.0": "Ashburn",

"71.0": "Auburn Gresham",

"72.0": "Beverly",

"73.0": "Washington Heights",

"74.0": "Mount Greenwood",

"75.0": "Morgan Park",

"76.0": "O'Hare",

"77.0": "Edgewater"

}
data['Community Area'] = data['Community Area'].astype(str)

data['Area Name'] = data['Community Area'].map(areas)
data['Primary Type'].value_counts()/data.shape[0]
homicide = data[data['Primary Type'] == 'HOMICIDE'].copy()
homicide['Date2'] = homicide['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p'))

homicide['Updated2'] = homicide['Updated On'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p'))



homicide = homicide.drop(['Unnamed: 0',

                  'X Coordinate',

                  'Y Coordinate',

                  'Location',

                  'Date',

                  'Updated On'], axis = 1)



homicide.rename(columns = {'Date2':'Date',

            'Updated2':'Updated On'},

            inplace=True) 
bins = homicide[['Description', 'Year', 'Date']].copy()

title = 'Chicago Murder Count, 2012-2016'



def plot_crime_by_Year(bins, title):

    '''

    (dataframe, string) -> None

    Construct a line plot from the dataframe dataframe with title title.

    '''

    colors = ['#fe9929','#ec7014','#cc4c02','#993404','#662506']

    i = 0

    bins = bins[bins.Year < 2017]

    years = bins.groupby('Year')

    for a, b in years:

        b.index = b.Date

        aYear = b.resample('W').count()

        aYear.index = range(aYear.shape[0])

        plt.plot(aYear.Description.cumsum(), color=colors[i], label = a)

        i += 1

    plt.grid()

    plt.legend(loc = 'upper left')

    plt.title(title)

    plt.xlabel('Weeks')

    plt.ylabel('Cumulative Count')



plot_crime_by_Year(bins, title)
homicide_for_heatmap = homicide[homicide.Year < 2017].copy()



def plot_heatmap(data_row, data_column, title):

    '''

    (pd.Series, pd.Series, str) -> None

    Construct a pd.crosstab data frame from the two series and use that to plot a heatmap

    '''

    table = pd.crosstab(data_row, data_column)

    fig, ax = plt.subplots(1, 1, figsize = (15, 8))

    map = ax.imshow(table,

                   cmap='cool',

                   interpolation='nearest')

    plt.xticks(range(77), range(1,78))

    plt.yticks(range(5), range(2012, 2017))

    ax.set_title(title)

    ax.set_xlabel('Community Area')

    ax.set_ylabel('Year')

    plt.colorbar(map,

             orientation= 'horizontal',

             shrink = 0.25,

            pad = 0.10);



plot_heatmap(homicide['Year'], homicide['Community Area'], 'Heatmap of Chicago Murders')
homicide_area_year_group = homicide.groupby(['Year','Area Name'])

        

def top_ten_sorter(group):

    year_area = defaultdict(int)

    for a, b in group:

        year_area[a] = b.shape[0]



    counts = set(year_area.values())

    count_by_year_area = defaultdict(list)

    for c in counts:

        for a, b in year_area.items():

            if b == c:

                count_by_year_area[c].append(a)



    print ("COUNT YEAR  COMMUNITY AREA")

    for c in sorted(counts, reverse=True)[:10]:

        for item in count_by_year_area[c]:

            print ("{:5} {:4}  {:25}".format(c, item[0], item[1]))

        

top_ten_sorter(homicide_area_year_group)
theft = data[data['Primary Type']=='THEFT'].copy()



theft['Date2'] = theft['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p'))

theft['Updated2'] = theft['Updated On'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p'))



theft = theft.drop(['Unnamed: 0',

                  'X Coordinate',

                  'Y Coordinate',

                  'Location',

                  'Date',

                  'Updated On'], axis = 1)



theft.rename(columns = {'Date2':'Date',

            'Updated2':'Updated On'},

            inplace=True)
bins = theft[['Description', 'Year', 'Date']].copy()

plot_crime_by_Year(bins, 'Theft in Chicago, 2012-2016')
theft_for_crosstab = theft[theft.Year < 2017].copy()

theft_area_v_year = pd.crosstab(theft_for_crosstab.Year, theft_for_crosstab['Community Area']).copy()



plot_heatmap(theft_for_crosstab.Year, theft_for_crosstab['Community Area'], "HeatMap of Chicago Theft")
theft_by_year_area_group = theft.groupby(['Year', 'Area Name'])

top_ten_sorter(theft_by_year_area_group)
theft.Description.value_counts()/theft.shape[0]
cars = data[data['Primary Type']=='MOTOR VEHICLE THEFT'].copy()



cars['Date2'] = cars['Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p'))

cars['Updated2'] = cars['Updated On'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p'))



cars = cars.drop(['Unnamed: 0',

                  'X Coordinate',

                  'Y Coordinate',

                  'Location',

                  'Date',

                  'Updated On'], axis = 1)



cars.rename(columns = {'Date2':'Date',

            'Updated2':'Updated On'},

            inplace=True)
bins = cars[['Description', 'Year', 'Date']].copy()

title = 'Car Theft in Chicago, 2012-2016'

plot_crime_by_Year(bins, title)
cars_ct = cars[cars.Year < 2017].copy()

cars_area_v_year = pd.crosstab(cars_ct['Year'], cars_ct['Community Area']).copy()

plot_heatmap(cars_ct['Year'], cars_ct['Community Area'], 'Heatmap of Chicago Car Theft')
cars_area_by_year_group = cars.groupby(['Year', 'Area Name'])

top_ten_sorter(cars_area_by_year_group)