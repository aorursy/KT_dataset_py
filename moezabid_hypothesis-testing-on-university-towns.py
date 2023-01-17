import pandas as pd

import numpy as np

from scipy.stats import ttest_ind

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading Data

all_homes_at_a_city_level = pd.read_csv("/kaggle/input/zillow-all-homes-data/City_Zhvi_AllHomes.csv")

gdp_over_time = pd.read_excel("/kaggle/input/zillow-all-homes-data/gdplev.xls", skiprows=7)

universtiy_towns = pd.read_csv("/kaggle/input/zillow-all-homes-data/university_towns.txt", sep="\t", header=None)
all_homes_at_a_city_level.head()
gdp_over_time.head()
universtiy_towns.head()
# We will use this dictionary to map state names to two letter acronyms

states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
def get_list_of_university_towns():

    """Returns a DataFrame of towns and the states they are in from the 

    university_towns.txt list. The format of the DataFrame should be:

    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 

    columns=["State", "RegionName"]  )

    

    The following cleaning needs to be done:



    1. For "State", removing characters from "[" to the end.

    2. For "RegionName", when applicable, removing every character from " (" to the end."""

    

    with open('/kaggle/input/zillow-all-homes-data/university_towns.txt') as f:

        unviersity_towns = f.readlines()

    data = []

    for line in unviersity_towns:

        uni = line[:-1] # Fallback in case it isn't a state and doesn't have a " ("

        if line[-7:] == '[edit]\n':

            state = line[:-7]

            continue

        elif '(' in line:

            uni = line[:line.find('(') - 1]



        data.append((state, uni))

    

    return pd.DataFrame(data=data, columns=['State', 'RegionName'])
get_list_of_university_towns().head()
def get_recession_start():

    '''Returns the year and quarter of the recession start time as a 

    string value in a format such as 2005q3'''    

    df = gdp_over_time[['Unnamed: 4','Unnamed: 5']]

    df.columns = ['Quarter','GDP']

    # We are only going to look for data from the first quarter of 2000 onward so :

    df = df.iloc[212:]

    recession_start = ""

    recession = list()

    recession_start = []

    for i in range(len(df) - 4):

        if ((df.iloc[i][1] > df.iloc[i+1][1]) & (df.iloc[i+1][1] > df.iloc[i+2][1])):

            recession_start.append(df.iloc[i][0])

    return recession_start[0]
get_recession_start()
def get_recession_end():

    '''Returns the year and quarter of the recession end time as a 

    string value in a format such as 2005q3'''

    df = gdp_over_time[['Unnamed: 4','Unnamed: 5']]

    df.columns = ['Quarter','GDP']

    # We are only going to look for data from the first quarter of 2000 onward so :

    df = df.iloc[212:]

    recession_end = ""

    for i in range(len(df)-4):

        if((df.iloc[i][1] > df.iloc[i+1][1]) & (df.iloc[i+1][1] > df.iloc[i+2][1]) & (df.iloc[i+3][1] > df.iloc[i+2][1]) & (df.iloc[i+4][1] > df.iloc[i+3][1])):

            recession_end = df.iloc[i+4][0]

    return recession_end
get_recession_end()
def get_recession_bottom():

    '''Returns the year and quarter of the recession bottom time as a 

    string value in a format such as 2005q3'''

    df = gdp_over_time[['Unnamed: 4','Unnamed: 5']]

    df.columns = ['Quarter','GDP']

    # We are only going to look for data from the first quarter of 2000 onward so :

    df = df.iloc[212:]

    recession_bottom = ""

    recession = list()

    for i in range(len(df)-4):

        if((df.iloc[i][1] > df.iloc[i+1][1]) & (df.iloc[i+1][1] > df.iloc[i+2][1]) & (df.iloc[i+3][1] > df.iloc[i+2][1]) & (df.iloc[i+4][1] > df.iloc[i+3][1])):

            recession.append([df.iloc[i][0], df.iloc[i+1][0], df.iloc[i+2][0], df.iloc[i+3][0], df.iloc[i+4][0]])

            # We will select the element with index 2 in the recession array since it represents the bottom of the recession.

            recession_bottom = recession[0][2]

    return recession_bottom
get_recession_bottom()
def convert_housing_data_to_quarters():

    '''Converts the housing data to quarters and returns it as mean 

    values in a dataframe. This dataframe should be a dataframe with

    columns for 2000q1 through 2016q3, and should have a multi-index

    in the shape of ["State","RegionName"].

    '''

    

    # Getting rid of the RegionID, and the years before the year 2000

    df = all_homes_at_a_city_level.drop(all_homes_at_a_city_level.columns[[0] + list(range(3,51))], axis=1)

    # Replacing all the states names

    df["State"] = df["State"].replace(states)

    df.set_index(["State", "RegionName"], inplace=True)

   

    # Createing a new DataFrame that contains means for each quarter from 2000 to 2015.

    new_data = pd.DataFrame()

    # We are not adding the year 2016 because we don't have all the 4 quarters data available.

    for year in range(2000,2016):

        new_data[str(year) + 'q1'] = df[[str(year) + '-01', str(year) + '-02', str(year) + '-03']].mean(axis = 1)

        new_data[str(year) + 'q2'] = df[[str(year) + '-04', str(year) + '-05', str(year) + '-06']].mean(axis = 1)

        new_data[str(year) + 'q3'] = df[[str(year) + '-07', str(year) + '-08', str(year) + '-09']].mean(axis = 1)

        new_data[str(year) + 'q4'] = df[[str(year) + '-10', str(year) + '-11', str(year) + '-12']].mean(axis = 1)

    # Now adding the year 2016's remaining quarters since they have not been added in the previous loop

    new_data[str(2016) + 'q1'] = df[[str(2016) + '-01', str(year) + '-02', str(2016) + '-03']].mean(axis = 1)

    new_data[str(2016) + 'q2'] = df[[str(2016) + '-04', str(year) + '-05', str(2016) + '-06']].mean(axis = 1)

    new_data[str(2016) + 'q3'] = df[[str(2016) + '-07', str(year) + '-08']].mean(axis = 1)

    return new_data
#convert_housing_data_to_quarters().loc["Texas"].loc["Austin"].loc["2010q3"]

convert_housing_data_to_quarters()
def run_ttest():

    '''First creates new data showing the decline or growth of housing prices

    between the recession start and the recession bottom. Then runs a ttest

    comparing the university town values to the non-university towns values, 

    return whether the alternative hypothesis (that the two groups are the same)

    is true or not as well as the p-value of the confidence. 

    

    Returns the tuple (different, p, better) where different=True if the t-test is

    True at a p<0.01 (we reject the null hypothesis), or different=False if 

    otherwise (we cannot reject the null hypothesis). The variable p should

    be equal to the exact p value returned from scipy.stats.ttest_ind(). The

    value for better should be either "university town" or "non-university town"

    depending on which has a lower mean price ratio (which is equivilent to a

    reduced market loss).'''

    

     

    unitowns = get_list_of_university_towns()

    bottom = get_recession_bottom()

    start = get_recession_start()

    housing_data = convert_housing_data_to_quarters()

    # Selecting the quarter before the recession

    bstart = housing_data.columns[housing_data.columns.get_loc(start) - 1]

        

    """ 

    The formula for price ratio is :

    price_ratio=quarter_before_recession/recession_bottom

    """

    housing_data['ratio'] =  housing_data[bstart] / housing_data[bottom]

    housing_data = housing_data[[bottom, bstart, 'ratio']]

    housing_data = housing_data.reset_index()

    

    unitowns_hdata = pd.merge(housing_data,unitowns,how='inner',on=['State','RegionName'])

    unitowns_hdata['IsUniversityTown'] = True

    housing_data_complete = pd.merge(housing_data, unitowns_hdata, how='outer', on=['State','RegionName',bottom, bstart, 'ratio'])

    housing_data_complete['IsUniversityTown'] = housing_data_complete['IsUniversityTown'].fillna(False)

    

    university_towns = housing_data_complete[housing_data_complete['IsUniversityTown'] == True]

    non_universtiy_towns = housing_data_complete[housing_data_complete['IsUniversityTown'] == False]

    

    # Executing the Hypothesis test all while dropping the na values

    t,p = ttest_ind(university_towns['ratio'].dropna(), non_universtiy_towns['ratio'].dropna())

    # We are comparing the p value to 0.01 in our case

    different_prices = True if p<0.01 else False

    # Better contains the tag of the family of towns that have the better prices

    better_family = "university town" if university_towns['ratio'].mean() < non_universtiy_towns['ratio'].mean() else "non-university town"

    

    return (different_prices, p, better_family)
run_ttest()