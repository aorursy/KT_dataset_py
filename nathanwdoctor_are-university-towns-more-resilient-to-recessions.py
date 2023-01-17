import pandas as pd

import numpy as np

from scipy.stats import ttest_ind

# %cd "C:\Users\ndoctor\PYTHON_CODING\Data"
states_dict = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 

               'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 

               'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 

               'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 

               'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 

               'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 

               'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 

               'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 

               'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 

               'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 

               'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas','NY': 'New York', 

               'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 

               'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 

               'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 

               'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 

               'ND': 'North Dakota', 'VA': 'Virginia'}
colnames = ['City', 'State']

univ_towns = pd.read_csv('../input/university-towns/university_towns.txt', sep='\n', header=None, names=colnames)
univ_towns.head()
univ_towns['City'] = univ_towns['City'].str.replace('\[.*$','')
univ_towns['City'] = univ_towns['City'].str.replace('\ \(.*$','')
univ_towns.head()
univ_towns['State'] = univ_towns.loc[univ_towns['City'].isin(states_dict.values())]['City']
univ_towns.head()
univ_towns['City'] = univ_towns.loc[~univ_towns['City'].isin(states_dict.values())]['City']
univ_towns['State'] = univ_towns['State'].ffill()
univ_towns
univ_towns = univ_towns.dropna()
univ_towns = univ_towns.set_index(['State','City'])
gdp = pd.read_excel('../input/university-towns/gdplev.xls', skiprows=219)

gdp.head()
gdp = gdp.rename(columns={'1999q4': 'Quarter', 12323.3:'GDP'})

gdp = gdp[['Quarter','GDP']]
gdp.head()
def get_recession_start():

    i = 2

    length = len(gdp['GDP'])

    while i < length:

        if gdp.GDP[i] < gdp.GDP[i-1] and gdp.GDP[i-1] < gdp.GDP[i-2]:

            return gdp.Quarter[i-1]

        i += 1



get_recession_start()
def get_recession_end():

    i = 2

    length = len(gdp['GDP'])

    in_recession = False

    while i < length:

        if gdp.GDP[i] < gdp.GDP[i-1] and gdp.GDP[i-1] < gdp.GDP[i-2]:

            in_recession = True

        if in_recession == True and gdp.GDP[i] > gdp.GDP[i-1] and gdp.GDP[i-1] > gdp.GDP[i-2]:

            return gdp.Quarter[i]

        i += 1

    return recession_end



get_recession_end()
def get_recession_bottom():

    i = 2

    length = len(gdp['GDP'])

    in_recession = False

    lowest_gdp = np.inf

    while i < length:

        if gdp.GDP[i] < gdp.GDP[i-1] and gdp.GDP[i-1] < gdp.GDP[i-2]:

            in_recession = True

        if in_recession == True and gdp.GDP[i] < lowest_gdp:

            lowest_gdp = gdp.GDP[i]

            return gdp.Quarter[i]

        if in_recession == True and gdp.GDP[i] > gdp.GDP[i-1] and gdp.GDP[i-1] > gdp.GDP[i-2]:

            in_recession = False

        i+=1

    return recession_bottom



get_recession_bottom()
def convert_housing_data_to_quarters():

    """

    Converts the housing data to quarters and returns it as mean 

    values in a dataframe. 

    """

    

    housing_data = pd.read_csv('../input/university-towns/City_Zhvi_AllHomes.xls',encoding='latin-1')

    housing_data['State'].replace(states_dict, inplace=True)

    housing_data = housing_data.set_index(["State","RegionName"])

    housing_data = housing_data.iloc[:,49:250]

    

    def quarters(col):

        if col.endswith(("01", "02", "03")):

            s = col[:4] + "q1"

        elif col.endswith(("04", "05", "06")):

            s = col[:4] + "q2"

        elif col.endswith(("07", "08", "09")):

            s = col[:4] + "q3"

        else:

            s = col[:4] + "q4"

        return s  

    

    housing_data = housing_data.groupby(quarters, axis=1).mean()

    housing_data = housing_data.sort_index()

    return housing_data



convert_housing_data_to_quarters()
def run_ttest():

    housing_data = convert_housing_data_to_quarters()

    better = None

    isin_list = housing_data.index.isin(univ_towns.index).astype(bool)

    housing_data.insert(loc=0, column='isit_univ', value=isin_list)

    

    univ_town = housing_data[housing_data['isit_univ']==True]

    not_univ_town = housing_data[housing_data['isit_univ']==False]

    

    univ_ratio = univ_town[get_recession_start()] / univ_town[get_recession_bottom()]

    not_univ_ratio = not_univ_town[get_recession_start()] / not_univ_town[get_recession_bottom()]

    p_value = ttest_ind(univ_ratio.dropna(), not_univ_ratio.dropna())[1]

    different = p_value < .01

    

    if univ_ratio.mean() < not_univ_ratio.mean():

        better = 'university town'

    else:

        better = 'non-university town'

    return (different, p_value, better)



run_ttest()