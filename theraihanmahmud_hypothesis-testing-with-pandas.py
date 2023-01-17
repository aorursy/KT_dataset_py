# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def get_list_of_university_towns():

    '''This function returns a dataFrame of towns and the states they are in from the 

    university_towns.txt list. The format of the DataFrame is:

    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 

    columns=["State", "RegionName"]  )

    

    The following cleaning  will be done:



    1. For "State", removing characters from "[" to the end.

    2. For "RegionName", when applicable, removing every character from " (" to the end. '''



    uni_towns = pd.read_fwf('/kaggle/input/university-towns-of-usa/university_towns.txt',names=['RegionName']) #it's easy to convert text data to dataframe using read_fwf. It does all the formatting.

    

    #uni_towns = pd.read_csv('/kaggle/input/university-towns-of-usa/university_towns.txt',sep='\n', header=None, names=['RegionName']) # you can use this format too

        



    uni_towns['State'] = np.where(uni_towns['RegionName'].str.contains('edit'),uni_towns['RegionName'],np.nan) # In the test file there state names and university town names but they aren't classified. There is a word 'edit' after every state name . So we will create a new column 'State' and separate all the state names using the condition.

    

    uni_towns['State'].fillna(method='ffill',inplace=True)



    uni_towns=uni_towns[['State','RegionName']]



    uni_towns=uni_towns[uni_towns['State'] != uni_towns['RegionName']]



    for col in uni_towns:

        uni_towns[col]= uni_towns[col].str.replace(r"\(.*\)","").str.replace(r"\[.*\]","").str.rstrip()



    return uni_towns

get_list_of_university_towns()
def get_recession_start():

    '''Returns the year and quarter of the recession start time as a 

    string value in a format such as 2005q3'''

    gdp = pd.read_excel('/kaggle/input/gdp-of-different-towns-in-usa/gdplev.xls',header=None)

    gdp=gdp.dropna(how='all')

    gdp=gdp.dropna(axis=1,how='all')

    #gdp=gdp.drop(axis=7,columns=7)

    gdp=(gdp[gdp.index>219].drop(axis=1,columns=[0,1,2])).reset_index().drop(columns=['index',5]).rename(columns={4:'Quarter',6:'GDP'})



    gdp['GDP_dec']=gdp['GDP'].diff()

    gdp = gdp[gdp['GDP_dec']<0]

    gdp=gdp.reset_index()

    recession_start = gdp['Quarter'][(gdp['index'].diff().idxmin())-1]

    

    

    return recession_start

get_recession_start()
def get_recession_end():

    '''Returns the year and quarter of the recession end time as a 

    string value in a format such as 2005q3'''

    gdp = pd.read_excel('/kaggle/input/gdp-of-different-towns-in-usa/gdplev.xls',header=None)

    gdp=gdp.dropna(how='all')

    gdp=gdp.dropna(axis=1,how='all')

    #gdp=gdp.drop(axis=7,columns=7)

    gdp=(gdp[gdp.index>219].drop(axis=1,columns=[0,1,2])).reset_index().drop(columns=['index',5]).rename(columns={4:'Quarter',6:'GDP'})



    gdp['GDP_dec']=gdp['GDP'].diff()

    gdp = gdp[gdp['GDP_dec']>0]

    gdp=gdp.reset_index()

    recession_end=gdp['Quarter'].values[gdp['index'].diff().idxmax() + 1]

    return recession_end

get_recession_end()
def get_recession_bottom():

    '''Returns the year and quarter of the recession bottom time as a 

    string value in a format such as 2005q3'''

    gdp = pd.read_excel('/kaggle/input/gdp-of-different-towns-in-usa/gdplev.xls',header=None)

    gdp=gdp.dropna(how='all')

    gdp=gdp.dropna(axis=1,how='all')

    #gdp=gdp.drop(axis=7,columns=7)

    gdp=(gdp[gdp.index>219].drop(axis=1,columns=[0,1,2])).reset_index().drop(columns=['index',5]).rename(columns={4:'Quarter',6:'GDP'})

    rec_st = get_recession_start()

    rec_end= get_recession_end()

    gdp = gdp.iloc[(gdp.index[gdp['Quarter']== rec_st].tolist()[0]) : (gdp.index[gdp['Quarter']== rec_end].tolist()[0])]

    bottom = gdp['Quarter'].values[gdp['GDP'].values.argmin()]

    return bottom

get_recession_bottom()
def convert_housing_data_to_quarters():

    '''Converts the housing data to quarters and returns it as mean 

    values in a dataframe. This dataframe should be a dataframe with

    columns for 2000q1 through 2016q3, and should have a multi-index

    in the shape of ["State","RegionName"].

    

    Note: Quarters are defined in the assignment description, they are

    not arbitrary three month periods.

    

    The resulting dataframe should have 67 columns, and 10,730 rows.

    '''

    house_data = pd.read_csv('/kaggle/input/housing-price-in-different-towns-of-usa/City_Zhvi_AllHomes.csv')



    house_data['State']=house_data['State'].replace({'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'})



    house_data.drop(['RegionID','Metro','CountyName','SizeRank'],axis=1,inplace=True)



    house_data=house_data.drop(house_data.loc[:,'1996-04':'1999-12'],axis=1)



    house_data=house_data.set_index(['State','RegionName']).sort_values(by=['State','RegionName'])



    def change_to_quarter(date:str):

        date=date.split('-')

        month = int(date[1])

        quart = int((month-1)/3)+1

        return date[0]+'q'+str(quart)



    house_data=house_data.groupby(change_to_quarter,axis=1).mean()



    return house_data

convert_housing_data_to_quarters()
from scipy.stats import ttest_ind
def run_ttest():

    '''First creates new data showing the decline or growth of housing prices

    between the recession start and the recession bottom. Then runs a ttest

    comparing the university town values to the non-university towns values, 

    return whether the alternative hypothesis (that the two groups are the same)

    is true or not as well as the p-value of the confidence. 

    

    Return the tuple (different, p, better) where different=True if the t-test is

    True at a p<0.01 (we reject the null hypothesis), or different=False if 

    otherwise (we cannot reject the null hypothesis). The variable p should

    be equal to the exact p value returned from scipy.stats.ttest_ind(). The

    value for better should be either "university town" or "non-university town"

    depending on which has a lower mean price ratio (which is equivilent to a

    reduced market loss).'''

     

    uni_towns = get_list_of_university_towns()

    house_data = convert_housing_data_to_quarters()



    uni_towns=uni_towns.set_index(['State','RegionName'])



    start = get_recession_start()

    bottom = get_recession_bottom()

    bef_start =house_data.columns[house_data.columns.get_loc(start) -1]





    house_data=house_data.loc[:,bef_start:bottom]

    house_data['ratio'] = house_data[bottom]-house_data[bef_start]





    unid = pd.merge(house_data,uni_towns,how='inner',left_index=True,right_index=True)

    nounid= house_data.drop(unid.index)









    t,p = ttest_ind(unid['ratio'].dropna(),nounid['ratio'].dropna())



    different = True if p < 0.01 else False



    better = "non-university town" if unid['ratio'].mean() < nounid['ratio'].mean() else "university town"

    return different, p, better



run_ttest()