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
import pandas as pd
import numpy as np
#hypothesis test 
#university towns data cleaning
uni = pd.read_table('../input/zillow-all-homes-data/university_towns.txt',header = None, names = ['name'])
uni.name = uni.name.str.split('(',expand = True)[0]
mask = uni.name.str.contains('edit')
index = mask[mask == True].index
state = []
for i in uni.index:
    if i in index:
        state.append(uni.loc[i]['name'])
    if i not in index:
        state.append(state[i-1])
uni['state'] = state
uni.drop(index, inplace = True)
uni.state = uni.state.str.split('[', expand = True)[0]
uni = uni.rename(columns = {'name':'RegionName', 'state':'State'})
uni['RegionName'] = uni.RegionName.apply(lambda x: x.strip())
#drive recession period after 2000
gdp = pd.read_excel('../input/zillow-all-homes-data/gdplev.xls', skiprows = [0,1,2,3,4,5,6,7],header = None, 
                    usecols = [4,5], names = ['date','gdp'])
gdp = gdp.iloc[gdp[gdp['date'] == '2000q1'].index[0]:]
#find recession start
for i in range(len(gdp)-3):
    if gdp.iloc[i][1]<gdp.iloc[i+1][1] and gdp.iloc[i+1][1]>gdp.iloc[i+2][1] and gdp.iloc[i+2][1]>gdp.iloc[i+3][1]:
        recession_start = gdp.iloc[i+1][0]
gdp = gdp.reset_index().set_index('date').drop('index', axis = 1)
#find recession bottom and recession end 
loca = recession_start[0]
for i in range(len(gdp)-4):
    if gdp.iloc[i][0]>gdp.iloc[i+1][0] and gdp.iloc[i+1][0]>gdp.iloc[i+2][0] and gdp.iloc[i+2][0]<gdp.iloc[i+3][0] and gdp.iloc[i+3][0]<gdp.iloc[i+4][0]:
        recession_bottom  = gdp.iloc[i+2].name
        recession_end = gdp.iloc[i+4].name
print(recession_start, recession_bottom, recession_end)
#housing data
housing = pd.read_csv('../input/zillow-all-homes-data/City_Zhvi_AllHomes.csv')
housing = housing.set_index(['State','RegionName']).drop(['RegionID','CountyName','Metro','SizeRank'], axis = 1)
housing = housing.T
housing.index = pd.to_datetime(housing.index, format = '%Y/%m')
housing = housing['2000':]
#change the date into quarter format
housing = housing.resample('q',label = 'right',closed = 'left').mean()
housing.index = housing.index.strftime('%Y')
quarter = []
for i in range(len(housing)):
    if i%4 == 0:
        quarter.append('q1')
    if i%4 == 1:
        quarter.append('q2')
    if i%4 == 2:
        quarter.append('q3')
    if i%4 == 3:
        quarter.append('q4')
housing.index = housing.index+quarter
housing = housing.T
#map state into full name
housing = housing.reset_index()
states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}
def mapping(state):
    for i in states.keys():
        if state == i:
            state = states[i]
            return state
housing['State'] = housing.State.apply(mapping)
housing.head()
#split housing data into university town and non-university_town
uni_housing = pd.merge(uni, housing, right_on = ['State', 'RegionName'], left_on=['State','RegionName'], how = 'inner')
to_be_dropped = uni_housing.set_index(['State','RegionName']).index
uni_housing = uni_housing.set_index(['State','RegionName'])
non_uni_housing = housing.set_index(['State','RegionName']).drop(to_be_dropped)
#get data in recession period
cols_to_keep = [recession_start, recession_bottom]
uni_housing_re = uni_housing[cols_to_keep]
non_uni_housing_re = non_uni_housing[cols_to_keep]
uni_housing_re['diff'] = uni_housing_re[recession_start] - uni_housing_re[recession_bottom]
uni_housing_re = uni_housing_re.dropna()
non_uni_housing_re = non_uni_housing_re.dropna()
non_uni_housing_re['diff'] = non_uni_housing_re[recession_start] - non_uni_housing_re[recession_bottom]
uni_housing_re.head()
#T test between university town housing data and non-university housing data during recession period, 
#if p value is larger than 0.05, then means of both have no difference
import scipy.stats as stats
p = stats.ttest_ind(uni_housing_re['diff'], non_uni_housing_re['diff']).pvalue
t = stats.ttest_ind(uni_housing_re['diff'], non_uni_housing_re['diff']).statistic
print(t,p)

