# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
!pip install pyzipcode
import numpy as np # linear algebra
import pandas as pd # data processiInng, CSV file I/O (e.g. pd.read_csv)
from pyzipcode import ZipCodeDatabase
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/us-population-by-zip-code/population_by_zip_2010.csv", dtype={'zipcode':'category'})
df = df.dropna()
df.head(5)
zip_df = pd.read_csv("/kaggle/input/zipcodes-county-fips-crosswalk/ZIP-COUNTY-FIPS_2017-06.csv", dtype={"ZIP":'category'})
zip_df.head(5)
pop_df = pd.read_csv("/kaggle/input/us-census-demographic-data/acs2015_county_data.csv")
pop_df.head(5)
income_df = pd.read_csv('/kaggle/input/kaggle-income/kaggle_income.csv',encoding='latin-1')
income_df = income_df.rename(columns={'State_ab':'State'})
income_df.head(5)
def zip_finder(CountNum):
    try:
        return (((zip_df[zip_df['STCOUNTYFP']==int(CountNum)])['ZIP']).values)[0]
    except:
        print(CountNum)
        return('None')
def Zip_2_City(zipval):
    zcdb = ZipCodeDatabase()
    if zipval != 'None':
        zipcode = zcdb[zipval]
        if zipcode.city != None:
            return zipcode.city, zipcode.state
        else: return "None",'None'
    else:
        return "None"
def City_State_list_maker(zip_list):   
    City_list = list()
    State_list = list()
    for zips in zip_list:
        if zips == 'None':
            City_list.append('None')
            State_list.append('None')
        else:
            try: 
                city, state = Zip_2_City(zips)
                City_list.append(city)
                State_list.append(state)
            except:
                City_list.append('None')
                State_list.append('None')
    return City_list, State_list
Mean_Income_df = income_df[['Mean','Median','Stdev','City','State']].groupby(['City','State'], as_index=False).mean()
Mean_Income_df.to_csv('Mean_Income_df.csv',index=False)
Mean_Income_df.head()
zip_array = list(df['zipcode'].values)
print(type(zip_array))
print(zip_array[:5])
City_list = list()
State_list = list()
City_list, State_list = City_State_list_maker(zip_array)
City_State_df = pd.DataFrame(data={'zipcode':zip_array,
                                  'City':City_list, 'State':State_list})
City_State_df.head()
new_df = df.merge(City_State_df, on='zipcode')
new_df = new_df[['population', 'minimum_age','maximum_age','City','State']]
pop_count = new_df[['population','City','State' ]].groupby(['City','State'], as_index=False).sum()
pop_min = new_df[['minimum_age','City','State' ]].groupby(['City','State'], as_index=False).min()
pop_max = new_df[['minimum_age','City','State' ]].groupby(['City','State'], as_index=False).max()
Population_df = pop_count.merge(pop_min, on=['State','City'])
Population_df = Population_df.merge(pop_max, on=['State','City'])
Population_df = Population_df.reset_index()
Population_df.to_csv('Population_df.csv',index=False)
Population_df.head()
County_names = pop_df['CensusId'].values
zip_list = list()
for i, county_name in enumerate(County_names):
    zip_list.append(zip_finder(county_name))
print(np.where(County_names==2158))
print(np.where(County_names==46102))
zip_list[81] == 'None'
zip_list[2412] == 'None'
City_list = list()
State_list = list()
City_list, State_list = City_State_list_maker(zip_list)
sum(x is 'None' for x in State_list) == sum(x is 'None' for x in City_list)
Zip_and_city_county = pd.DataFrame(data={'City':City_list,'State':State_list, 'Zipcode':zip_list, 'CensusId': County_names})
Zip_and_city_county = Zip_and_city_county.replace(to_replace='None', value=np.nan)
pop_complete_df =  pop_df[pop_df.columns[pop_df.columns != 'State']].merge(Zip_and_city_county, on='CensusId')
print(pop_complete_df.shape)
print(pop_complete_df['Zipcode'].iloc[81])
noneless_df = pop_complete_df.dropna()
print(noneless_df.shape)
print(noneless_df['Zipcode'].iloc[81])
noneless_df.columns
Diversity_df = noneless_df[['Hispanic','White', 'Black', 'Native', 'Asian', 'Pacific','City','State']].groupby(['City','State'],as_index=False).mean()
Diversity_df.to_csv("Diversity_df.csv",index=False)
Diversity_df.head()
Income_per_capita_df = noneless_df[['Income','IncomeErr', 'IncomePerCap', 'IncomePerCapErr','City','State']].groupby(['City','State'],as_index=False).mean()
Income_per_capita_df.to_csv("Income_per_capita_df.csv",index=False)
Income_per_capita_df.head()
Poverty_df = noneless_df[['Poverty','ChildPoverty','City','State']].groupby(['City','State'],as_index=False).sum()
Poverty_df.to_csv("Poverty_df.csv",index=False)
Poverty_df.head()
Industry_df = noneless_df[['Professional', 'Service', 'Office', 'Construction','Production','City','State']].groupby(['City','State'],as_index=False).sum()
Industry_df.to_csv("Industry_df.csv",index=False)
Industry_df.head()
Trasportation_df = noneless_df[['Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp','WorkAtHome', 'MeanCommute','City','State']].groupby(['City','State'],as_index=False).mean()
Trasportation_df.to_csv("Trasportation_df.csv",index=False)
Trasportation_df.head()
Employment_df = noneless_df[['Employed','Unemployment','City','State']].groupby(['City','State'],as_index=False).sum()
Employment_df.to_csv('Employment_df.csv',index=False)
Employment_df.head()
Employment_ratio_df = noneless_df[['PrivateWork', 'PublicWork','SelfEmployed', 'FamilyWork','City','State']].groupby(['City','State'],as_index=False).mean()
Employment_ratio_df.to_csv('Employment_ratio_df.csv',index=False)
Employment_ratio_df.head()
total_df = Population_df.merge(Diversity_df, on=['State','City'])
total_df = total_df.merge(Poverty_df, on=['State','City'])
total_df = total_df.merge(Industry_df, on=['State','City'])
total_df = total_df.merge(Employment_df, on=['State','City'])
total_df = total_df.merge(Trasportation_df, on=['State','City'])
total_df = total_df.merge(Employment_ratio_df, on=['State','City'])
total_df = total_df.merge(Income_per_capita_df, on=['State','City'])
total_df = total_df.merge(Mean_Income_df, on=['State','City'])
total_df.to_csv('Total_Search_df.csv',index=False)
total_df.head(5)