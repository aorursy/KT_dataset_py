# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



## 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





index_by = ['city','state' ]


geo_data = pd.read_csv("../input/state-fipsstatestate-abbrzipcodecountycity/geo-data.csv")[['state', 'zipcode','city']]

#geo_data['city_comma_state'] = geo_data.apply(lambda row: str(row.city) + ", " + str(row.state), axis = 1)

city_comma_state_zips = geo_data[['city','state','zipcode']]

city_comma_state_zips.set_index(['city'],inplace=True)



city_comma_state_zips.loc['New york',:]
def get_city(city_comma_state):

    [city,_] =  city_comma_state.split(', ')

    return city

def get_state(city_comma_state):

    [_,state] =  city_comma_state.split(', ')

    return state

    

car_ownership = pd.read_csv("../input/car-ownership/car-ownership.csv")

car_ownership['city'] = car_ownership.apply(lambda row: get_city(row.city_comma_state), axis = 1)

car_ownership['state'] = car_ownership.apply(lambda row: get_state(row.city_comma_state), axis = 1)

car_ownership.drop(['city_comma_state'],axis=1, inplace=True)

car_ownership.reset_index()

car_ownership.set_index(['city'],inplace=True)



car_ownership

#state


city_stat_2 = pd.read_csv("../input/us-metropolitan-population-density-2016/city_stat_2.csv")[['city', 'State','population_density']]

#city_stat_2['city_comma_state'] = city_stat_2.apply(lambda row: row.city + ", " + row.State, axis = 1)

city_stat_2['state'] = city_stat_2.apply(lambda row: row.State, axis = 1)

population_density = city_stat_2[['city','state','population_density']]

population_density.set_index(['city'],inplace=True)



#state['state']

population_density

rc = population_density.merge(car_ownership, on= index_by) #.drop(['city_comma_state'],axis=1, inplace=True)

rc
kaggle_income = pd.read_csv("../input/kaggle-income/kaggle_income.csv")[['State_Name', 'City','Median']].groupby(['State_Name', 'City']).agg(np.mean).add_suffix('_Income').reset_index()

kaggle_income['city'] = kaggle_income.apply(lambda row: row.City, axis = 1) 

kaggle_income['state'] = kaggle_income.apply(lambda row: row.State_Name, axis = 1)

city_income = kaggle_income[['city','state','Median_Income']]

city_income.set_index(['city'],inplace=True)

city_income.to_csv("city_income.csv", index=False )



def pull_state(city):

    try:

        rc = population_density.loc[city,:]['state']

        return rc if type(rc) is str else rc.take([1])[0]

    except KeyError as e:

        try:

            # rc.take([1])[0]

            rc1 = car_ownership.loc[city,:]['state']

            return rc1 if type(rc1) is str else rc1.take([1])[0]

        except KeyError as e:

            rc2 = city_income.loc[city,:]['state']

            return rc2 if type(rc1) is str else rc2.take([1])[0]

            





pull_state('Winston-Salem')
city_state_ins = pd.read_json("../input/city-state-ins/city_ins.json")

city_state_ins.to_csv("city_state_ins.csv", index=False )

city_state_ins.drop(['city_comma_state'],axis=1, inplace=True)

#city_state_ins.set_index(['city'],inplace=True) 

city_state_ins['state'] = city_state_ins.apply(lambda row: pull_state(row.city) if len(row.state) == 0 else row.state, axis = 1)

#city_state_ins.set_index(index_by,inplace=True)

city_state_ins#.loc['New York',:]

rc1 = rc.merge(city_income, on= index_by)

rc1

rc2 = rc1.merge(city_state_ins, on= index_by)

rc2
rc2.to_csv("city_vehicle_ownership_insurance.csv", index=False )