# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/cities_r2.csv')
rawData = pd.value_counts(data['state_name'])



cityNumber_Of_State = pd.DataFrame(columns=['woe-name', 'value'])

cityNumber_Of_State['woe-name'] = rawData.index

cityNumber_Of_State['value'] = rawData.values

cityNumber_Of_State.sort_values
literacy_rate_total_of_states = pd.DataFrame(columns=['woe-name','value'])

rawData = data[['state_name','population_total','literates_total']]

rawData.head()
rawData = rawData.groupby('state_name').apply(lambda data: data['literates_total'].sum()/data['population_total'].sum() * 100)

rawData.sort_values(ascending=False)
literacy_rate_total_of_states['woe-name'] = rawData.index

literacy_rate_total_of_states['value'] = rawData.values;literacy_rate_total_of_states.head()
female_graduates_of_states = pd.DataFrame(columns=['woe-name','value'])

rawData = data[['state_name','female_graduates','population_female']];rawData.head()
rawData = rawData.groupby('state_name').apply(lambda data: data['female_graduates'].sum()/data['population_female'].sum() * 100)

rawData.sort_values(ascending=False)
female_graduates_of_states['woe-name'] = rawData.index

female_graduates_of_states['value'] = rawData.values;female_graduates_of_states.sort_values('value',ascending=False).head()
total_graduates_of_states = pd.DataFrame(columns=['woe-name','value'])

rawData = data[['state_name','total_graduates','population_total']]

rawData.head()
rawData = rawData.groupby('state_name').apply(lambda data: data['total_graduates'].sum()/data['population_total'].sum() * 100)

rawData.sort_values(ascending=False)
total_graduates_of_states['woe-name'] = rawData.index

total_graduates_of_states['value'] = rawData.values;total_graduates_of_states.sort_values('value', ascending=False).head()
cityPopulation = pd.DataFrame(columns=['city', 'state_name', 'lat','lon','population'])

cityPopulation['city'] = data.name_of_city

cityPopulation['population'] = data.population_total

cityPopulation['state_name'] = data.state_name



temp = data.location

for index in temp.index:

    tempArr=temp[index].split(',')

    #print(   tempArr)

    cityPopulation.at[index,'lat'] = float(tempArr[0])

    cityPopulation.at[index,'lon'] = float(tempArr[1])

#mapData.at[1,'population']

cityPopulation.head()
effective_literacy_rate_total = pd.DataFrame(columns=['city', 'state_name', 'lat','lon','effective_literacy_rate_total'])

effective_literacy_rate_total['city'] = data.name_of_city

effective_literacy_rate_total['effective_literacy_rate_total'] = data.effective_literacy_rate_total

effective_literacy_rate_total['state_name'] = data.state_name



temp = data.location

for index in temp.index:

    tempArr=temp[index].split(',')

    #print(   tempArr)

    effective_literacy_rate_total.at[index,'lat'] = float(tempArr[0])

    effective_literacy_rate_total.at[index,'lon'] = float(tempArr[1])



effective_literacy_rate_total.sort_values(by='effective_literacy_rate_total',ascending=False).head()
effective_literacy_rate_total.sort_values(by='effective_literacy_rate_total',ascending=False).tail()
effective_literacy_rate_total.describe()
top = effective_literacy_rate_total.nlargest(125,'effective_literacy_rate_total')

top.head()
tail = effective_literacy_rate_total.nsmallest(125,'effective_literacy_rate_total')

tail.head()
female_graduates = pd.DataFrame(columns=['city', 'state_name', 'lat','lon','female_graduates_ratio'])

female_graduates['city'] = data.name_of_city

female_graduates['female_graduates_ratio'] = data.female_graduates/data.population_female *100

female_graduates['state_name'] = data.state_name



temp = data.location

for index in temp.index:

    tempArr=temp[index].split(',')

    female_graduates.at[index,'lat'] = float(tempArr[0])

    female_graduates.at[index,'lon'] = float(tempArr[1])

female_graduates.describe()
top_125_city_by_female_graduates = female_graduates.nlargest(125,'female_graduates_ratio')

top_125_city_by_female_graduates.head()
tail_125_city_by_female_graduates = female_graduates.nsmallest(125,'female_graduates_ratio')

tail_125_city_by_female_graduates.head()
total_graduates_ratio_each_city = pd.DataFrame(columns=['city', 'state_name', 'lat','lon','total_graduates_ratio'])

total_graduates_ratio_each_city['city'] = data.name_of_city

total_graduates_ratio_each_city['total_graduates_ratio'] = data.total_graduates/data.population_total * 100

total_graduates_ratio_each_city['state_name'] = data.state_name



temp = data.location

for index in temp.index:

    tempArr=temp[index].split(',')

    #print(   tempArr)

    total_graduates_ratio_each_city.at[index,'lat'] = float(tempArr[0])

    total_graduates_ratio_each_city.at[index,'lon'] = float(tempArr[1])

#mapData.at[1,'population']

total_graduates_ratio_each_city.describe()
top_125_city_by_total_graduates_ratio = total_graduates_ratio_each_city.nlargest(125,'total_graduates_ratio')

top_125_city_by_total_graduates_ratio.head()
tail_125_city_by_total_graduates_ratio = total_graduates_ratio_each_city.nsmallest(125,'total_graduates_ratio')

tail_125_city_by_total_graduates_ratio.head()