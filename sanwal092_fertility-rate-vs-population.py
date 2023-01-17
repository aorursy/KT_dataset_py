
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
population = pd.read_csv('../input/country_population.csv')
fertility = pd.read_csv('../input/fertility_rate.csv')
life = pd.read_csv('../input/life_expectancy.csv')

temp_pop= population
temp_pop.drop(columns=['Country Name','Country Code', 'Indicator Name', 'Indicator Code'],axis =1, inplace=True)
pop_sum=temp_pop.sum()
pop_sum=pd.DataFrame(pop_sum).reset_index()
pop_sum.columns= ['Year','Total Population']
# type(pop_sum)
pop_sum.head()
pop_sum.shape

#  Stack overflow method 

plt.figure(figsize=(25,10))
plt.plot(pop_sum['Year'], pop_sum['Total Population'])
plt.title('Global Population from 1960 to 2016')
plt.xticks(np.arange(1960,2017))

plt.show()

temp_fert= fertility
temp_fert.head()
temp_fert.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)
new_fert =temp_fert.dropna()
# new_fert.isnull().values.any() # no empty values 
new_fert.head()
fert_sum = new_fert.mean()
fert_sum = pd.DataFrame(fert_sum).reset_index()
fert_sum.columns=['Year', 'Fertility']
fert_sum.describe()
# fert_sum.plot()
plt.figure(figsize=(25,10))
plt.plot(fert_sum['Year'], fert_sum['Fertility'])
plt.xticks(np.arange(1960,2017))
plt.title('Fertility from 1960 to 2016')
plt.show()

df_life = life
# df_life.head()
df_life.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)
new_life = df_life.dropna()
new_life.head()

new_life.isnull().values.any()# no empty values 
life_mean =  new_life.mean()
life_mean = pd.DataFrame(life_mean).reset_index()
life_mean.columns= ['Year', 'Life expectancy']
# life_mean.plot()
life_mean.head()
plt.figure(figsize=(25,10))
plt.plot(life_mean['Year'], life_mean['Life expectancy'])
plt.xticks(np.arange(1960,2017))
plt.title('Life Expectancy from 1960 to 2016')
plt.show()

def make_df(df,value_name):
    
    # First off, I will drop the useless columns 
    df.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)
    
    df.dropna()
    
#     while df.isnunll().values.isany()== True:
#           df.dropna()
#     else:
#           continue
        
    if value_name == 'Population':
        df_stat =df.sum()
    else:
        df_stat =df.mean()
    
    df = pd.DataFrame(df_stat).reset_index()
    
    if 'Population' in value_name:
        df.columns= ['Year', 'Population']
    elif 'Fertility' in value_name:
        df.columns = ['Year', 'Fertility']
    else:
        df.columns = ['Year', 'Life Expectancy']
    
    return df
    
    
population = pd.read_csv('../input/country_population.csv')
pop_df = make_df(population, 'Population')
pop_df.plot()
# pop_df.head()

fert_sum.describe()

life_mean.describe()

pop_sum.describe()
# BUILD A MASSIVE DATA FRAME CONTAING THE DATA FOR ALL THREE CRITERIA
# world_data = pd.merge(pd.merge(pop_sum, life_mean, on='Year'), fert_sum, on='Year')
test = pd.merge(pop_sum,life_mean, on='Year')
world_data = pd.merge(test, fert_sum, on='Year')
world_data.columns

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
world_data[['Total Population', 'Life expectancy', 'Fertility']] = scaler.fit_transform(world_data[['Total Population', 'Life expectancy', 'Fertility']])
world_data.head()
# world_data.plot(x="Year", y=["Total Population","Life expectancy","Fertility"], figsize = (10,10) )
population = pd.read_csv('../input/country_population.csv')
fertility = pd.read_csv('../input/fertility_rate.csv')
life = pd.read_csv('../input/life_expectancy.csv')
    
population = pd.read_csv('../input/country_population.csv')
population.head()
pop_year = population.drop(['Country Code','Indicator Name', 'Indicator Code'],axis=1)
pop_year= pop_year.T
# population.head()
pop_year.head()

pop_year.shape
# pop_year.isnull().values.any()
t = pop_year.isnull().apply(sum, axis=0)
t

# for col in pop_year:
#     if t:
#         pop_year.fillna(pop_year.mean())

pop_year = pop_year.fillna(pop_year.mean())        
# pop_year.isnull().values.any() # The answer is false
pop_year.shape
# pop_year
print(fertility.T.shape[1]-fert_year.shape[1],'countries have been dropped from the dataset because there were atleast 40 or more missing NaN values')
fertility = pd.read_csv('../input/fertility_rate.csv')

fertility=fertility.drop(['Country Code','Indicator Name', 'Indicator Code'],axis=1)
fert_year = fertility
# # fert_year.reset_index()
# t = fert_year.isnull().apply(sum, axis=0)
# t

# for col in fert_year:
#     if t[col]>=40:
#         del fert_year[col]
# #         fert_year.fillna(fert_year[col].mean)
#     else:
#         fert_year.fillna(fert_year[col].mean)
        
fert_year= fert_year.dropna()
fert_year


fert_temp = fert_year

fert_temp= fert_temp.dropna()
# fert_temp=fert_temp.drop('mean of year', axis =1)
fert_temp

country_names = pd.DataFrame(fert_temp['Country Name'])
country_names
fert_ignore =pd.DataFrame(fert_temp.iloc[:,:]) #Ignores the names of the country
# fert_ignore.columns = range(fert_ignore.shape[1])
fert_ignore
if 'mean of year' in fert_ignore.columns:
    fert_ignore =fert_ignore.drop('mean of year',axis =1)
fert_ignore.T
fert_mean = fert_ignore.mean(axis =0)
type(fert_mean)
fert_mean
# print(fert_mean.shape)
# print(fert_ignore.T.shape)
# # #Need to add the mean column to DataFrame as well. I will add it at the very begining
idx= 0
fert_add = fert_ignore.T
fert_add.insert(loc=idx, column='mean of year', value=fert_mean)
fert_add.columns[0]
# fert_add.columns[1:] #= np.arange(1960,2017)
fert_add = fert_add.T
# len(fert_add.T.columns)
# pd.DataFrame(fert_add['1960']).reset_index().drop('index',axis =1)
# column_interest= fert_add.columns[41:]
fert_add.iloc[1]
column_drop = fert_add.columns[1:41] #1960 to 1999
column_interest= fert_add.columns[41:] #2000 to 2016
mil_fert = fert_add.drop(column_drop,axis=1) #fertility data since 2000

mil_fert.iloc[:,1:]#.mean().mean()
mil_filter = mil_fert[mil_fert[column_interest]>= 3.0]
# mil_temp = mil_filter[]
count_try= mil_filter.drop('Country Name', axis=1).dropna()
count_try

more_filter = count_try[count_try[column_interest]>=5.0]
more_filter= more_filter.dropna()
more_filter
filter_six = more_filter[more_filter[column_interest]>=5.5]
filter_six = filter_six.dropna()
filter_six
country_indices=filter_six.index.values
country_indices
type(country_indices)

insert_names = country_names[country_names.index.isin(country_indices)]
insert_names['Country Name']

idx= 0
country_fert = filter_six
country_fert.insert(loc=idx, column='Country Name', value=insert_names)
country_fert
# fert_add.columns[0]
country_fert.plot(figsize= (10,10), title= '10 countries with highest fertility rate 2000-2016')
fertility[column_interest].dropna().plot(kind='line',figsize=(10,10), title = 'World fertility 2000-2016')
population[column_interest].plot(figsize=(10,10), title='World population 2000-2016')
population = population = pd.read_csv('../input/country_population.csv')
population= population.drop(['Country Code','Indicator Name','Indicator Code'], axis=1)
population['2016'].sum()
world_16 = 78856789486.0

country_pop= population[column_interest].iloc[country_indices]
country_pop['2016'].sum()
ten_16= 414258372.0
country_pop
#population of the 10 countries in 2016 was 414258372.0

idx= 0
country_pop.insert(loc=idx, column='Country Name', value=insert_names)
country_pop.plot(figsize= (10,10), title = 'population of 10 mos fertile countries')
percent = np.multiply(np.divide(ten_16,world_16),100)
print('The top 10 most fertile countries in 2016, form ', percent,'%  of the world population in 2016')
# percent 