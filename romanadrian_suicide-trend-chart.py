# imports and so on

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



print("Setup Complete")
file_path='../input/suicide-rates-overview-1985-to-2016/master.csv'



my_data = pd.read_csv(file_path)
my_data.head(15)
my_data.describe()
my_data.info()
countries = my_data.groupby('country').country.unique()

print("There are", countries.count(), "countries")



years_list = sorted(my_data.year.unique())

years = len(years_list)

print("There are", years, "years in the dataset:", years_list)
my_data_trimmed = my_data.drop(['suicides/100k pop', 'country-year'], axis=1)
my_data_trimmed.columns = my_data_trimmed.columns.str.strip()

my_data_trimmed['gdp_for_year ($)'] = my_data_trimmed['gdp_for_year ($)'].str.replace(',','').astype('int64')

my_data_trimmed.head()
colNames = ['suicides_no', 'population']

countriesData = my_data_trimmed.groupby(['country', 'year'])[colNames].sum()



print(countriesData.head())
fieldToChart = 'suicides_avg' 

countriesData[fieldToChart] = countriesData.apply(lambda x: x['suicides_no']/x['population'] * 100000, axis =  1)

countriesData = countriesData.drop(colNames, axis=1)



print(countriesData.head(10))
worldData = my_data_trimmed.groupby('year')[colNames].sum()



print(worldData.head())



worldData[fieldToChart] = worldData.apply(lambda x: x['suicides_no']/x['population'] * 100000, axis =  1)

worldData.drop(colNames, axis=1, inplace=True)



print(worldData.head())

countries_to_chart = ['Albania', 'Romania', 'France', 'Hungary', 'United States']



plt.figure(figsize=(14,8))

plt.title('Average suicide rates by year')

plt.xlabel('Year')

plt.ylabel('Suicides')

for country in countries_to_chart:

    data = countriesData.loc[country]

    sns.lineplot(data=data[fieldToChart], label=country)

    

sns.lineplot(data=worldData[fieldToChart], label='World')
genData = my_data_trimmed.groupby('generation')[colNames].sum()



print(genData.head())



genData[fieldToChart] = genData.apply(lambda x: x['suicides_no']/x['population'] * 100000, axis =  1)

genData.drop(colNames, axis=1, inplace=True)



print(genData.head())
plt.figure(figsize=(14,8))

plt.title('Avg suicides by generation')

sns.barplot(x=genData.index, y=genData['suicides_avg'])
my_detailed_data = my_data_trimmed.copy()

my_detailed_data[fieldToChart] = my_detailed_data.apply(lambda x: x['suicides_no']/x['population'] * 100000, axis =  1)





plt.figure(figsize=(14,8))

plt.title('Suicides by sex')

sns.scatterplot(x=my_detailed_data['gdp_per_capita ($)'], y=my_detailed_data[fieldToChart], hue=my_detailed_data['sex'])



#sns.lmplot(x='gdp_per_capita ($)', y=fieldToChart, hue='sex', data=my_detailed_data, aspect=1.5, height=8)
worldData = my_data_trimmed.groupby('sex')[colNames].sum()

worldData[fieldToChart] = worldData.apply(lambda x: x['suicides_no']/x['population'] * 100000, axis =  1)

worldData.drop(colNames, axis=1, inplace=True)



worldData
worldData.plot.pie(y='suicides_avg', figsize=(10,10))
# remove records for countries that we're not interested in

countriesData = my_data_trimmed[my_data_trimmed['country'].isin(countries_to_chart)] 



sexData = countriesData.groupby(['country','sex'])[colNames].sum()



print(sexData.head())



sexData[fieldToChart] = sexData.apply(lambda x: x['suicides_no']/x['population'] * 100000, axis =  1)

sexData.drop(colNames, axis=1, inplace=True)



print(sexData.head())
sexData = sexData.reset_index()



#switchToNumerical = { 'sex' : { 'male' : 0 , 'female' : 1 }}

#sexData.replace(switchToNumerical, inplace=True)



sexData.head()
def pie(vals, lab, color=None):

    plt.pie(vals, labels=lab.values)

    

#grid = sns.FacetGrid(sexData, col='country')

#grid.map(pie, 'suicides_avg', 'sex')



#plt.show()



sexData.pivot('sex', 'country', 'suicides_avg').plot.pie(subplots=True, figsize=(35, 15))
sns.set()

sns.pairplot(my_detailed_data, height = 2.3)

plt.show();
correlations = my_detailed_data.corr()



correlations
plt.figure(figsize=(14,14))

plt.title('Correlation matrix')

sns.set(font_scale=1.5)

hm = sns.heatmap(correlations, cbar=True, annot=True, square=True, fmt='.4f', annot_kws={'size': 17}, yticklabels=correlations.columns.values, xticklabels=correlations.columns.values)

plt.show()