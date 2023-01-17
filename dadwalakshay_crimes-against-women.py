import pandas as pd

import matplotlib.pyplot as plt
crime12 = pd.read_csv('../input/crime/42_District_wise_crimes_committed_against_women_2001_2012.csv')
crime13 = pd.read_csv('../input/crime/42_District_wise_crimes_committed_against_women_2013.csv')
crime12.columns = crime12.columns.str.upper()
crime13.columns = crime13.columns.str.upper()

crime13['STATE/UT'] = crime13['STATE/UT'].str.upper()

crime13['DISTRICT'].replace('ZZ TOTAL', 'TOTAL', inplace = True)
dataframe = pd.concat([crime12, crime13])
dataframe.fillna(value = 0, inplace= True)
total_crime = dataframe[dataframe['DISTRICT'] == 'TOTAL']
punjab_crime = total_crime[total_crime['STATE/UT'] == 'PUNJAB']
punjab_crime.set_index('YEAR')[['DOWRY DEATHS', 'CRUELTY BY HUSBAND OR HIS RELATIVES','ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY']].plot(kind = 'line', figsize = (12,8))

plt.xlabel('Years')

plt.ylabel('No. of Cases in Punjab')

plt.title('Domestic Violence against Women')

plt.show()
data24 = dataframe[(dataframe['DISTRICT'] == 'TOTAL') & (dataframe['YEAR'] == 2013)]
allstates24 = data24[['RAPE', 'KIDNAPPING AND ABDUCTION','INSULT TO MODESTY OF WOMEN']].plot(kind = 'barh', figsize = (10,13), width = 1)

allstates24.set_xlabel('No. of Cases in 2013')

allstates24.set_yticklabels(data24['STATE/UT'])

plt.show()
delhi_crime = dataframe[dataframe['STATE/UT'] == 'DELHI']

delhi_crime_tot = delhi_crime.groupby('YEAR').sum()
graph_compare = plt.figure(figsize= (12,8))

punjab_graph = graph_compare.add_subplot(211)

delhi_graph = graph_compare.add_subplot(212)

graph_compare.subplots_adjust(hspace = 0.3)



d = delhi_crime_tot[['RAPE', 'KIDNAPPING AND ABDUCTION']]

delhi_graph.set_xlabel('Years')

delhi_graph.set_ylabel('No. of Cases')

delhi_graph.set_title('Rapes and Kidnaps during past Years in Delhi')

delhi_graph.plot(d)

delhi_graph.legend(['Rape', 'Kidnap'])



p = punjab_crime.set_index('YEAR')[['RAPE', 'KIDNAPPING AND ABDUCTION']]

punjab_graph.set_xlabel('Years')

punjab_graph.set_ylabel('No. of Cases')

punjab_graph.set_title('Rapes and Kidnaps during past Years in Punjab')

punjab_graph.plot(p)

punjab_graph.legend(['Rape', 'Kidnap'])
crimes = dataframe.groupby('STATE/UT').sum()
crimes.drop('YEAR', axis= 1, inplace= True)
crimes['TOTAL'] = 0
for i in range(len(crimes.index)):

    crimes['TOTAL'][i] = crimes.iloc[i].sum()

    
max_crimes = crimes['TOTAL'].nlargest(10).plot(kind = 'barh', title = 'Most Unsafe States for Women(2001-13)', figsize = (6,5))

plt.xlabel('No. of Crimes')

plt.show()

min_crimes = crimes['TOTAL'].nsmallest(10).plot(kind = 'barh', title = 'Safest States for Women(2001-13)', figsize = (6,5))

plt.xlabel('No. of Crimes')

plt.show()