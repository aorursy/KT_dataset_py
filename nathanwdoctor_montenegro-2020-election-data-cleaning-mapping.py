import pandas as pd

import geopandas as gpd

import unicodedata
results = pd.read_html('https://rezultati.dik.co.me/')[0]

results.head()
# List of Montenegrin political parties in English language

party_list_in_english = pd.read_excel('../input/montenegro-info/Parties.xlsx').columns

party_list_in_english
# Dictionary of parties translated and shortened from Montenegrin to English

translated_parties = dict(zip(results.columns, party_list_in_english))

translated_parties
# Translates political parties to English

results = results.rename(columns=translated_parties)

results.head()
# Fills missing entries with correct values from election website

results['Municipality'] = results['Municipality'].fillna('Total', limit=1)

results['Registered_Voters'] = results['Registered_Voters'].fillna('540026 / 540026', limit=1)

results['Total_Votes'] = results['Total_Votes'].fillna('413952 76,65 %', limit=1)

results['Total_Valid_Votes'] = results['Total_Valid_Votes'].fillna('409451 98,91 %', limit=1)
results.head()
results['Municipality']
def unicode_normalize(input_string):

    """

    Converts special alphabetical characters with the base character

    e.g.: 'hétérogénéité'  --> 'heterogeneite'

    """

    return ''.join((c for c in unicodedata.normalize('NFD', input_string)

                    if unicodedata.category(c) != 'Mn'))
results['Municipality'] = results['Municipality'].apply(lambda x: unicode_normalize(x))
results['Municipality']
results = results.set_index('Municipality')
# Cleans 'results' dataframe keeping only the relevant info

results.loc['Total'] = results.apply(lambda x: x.loc['Total'].split()[0], axis=0)

results['Registered_Voters'] = results['Registered_Voters'].apply(lambda x: x.split()[0])

results.head()
results.dtypes
results = results.astype('int32')
results['winner'] = results.loc[:,'Social_Democrats':'Peace_is_Our_Nation'

                               ].apply(lambda x: x.idxmax(), axis=1)
results['winner']
results['winner'].value_counts()
# New dataframe for handling percentages

results_perc = results.copy()
# Percent of invalid votes in each municipality

results_perc['percent_invalid'] = ((results_perc['Total_Votes'] - 

                                    results_perc['Total_Valid_Votes']) / 

                                   results_perc['Total_Votes'] * 100)
# The turnout rate in each municipality

results_perc['turnout'] = results_perc['Total_Votes'] / results_perc['Registered_Voters'] * 100
# Converts the party columns to percentages

results_perc.loc[:,'Social_Democrats':'Peace_is_Our_Nation'] = (

    results_perc.loc[:,'Social_Democrats':'Peace_is_Our_Nation'].apply(

        lambda x: x / results_perc['Total_Valid_Votes'] * 100))



# Converts the Registered_Voters, Total_Votes, and Total_Valid_Votes columns to percentages

results_perc.loc[:,'Registered_Voters':'Total_Valid_Votes'] = (

    results_perc.loc[:,'Registered_Voters':'Total_Valid_Votes'].apply(

        lambda x: x / x[0] * 100))
print ('Democratic Party of Socialists:', 

       results_perc.loc['Total','Democratic_Party_of_Socialists_of_Montenegro'].round(2))

print ()

print ('For the Future of Montenegro:', 

       results_perc.loc['Total','For_the_Future_of_Montenegro'].round(2))

print ('Peace is Our Nation:', 

       results_perc.loc['Total','Peace_is_Our_Nation'].round(2))

print ('United_Reform Action:', 

       results_perc.loc['Total','Civic_Movement_United_Reform_Action'].round(2))

print ('Opposition coalition total:', 

       results_perc.loc['Total',['Civic_Movement_United_Reform_Action', 

                                 'Peace_is_Our_Nation', 

                                 'For_the_Future_of_Montenegro']].sum().round(2))
mont_map = gpd.read_file('../input/montenegro-map/montenegro_map.txt')

mont_map.head()
mont_map = mont_map.rename(columns={'name': 'Municipality'})
# Removes extra words in Municipality column to allow for future merging

mont_map['Municipality'] = mont_map['Municipality'].str.replace(' Municipality', '')

mont_map['Municipality'] = mont_map['Municipality'].str.replace('Old Royal Capital ', '')

mont_map['Municipality'] = mont_map['Municipality'].str.replace(' Capital City', '')
print ('results_perc shape:', results_perc.shape, 'mont_map shape:', mont_map.shape)
def elements_missing(li1, li2):

    """

    Returns the elements missing from each input list compared to the other list

    """

    return [elem for elem in li1 + li2 if elem not in li1 or elem not in li2]
# Finds the municipalities in results_perc that are not in mont_map and vice versa

elements_missing(list(results_perc.index), list(mont_map['Municipality']))
mont_map['Municipality'] = mont_map['Municipality'].apply(lambda x: unicode_normalize(x))
elements_missing(list(results_perc.index), list(mont_map['Municipality']))
overall_map = mont_map.merge(results_perc, left_on='Municipality', right_index=True)
overall_map.head()
population = pd.read_html('https://en.wikipedia.org/wiki/Municipalities_of_Montenegro')[1]

population.head()
population.columns
population.columns = population.columns.droplevel([0])
population.columns
population = population.loc[:,['Municipality', 

                               'Total', 

                               'Ethnic Majority (2011)', 

                               'Predominant language', 

                               'Predominant religion']]
population['Ethnic Majority (2011)'] = (population['Ethnic Majority (2011)']

                                        .apply(lambda x: x.split()[0]))
population['Municipality'] = population['Municipality'].apply(lambda x: unicode_normalize(x))
elements_missing(list(population['Municipality']), list(overall_map['Municipality']))
overall_map = overall_map.merge(population)
overall_map.head()
type(overall_map)
#overall_map.to_file(r'M:/<<your/location/here>>')