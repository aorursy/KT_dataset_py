# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Preliminary step reading Olympic Games as DataFrame and inspecting df.head()

medals = pd.read_csv('../input/.csv')

print(medals.info())

medals.head()
# Select the 'NOC' column of medals and .values_counts() it

medal_counts = medals['NOC'].value_counts()



# Print the total amount of medals using .sum()

print('The total medals: %d' % medal_counts.sum())



# Print top 15 countries ranked by medals

# Remember, here the .type is a pandas.Series to conver to DataFrame use medal_counts.to_frame()

print('\nTop 15 countries:\n', medal_counts.head(15))
# Construct the pivot table: counted

counted = medals.pivot_table(index='NOC', values='Athlete', columns='Medal', aggfunc='count')



# Create the new column: counted['totals']

counted['totals'] = counted.sum(axis='columns')



# Sort counted by the 'totals' column

counted = counted.sort_values('totals', ascending=False)



# Print the top 15 rows of counted

counted.head(15)
# Construct the pivot table using the parameter margins=True: counted

counted = medals.pivot_table(index='NOC', values='Athlete', columns='Medal', aggfunc='count', margins=True, margins_name='Totals_all')



# Sort counted by the 'totals' column

counted = counted.sort_values('Totals_all', ascending=False)



counted.head()
# Select columns: ev_gen

ev_gen_uniques = medals[['Event_gender', 'Gender']].drop_duplicates()



# Print ev_gen_uniques, the index here is the index from the rows those values appear

ev_gen_uniques
# Group medals by the two columns: medals_by_gender

medals_by_gender = medals.groupby(['Event_gender', 'Gender']).count()



# Print medal_count_by_gender

medals_by_gender
# Create a boolean series mask to be used as a filter: boolean_filter

boolean_filter = (medals.Event_gender == 'W') & (medals.Gender == 'Men')



# Use the mask to select the wrong entry that matches the filter

medals[boolean_filter]
# Selecting the wrong entry with .iloc[]

medals.iloc[[23675]]
# Correcting the 'Gender', column index 6 on data set

medals.iloc[[23675], [6]] = 'Women'

medals.iloc[[23675]]
# Group medals by 'NOC'

# Compute the number of distinct sports in which each country won medals

# Sort the values of Nsports in descending order



# Conventional .groupby()

# Nsports = medals.groupby('NOC')['Sport'].nunique().sort_values(ascending=False)



# Sophisticated .groupby()

Nsports = medals[['NOC', 'Sport']].groupby('NOC', as_index=False).agg({'Sport':'nunique'}).sort_values('Sport', ascending=False)



# Print the top 15 rows of Nsports, notice that it is a Data Frame with no index

Nsports.head(15)
# Extract all rows for which the 'Edition' is between 1952 & 1988: during_cold_war

during_cold_war = (medals.Edition >= 1952) & (medals.Edition <= 1988)



# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs

is_usa_urs = medals.NOC.isin(['USA', 'URS'])



# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals

cold_war_medals = medals.loc[during_cold_war & is_usa_urs]



# Group cold_war_medals by 'NOC'

country_grouped = cold_war_medals.groupby('NOC')



# Create Nsports

Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)

print(Nsports)
# Create the pivot table: medals_won_by_country

medals_won_by_country = medals.pivot_table(index='Edition', columns='NOC', values='Athlete', aggfunc='count')



# Slice medals_won_by_country: cold_war_usa_usr_medals

cold_war_usa_usr_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]

print('Consistency during cold war\n', cold_war_usa_usr_medals.idxmax(axis='columns'))

print('\nTotal counts\n', cold_war_usa_usr_medals.idxmax(axis='columns').value_counts())
# Redefine 'Medal' as an ordered categorical

medals.Medal = pd.Categorical(values=medals.Medal, categories=['Bronze', 'Silver', 'Gold'], ordered=True)



# Create the DataFrame: usa

usa = medals[medals.NOC == 'USA']



# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'

usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Note that usa.pivot_table(index=['Edition', 'Medal'], values='Athlete', aggfunc='count')

# Produces the same output!!



# Reshape usa_medals_by_year by unstacking

usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')



# Plot the DataFrame usa_medals_by_year

usa_medals_by_year.plot.area(figsize=(12,8), title='USA medals over time in Olympic games')

plt.show()
# Create the DataFrame: urs

urs = medals[medals.NOC == 'URS']



# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'

usa_medals_by_year = urs.groupby(['Edition', 'Medal'])['Athlete'].count()

# Note that usa.pivot_table(index=['Edition', 'Medal'], values='Athlete', aggfunc='count')

# Produces the same output!!



# Reshape usa_medals_by_year by unstacking

usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')



# Plot the DataFrame usa_medals_by_year

usa_medals_by_year.plot.area(figsize=(12,8), title='URS medals over time in Olympic games')

plt.show()
# Finding Hungary on the ranking!

for place, country in enumerate(counted.index):

    if country == 'HUN':

        print('Hungary is the ' + str(place+1) + ' country in the total Olympic medals ranking')

        break
# What are the Hungarian top sports?

# You can check on hun_medals the margins=True is used to make sure

# we have the same total amount of medals as stated on counted.head(), Question 1

# 1053 is the right total so we are on the right track!

hun_medals = medals[medals.NOC == 'HUN'].pivot_table(index=['Sport'], columns='Medal', values='Athlete', aggfunc='count', dropna=True, fill_value=0, margins=True)

hun_medals_sort = hun_medals.sort_values('All', ascending=False)

print('Top 3 Hungarian sports according to Olympic medals:')

hun_medals_sort.head(4)
hun_medals = medals[medals.NOC == 'HUN'].groupby(['Sport', 'Discipline'])[['Medal']].agg('count').sort_values('Medal', ascending=False).head(3)

hun_medals_nosort = medals[medals.NOC == 'HUN'].groupby(['Sport', 'Discipline'])[['Medal']].agg('count')



#hun_medals = medals[medals.NOC == 'HUN'].groupby(['Sport', 'Discipline']).agg('count')['Medal'].nlargest(6).to_frame()

fen_tot = (hun_medals_sort.All.loc['Fencing'])

aqu_tot = (hun_medals_sort.All.loc['Aquatics'])

can_tot = (hun_medals_sort.All.loc['Canoe / Kayak'])



print('Fencing has ' + str((hun_medals.Medal['Fencing'].values/fen_tot)*100)[2:6] + '% of efficiency')

print('Water polo has ' + str((hun_medals.Medal.loc['Aquatics'].values/aqu_tot)*100)[2:6] + '% of efficiency')

print('Canoe / Kayak has ' + str((hun_medals.Medal.loc['Canoe / Kayak'].values/can_tot)*100)[2:6] + '% of efficiency')



hun_medals
# Not very optimized

def hun_sorted_discipline(sport1, sport2, sport3):

    disc_mask = (medals.Discipline == sport1) | (medals.Discipline == sport2) | (medals.Discipline == sport3)

    hun_comp = medals[['NOC', 'Discipline', 'Medal']][disc_mask].groupby(['NOC', 'Discipline']).count().sort_values('Medal', ascending=False).reset_index()

    hun_comp_piv = hun_comp.pivot_table(index='NOC', columns='Discipline', values='Medal', fill_value=0)

    hun_s1 = hun_comp_piv.sort_values(by=[sport1], ascending=False)[[sport1]].head()

    hun_s2 = hun_comp_piv.sort_values(by=[sport2], ascending=False)[[sport2]].head()

    hun_s3 = hun_comp_piv.sort_values(by=[sport3], ascending=False)[[sport3]].head()

    

    print('World top 5 countries per Olympic medals \n')

    print('\n',hun_s1)

    print('\n', hun_s2)

    print('\n', hun_s3)

    

hun_sorted_discipline('Canoe / Kayak F', 'Water polo', 'Fencing')
# Create the DataFrame: urs

urs = medals[medals.NOC == 'HUN']



# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'

usa_medals_by_year = urs.groupby(['Edition', 'Medal'])['Athlete'].count()

# Note that usa.pivot_table(index=['Edition', 'Medal'], values='Athlete', aggfunc='count')

# Produces the same output!!



# Reshape usa_medals_by_year by unstacking

usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')



# Plot the DataFrame usa_medals_by_year

usa_medals_by_year.plot.area(figsize=(12,8), title='Hungary medals over time in Olympic games')

plt.show()