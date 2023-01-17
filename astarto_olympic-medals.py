import pandas as pd
import matplotlib.pyplot as plt
medals = pd.read_csv('../input/all_medalists.csv')
medals.head(15)
GRE_edition_grouped = medals.loc[medals.NOC=='GRE'].groupby('Edition')
GRE_edition_grouped['Medal'].count()
# Select the 'NOC' column of medals: country_names
country_names = medals['NOC']

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))
# Construct the pivot table: counted
counted = medals.pivot_table(index='NOC',columns= 'Medal',values='Athlete',aggfunc= 'count')
counted.head()
# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis = 'columns')
counted.head()
# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)
counted.head(15)
medals.info()
# Select columns: ev_gen
ev_gen = medals[['Event_gender','Gender']]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)
# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender','Gender'])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)
# Create the Boolean Series: sus
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')

# Create a DataFrame with the suspicious row: suspect
suspect = medals[(medals.Event_gender == 'W') & (medals.Gender == 'Men')]

# Print suspect
print(suspect)
# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped['Sport'].nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending = False)

# Print the top 15 rows of Nsports
print(Nsports.head(15))
# Extract all rows for which the 'Edition' is between 1952 & 1988: during_cold_war
during_cold_war = (medals['Edition'] >= 1952) & (medals['Edition']<= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA','URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]
cold_war_medals.head()
cold_war_medals.tail()
# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending = False)

# Print Nsports
print(Nsports)
# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index= 'Edition',columns='NOC',values= 'Athlete',aggfunc='count')
medals_won_by_country.head()
# Slice medals_won_by_country: cold_war_usa_urs_medals
cold_war_usa_urs_medals = medals_won_by_country.loc['1952':'1988', ['USA','URS']]
cold_war_usa_urs_medals
# Create most_medals 
most_medals = cold_war_usa_urs_medals.idxmax(axis ='columns')
most_medals
most_medals.value_counts()
# Create the DataFrame: usa
usa = medals.loc[medals.NOC=='USA']
usa.head()
# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition','Medal'])['Athlete'].count()
usa_medals_by_year
# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level= 'Medal')
print(usa_medals_by_year)
# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()
# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values=medals.Medal, categories= ['Bronze','Silver','Gold'],ordered=True)

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()

