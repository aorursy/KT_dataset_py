import pandas as pd
# Change the path to the dataset file if needed. 

PATH = '../input/athlete_events.csv'
data = pd.read_csv(PATH)

data.head()
data.describe()
data.sort_values('Age', ascending=False).head()
data.sort_values('Weight', ascending=False).head()
mask_1992 = data['Year'] == 1992

mask_man = data['Sex'] == 'M'

mask_women = data['Sex'] == 'F'
min_man_age = data[mask_man & mask_1992]['Age'].min()
min_woman_age = data[mask_women & mask_1992]['Age'].min()
print('Answer: ' + str(min_man_age) + ' and ' + str(min_woman_age))
mask_2012 = data['Year'] == 2012

mask_males = data['Sex'] == 'M'

mask_basketball_players = data['Sport'] == 'Basketball'
data_males_2012 = data[mask_2012 & mask_males].drop_duplicates('ID')

data_basketball_males_2012 = data[mask_2012 & mask_males & mask_basketball_players].drop_duplicates('ID')
count_of_basketball_males_2012 = len(data_basketball_males_2012)

count_of_males_2012 = len(data_males_2012)

result = round(count_of_basketball_males_2012 / count_of_males_2012 * 100, 1)
print('Answer is: ' + str(result))
data_females_2000 = data[(data['Year'] == 2000) & (data['Sex'] == 'F') & (data['Sport'] == 'Tennis')].drop_duplicates('ID')
data_females_2000['Height'].agg(['mean', 'std'])
mask_2006 = data['Year'] == 2006

fatest_sportsman_2006 = data[mask_2006].sort_values('Weight', ascending=False).iloc[0]
print('Answer is: ' + str(fatest_sportsman_2006['Sport']))
mask_john_aalberg = data['Name'] == 'John Aalberg'

unique_year_john_aalberg_participations = data[mask_john_aalberg].drop_duplicates('Year')
print('Answer is: ' + str(len(unique_year_john_aalberg_participations)))
mask_switzerland = data['Team'] == 'Switzerland'

mask_gold_medals = data['Medal'] == 'Gold'

mask_tennis = data['Sport'] == 'Tennis'

mask_2008 = data['Year'] == 2008
swetzerland_tennis_gold_medals_2008 = data[mask_switzerland & mask_gold_medals & mask_tennis & mask_2008]
print('Answer: ' + str(len(swetzerland_tennis_gold_medals_2008)))
mask_2016 = data['Year'] == 2016

mask_italy = data['Team'] == 'Italy'

mask_spain = data['Team'] == 'Spain'

mask_medals_not_nan = data['Medal'].notna()
italy_medals_count = len(data[mask_italy & mask_2016 & mask_medals_not_nan])

spain_medals_count = len(data[mask_spain & mask_2016 & mask_medals_not_nan])
print('Answer: ' + str(spain_medals_count < italy_medals_count))
mask_15_25 = (data['Age'] >= 15) & (data['Age'] < 25)

mask_25_35 = (data['Age'] >= 25) & (data['Age'] <  35)

mask_35_45 = (data['Age'] >= 35) & (data['Age'] < 45)

mask_45_55 = (data['Age'] >= 45) & (data['Age'] <= 55)

mask_2008 = data['Year'] == 2008
participant_counts_by_age_categoy = {

    '[15-25)': len(data[mask_15_25 & mask_2008].drop_duplicates('ID')),

    '[25-35)': len(data[mask_25_35 & mask_2008].drop_duplicates('ID')),

    '[35-45)': len(data[mask_35_45 & mask_2008].drop_duplicates('ID')),

    '[45-55]': len(data[mask_45_55 & mask_2008].drop_duplicates('ID')),

}
min_participants_category = min(participant_counts_by_age_categoy, key=participant_counts_by_age_categoy.get)

max_participants_category = max(participant_counts_by_age_categoy, key=participant_counts_by_age_categoy.get)
print('Answer is: ' + str(min_participants_category) + ' and ' + str(max_participants_category) + ' correspondinngly')
summer_olympics_mask = data['Season'] == 'Summer'

winter_olympics_mask = data['Season'] == 'Winter'

atlanta_mask = data['City'] == 'Atlanta'

squaw_valley_mask = data['City'] == 'Squaw Valley'
any_summer_olympics_in_atlanta = len(data[summer_olympics_mask & atlanta_mask]) > 0

any_winter_olympics_in_squaw_valley = len(data[winter_olympics_mask & squaw_valley_mask]) > 0
print('Answer is ' + str(any_summer_olympics_in_atlanta) + ', ' + str(any_winter_olympics_in_squaw_valley))
mask_1986 = data['Year'] == 1986

mask_2002 = data['Year'] == 2002
count_of_unique_sports_1986 = len(data[mask_1986].drop_duplicates('Sport'))

count_of_unique_sports_2002 = len(data[mask_2002].drop_duplicates('Sport'))

absolute_difference_between_counts_of_sports = abs(count_of_unique_sports_1986 - count_of_unique_sports_2002)
print('Answer is: ' + str(absolute_difference_between_counts_of_sports))