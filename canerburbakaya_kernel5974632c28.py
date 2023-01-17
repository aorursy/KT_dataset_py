from IPython.display import Image
print("World Heat Map That Displays Player Distribution by Country")
Image(filename='../input/soccer-national-player-prediction/world_heat_map_player_count_distrubution.png')

#sources 
#https://www.kaggle.com/ajinkyablaze/football-manager-data
#https://www.kaggle.com/karangadiya/fifa19
#https://footballdatabase.com/ranking/world/1
#https://github.com/python-visualization/folium/blob/master/examples/data/world-countries.json


from pandas.plotting import scatter_matrix
from time import time
from datetime import datetime  
from datetime import timedelta 
from IPython.display import display 
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.cluster import KMeans
import math 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from statistics import mean

#https://github.com/nalepae/pandarallel
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=5)


# player data from Football Manager 2017 dataset
data_fm = pd.read_csv('../input/football-manager-data')

# player data from Fifa 19 dataset
data = pd.read_csv('../input/fifa19/data.csv')

#countries world-countries.json
world_countries_json_source = '../input/soccer-national-player-prediction/world-countries.json'


import seaborn as sns
fig, ax = plt.subplots(figsize=(40,40))
sns.heatmap(data.corr(), annot=True, ax=ax, cmap='BrBG').set(title='Feature Correlation', xlabel='Columns', ylabel='Columns')
plt.show()
import requests
from bs4 import BeautifulSoup
data_club_rankings = pd.DataFrame(columns=['ranking', 'club', 'country', 'points'])


club_data = 4*[None]

for page_id in range(1,53):
    link = 'https://footballdatabase.com/ranking/world/'+str(page_id)
    print(link)
    result = requests.get(link)
    
    src = result.content
    soup = BeautifulSoup(src, 'lxml')

    content_rank = soup.find_all('td', attrs={'class':'rank'})
    content_club_text_left = soup.find_all('td', attrs={'class':'club text-left'})

    for i, e in enumerate(content_club_text_left):
        club_data[0] = content_rank[i*2].text
        club_data[1] = e.find_all('a')[0].text
        club_data[2] = e.find_all('a')[1].text
        club_data[3] = content_rank[i*2+1].text
        data_club_rankings.loc[data_club_rankings.shape[0]] = club_data
        #print("ranking: {}  Club: {}  Country: {}  Points: {}".format(club_data[0], club_data[1], club_data[2], club_data[3]))
        print("{}".format(club_data[1]))

len(data.Club.unique())
data_club_rankings.head()
# from data to data_club_rankings
dict_club = {}
u_clubs = data.Club.unique()
ranked_clubs = list(data_club_rankings.club)
clubs_not_found = []

# u_club in data,  club in ranked_clubs
for u_club in u_clubs:
    if u_club not in list(ranked_clubs):
        is_found=False
        for club in ranked_clubs:
            condition_A = str(club) in str(u_club)
            condition_B = str(u_club) in str(club)
            if condition_A or condition_B:
                is_found = True
                dict_club[u_club] = club
                ranked_clubs.remove(club)
                break
        if not is_found:
            # check longest
            is_lengest_word_contained = False
            words_in_u_club = u_club.split(' ')
            len_words = [len(n) for n in words_in_u_club]
            np_list_len_words_in_u_club = np.array(len_words)
            longest_word_in_u_club = words_in_u_club[np_list_len_words_in_u_club.argmax()]
            for club in ranked_clubs:
                if longest_word_in_u_club in club:
                    dict_club[u_club] = club
                    ranked_clubs.remove(club)
                    is_lengest_word_contained = True
                    break
            if not is_lengest_word_contained:
                clubs_not_found.append(u_club)
    else:
        dict_club[u_club] = u_club
        ranked_clubs.remove(u_club)
clubs_not_found[:5]
dict_club
data_club_rankings[data_club_rankings.club.str.lower().str.contains('bren')]
#data_club_rankings[data_club_rankings.country=='France']
# correction on dictionary
dict_club[None] = 'unknown'
dict_club['Bayer 04 Leverkusen'] = 'Bayer Leverkusen'
dict_club['PFC CSKA Moscow'] = 'CSKA Moskva'
dict_club['Tigres U.A.N.L.'] = 'Tigres UANL'
dict_club['Derby County'] = 'Ross County'
dict_club['Club Atlético Talleres'] = 'Talleres de Cordoba'
dict_club['Ceará Sporting Club'] = 'Ceará SC'
dict_club['Rionegro Águilas'] = 'Águilas Doradas'
dict_club['América de Cali'] = 'América de Cali'
dict_club['Central Coast Mariners'] = 'Central Coast Mariners FC'
dict_club['US Orléans Loiret Football'] = 'unknown'
dict_club['Jaguares de Córdoba'] = 'CD Jaguares'
dict_club['Oldham Athletic'] = 'unknown'
dict_club['Notts County'] = 'unknown'
dict_club['Port Vale'] = 'unknown'
dict_club['Forest Green Rovers'] = 'unknown'
dict_club['Inter'] = 'Inter Milan'
dict_club['Internacional'] = 'Internacional'
# adding 'unknown' record to 'data_club_rankings' data frame
club_data = [data_club_rankings.shape[0], 'unknown', 'unknown', int(data_club_rankings.points.min())-20]
data_club_rankings.loc[data_club_rankings.shape[0]] = club_data
starting_position_for_cr_bin = 0
cr_bin_size = 10
extension_factor = 1.42
bin_no = 0
for i in range(data_club_rankings.shape[0]):
    if i == data_club_rankings.shape[0]-1:
        data_club_rankings.at[i, 'segment_point'] = bin_no+1
        break
    if i < starting_position_for_cr_bin + cr_bin_size:
        data_club_rankings.at[i, 'segment_point'] = bin_no
    elif i == starting_position_for_cr_bin + cr_bin_size:
        starting_position_for_cr_bin = i
        cr_bin_size = int(cr_bin_size * extension_factor)
        bin_no += 1
        data_club_rankings.at[i, 'segment_point'] = bin_no

for i in range(data_club_rankings.shape[0]):
    data_club_rankings.at[i, 'segment_point'] = data_club_rankings.segment_point.max() - data_club_rankings.segment_point[i]
data_club_rankings.head()
list(data_club_rankings[data_club_rankings.club == 'Inter Milan'].country)[0]
for i in range(data.shape[0]):
    match_in_ranked_clubs_df = 'unknown'
    if data.Club[i] in dict_club:
        match_in_ranked_clubs_df = dict_club[data.Club[i]]
    print("club: {},  match_in_ranked_clubs_df: {}".format(data.Club[i], match_in_ranked_clubs_df))
    cr_segment_point = list(data_club_rankings[data_club_rankings.club == match_in_ranked_clubs_df].segment_point)[0]
    cr_country = list(data_club_rankings[data_club_rankings.club == match_in_ranked_clubs_df].country)[0]
    #print("club: {},  match_in_ranked_clubs_df: {},  cr_segment_point: {},  cr_country: {}".format(data.Club[i], match_in_ranked_clubs_df, cr_segment_point, cr_country))
    data.at[i, 'club_segment_point'] = cr_segment_point
    data.at[i, 'club_country'] = cr_country
    
data.head()
data_fm['national_player'] = data_fm['IntCaps'].map(lambda x: 1 if x>0 else 0)
data['Preferred Foot'].mode()
#First let's drop evidently irrelevant features
columns_to_be_dropped = ['Joined', 'Real Face', 'Photo', 'Flag', 'Club Logo']
data = data.drop(columns_to_be_dropped, axis=1)
number_of_columns_with_null_values = len(data.isnull().any()[data.isnull().any()==True])
number_of_all_columns = len(data.columns)

print("{0} features have null values out of total {1} columns.".format(number_of_columns_with_null_values, number_of_all_columns))
(data.isnull().sum() / data.shape[0])[data.isnull().sum() / data.shape[0] > 0][:35]
(data.isnull().sum() / data.shape[0])[data.isnull().sum() / data.shape[0] > 0][35:]
data['Height']
data['ST'].mode()
# remove € sign and convert into numeric value K 1000 and M 1.000.000
data['Release Clause'].fillna(value='€0.0K', inplace=True)
data['Wage'].fillna(value='€0.0K', inplace=True)
def convert_release_clause_into_numeric(x):
    x = str(x).replace('€', '')
    if 'M' in x:
        x = x.replace('M', '')
        x = float(x) * 1000000
    elif 'K' in x:
        x = x.replace('K', '')
        x = float(x) * 1000
    return x

monetary_features = ['Release Clause', 'Value', 'Wage']
for col in monetary_features:
    data[col] = data[col].map(convert_release_clause_into_numeric)
    data[col] = pd.to_numeric(data[col])

from numpy import nanmean
data['Value'] = data['Value'].apply(lambda x: nanmean(data.groupby('Club')['Value']) if x == None else x)


data.dtypes[:10]
# remove lbs expression from weight
def remove_lbs(x):
    x = str(x).replace('lbs', '')
    return x

data['Weight'] = data['Weight'].map(remove_lbs)
data['Weight'].mode()
# remove ' and convert numeric 
# Note: 1 foot height is equal to 12 inches
data['Height'].mode()
def convert_height_into_numeric(x):
    integer_part = float(str(x).split('\'')[0])
    float_part = 0.0
    if len(str(x).split('\'')) > 1:
        float_part = float(str(x).split('\'')[1])/12.0
    return integer_part + float_part

data['Height'] = data['Height'].map(convert_height_into_numeric)
#remove + sign and sum up both Operand
malformed_columns_with_plus_sign = [ 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 
                                    'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB' ]
def remove_plus(x):
    if '+' in str(x):
        x0 = float(str(x).split('+')[0])
        x1 = float(str(x).split('+')[1])
        return x0+x1
    return float(x)
for col in malformed_columns_with_plus_sign:
    data[col] = data[col].map(remove_plus)

data['ST'].mode()
missing_columns_to_be_assigned_median = [ 'Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes', 'LS', 'ST', 'RS', 'LW', 'LF',
       'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 
       'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB',
       'Height', 'Weight', 'International Reputation' ]


for col in missing_columns_to_be_assigned_median:
    data[col].fillna(value=data[col].median(), inplace=True)
missing_columns_to_be_assigned_mode = ['Body Type', 'Contract Valid Until', 'Weak Foot', 
                                       'Jersey Number', 'Preferred Foot', 'Skill Moves', 'Work Rate']

for col in missing_columns_to_be_assigned_mode:
    data[col].fillna(value=data[col].mode()[0], inplace=True)
data['Loaned From'].value_counts()
missing_columns_to_be_assigned_unknown = ['Loaned From', 'Position', 'Club']


for col in missing_columns_to_be_assigned_unknown:
    data[col].fillna(value='unknown', inplace=True)
data.head()
data.dtypes[:10]
data[data.Value>5000000].groupby(['Nationality'])['ID'].count()['United States']
data_heat = data[['Nationality', 'ID', 'Value']]
#data_heat['player_count_by_country'] = 

dict_country_name_from_df_to_follium = {}
dict_country_name_from_df_to_follium['United States'] = 'United States of America'
dict_country_name_from_df_to_follium['China PR'] = 'China'
dict_country_name_from_df_to_follium['Korea Republic'] = 'South Korea'
dict_country_name_from_df_to_follium['Korea DPR'] = 'North Korea'
dict_country_name_from_df_to_follium['Bosnia Herzegovina'] = 'Bosnia and Herzegovina'
dict_country_name_from_df_to_follium['FYR Macedonia'] = 'Macedonia'
dict_country_name_from_df_to_follium['Central African Rep.'] = 'Central African Republic'
dict_country_name_from_df_to_follium['Trinidad & Tobago'] = 'Trinidad and Tobago'
dict_country_name_from_df_to_follium['DR Congo'] = 'Democratic Republic of the Congo'
dict_country_name_from_df_to_follium['Congo'] = 'Republic of the Congo'
dict_country_name_from_df_to_follium['England'] = 'United Kingdom'
dict_country_name_from_df_to_follium['Republic of Ireland'] = 'Ireland'

data_heat.head(35)



series_player_counts = data_heat.groupby(['Nationality'])['ID'].count()
data_heat['Value'] = data_heat.Value.astype(float)

np_array_stats_by_country = np.hstack((np.array(series_player_counts.index), np.array(series_player_counts.values))).reshape(2,-1).T


data_heat = pd.DataFrame(np_array_stats_by_country, columns=['country', 'player_counts_by_country'])

data.Value = (data.Value).astype(float)
series_precious_players_by_country = data[data.Value>5000000].groupby(['Nationality'])['ID'].count()
series_precious_players_by_country

data.Overall = (data.Overall).astype(float)
series_average_overall_skill_of_players_by_country = data[data.national_player == 1].groupby(['Nationality'])['Overall'].mean()
series_average_overall_skill_of_players_by_country


# shrink data logarithmically to obtain a more representative distribution of values through different polars (coumtries)
data_heat['precious_player_counts_by_country'] = data_heat['country'].apply(lambda x: series_precious_players_by_country[x] if x in series_precious_players_by_country else 0)
data_heat['average_overall_skill_of_players_by_country'] = data_heat['country'].apply(lambda x: series_average_overall_skill_of_players_by_country[x] if x in series_average_overall_skill_of_players_by_country else 0)

print(data_heat.sort_values(by=['precious_player_counts_by_country'], ascending=False).head(6))

data_heat['precious_player_counts_by_country'] = data_heat['precious_player_counts_by_country'].apply(lambda x: math.log(x+1) if x+1!=0 else 0.0)
data_heat['average_overall_skill_of_players_by_country'] = data_heat['average_overall_skill_of_players_by_country'].apply(lambda x: x**2 if x>0 else x)

print(data_heat.sort_values(by=['average_overall_skill_of_players_by_country'], ascending=False).head(6))

data_heat['player_counts_by_country'] = data_heat['player_counts_by_country'].apply(lambda x: math.log(x) if x!=0 else 0.0)

# revise country names and modify if needed based on Folium API

data_heat['country'] = data_heat['country'].apply(lambda x: dict_country_name_from_df_to_follium[x] if x in dict_country_name_from_df_to_follium.keys() else x)


scaler = MinMaxScaler()
heat_columns = ['player_counts_by_country', 'precious_player_counts_by_country', 'average_overall_skill_of_players_by_country']

for col in heat_columns:
    data_heat[col] = scaler.fit_transform(np.array(data_heat[col]).reshape(-1, 1))

#data_heat['average_overall_skill_of_players_by_country'] = data_heat['average_overall_skill_of_players_by_country'].apply(lambda x: math.log(x+1) if x+1!=0 else 0.0)

print(data_heat.sort_values(by=['average_overall_skill_of_players_by_country'], ascending=False).head(6))

data[data.Value>5000000].groupby(['Nationality'])['ID'].count().describe()
import folium

world_heat_map_player_count_by_country = folium.Map(
    location=[28.220860, 16.297040],
    #tiles='Stamen Terrain',
    zoom_start=1.5
)
#  One of the following color brewer palettes: 
# ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’

world_heat_map_player_count_by_country.choropleth(geo_data=world_countries_json_source, data=data_heat, columns=['country', 'player_counts_by_country'], key_on='feature.properties.name',
                         fill_color='OrRd', fill_opacity=0.7,  line_opacity=0.2, nan_fill_color='black', nan_fill_opacity=0.2)

world_heat_map_player_count_by_country
world_heat_map_player_average_market_value_by_country = folium.Map(
    location=[28.220860, 16.297040],
    #tiles='Stamen Terrain',
    zoom_start=1.5
)
#  One of the following color brewer palettes: 
# ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’

world_heat_map_player_average_market_value_by_country.choropleth(geo_data=world_countries_json_source, data=data_heat, columns=['country', 'precious_player_counts_by_country'], key_on='feature.properties.name',
                         fill_color='OrRd', fill_opacity=0.7,  line_opacity=0.2, nan_fill_color='black', nan_fill_opacity=0.2)

world_heat_map_player_average_market_value_by_country
data.head()
data.columns
data.Value
data.shape
nationalities_in_fifa19 = len(data.Nationality.unique())
nationalities_in_fm = len(data_fm.NationID.unique())
print("nationalities_in_fifa19: {0}\nnationalities_in_fm: {1}".format(nationalities_in_fifa19, nationalities_in_fm))
nationality_distribution_fifa = data.Nationality.value_counts()
sum_nat_55_plus = sum(nationality_distribution_fifa[nationality_distribution_fifa>55])
no_nat_55_plus = len(nationality_distribution_fifa[nationality_distribution_fifa>55])
total_players = len(data.ID.unique())
print("The number of nationalities with more than 55 players are {0} and covers {1} players out of {2}".format(no_nat_55_plus, sum_nat_55_plus, total_players))
# find the players with similar names and obtain player's full name
def find_str(x, name):
    if name in x:
        print(x)
    return x
a = data_fm['Name'].apply(find_str, name='De Bruyne')
data[data['Name'].str.contains('De Bruyne')]
# to display the player to obtain his NationID
#data_fm[data_fm['Name'] == 'Zhiyi']
data_fm[data_fm['Name'].str.contains('De Bruyne')]
dict_nationality = {}
dict_nationality[788] = 'Portugal'
dict_nationality[1649] = 'Argentina'
dict_nationality[1651] = 'Brazil'
dict_nationality[796] = 'Spain'
dict_nationality[757] = 'Belgium'
dict_nationality[761] = 'Croatia'
dict_nationality[795] = 'Slovenia'
dict_nationality[787] = 'Poland'
dict_nationality[771] = 'Germany'
dict_nationality[1657] = 'Uruguay'
dict_nationality[769] = 'France'
dict_nationality[776] = 'Italy'
dict_nationality[16] = 'Egypt'
dict_nationality[764] = 'Denmark'
dict_nationality[19] = 'Gabon'
dict_nationality[801] = 'Wales'
dict_nationality[41] = 'Senegal'
dict_nationality[366] = 'Costa Rica'
dict_nationality[794] = 'Slovakia'
dict_nationality[784] = 'Netherlands'
dict_nationality[765] = 'England'
dict_nationality[759] = 'Bosnia Herzegovina'
dict_nationality[34] = 'Morocco'
dict_nationality[802] = 'Serbia'
dict_nationality[5] = 'Algeria'
dict_nationality[755] = 'Austria'
dict_nationality[772] = 'Greece'
dict_nationality[1652] = 'Chile'
dict_nationality[797] = 'Sweden'
dict_nationality[135] = 'Korea Republic'
dict_nationality[1653] = 'Colombia'
dict_nationality[22] = 'Guinea'
dict_nationality[379] = 'Mexico'
dict_nationality[11] = 'Cameroon'
dict_nationality[31] = 'Mali'
dict_nationality[799] = 'Turkey'
dict_nationality[1435] = 'Australia'
dict_nationality[24] = 'Ivory Coast'
dict_nationality[793] = 'Scotland'
dict_nationality[53] = 'DR Congo'
dict_nationality[1651] = 'Japan'
dict_nationality[110] = 'China PR'
dict_nationality[376] = 'Honduras'
dict_nationality[791] = 'Russia'
dict_nationality[789] = 'Republic of Ireland'
dict_nationality[390] = 'United States'
dict_nationality[786] = 'Norway'
dict_nationality[133] = 'Saudi Arabia'
dict_nationality[798] = 'Switzerland'
dict_nationality[21] = 'Ghana'


dict_nationality
data['UID'] = data.shape[0] * [-1]
data['national_player'] = data.shape[0] * [None]
data.head()


def match_players_from_2_datasets(row):
    fname = row['Name'].replace('ć', 'c').replace('č', 'c').replace('š', 's').replace('í', 'i')
    elm = fname.split(' ')
    length_checker = np.vectorize(len) 
    lenelm = length_checker(elm)
    shortest_word_in_name = elm[np.where(lenelm == lenelm.min())[0][0]]
    wanted_name = ''
    initial = ''
    
    
    if len(shortest_word_in_name)==2 and shortest_word_in_name[1]=='.':
        initial = shortest_word_in_name[0]
    
    
    if len(elm) > 1 and len(shortest_word_in_name)<3:
        wanted_name = fname.replace(shortest_word_in_name,'',1)
    else:
        wanted_name = fname
    
    #trims the white space at the start and end
    wanted_name = wanted_name.strip()
    
    
    print("{0}  >>>  wanted_name: {1}. {2}, row.Age: {3}".format(row['Unnamed: 0'], initial, wanted_name, row.Age))
    
    
    if wanted_name == '':
        return row
    
    
    cond_a = (data_fm.Name.str.contains(wanted_name))
    
    cond_b = data_fm.shape[0] * [True]
    
    if (initial!=''):
        cond_b = np.array(data_fm.Name.apply(lambda x: str(x).upper())).astype('<U1') == initial
    cond_c = (abs(row['Age']-data_fm.Age-2) < 2)
    
    nationality_contains_list = data_fm.NationID.apply(lambda x: True if x in dict_nationality.keys() and dict_nationality[x] == row['Nationality'] else False)
    
    cond_d = data_fm.shape[0] * [True]
    if row['Nationality'] in dict_nationality.values():
        cond_d = nationality_contains_list
    
    
    condition = cond_a & cond_b & cond_c & cond_d
    
    cond_a = np.array(cond_a, dtype=np.float64)
    cond_b = np.array(cond_b, dtype=np.float64)
    cond_c = np.array(cond_c, dtype=np.float64)
    
    cond_a_result = (cond_a.astype(float)).sum()
    cond_b_result = (cond_b.astype(float)).sum()
    cond_c_result = (cond_c.astype(float)).sum()
    
    condition_result = (condition.astype(float)).sum()
    
    
    
    print("CONDITIONS  a: {0}, b: {1}, c: {2}, condition: {3}".format(cond_a_result, cond_b_result, cond_c_result, condition_result))
    
    result_collection = data_fm[condition]
    
    np_value = result_collection['national_player']
    if np_value.shape[0] == 1:
        row['national_player'] = np_value.iloc[0]
    uid_value = result_collection['UID']
    if uid_value.shape[0] == 1:
        row['UID'] = uid_value.iloc[0]
        print ("UID: {0},  national_player: {1}".format(uid_value.iloc[0], np_value.iloc[0]))
    else:
        print(uid_value.shape[0])
    
    return row

from pandarallel import pandarallel

initial_date_time = datetime.now()
print("initial_date_time: {0}\n\n\n\n".format(initial_date_time))



starttime = time()

data = data.parallel_apply(match_players_from_2_datasets, axis=1)


print('That took {} seconds'.format(time() - starttime))
data.head()
(data.isnull().sum() / data.shape[0])[data.isnull().sum() / data.shape[0] > 0][:45]
data.to_csv(r'fifa19_processed.csv', index = False)
data = pd.read_csv('fifa19_processed.csv')
data.shape
data.columns
data = data.dropna(subset=['national_player', 'UID'], how='any').reset_index()
data.shape
data = data.drop(['Unnamed: 0', 'ID', 'Name', 'UID'], axis=1)
data.shape
# data_fm
labels = 'National Players', 'Not National Players'
sizes = [data_fm.national_player.value_counts()[1], data_fm.national_player.value_counts()[0]]
colors = ['yellowgreen', 'lightskyblue']
explode = (0.1, 0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

data_fm.national_player.value_counts()

# data
labels = 'National Players', 'Not National Players'
sizes = [data.national_player.value_counts()[1], data.national_player.value_counts()[0]]
colors = ['yellowgreen', 'lightskyblue']
explode = (0.1, 0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

data.national_player.value_counts()
import folium
world_heat_map_player_average_market_value_by_country = folium.Map(
    location=[28.220860, 16.297040],
    #tiles='Stamen Terrain',
    zoom_start=1.5
)
#  One of the following color brewer palettes: 
# ‘BuGn’, ‘BuPu’, ‘GnBu’, ‘OrRd’, ‘PuBu’, ‘PuBuGn’, ‘PuRd’, ‘RdPu’, ‘YlGn’, ‘YlGnBu’, ‘YlOrBr’, and ‘YlOrRd’

world_heat_map_player_average_market_value_by_country.choropleth(geo_data=world_countries_json_source, data=data_heat, columns=['country', 'average_overall_skill_of_players_by_country'], key_on='feature.properties.name',
                         fill_color='OrRd', fill_opacity=0.7,  line_opacity=0.2, nan_fill_color='black', nan_fill_opacity=0.2)

world_heat_map_player_average_market_value_by_country
plt.hist(data['Value'])
plt.xlabel('Market Value of The Players')
plt.ylabel('Number of The Players')
plt.show()

plt.hist(data[data.Value > 20000000]['Value'])
plt.xlabel('Market Value of The Players (Higher Than $20M)')
plt.ylabel('Number of The Players')
plt.show()
plt.hist(data['Wage'])
plt.xlabel('Wage of The Players')
plt.ylabel('Number of The Players')
plt.show()

plt.hist(data[data.Wage > 100000]['Value'])
plt.xlabel('Wage of The Players (Higher Than $100K)')
plt.ylabel('Number of The Players')
plt.show()
plt.hist(data['Release Clause'])
plt.xlabel('Release Clause for The Players')
plt.ylabel('Number of The Players')
plt.show()

plt.hist(data[data['Release Clause'] > 2000000]['Release Clause'])
plt.xlabel('Release Clause for The Players (Higher Than $2M)')
plt.ylabel('Number of The Players')
plt.show()
# Assign 0 market value and wages as the median of the series. 
# Because, 0 wage is unrealistic and may be misleading
numeric_columns_to_be_shrunk = ['Value', 'Wage']
for col in numeric_columns_to_be_shrunk:
    data[col] = data[col].apply(lambda x: data[col].median() if x==0.0 else x)
numeric_columns_to_be_shrunk = ['Value', 'Wage', 'Release Clause']
for col in numeric_columns_to_be_shrunk:
    data[col+'_shrunk'] = data[col].apply(lambda x: math.log(x+0.01))
data = data.drop(numeric_columns_to_be_shrunk, axis=1)
data.head()
plt.hist(data['Value_shrunk'])
plt.xlabel('Shrunk Market Value of The Players')
plt.ylabel('Number of The Players')
plt.show()

plt.hist(data['Wage_shrunk'])
plt.xlabel('Shrunk Wage of The Players')
plt.ylabel('Number of The Players')
plt.show()

plt.hist(data['Release Clause_shrunk'])
plt.xlabel('Shrunk Release Clause for The Players')
plt.ylabel('Number of The Players')
plt.show()
position_columns = ['LS', 'ST', 'RS', 'LW',
           'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
           'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
data[position_columns].head()
for col in position_columns:
    data[col] = data[col].apply(lambda x: x/100.0)
from sklearn.cluster import KMeans

X = np.array(data[position_columns])

kmeans = KMeans(n_clusters=11, random_state=0).fit(X)

for i in range(0, data.shape[0]):
    data.at[i, 'PositionID'] = kmeans.labels_[i]

data.head()
data.PositionID.value_counts()
columns_for_segment_comparison = ['Value_shrunk', 'Wage_shrunk', 'Release Clause_shrunk']

for col in columns_for_segment_comparison:
    for i in range(0, data.shape[0]):
        data.at[i, 'DistanceToMaxInSegment_'+col] = data.at[i, col] - data[(data.Nationality == data.at[i, 'Nationality']) &  (data.PositionID == data.at[i, 'PositionID'])][col].max()
        data.at[i, 'DistanceToMedianInSegment_'+col] = data.at[i, col] - data[(data.Nationality == data.at[i, 'Nationality']) &  (data.PositionID == data.at[i, 'PositionID'])][col].median()
data = data.drop(['index'], axis=1)
data.tail()
data.to_csv(r'fifa19_processed2.csv', index = False)
data = pd.read_csv('fifa19_processed2.csv')
data = data.drop(['International Reputation'], axis=1)
data['Weight'].fillna(value=data['Weight'].median(), inplace=True)
number_of_national_players_by_nation = data.groupby("Nationality").national_player.sum()
data['number_of_national_players_by_nation'] = data['Nationality'].apply(lambda x: number_of_national_players_by_nation[x])
data.head(3)
# gather non-numeric columns
non_numeric_columns = []
s = data.dtypes
for i in range(0, len(data.dtypes)):
    e = data.dtypes[i]
    if e!=float and e!=int:
        non_numeric_columns.append(s.index[i])

non_numeric_columns = non_numeric_columns + ['PositionID']
non_numeric_columns
data['Loaned From'].value_counts()
data.iloc[70:76]
data = pd.get_dummies(data, columns=non_numeric_columns)
scaler = MinMaxScaler()
for col in data.columns:
    data[col] = scaler.fit_transform(np.array(data[col]).reshape(-1, 1))
data.shape

data_pos = data[data.national_player == 1.0]
data_neg = data[data.national_player == 0.0]

validation_spilt_ratio = 0.25

data_pos_0 = data_pos.sample(frac=validation_spilt_ratio)
data_pos_1 = data_pos.drop(data_pos_0.index)

data_neg_0 = data_neg.sample(frac=validation_spilt_ratio)
data_neg_1 = data_neg.drop(data_neg_0.index)

data_train = pd.concat([data_pos_1, data_neg_1], ignore_index=True)
data_train = data_train.sample(frac=1).reset_index(drop=True)
data_validation = pd.concat([data_pos_0, data_neg_0], ignore_index=True)
data_validation = data_validation.sample(frac=1).reset_index(drop=True)
data_validation.shape

X = np.array(data_train.drop(['national_player'], 1)).astype(float)
y = np.array(data_train['national_player']).astype(float)
def up_sample_minority_class(df, random_state, is_future_selection_to_be_made):
    # Entries of the both minority and majority classes
    data_majority = df.loc[df['national_player'] == 0.0]
    data_minority = df.loc[df['national_player'] == 1.0]
    
    print("data_majority: {0} @ data_minority: {1}".format(len(data_majority), len(data_minority)))
    
    #populates the minority portion of the samples up to the size of majority portion
    data_minority_up_sampled = resample(data_minority, 
                                     replace=True,
                                     n_samples=len(data_majority),
                                     random_state=random_state)
    
    # Combine majority class with upsampled minority class
    data_up_sampled = pd.concat([data_majority, data_minority_up_sampled])
    
    # Display new class counts
    print(data_up_sampled.national_player.value_counts())
    
    X_up_sampled = np.array(data_up_sampled.drop(['national_player'], 1).astype(float))
    y_up_sampled = np.array(data_up_sampled['national_player']).astype(float)
    
    if is_future_selection_to_be_made:
        X_up_sampled = SelectKBest(chi2, k=10).fit_transform(X_up_sampled, y_up_sampled)
    
    X_train_up_sampled, X_test_up_sampled, y_train_up_sampled, y_test_up_sampled = train_test_split(X_up_sampled, y_up_sampled, random_state=random_state)
    
    return X_train_up_sampled, X_test_up_sampled, y_train_up_sampled, y_test_up_sampled, X_up_sampled, y_up_sampled
#X = SelectKBest(chi2, k=40).fit_transform(X, y)

X_train, X_test, y_train, y_test, X, y = up_sample_minority_class(data_train, 28, False)
#train_test_split(X, y, test_size=0.25)

# Tried Principal Component Analysis to make the calculations faster 
# and see if it improves the performance by any chance


# from sklearn.decomposition import PCA

# # feature extraction
# pca = PCA(n_components=X_train.shape[1]-0)
# fit = pca.fit(X_train)

# # # summarize components
# # #print("Explained Variance: %s" % fit.explained_variance_ratio_)
# # #print(fit.components_)

# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)





clf = RandomForestClassifier(max_depth=24, n_estimators=60, n_jobs=-2)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)


# threshold = 0.35

# prediction_probablities = clf.predict_proba(X_test)
# predictions = (prediction_probablities [:,1] >= threshold).astype('float')


print("\nAccuracy Score: {0}".format(accuracy_score(y_test, predictions)))
print("Precision Score: {0}".format(precision_score(y_test, predictions)))
print("Recall Score: {0}".format(recall_score(y_test, predictions)))
print("F-Beta Score: {0}".format(fbeta_score(y_test, predictions, beta=1.0)))

X_validation = np.array(data_validation.drop(['national_player'], 1)).astype(float)
y_validation = np.array(data_validation['national_player']).astype(float)


X_train_validation, X_test_validation, y_train_validation, y_test_validation, X_validation, y_validation = up_sample_minority_class(data_validation, 145, False)
#train_test_split(X_validation, y_validation, test_size=0.25)
#train_test_split(X, y, test_size=0.25)

# print("X_test_validation: "+str(X_test_validation.shape))


# X_test_validation = pca.transform(X_test_validation)

# print("X_test_validation: "+str(X_test_validation.shape))



predictions_validation = clf.predict(X_test_validation)


threshold = 0.33

prediction_probablities_validation = clf.predict_proba(X_test_validation)
predictions_validation = (prediction_probablities_validation [:,1] >= threshold).astype('float')


print("\nValidation Accuracy Score: {0}".format(accuracy_score(y_test_validation, predictions_validation)))
print("Validation Precision Score: {0}".format(precision_score(y_test_validation, predictions_validation)))
print("Validation Recall Score: {0}".format(recall_score(y_test_validation, predictions_validation)))
print("Validation F-Beta Score: {0}".format(fbeta_score(y_test_validation, predictions_validation, beta=2.0)))

prediction_probablities_validation = clf.predict_proba(X_test_validation)
predictions_validation = (prediction_probablities_validation [:,1] >= threshold).astype('float')


wrongly_predicted_sample_indices = []
for i in range(len(predictions_validation)):
    if predictions_validation[i] != y_test_validation[i]:
        wrongly_predicted_sample_indices.append(i)

len(wrongly_predicted_sample_indices)
data_y_test_validation = pd.DataFrame(data = y_test_validation[wrongly_predicted_sample_indices], columns = ['national_player']).reset_index()
data_X_test_validation = pd.DataFrame(data = X_test_validation[wrongly_predicted_sample_indices], columns = data.drop(['national_player'], axis = 1).columns).reset_index()
data_wrongly_predicted = pd.merge(data_X_test_validation, data_y_test_validation, how = 'left', left_index = True, right_index = True).drop(['index_x', 'index_y'], axis=1)
data_wrongly_predicted.head()
print("Age.mean for data_wrongly_predicted: {}, Age.mean for data_validation: {}"
      .format(data_wrongly_predicted.Age.mean(), data_validation.Age.mean()))
comparison_table = pd.DataFrame(None, columns = ['row_id'] + list(data.columns))
data_wrongly_predicted.name = 'data_wrongly_predicted'
data_validation.name = 'data_validation'


for df in [data_wrongly_predicted, data_validation]:
    s = df.mean()
    s['row_id'] = df.name + '.mean()'
    comparison_table = comparison_table.append(s, ignore_index=True)


for df in [data_wrongly_predicted, data_validation]:
    s = df.sum()
    s['row_id'] = df.name + '.sum()'
    comparison_table = comparison_table.append(s, ignore_index=True)
i = 4
comparison_table.iloc[:,i*20:(i+1)*20]

for col in comparison_table.columns:
    if col in ['row_id']:
        comparison_table.at[4, col] = 'wrong_ratio/all_ratio'
        comparison_table.at[5, col] = 'wrong_count/all_count'
    elif comparison_table.at[1, col] != 0.0 and comparison_table.at[3, col] != 0.0:
        comparison_table.at[4, col] = comparison_table.at[0, col] / comparison_table.at[1, col]
        comparison_table.at[5, col] = comparison_table.at[2, col] / comparison_table.at[3, col]
        
comparison_table[['row_id', 'number_of_national_players_by_nation']]
comparison_table_T = comparison_table.T
comparison_table_T.head()
comparison_table_T.columns = list(np.array(comparison_table_T[:1]))[0]
comparison_table_T.shape
comparison_table_T.head()
comparison_table_T = comparison_table_T.drop(comparison_table_T.index[0])
comparison_table_T.head()
comparison_table_T = comparison_table_T.dropna( how='any')
comparison_table_T.head()
comparison_table_T = comparison_table_T.sort_values(by=['data_wrongly_predicted.sum()'], ascending=False)
i=5
comparison_table_T.iloc[10*i:10*(i+1),:]
contribution_to_wrong_predicitons = 0
count = 0
for col in comparison_table_T.index:
    if 'Nationality_' in col and comparison_table_T['wrong_ratio/all_ratio'][col] > 1.1:
        print("col: {}  ---  {}  ---  {}".format( col,  comparison_table_T['data_wrongly_predicted.sum()'][col] ,  comparison_table_T['wrong_ratio/all_ratio'][col] )) 
        contribution_to_wrong_predicitons += comparison_table_T['data_wrongly_predicted.sum()'][col]
        count = count + 1

i=5
comparison_table.iloc[:10,15*i:15*(i+1)]
data_wrongly_predicted.head()
#TODO