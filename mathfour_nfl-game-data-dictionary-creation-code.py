import pandas as pd
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
pd.options.display.float_format = lambda x: f' {x:,.2f}'
import warnings
warnings.filterwarnings("ignore")
games = pd.read_csv('../input/game_data.csv', parse_dates=['Game_Date'])
games.head(2)
# assign our variable so the rest of the code works nicely
data = games

# helper function to count the blanks (not the NaN's, but the actual blank values)
def count_blanks(column):
    count = 0
    for thing in column:
        if not thing:
            count += 1
    return count

# create the data dictionary directly (mostly empty info at this point)
data_dictionary = pd.DataFrame(index=data.columns, columns=['unique', 'missing', 'blank', 'read type',
                                                            'act type', 'description'])
for name in data.columns:
    data_dictionary.loc[name,'missing'] = data[name].isna().sum()
    data_dictionary.loc[name, 'unique'] = data[name].nunique()
    data_dictionary.loc[name,'blank'] = count_blanks(data[name])
    data_dictionary.loc[name, 'values'] = str(data[name].unique())
    data_dictionary.loc[name,'act type'] = 'string'
    data_dictionary.loc[name,'read type'] = str(data[name].dtype)
    
# just to make sure it's looking like I want
data_dictionary
# defining manually which columns should be which kind of data type

# the data that should be dtype string
data_strings = ['Season_Type', 'Game_Day','Game_Site', 'Start_Time', 
                'Home_Team', 'HomeTeamCode', 'Visit_Team', 'VisitTeamCode', 
                'Stadium', 'StadiumType', 'Turf', 'GameWeather', 'OutdoorWeather']
data_bools = [] # data that should be dtype boolean (none in this dataset)
data_ints = ['GameKey', 'Season_Year', 'Week'] # data that should be dtype int
data_floats = ['Temperature'] # data that should be dtype float
data_dates = ['Game_Date'] # data that should be dtype datetime

# defining manually what each column actually is
data_dictionary.loc['GameKey','description'] = (
            'Looks like just an index from 1 instead of 0')
data_dictionary.loc['Season_Year','description'] = (
            'One of the two years covered in this dataset')
data_dictionary.loc['Season_Type','description'] = (
            'Pre, post or regular season')
data_dictionary.loc['Week','description'] = (
            'The week in which the game is played')
data_dictionary.loc['Game_Date','description'] = (
            'Date on which the game is played')
data_dictionary.loc['Game_Day','description'] = (
            'Day of the week (word) in which the game is played')
data_dictionary.loc['Game_Site','description'] = (
            'City, country or other location of the stadium')
data_dictionary.loc['Start_Time','description'] = (
            'Start of the game (kickoff?) in 24 hour scale')
data_dictionary.loc['Home_Team','description'] = (
            'Full name of the home team including loc and mascot')
data_dictionary.loc['HomeTeamCode','description'] = (
            'Two or three letter team identifier')
data_dictionary.loc['Visit_Team','description'] = (
            'Full name of the visiting team including loc and mascot')
data_dictionary.loc['VisitTeamCode','description'] = (
            'Two or three letter team identifier')
data_dictionary.loc['Stadium','description'] = (
            'Full name of stadium')
data_dictionary.loc['StadiumType','description'] = (
            'Type as well as status of stadium. Many dup\'s b/c of typos')
data_dictionary.loc['Turf','description'] = (
            'Type of turf. Many dup\'s b/c of typos')
data_dictionary.loc['GameWeather','description'] = (
            'Sometimes detailed sometimes not. Should be cleaned')
data_dictionary.loc['Temperature','description'] = (
            'Temperature, assumingly in Fahrenheit')
data_dictionary.loc['OutdoorWeather','description'] = (
            'Sometimes detailed sometimes not. Not sure relation to GameWeather')
# insert into the dictionary df the types they should be
data_dictionary.loc[data_bools,'act type'] = 'bool'
data_dictionary.loc[data_ints,'act type'] = 'int'
data_dictionary.loc[data_floats,'act type'] = 'float'
data_dictionary.loc[data_dates,'act type'] = 'date'

data_dictionary
# save data dictionary to a file for later use if you want
# data_dictionary.to_csv('../output/NFL_Game_Data_Dictionary.csv')