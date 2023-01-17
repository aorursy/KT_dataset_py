# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv", low_memory=False)
df["WindSpeed"].unique()
WindSpeed_dict = {
       'SSW':np.nan, '11-17':14, '14-23':18.5, '13 MPH':13, '12-22':17, '4 MPh':4, '15 gusts up to 25':15,
       '10MPH':10, '10mph':10, 'E':np.nan, '7 MPH':7, 'Calm':np.nan, '6 mph':6, 'SE':np.nan, '10-20':15, '12mph':12,
       '6mph':6, '9mph':9, 'SSE':np.nan, '14 Gusting to 24':14, '6 mph, Gusts to 10':6, '2 mph, gusts to 5':2,
       '12 mph':12,'9 mph, gusts to 13':9, '10 mph, gusts to 15':10
}
df["WindSpeed"] = df["WindSpeed"].replace(WindSpeed_dict).astype(float)
df[(df['PossessionTeam']!=df['HomeTeamAbbr'])&(df['PossessionTeam']!=df['VisitorTeamAbbr'])]['PossessionTeam'].unique()
df[(df['FieldPosition']!=df['HomeTeamAbbr'])&(df['FieldPosition']!=df['VisitorTeamAbbr'])]['FieldPosition'].unique()
df['HomeTeamAbbr'].unique()
TeamAbbr_dict = {
    'BLT': 'BAL', 'CLV': 'CLE', 'ARZ':'ARI', 'HST':'HOU'
}
df['FieldPosition'].fillna("None", inplace=True)
df['PossessionTeam'].replace(TeamAbbr_dict, inplace=True)
df['FieldPosition'].replace(TeamAbbr_dict, inplace=True)
df['PlayerBirthDate'] = pd.to_datetime(df['PlayerBirthDate'], infer_datetime_format=True)
df['Position'].replace({'SAF':'S'}, inplace=True)
Stadium_dict = {
       'Broncos Stadium at Mile High' : 'Broncos Stadium At Mile High', 
       'CenturyField' : 'CenturyLink Field',
       'Tottenham Hotspur' : 'Tottenham Hotspur Stadium',
       'Azteca Stadium' : 'Estadio Azteca',
       'Twickenham' : 'Twickenham Stadium',
       'MetLife' : 'MetLife Stadium',
       'CenturyLink' : 'CenturyLink Field',
       'M&T Stadium':'M&T Bank Stadium',
       'First Energy Stadium' : 'FirstEnergy Stadium',
       'Los Angeles Memorial Coliesum':'Los Angeles Memorial Coliseum',
       'M & T Bank Stadium' : 'M&T Bank Stadium',
       'FirstEnergyStadium' : 'FirstEnergy Stadium',
       'Paul Brown Stdium' : 'Paul Brown Stadium', 
       'FedexField': 'FedExField',
       'FirstEnergy' : 'FirstEnergy Stadium',
       'Everbank Field' : 'EverBank Field',
       'Mercedes-Benz Dome' : 'Mercedes-Benz Superdome',
       'Lambeau field' : 'Lambeau Field',
       'NRG' : 'NRG Stadium'
}
df['Stadium'].replace(Stadium_dict, inplace=True)
df['Location'].unique()
Location_dict = {
    'Foxborough, MA' : 'MA',
    'Orchard Park NY': 'NY',
    'Chicago. IL' : 'IL',
    'Cincinnati, Ohio': 'OH',
    'Cleveland, Ohio' : 'OH',
    'Detroit, MI': 'MI',
    'Houston, Texas': 'TX',
    'Nashville, TN' : 'TN',
    'Landover, MD' : 'MD',
    'Los Angeles, Calif.' : 'CA',
    'Green Bay, WI' : 'WI',
    'Santa Clara, CA' : 'CA',
    'Arlington, Texas': 'TX',
    'Minneapolis, MN' : 'MN',
    'Denver, CO' : 'CO',
    'Baltimore, Md.' : 'MD',
    'Charlotte, North Carolina' : 'NC',
    'Indianapolis, Ind.' : 'ID',
    'Jacksonville, FL' : 'FL',
    'Kansas City, MO' : 'MO',
    'New Orleans, LA' : 'LA',
    'Pittsburgh' : 'PA',
    'Tampa, FL' : 'FL',
    'Carson, CA' : 'CA',
    'Oakland, CA' : 'CA',
    'Seattle, WA' : 'WA',
    'Atlanta, GA' : 'GA',
    'East Rutherford, NJ': 'NJ',
    'London, England' : 'ENG',
    'Chicago, IL' : 'IL',
    'Detroit' : 'MI',
    'Philadelphia, Pa.' : 'PA',
    'Glendale, AZ' : 'AZ',
    'Cleveland, OH' : 'OH',
    'Foxborough, Ma' : 'MA',
    'E. Rutherford, NJ' : 'NJ',
    'Miami Gardens, Fla.' : 'FL',
    'Houston, TX' : 'TX',
    'London':'ENG',
    'New Orleans, La.' : 'LA',
    'Mexico City' : 'MEX',
    'Baltimore, Maryland':'MA',
    'Arlington, TX' : 'TX',
    'Jacksonville, Fl' : 'FL',
    'Jacksonville, Florida' : 'FL',
    'Pittsburgh, PA': 'PA',
    'Charlotte, NC' : 'NC',
    'Cleveland,Ohio' : 'OH',
    'East Rutherford, N.J.' : 'NJ',
    'Philadelphia, PA' : 'PA',
    'Seattle' : 'WA',
    'Cleveland Ohio' : 'OH',
    'Miami Gardens, FLA' : 'FL',
    'Orchard Park, NY' : 'NY',
    'Cleveland' : 'OH',
    'Cincinnati, OH' : 'OH',
    'Kansas City,  MO' : 'MO',
    'Jacksonville Florida' : 'FL',
    'Los Angeles, CA' : 'CA',
    'New Orleans' : 'LA',
    'Chicago' : 'IL',
    'Charlotte North Carolina' : 'NC',
    'Miami Gardens, FL' : 'FL',
    'Denver CO' : 'CO',
    'Santa Clara, CSA' : 'CA',
    'Baltimore, MD' : 'MD',
    'Mexico City, Mexico' : 'MEX'
}
df['Location'].replace(Location_dict, inplace=True)
df['StadiumType'].sort_values().unique()
df['StadiumType'].fillna("None", inplace=True)
StadiumType_dict = {
    'Bowl' : 'Outdoor',
    'Closed Dome': 'Closed',
    'Cloudy' : 'None',
    'Dome' : 'Closed',
    'Dome, closed' : 'Closed',
    'Domed' : 'Closed',
    'Domed, Open' : 'Open',
    'Domed, closed' : 'Closed',
    'Domed, open' : 'Open',
    'Heinz Field' : 'Outdoor',
    'Indoor' : 'Closed',
    'Indoor, Open Roof' : 'Open',
    'Indoor, Roof Closed' : 'Closed',
    'Indoor, roof open' : 'Open',
    'Indoors' : 'Closed',
    'OUTDOOR' : 'Outdoor',
    'Open' : 'Outdoor',
    'Oudoor' : 'Outdoor',
    'Ourdoor' : 'Outdoor',
    'Outddors' : 'Outdoor',
    'Outdoor Retr Roof-Open' : 'Open',
    'Outdoors' : 'Outdoor',
    'Outdor' : 'Outdoor',
    'Outside' : 'Outdoor',
    'Retr. Roof - Closed' : 'Closed',
    'Retr. Roof - Open' : 'Open',
    'Retr. Roof Closed' : 'Closed',
    'Retr. Roof-Closed' : 'Closed',
    'Retr. Roof-Open' : 'Open',
    'Retractable Roof' : 'Closed',
    'Retractable Roof - Closed' : 'Closed',
    'indoor' : 'Closed'
}
df['StadiumType'].replace(StadiumType_dict, inplace=True)
df['Turf'].sort_values().unique()
Turf_dict = {
    'A-Turf Titan' : 'Artificial',
    'Artifical' : 'Artificial',
    'DD GrassMaster' : 'Artificial',
    'Field Turf' : 'Artificial',
    'Field turf' : 'Artificial',
    'FieldTurf' : 'Artificial',
    'FieldTurf 360' : 'Artificial',
    'FieldTurf360' : 'Artificial',
    'Grass' : 'Natural',
    'Natural Grass' : 'Natural' ,
    'Natural grass' : 'Natural',
    'Naturall Grass' : 'Natural',
    'SISGrass' : 'Artificial',
    'Turf' : 'Artificial',
    'Twenty Four/Seven Turf' : 'Artificial',
    'Twenty-Four/Seven Turf' : 'Artificial',
    'UBU Speed Series-S5-M' : 'Artificial',
    'UBU Sports Speed S5-M' : 'Artificial',
    'UBU-Speed Series-S5-M' : 'Artificial',
    'grass' : 'Natural',
    'natural grass' : 'Natural'
}
df['Turf'].replace(Turf_dict, inplace=True)
df['GameWeather'].sort_values().unique()
df['GameWeather'].fillna('None', inplace=True)
GameWeather_dict = {
    '30% Chance of Rain' : 'None',
    'Breezy': 'Clear',
    'Clear Skies' : 'Clear',
    'Clear and Cool' : 'Clear',
    'Clear and Sunny' : 'Clear',
    'Clear and cold' : 'Clear',
    'Clear and sunny' : 'Clear',
    'Clear and warm' : 'Clear',
    'Clear skies' : 'Clear',
    'Cloudy and Cool' : 'Cloudy',
    'Cloudy and cold' : 'Cloudy',
    'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.' : 'Cloudy',
    'Cloudy with showers and wind' : 'Cloudy',
    'Cloudy, 50% change of rain' : 'Cloudy',
    'Cloudy, Rain' : 'Cloudy',
    'Cloudy, chance of rain' : 'Cloudy',
    'Cloudy, fog started developing in 2nd quarter' : 'Cloudy',
    'Cloudy, light snow accumulating 1-3"' : 'Snow',
    'Cold' : 'None',
    'Controlled Climate' : 'None',
    'Coudy' : 'Cloudy',
    'Fair' : 'Clear',
    'Hazy' : 'Cloudy',
    'Heavy lake effect snow' : 'Snow',
    'Indoor' : 'None',
    'Indoors' : 'None',
    'Light Rain' : 'Rain',
    'Light rain' : 'Rain',
    'Mostly Clear' : 'Clear',
    'Mostly Cloudy' :'Cloudy',
    'Mostly Coudy' : 'Cloudy',
    'Mostly Sunny' : 'Clear',
    'Mostly Sunny Skies' : 'Clear',
    'Mostly clear' : 'Clear',
    'Mostly cloudy' : 'Cloudy',
    'Mostly sunny' : 'Clear',
    'N/A (Indoors)' : 'None',
    'N/A Indoor' : 'None',
    'N/A Indoors' : 'None',
    'Overcast' : 'Cloudy',
    'Partly Cloudy' : 'Cloudy',
    'Partly Clouidy' : 'Cloudy',
    'Partly Sunny' : 'Clear',
    'Partly clear' : 'Clear',
    'Partly cloudy' : 'Cloudy',
    'Partly cloudy and mild' : 'Cloudy',
    'Partly sunny' : 'Clear',
    'Party Cloudy' : 'Cloudy',
    'Rain' : 'Rain',
    'Rain Chance 40%' : 'None',
    'Rain and Wind': 'Rain',
    'Rain likely, temps in low 40s.' : 'None',
    'Rain shower' : 'Rain',
    'Raining' : 'Rain',
    'Rainy' : 'Rain',
    'Scattered Showers' : 'Rain',
    'Showers' : 'Rain',
    'Sun & clouds' : 'Cloudy',
    'Sunny' : 'Clear',
    'Sunny Skies' : 'Clear',
    'Sunny and clear' : 'Clear',
    'Sunny and cold' : 'Clear',
    'Sunny and warm' : 'Clear',
    'Sunny, Windy' : 'Clear',
    'Sunny, highs to upper 80s' : 'Clear',
    'T: 51; H: 55; W: NW 10 mph' : 'None',
    'cloudy' : 'Cloudy',
    'overcast' : 'Cloudy',
    'partly cloudy' : 'Cloudy',
    'sUNNY' : 'Clear'
}
df['GameWeather'].replace(GameWeather_dict, inplace=True)
df['WindDirection'].sort_values().unique()
df['WindDirection'].fillna("None", inplace=True)
WindDirection_dict = {
    '1' : 'None',
    '13' : 'None',
    '8' : 'None',
    'Calm' : 'None',
    'EAST' : 'E',
    'East' : 'E',
    'East North East' : 'ENE',
    'East Southeast' : 'ESE',
    'From ESE' : 'ESE',
    'From NE' : 'NE',
    'From NNE' : 'NNE',
    'From NNW' : 'NNW',
    'From S' : 'S',
    'From SSE' : 'SSE',
    'From SSW' : 'SSW',
    'From SW' : 'SW',
    'From W' : 'W',
    'From WSW' : 'WSW',
    'N-NE' : 'NNE',
    'North' : 'N',
    'North East' : 'NE',
    'North/Northwest' : 'NNW',
    'NorthEast' : 'NE',
    'Northeast' : 'NE',
    'Northwest' : 'NW',
    'S-SW' : 'SSW',
    'South' : 'S',
    'South Southeast' : 'SSE',
    'South Southwest' : 'SSW',
    'South west' : 'SW',
    'South, Southeast' : 'SSE',
    'SouthWest' : 'SW',
    'Southeast' : 'SE',
    'Southerly' : 'S',    
    'Southwest' : 'SW',
    'W-NW' : 'WNW',
    'W-SW' : 'WSW',
    'West' : 'W',
    'West Northwest' : 'WNW',
    'West-Southwest' : 'WSW',
    'from W' : 'W',
    's' : 'S'
}
df['WindDirection'].replace(WindDirection_dict, inplace=True)
df['OffenseFormation'].fillna("EMPTY", inplace=True)
aux = df['GameClock'].str.split(':', expand=True).astype(int)
df['GameClockSeconds'] = aux[0]*60+aux[1]
df.drop(columns='GameClock', inplace=True)
aux = df['PlayerHeight'].str.split('-', expand=True).astype('int')
df['PlayerHeightFt'] = aux[0]*12+aux[1]
df.drop(['PlayerHeight'], axis=1, inplace=True)
df['NewYardLine'] = 0
df.loc[df['FieldPosition']=="None", ['NewYardLine']] = 50
df.loc[df['FieldPosition']==df['PossessionTeam'], ['NewYardLine']] = df['YardLine']
df.loc[df['FieldPosition']!=df['PossessionTeam'], ['NewYardLine']] = 100 - df['YardLine']
df['Rusher'] = 0
df.loc[df['NflIdRusher']==df['NflId'],['Rusher']] = 1
df['PossessionFlag'] = 0
df.loc[(df['PossessionTeam']==df['HomeTeamAbbr']) & (df['Team']=='home'), ['PossessionFlag']] = 1
df.loc[(df['PossessionTeam']==df['VisitorTeamAbbr']) & (df['Team']=='away'), ['PossessionFlag']] = 1
df["Orientation"].fillna(0, inplace=True)
df["Dir"].fillna(0, inplace=True)
df["Humidity"].replace(0, np.nan, inplace=True)
df["Humidity"].fillna(df["Humidity"].mean(), inplace=True)
df["DefendersInTheBox"].fillna(df["DefendersInTheBox"].mean(), inplace=True)
df["Temperature"].fillna(df["Temperature"].mean(), inplace=True)
df["WindSpeed"].fillna(df["WindSpeed"].mean(), inplace=True)
df.columns
play_columns = ['GameId', 'PlayId', 'Season', 'Quarter', 'PossessionTeam',
       'Down', 'Distance', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'NflIdRusher',
       'OffenseFormation', 'OffensePersonnel', 'DefendersInTheBox',
       'DefensePersonnel', 'PlayDirection', 'Yards',
       'HomeTeamAbbr', 'VisitorTeamAbbr', 'Week', 'Stadium', 'Location',
       'StadiumType', 'Turf', 'GameWeather', 'Temperature', 'Humidity',
       'WindSpeed', 'WindDirection', 'GameClockSeconds',
       'NewYardLine']
player_columns = ['PlayId', 'Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation','Dir','PlayerWeight',
                  'PlayerBirthDate', 'PlayerCollegeName', 'Position',  'PlayerHeightFt', 'Rusher', 'PossessionFlag']
plays = df[play_columns].drop_duplicates()
plays.shape[0]
players = df[player_columns]
players
players['N'] = players.groupby('PlayId').cumcount()
players = players.pivot_table(
    values = [
        'Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation',
        'Dir', 'PlayerWeight', 'PlayerBirthDate',
        'PlayerCollegeName', 'Position', 'PlayerHeightFt',
        'Rusher', 'PossessionFlag', 
             ],
    index = [
        'PlayId'
    ],
    columns = [
        'N'
    ],
    aggfunc = 'first'
)
players.columns = players.columns.get_level_values(0).astype(str)+'_'+players.columns.get_level_values(1).astype(str)
categ_players_cols = players.select_dtypes(include='object').columns
num_players_cols = players.select_dtypes(exclude='object').columns
players = pd.concat([players[num_players_cols],pd.get_dummies(players[categ_players_cols])], axis=1).reset_index()
categ_plays_cols = [
    'PossessionTeam', 'OffenseFormation', 'OffensePersonnel', 
    'DefensePersonnel', 'PlayDirection', 'Stadium',
    'Location', 'StadiumType', 'Turf', 'GameWeather',
    'WindDirection'
]
num_plays_cols = [
    'PlayId', 'Season', 'Quarter', 'Down', 'Distance',
    'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
    'DefendersInTheBox', 'Week', 'Temperature', 'Yards',
    'Humidity', 'WindSpeed', 'GameClockSeconds', 'NewYardLine'
]
plays = pd.concat([plays[num_plays_cols], pd.get_dummies(plays[categ_plays_cols])], axis=1).reset_index(drop=True)
model_df = pd.concat([plays, players], axis=1)
aux = pd.DataFrame(columns=['Yards' + str(i) for i in range(-99, 100)])
yards_df = pd.concat([aux, pd.concat([pd.DataFrame([1], columns=['Yards'+str(i)]) for i in plays['Yards']], ignore_index=True)]).fillna(0).astype(int)
model_df = pd.concat([model_df, yards_df], axis=1)
model_df.head()
