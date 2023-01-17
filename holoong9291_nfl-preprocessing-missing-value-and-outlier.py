import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', parse_dates=['TimeHandoff','TimeSnap'], infer_datetime_format=True, low_memory=False)
df_train.isnull().sum()
df_train[['GameWeather','Temperature','Humidity','WindSpeed','WindDirection']] = df_train[['GameWeather','Temperature','Humidity','WindSpeed','WindDirection']].fillna(method='ffill')



def fill_stadiumtype(row):

    if row['Stadium'] in ['StubHub Center','MetLife Stadium'] and pd.isnull(row['StadiumType']):

        return 'Outdoor'

    return row['StadiumType']



df_train.StadiumType = df_train.apply(fill_stadiumtype, axis=1)

df_train.StadiumType = df_train.StadiumType.fillna(method='ffill')



df_train.FieldPosition = df_train.FieldPosition.fillna('Middle')



df_train.OffenseFormation = df_train.OffenseFormation.fillna('SINGLEBACK')



defendersInTheBox = df_train.groupby(['Team','HomeTeamAbbr','VisitorTeamAbbr','DefensePersonnel']).DefendersInTheBox.median()

def fill_defendersinthebox(row):

    if pd.isnull(row['DefendersInTheBox']):

        return defendersInTheBox[row['Team']][row['HomeTeamAbbr']][row['VisitorTeamAbbr']][row['DefensePersonnel']]

    return row['DefendersInTheBox']

df_train.DefendersInTheBox = df_train.apply(fill_defendersinthebox, axis=1)



df_train.Orientation = df_train.Orientation.fillna(df_train.Orientation.mean())

df_train.Dir = df_train.Dir.fillna(df_train.Dir.mean())
df_train.isnull().sum()
df_train.StadiumType.unique()
stadiumtype_map = {

    'Outdoor':'Outdoor','Outdoors':'Outdoor','Outddors':'Outdoor','Oudoor':'Outdoor','Ourdoor':'Outdoor','Outdor':'Outdoor','Outside':'Outdoor',

    'Indoors':'Indoor','Indoor':'Indoor',

    'Retractable Roof':'Retractable Roof',

    'Retr. Roof-Closed':'Retr. Roof-Closed','Retr. Roof - Closed':'Retr. Roof-Closed','Retr. Roof Closed':'Retr. Roof-Closed',

    'Retr. Roof-Open':'Retr. Roof-Open','Retr. Roof - Open':'Retr. Roof-Open',

    'Open':'Open',

    'Indoor, Open Roof':'Indoor, Open Roof',

    'Indoor, Roof Closed':'Indoor, Roof Closed',

    'Outdoor Retr Roof-Open':'Outdoor Retr Roof-Open',

    'Dome':'Dome','Domed':'Dome',

    'Domed, closed':'Domed, closed','Closed Dome':'Domed, closed','Dome, closed':'Domed, closed',

    'Domed, Open':'Domed, Open','Domed, open':'Domed, Open',

    'Heinz Field':'Heinz Field',

    'Cloudy':'Cloudy',

    'Bowl':'Bowl',

}

df_train.StadiumType = df_train.StadiumType.map(stadiumtype_map)
possessionteam_map = {

    'BLT':'BAL',

    'CLV':'CLE',

    'ARZ':'ARI',

    'HST':'HOU'

}

df_train.PossessionTeam = df_train.PossessionTeam.apply(lambda pt:possessionteam_map[pt] if pt in possessionteam_map.keys() else pt)
location_map = {

    'Foxborough, MA':'Foxborough',

    'Orchard Park NY':'Orchard Park','Orchard Park, NY':'Orchard Park',

    'Chicago. IL':'Chicago','Chicago, IL':'Chicago',

    'Cincinnati, Ohio':'Cincinnati','Cincinnati, OH':'Cincinnati',

    'Cleveland, Ohio':'Cleveland','Cleveland, OH':'Cleveland','Cleveland,Ohio':'Cleveland','Cleveland Ohio':'Cleveland','Cleveland':'Cleveland',

    'Detroit, MI':'Detroit','Detroit':'Detroit',

    'Houston, Texas':'Houston','Houston, TX':'Houston',

    'Nashville, TN':'Nashville',

    'Landover, MD':'Landover',

    'Los Angeles, Calif.':'Los Angeles','Los Angeles, CA':'Los Angeles',

    'Green Bay, WI':'Green Bay',

    'Santa Clara, CA':'Santa Clara',

    'Arlington, Texas':'Arlington','Arlington, TX':'Arlington',

    'Minneapolis, MN':'Minneapolis',

    'Denver, CO':'Denver',

    'Baltimore, Md.':'Baltimore','Baltimore, Maryland':'Baltimore',

    'Charlotte, North Carolina':'Charlotte','Charlotte, NC':'Charlotte',

    'Indianapolis, Ind.':'Indianapolis',

    'Jacksonville, FL':'Jacksonville','Jacksonville, Fl':'Jacksonville','Jacksonville, Florida':'Jacksonville','Jacksonville Florida':'Jacksonville',

    'Kansas City, MO':'Kansas City','Kansas City,  MO':'Kansas City',

    'New Orleans, LA':'New Orleans','New Orleans, La.':'New Orleans','New Orleans':'New Orleans',

    'Pittsburgh':'Pittsburgh','Pittsburgh, PA':'Pittsburgh',

    'Tampa, FL':'Tampa',

    'Carson, CA':'Carson',

    'Oakland, CA':'Oakland',

    'Seattle, WA':'Seattle','Seattle':'Seattle','Cleveland Ohio':'Seattle',

    'Atlanta, GA':'Atlanta',

    'East Rutherford, NJ':'East Rutherford','E. Rutherford, NJ':'East Rutherford','East Rutherford, N.J.':'East Rutherford',

    'London, England':'London','London':'London',

    'Philadelphia, Pa.':'Philadelphia','Philadelphia, PA':'Philadelphia',

    'Glendale, AZ':'Glendale',

    'Foxborough, Ma':'Foxborough',

    'Miami Gardens, Fla.':'Miami Gardens','Miami Gardens, FLA':'Miami Gardens',

    'Mexico City':'Mexico City',

    

}

df_train.Location = df_train.Location.map(location_map)