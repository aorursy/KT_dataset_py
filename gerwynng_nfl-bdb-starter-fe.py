import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from kaggle.competitions import nflrush



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
##### FE PART 1
def player_features(df):

    df['X'] = np.where(df['PlayDirection'] == 'left', df['X'].apply(lambda x: 120 - x), df['X'])

    df['Y'] = np.where(df['PlayDirection'] == 'left', df['Y'].apply(lambda x: 53.3 - x), df['Y'])

    df['Dir'] = np.where(df['PlayDirection'] == 'left', df['Dir'].apply(lambda x: x + 180), df['Dir'])

    df['Dir'] = df['Dir'].apply(lambda x: x if x <= 360 else x-360)

    df['Orientation'] = np.where(df['PlayDirection'] == 'left', df['Orientation'].apply(lambda x: x + 180), df['Orientation'])

    df['Orientation'] = df['Orientation'].apply(lambda x: x if x <= 360 else x-360)

    

    df['PlayerHeight'] = df['PlayerHeight'].str.split('-',expand=True)[0].astype('int16') * 12 + df['PlayerHeight'].str.split('-',expand=True)[1].astype('int16')

    df['PlayerAge'] = df['Season'].astype('int16') - df['PlayerBirthDate'].str.split('/',expand=True)[2].astype('int16')

    df['PlayerBMI'] = 703*df['PlayerWeight']/df['PlayerHeight']**2

    

    bins = [9 + i*10 for i in range(0,10)]

    df['JerseyNumber'] = pd.cut(df['JerseyNumber'], bins=bins)

    

    df.drop('PlayerBirthDate', axis=1, inplace=True)

    return df
def team_features(df):

    arr = [[int(s[0]) for s in t.split(", ")] for t in df["DefensePersonnel"]]

    df["DL"] = pd.Series([a[0] for a in arr])

    df["LB"] = pd.Series([a[1] for a in arr])

    df["DB"] = pd.Series([a[2] for a in arr])



    for op in ['RB', 'TE', 'WR', 'DB', 'DL', 'LB', 'QB', 'OL']:

        df[op] = df['OffensePersonnel'].apply(lambda x: x.split(' '+str(op))[0][-1] if str(op) in x else 0).astype('int16')

        

    df.drop(['OffensePersonnel', 'DefensePersonnel'], axis=1, inplace = True)

    return df
def get_seconds(x):

    timestamp = x.split('T')[1].split('.')[0]

    timestamp = int(timestamp.split(':')[0])*3600 + int(timestamp.split(':')[1])*60 + int(timestamp.split(':')[2]) 

    return timestamp





def time_features(df):

    df['TimeDelta'] = df['TimeHandoff'].apply(get_seconds) - df['TimeSnap'].apply(get_seconds)

    df['TimeDelta'] = df['TimeDelta'].apply(lambda x: x if x >= 0 else x + 24*60*60)

    

    df["GameHour"] = df['GameClock'].str.split(':', expand=True)[0].astype('int16')

    df["GameMinute"] = df['GameClock'].str.split(':', expand=True)[1].astype('int16')

    

    df.drop(['TimeHandoff', 'TimeSnap', 'GameClock'], axis = 1, inplace = True)

    return df