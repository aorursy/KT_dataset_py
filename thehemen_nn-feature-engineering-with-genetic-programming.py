# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
def get_stadium_type(df_col):

    replacements = {

        'Outdoors': 'Outdoor',

        'Oudoor': 'Outdoor',

        'Outddors': 'Outdoor',

        'Outdor': 'Outdoor',

        'Ourdoor': 'Outdoor',

        'Outside': 'Outdoor',

        'Indoors': 'Indoor',

        'Retractable': 'Retr.',

        'closed': 'Closed',

        'open': 'Open',

        'Closed Dome': 'Domed, Closed',

        'Dome, Closed': 'Domed, Closed',

        'Domed': 'Dome',

        ' - ': ' ',

        '-': ' '

    }



    # fix stadium description

    for replaced, replacing in replacements.items():

        df_col = df_col.str.replace(replaced, replacing)



    df_col = pd.factorize(df_col)[0]

    return df_col
def get_stadium_turf(df_col):

    df_col = df_col.str.lower()



    replacements = {

        'natural grass': 'grass',

        'natural': 'grass',

        'naturall grass': 'grass',

        'fieldturf': 'field turf',

        'fieldturf 360': 'fieldturf360',

        'artifical': 'artificial'

    }



    # fix staidum turf types

    df_col = df_col.replace(replacements)

    df_col = pd.factorize(df_col)[0]

    return df_col
def get_game_weather(df_col):

    df_col = df_col.str.lower()



    fixes = {

        ' skies': '',

        'coudy': 'cloudy',

        'clouidy': 'cloudy',

        'party': 'partly'

    }



    # fix stadium descriptions

    for replaced, replacing in fixes.items():

        df_col = df_col.str.replace(replaced, replacing)



    weather_coeffs = {

        'sunny': 2.5, 

        'clear': 2.5, 

        'warm': 1.5, 

        'cold': -1.5,

        'hazy': -1.5,

        'cloud': -2.5, 

        'rain': -2.5, 

        'snow': -5.0

    }



    replacements = {}

    for weather_description_raw in df_col.unique():

        weather_description = str(weather_description_raw)

        weather = 0.0

        for weather_type, weather_coeff in weather_coeffs.items():

            if weather_type in weather_description:

                weather += weather_coeffs[weather_type]

        if 'partly' in weather_description:

            weather *= 0.5

        replacements[weather_description] = weather



    df_col = df_col.replace(replacements)

    return df_col
def get_wind_speed(df_col):

    df_col = df_col.astype(str)

    df_col = df_col.replace('15 gusts up to 25', '20')  # replace expression with average ;)

    df_col = df_col.str.replace(r'[^0-9\-]', '')  # remove all non-digits except a range sign

    df_col = df_col.replace('', '0')  # replace empty with zero



    range_vals = {}



    for range_val in [x for x in df_col.unique() if '-' in str(x)]:

        min_val = int(range_val.split('-')[0])

        max_val = int(range_val.split('-')[1])

        aver_val = int(min_val + (max_val - min_val) / 2)

        range_vals[range_val] = aver_val



    df_col = df_col.replace(range_vals)  # replace range expression with average value

    df_col = df_col.replace(np.nan, 0)  # replace nans with zero

    df_col = df_col.astype(np.int64)

    return df_col
def get_wind_direction(df_col):

    direction_signs = {

        'north': 'N',

        'east': 'E',

        'south': 'S',

        'west': 'W'

    }



    df_col = df_col.astype(str)

    df_col = df_col.str.lower()



    # replace long direction signs to short ones

    for replaced, replacing in direction_signs.items():

        df_col = df_col.str.replace(replaced, replacing)



    df_col = df_col.str.upper()

    df_col = df_col.str.replace(r'[^WNSE]', '')  # remove all unnecessary characters

    df_col = df_col.replace(np.nan, 0.0)  # replace nans with zero

    df_col = df_col.replace('', 0.0)  # replace empty with zero



    compass_rose = {

        'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5,

        'E': 90.0, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,

        'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,

        'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5

    }



    df_col = df_col.apply(lambda x: x if x == np.string_ else 'N')  # replace non-string with N

    df_col = df_col.replace(compass_rose)  # replace wind signs with angles by the compass rose

    return df_col
def get_personnel_by_positions(df_col, unique_positions):

    df_col = df_col.str.replace(', ', ',')  # remove spaces after commas

    unique_personnels = df_col.unique().tolist()

    unique_personnels_indexed = {}



    for unique_personnel in unique_personnels:

        index_positions = unique_personnel.split(',')

        indexes = [0] * len(unique_positions)



        for index_position in index_positions:

            index = int(index_position.split(' ')[0])

            unique_position = index_position.split(' ')[1]

            

            if unique_position in unique_positions:

                position = unique_positions.index(unique_position)

                indexes[position] = index



        result = 0



        for index, position in enumerate(indexes):

            result += (10 ** index) * position



        unique_personnels_indexed[unique_personnel] = result



    df_col = df_col.replace(unique_personnels_indexed).astype(np.int64)

    return df_col
def get_index_by_position(df_col):

    positions = {

        'QB': 0, 'CB': 1, 'WR': 2, 'G': 3, 'T': 4, 'DE': 5, 'DT': 6, 'OLB': 7,

        'TE': 8, 'FS': 9, 'C': 10,'RB': 11, 'SS': 12, 'ILB': 13, 'MLB': 14, 'NT': 15,

        'LB': 16, 'OT': 17, 'FB': 18, 'OG': 19, 'DB': 20, 'S': 21, 'HB': 22, 'SAF': 23,

        'DL': 24, '0': 25

    }

    df_col = df_col.replace(positions)

    return df_col
player_num = 22



def feature_engineering(df, is_trained=False):

    df_len = len(df.index)

    df_len_final = df_len // player_num



    df = df.fillna('0')

    df['GameClock'] = df['GameClock'].apply(lambda x: (int(x.split(':')[0]) * 60 + int(x.split(':')[1])) // 60)

    df['PossessionTeam'] = df['PossessionTeam'].replace({'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'})

    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].replace({'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'})

    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].replace({'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'})

    df['OffensePersonnel'] = get_personnel_by_positions(df['OffensePersonnel'], ['OL', 'QB', 'RB', 'TE', 'WR'])

    df['DefensePersonnel'] = get_personnel_by_positions(df['DefensePersonnel'], ['DB', 'DL', 'LB'])

    df['PlayDirection'] = (df['PlayDirection'] == 'right').astype(np.uint8)

    df['TimeSnap'] = pd.to_datetime(df['TimeSnap'], format='%Y-%m-%dT%H:%M:%S.000Z')

    df['TimeHandoff'] = pd.to_datetime(df['TimeHandoff'], format='%Y-%m-%dT%H:%M:%S.000Z')

    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))

    df['PlayerBirthDate'] = pd.to_datetime(df['PlayerBirthDate'], format='%m/%d/%Y')

    df['PlayerCollegeName'] = pd.factorize(df['PlayerCollegeName'])[0]

    df['Position'] = get_index_by_position(df['Position'])

    df['Stadium'] = pd.factorize(df['Stadium'])[0]

    df['Turf'] = get_stadium_turf(df['Turf'])

    df['GameWeather'] = get_game_weather(df['GameWeather'])

    df['Location'] = pd.factorize(df['Location'])[0]

    df['StadiumType'] = get_stadium_type(df['StadiumType'])

    df['WindSpeed'] = get_wind_speed(df['WindSpeed'])

    df['WindDirection'] = get_wind_direction(df['WindDirection'])

    

    df['WindDirection'] = df['WindDirection'].astype(np.int)

    df['WindDirection_COS'] = np.cos(np.deg2rad(df['WindDirection']))

    df['WindDirection_SIN'] = np.sin(np.deg2rad(df['WindDirection']))



    for i, offense_column in enumerate(['Offense_OL', 'Offense_QB', 'Offense_RB', 'Offense_TE', 'Offense_WR']):

        df[offense_column] = ((df['OffensePersonnel'] % (10 ** (i + 1)) - df['OffensePersonnel'] % (10 ** i)) / (10 ** i)).astype(np.int64)

    for i, defense_column in enumerate(['Defense_DB', 'Defense_DL', 'Defense_LB']):

        df[defense_column] = ((df['DefensePersonnel'] % (10 ** (i + 1)) - df['DefensePersonnel'] % (10 ** i)) / (10 ** i)).astype(np.int64)

    for i, offense_form_column in enumerate(['OF_SHOTGUN', 'OF_SINGLEBACK', 'OF_JUMBO', 'OF_PISTOL', 'OF_I_FORM', 'OF_ACE', 'OF_WILDCAT', 'OF_EMPTY']):

        df[offense_form_column] = (df['OffenseFormation'] == offense_form_column.split('_')[-1]).astype(np.int64)



    df['QuarterGameClock'] = df['GameClock'] % (15 * 60)

    df['TimeDelta'] = (df['TimeHandoff'] - df['TimeSnap']).dt.total_seconds()

    

    # Player features

    df['PlayerAge'] = ((df['TimeHandoff'] - df['PlayerBirthDate']).dt.total_seconds() / (60 * 60 * 24 * 365)).astype(np.int64)

    df['IsRusher'] = (df['NflId'] == df['NflIdRusher']).astype(np.uint8)

    df['IsOffense'] = ((df['PossessionTeam'] == df['HomeTeamAbbr']) & (df['Team'] == 'home')) | ((df['PossessionTeam'] == df['VisitorTeamAbbr']) & (df['Team'] == 'away'))

    df['Team'] = (df['Team'] == 'home').astype(np.uint8)



    df['OffenseX_mean'] = df[df['IsOffense'] == True].groupby(['PlayId'])['X'].mean().values.repeat(22)

    df['OffenseY_mean'] = df[df['IsOffense'] == True].groupby(['PlayId'])['Y'].mean().values.repeat(22)

    df['DefenseX_mean'] = df[df['IsOffense'] == False].groupby(['PlayId'])['X'].mean().values.repeat(22)

    df['DefenseY_mean'] = df[df['IsOffense'] == False].groupby(['PlayId'])['Y'].mean().values.repeat(22)



    df['OffenseX_std'] = df[df['IsOffense'] == True].groupby(['PlayId'])['X'].std().values.repeat(22)

    df['OffenseY_std'] = df[df['IsOffense'] == True].groupby(['PlayId'])['Y'].std().values.repeat(22)

    df['DefenseX_std'] = df[df['IsOffense'] == False].groupby(['PlayId'])['X'].std().values.repeat(22)

    df['DefenseY_std'] = df[df['IsOffense'] == False].groupby(['PlayId'])['Y'].std().values.repeat(22)



    df['RusherX'] = df[df['IsRusher'] == True]['X'].values.repeat(22)

    df['RusherY'] = df[df['IsRusher'] == True]['Y'].values.repeat(22)



    df['QuaterbackX'] = df[df['Position'] == 0]['X'].values[:df_len_final].repeat(22)

    df['QuaterbackY'] = df[df['Position'] == 0]['Y'].values[:df_len_final].repeat(22)

    

    # Plauer features

    df['Dir'] = df['Dir'].astype(np.int)

    df['Dir_COS'] = np.cos(np.deg2rad(df['Dir']))

    df['Dir_SIN'] = np.sin(np.deg2rad(df['Dir']))



    df['Orientation'] = df['Orientation'].astype(np.int)

    df['Orientation_COS'] = np.cos(np.deg2rad(df['Orientation']))

    df['Orientation_SIN'] = np.sin(np.deg2rad(df['Orientation']))



    df['S_horizontal'] = df['S'] * df['Dir_COS']

    df['S_vertical'] = df['S'] * df['Dir_SIN']



    df['A_horizontal'] = df['A'] * df['Dir_COS']

    df['A_vertical'] = df['A'] * df['Dir_SIN']



    df['Distance_to_YardLine'] = abs(df['X'] - df['YardLine'])

    df['Distance_to_Offense'] = np.sqrt((df['X'] - df['OffenseX_mean']) ** 2 + (df['Y'] - df['OffenseY_mean']) ** 2)

    df['Distance_to_Defense'] = np.sqrt((df['X'] - df['DefenseX_mean']) ** 2 + (df['Y'] - df['DefenseY_mean']) ** 2)

    df['Distance_to_Rusher'] = np.sqrt((df['X'] - df['RusherX']) ** 2 + (df['Y'] - df['RusherY']) ** 2)

    df['Distance_to_Quarterback'] = np.sqrt((df['X'] - df['QuaterbackX']) ** 2 + (df['Y'] - df['QuaterbackY']) ** 2)



    df = df.drop(columns=['GameId', 'PlayId', 'DisplayName', 'PossessionTeam', 'FieldPosition',

                          'OffenseFormation', 'OffensePersonnel', 'DefensePersonnel', 'TimeHandoff', 'TimeSnap',

                          'PlayerBirthDate', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'NflId', 'NflIdRusher',

                          'WindDirection', 'Dir', 'Orientation'])

    

    player_columns = [

        'X', 'Y', 'S', 'A', 'Dir_COS', 'Dir_SIN',

        'Orientation_COS', 'Orientation_SIN',

        'S_horizontal', 'S_vertical', 'A_horizontal', 'A_vertical',

        'Distance_to_YardLine', 'Distance_to_Offense', 'Distance_to_Defense',

        'Distance_to_Rusher', 'Distance_to_Quarterback',

        'Dis', 'JerseyNumber', 'PlayerHeight', 'PlayerWeight',

        'PlayerCollegeName', 'Position', 'PlayerAge',

        'Team', 'IsOffense', 'IsRusher'

    ]



    common_columns = [col for col in df.columns if col not in player_columns]

    if is_trained:

        common_columns.pop(common_columns.index('Yards'))

    

    cols = list(df.columns.values)

    cols_to_end = []

    

    if is_trained:

        cols.pop(cols.index('Yards'))



    for column_name in player_columns:

        cols_to_end.append(column_name)

        cols.pop(cols.index(column_name))



    if is_trained:

        cols_to_end.append('Yards')



    df = df[cols + cols_to_end]

    return df, common_columns, player_columns



train_df, common_features, player_features = feature_engineering(train_df, is_trained=True)

pd.set_option('display.max_columns', 100)

train_df.head(n=22)
from sklearn.preprocessing import StandardScaler



common_feature_num = len(common_features)

player_feature_num = len(player_features)



def make_dataset(scaler, df, is_trained=False):

    df_numpy_arr = df.to_numpy()



    all_row_length = len(df.index)

    X_row_len = all_row_length // player_num

    X_col_len = common_feature_num + player_feature_num * player_num

    X_train = np.zeros((X_row_len, X_col_len), dtype=np.float)

    

    for i in range(X_row_len):

        X_train[i, :common_feature_num] = df_numpy_arr[i * player_num, :common_feature_num]



        for j in range(player_num):

            np_y_left = common_feature_num + j * player_feature_num

            np_y_right = common_feature_num + (j + 1) * player_feature_num

            df_y_left = common_feature_num

            df_y_right = common_feature_num + player_feature_num

            X_train[i, np_y_left: np_y_right] = df_numpy_arr[i * player_num + j, df_y_left: df_y_right]



    X_train = np.nan_to_num(X_train)



    if is_trained:

        scaler.fit(X_train)

    

    X_train = scaler.transform(X_train)

    

    if is_trained:

        Y_train = np.zeros((X_row_len, 199), dtype=np.float)



        for i in range(X_row_len):

            yard = int(df_numpy_arr[i * player_num, -1])

            Y_train[i, yard + 99:] = np.ones(shape=(1, 100 - yard))



        return X_train, Y_train

    else:

        return X_train, None



print('Index\t{0}\t{1:24}\t{2}\t\t{3}'.format('Dtype', 'Column name', '1st value', 'Is equal by play'))

for i, column_name in enumerate(train_df.columns):

    print('{0}\t{1}\t{2:24}\t{3:16}\t{4}'.format(i, train_df.dtypes[i], column_name, str(train_df[column_name][0]), column_name in player_features))



scaler = StandardScaler()

X_train, Y_train = make_dataset(scaler, train_df, is_trained=True)

print('X_train: {}'.format(X_train.shape))

print('Y_train: {}'.format(Y_train.shape))
from IPython.display import display



# List of [common] and [player] feature masks, extracted by genetic programming

masks = [

    [[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],

     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],

    [[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],

     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]],

    [[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],

     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]],

    [[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],

     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],

    [[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],

     [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]],

    [[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],

     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]],

    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],

     [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]],

    [[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],

     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],

    [[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],

     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]],

    [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],

     [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]]

]



def generate_input_mask(common_feature_mask, player_feature_mask):

    return np.concatenate([common_feature_mask, np.tile(player_feature_mask, player_num)])



feature_count = len(masks)

input_masks = [generate_input_mask(masks[i][0], masks[i][1]) for i in range(feature_count)]



def get_input_now(X_train, input_mask):

    non_zero_inds = np.where(input_mask != 0.0)[0]

    X_train_now = X_train[:, non_zero_inds]

    return X_train_now



hamming_distance = np.zeros((feature_count, feature_count), dtype=np.int)

for i in range(feature_count):

    for j in range(feature_count):

        distance_0 = np.count_nonzero(np.logical_xor(masks[i][0], masks[j][0]))

        distance_1 = np.count_nonzero(np.logical_xor(masks[i][1], masks[j][1]))

        hamming_distance[i, j] = distance_0 + distance_1

        

hd = pd.DataFrame(data=hamming_distance,

                  index=['{}'.format(i) for i in range(feature_count)],

                  columns=['{}'.format(i) for i in range(feature_count)])

display(hd)
from keras.models import Model

from keras.layers import *



def build_model(input_dim):

    inputs = Input(shape=(input_dim,))

    

    x = Dense(384, activation=None)(inputs)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.35)(x)

    

    x = Dense(256, activation=None)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.35)(x)

    

    x = Dense(192, activation=None)(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.05)(x)

    x = Dropout(0.35)(x)

    

    outputs = Dense(199, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer='adam', loss='mse')

    return model



test_model = build_model(input_dim=X_train.shape[1])

test_model.summary()
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



models = []

val_loss = []



VERBOSE_MODE = 0

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, min_lr=1e-6, mode='min',

                              verbose=VERBOSE_MODE)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=8, mode='min',

                               restore_best_weights=True, verbose=VERBOSE_MODE)



for i in range(feature_count):

    print('Feature Index: {}'.format(i))

    print('Common Features: {}'.format(masks[i][0]))

    print('Player Features: {}'.format(masks[i][1]))

    X_train_now = get_input_now(X_train, input_masks[i])

    model = build_model(input_dim=X_train_now.shape[1])

    history = model.fit(X_train_now, Y_train,

                        validation_split=0.15, batch_size=64, epochs=32,

                        callbacks=[reduce_lr, early_stopping], verbose=VERBOSE_MODE)

    models.append(model)

    val_loss.append(history.history['val_loss'])

    print('Validation Loss: {:.6} (epochs: {})\n'.format(min(history.history['val_loss']), len(history.history['val_loss'])))
import matplotlib.pyplot as plt



for i in range(feature_count):

    plt.plot(val_loss[i])



plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.grid()

plt.show()
from kaggle.competitions import nflrush

env = nflrush.make_env()



for (test_df, sample_prediction_df) in tqdm.tqdm(env.iter_test()):

    test_df, _, _ = feature_engineering(test_df)

    X_test, _ = make_dataset(scaler, test_df)

    

    y_preds = []

    for i in range(feature_count):

        X_test_now = get_input_now(X_test, input_masks[i])

        y_pred_now = models[i].predict(X_test_now)

        y_preds.append(y_pred_now)

    y_pred = np.mean(y_preds, axis=0)

    

    for pred in y_pred:

        prev = 0

        for i in range(len(pred)):

            if pred[i] < prev:

                pred[i] = prev

            prev = pred[i]



    y_pred[:, -1] = np.ones(shape=(y_pred.shape[0], 1))

    y_pred[:, 0] = np.zeros(shape=(y_pred.shape[0], 1))

    env.predict(pd.DataFrame(data=y_pred, columns=sample_prediction_df.columns))



env.write_submission_file()