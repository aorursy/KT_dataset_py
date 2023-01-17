# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from kaggle.competitions import nflrush



env = nflrush.make_env()

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from tqdm import tqdm, trange

tqdm.pandas()

warnings.filterwarnings('ignore')
train_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')

print (train_data.shape)

train_data.head()
train_data.columns
train_data[(train_data['PossessionTeam']!=train_data['HomeTeamAbbr']) & (train_data['PossessionTeam']!=train_data['VisitorTeamAbbr'])][['PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr']]
'''

Data Processing Functions

1. Stadium Type

2. Stadium Turf

3. Game Weather

4. Wind Direction

5. Wind Speed

6. Player Height

7. Offense Personnel

8. Defense Personnel

9. Data Processing

''' 



def stadium_type(stype):

    if (stype == 'Outdoor' or stype == 'Outdoors' or stype == 'Cloudy' or stype == 'Heinz Field' or stype == 'Outdor' or stype == 'Ourdoor' or stype == 'Outside' or stype == 'Outddors' or stype == 'Outdoor Retr Roof-Open' or stype == 'Oudoor' or stype == 'Bowl'):

        return ('outdoor')

    elif (stype == 'Indoors' or stype == 'Indoor' or stype == 'Indoor, Roof Closed' or stype == 'Retractable Roof' or stype == 'Retr. Roof-Closed' or stype == 'Retr. Roof - Closed' or stype == 'Retr. Roof Closed'):

        return ('indoor_closed')

    elif (stype == 'Indoor, Open Roof' or stype == 'Open' or stype == 'Retr. Roof-Open' or stype == 'Retr. Roof - Open'):

        return ('indoor_open')

    elif (stype == 'Dome' or stype == 'Domed, closed' or stype == 'Closed Dome' or stype == 'Domed' or stype == 'Dome, closed'):

        return ('dome_closed')

    elif (stype == 'Domed, Open' or stype == 'Domed, open'):

        return ('dome_open')

    else:

        return ('outdoor')



def refine_turf(turf):

    if (turf == 'natural grass' or turf == 'Naturall Grass' or turf == 'Natural Grass'):

        return ('natural_grass')

    elif (turf == 'Grass'):

        return ('grass')

    elif (turf == 'FieldTurf' or turf == 'Field turf' or turf == 'FieldTurf360' or turf == 'Field Turf'):

        return ('fieldturf')

    elif (turf == 'Artificial' or turf == 'Artifical'):

        return ('artificial')

    else:

        return ('grass')



def refine_gameweather(weather):

    if (weather == 'Clear and warm' or weather == 'Clear' or weather == 'Clear skies' or weather == 'Clear and sunny' or weather == 'Clear and Cool' or weather == 'Sunny and clear' or weather == 'Clear Skies' or weather == 'Clear and cold' or weather == 'Fair'):

        return ('clear')

    elif (weather == 'Sun & Clouds' or weather == 'Mostly Sunny' or weather == 'Partly Sunny' or weather == 'Sunny, highs to upper 80s' or weather == 'Partly sunny' or weather == 'Mostly sunny' or weather == 'Sunny and warm' or weather == 'Sunny, Windy' or weather == 'Moslty Sunny Skies' or weather == 'Sunny Skies' or weather == 'Sunny and cold' or weather == 'Sunny'):

        return ('sunny')

    elif (weather == 'Controlled Climate' or weather == 'Indoor' or weather == 'Indoors' or weather == 'N/A (Indoors)' or weather == 'N/A Indoor'):

        return ('indoor')

    elif (weather == 'Mostly Cloudy' or weather == 'Mostly Coudy' or weather == 'Partly Cloudy' or weather == 'Cloudy' or weather == 'Partly cloudy' or weather == 'Mostly Cloudy' or weather == 'Cloudy, fog started developing in 2nd quarter' or weather == 'Coudy' or weather == 'cloudy' or weather == 'Mostly cloudy' or weather == 'Party Cloudy' or weather == 'Cloudy, light snow accumulating 1-3' or weather == 'Cloudy and cold' or weather == 'Cloudy and Cool' or weather == 'Partly Clouidy' or weather == 'Overcast'):

        return ('cloudy')

    elif (weather == 'Light Rain' or weather == 'Showers' or weather == '30% Chance of Rain' or weather == 'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.' or weather == 'Rain' or weather == 'Rain likely, temps in low 40s.' or weather == 'Cloudy, chance of rain' or weather == 'Rain Chance 40%' or weather == 'Cloudy, 50% change of rain' or weather  == 'Cloudy, Rain' or weather == 'Rain Shower' or weather == 'Rainy'):

        return ('rainy')

    elif (weather == 'Cold' or weather == 'Heavy lake effect snow' or weather == 'Cloudy, light snow accumulating 1-3"' or weather == 'Snow'):

        return ('cold')

    else:

        return ('others')



def refine_winddirection(direction):

    if (direction == 'SW' or direction == 'SouthWest' or direction == 'SSW' or direction == 'West-Southwest' or direction == 'WSW' or direction == 'From SW' or direction == 'South Southwest' or direction == 'From WSW' or direction == 'From SSW' or direction == 'Southwest' or direction == 'W-SW'):

        return ('southwest')

    elif (direction == 'NNE' or direction == 'ENE' or direction == 'NE' or direction == 'Northeast' or direction == 'NorthEast' or direction == 'East North East' or direction == 'North East' or direction == 'From NNE' or direction == 'N-NE'):

        return ('northeast')

    elif (direction == 'SE' or direction == 'ESE' or direction == 'South Southeast' or direction == 'ESE' or direction == 'SSE' or direction == 'Southeast' or direction == 'From SSE' or direction == 'From ESE' or direction == 'East Southeast'):

        return ('southeast')

    elif (direction == 'East' or direction == 'E' or direction == 'EAST'):

        return ('east')

    elif (direction == 'North' or direction == 'N'):

        return ('north')

    elif (direction == 'S' or direction == 'From S' or direction == 'South' or direction == 's'):

        return ('south')

    elif (direction == 'Northwest' or direction == 'NW' or direction == 'NNW' or direction == 'WNW' or direction == 'W-NW' or direction == 'West Northwest' or direction == 'North/Northwest' or direction == 'From NNW'):

        return ('northwest')

    elif (direction == 'W' or direction == 'West' or direction == 'from W' or direction == 'From W'):

        return ('west')

    else:

        return ('still')

    

def refine_windspeed(speed):

    if (speed == '13 MPH'):

        return (13)

    elif (speed == '7 MPH'):

        return (7)

    elif (speed == '12-22'):

        return (17)

    elif (speed == '6 mph'):

        return (6)

    elif (speed == '6mph'):

        return (6)

    elif (speed == '9mph'):

        return (9)

    elif (speed == '12mph'):

        return (12)

    elif (speed == '14-23'):

        return (18)

    elif (speed == '4 MPh'):

        return (4)

    elif (speed == '10MPH'):

        return (10)

    elif (speed == '10mph'):

        return (10)

    elif (speed == '15 gusts up to 25'):

        return (20)

    elif (speed == '11-17'):

        return (14)

    elif (speed == '10-20'):

        return (15)

    elif (speed == 'SSW' or speed == 'Calm' or speed == 'SE' or speed == 'E'):

        return (0)

    else:

        pass

    try:

        speed = int(speed)

    except:

        speed = 0

    

def change_type(string):

    feet, inches = string.split('-')

    height = (int(feet)*12)+(int(inches))

    return (height)



def offense_personnel(string):

    rb_value = 0

    te_value = 0

    wr_value = 0

    ol_value = 0

    qb_value = 0

    lb_value = 0

    db_value = 0

    dl_value = 0

    string = string.split(',')

    for value in string:

        if 'RB' in value:

            rb_value = value[1]

        if 'TE' in value:

            te_value = value[1]

        if 'WR' in value:

            wr_value = value[1]

        if 'OL' in value:

            ol_value = value[1]

        if 'QB' in value:

            qb_value = value[1]

        if 'LB' in value:

            lb_value = value[1]

        if 'DB' in value:

            db_value = value[1]

        if 'DL' in value:

            dl_value = value[1]



    return (rb_value, te_value, wr_value, ol_value, qb_value, lb_value, db_value, dl_value)



def defense_personnel(string):

    dl_value = 0

    lb_value = 0

    db_value = 0

    ol_value = 0

    string = string.split(',')

    for value in string:

        if 'DL' in value:

            dl_value = value[1]

        if 'LB' in value:

            lb_value = value[1]

        if 'DB' in value:

            db_value = value[1]

        if 'OL' in value:

            ol_value = value[1]

            

    return (dl_value, lb_value, db_value, ol_value)

    

def data_processing(data):

    

    # Extracting age from PlayerBirthDate

    data['PlayerBirthDate'] = pd.to_datetime(data['PlayerBirthDate'])

    years = data['PlayerBirthDate'].progress_apply(lambda x: x.year)

    data['Age'] = pd.datetime.now().year - years

    

    # Player Height

    data['Playerheight'] = data['PlayerHeight'].progress_apply(lambda x: change_type(x))

    # TimeHandoff and TimeSnap

    data['TimeHandoff'] = pd.to_datetime(data['TimeHandoff'])

    data['TimeSnap'] = pd.to_datetime(data['TimeSnap'])

    data['TimeDelta'] = data['TimeHandoff'] - data['TimeSnap'] 

    data['TimeDelta'] = data['TimeDelta'].dt.total_seconds()

    

    #GameWeather

    data['RefineGameWeather'] = data['GameWeather'].progress_apply(lambda row: refine_gameweather(row))

    #WindDirection

    data['RefineWindDirection'] = data['WindDirection'].progress_apply(lambda row: refine_winddirection(row))

    #WindSpeed

    data['WindSpeed'].fillna(value = 0, inplace = True)

    data['RefineWindSpeed'] = data['WindSpeed'].progress_apply(lambda row: refine_windspeed(row))

    #StadiumType

    data['RefineStadiumType'] = data['StadiumType'].progress_apply(lambda row: stadium_type(row))

    #Turf

    data['RefineTurf'] = data['Turf'].progress_apply(lambda row: refine_turf(row))

    #PossessionTeam-HomeTeamAbbr-VisitorTeamAbbr Adjustments

    data['VisitorTeamAbbr'] = data['VisitorTeamAbbr'].replace('BAL', 'BLT')

    data['VisitorTeamAbbr'] = data['VisitorTeamAbbr'].replace('ARI', 'ARZ')

    data['HomeTeamAbbr'] = data['HomeTeamAbbr'].replace('BAL', 'BLT')

    data['HomeTeamAbbr'] = data['HomeTeamAbbr'].replace('ARI', 'ARZ')



    data['VisitorTeamAbbr'] = data['VisitorTeamAbbr'].replace('CLE', 'CLV')

    data['VisitorTeamAbbr'] = data['VisitorTeamAbbr'].replace('HOU', 'HST')

    data['HomeTeamAbbr'] = data['HomeTeamAbbr'].replace('CLE', 'CLV')

    data['HomeTeamAbbr'] = data['HomeTeamAbbr'].replace('HOU', 'HST')

    

    data['OffensePersonnel'] = ' '+data['OffensePersonnel']

    data['DefensePersonnel'] = ' '+data['DefensePersonnel']

    data['rb_offense'], data['te_offense'], data['wr_offense'], data['ol_offense'], data['qb_offense'], data['lb_offense'], data['db_offense'], data['dl_offense'] = zip(*data['OffensePersonnel'].map(offense_personnel))

    data['dl_defense'], data['lb_defense'], data['db_defense'], data['ol_defense'] = zip(*data['DefensePersonnel'].map(defense_personnel))

    data['rb_offense'] = data['rb_offense'].replace(' ', np.nan)

    data['te_offense'] = data['te_offense'].replace(' ', np.nan)

    data['wr_offense'] = data['wr_offense'].replace(' ', np.nan)

    data['ol_offense'] = data['ol_offense'].replace(' ', np.nan)

    data['qb_offense'] = data['qb_offense'].replace(' ', np.nan)

    data['lb_offense'] = data['lb_offense'].replace(' ', np.nan)

    data['db_offense'] = data['db_offense'].replace(' ', np.nan)

    data['dl_offense'] = data['dl_offense'].replace(' ', np.nan)



    data['rb_offense'].fillna(0, inplace = True)

    data['te_offense'].fillna(0, inplace = True)

    data['wr_offense'].fillna(0, inplace = True)

    data['ol_offense'].fillna(0, inplace = True)

    data['qb_offense'].fillna(0, inplace = True)

    data['lb_offense'].fillna(0, inplace = True)

    data['db_offense'].fillna(0, inplace = True)

    data['dl_offense'].fillna(0, inplace = True)

    

    data['dl_defense'] = data['dl_defense'].replace(' ', np.nan)

    data['lb_defense'] = data['lb_defense'].replace(' ', np.nan)

    data['ol_defense'] = data['ol_defense'].replace(' ', np.nan)

    data['db_defense'] = data['db_defense'].replace(' ', np.nan)



    data['dl_defense'].fillna(value=0, inplace = True)

    data['lb_defense'].fillna(value=0, inplace = True)

    data['ol_defense'].fillna(value=0, inplace = True)

    data['db_defense'].fillna(value=0, inplace = True)



    data['rb_offense'] = data['rb_offense'].astype(np.float64)

    data['te_offense'] = data['te_offense'].astype(np.float64)

    data['wr_offense'] = data['wr_offense'].astype(np.float64)

    data['ol_offense'] = data['ol_offense'].astype(np.float64)

    data['qb_offense'] = data['qb_offense'].astype(np.float64)

    data['lb_offense'] = data['lb_offense'].astype(np.float64)

    data['db_offense'] = data['db_offense'].astype(np.float64)

    data['dl_offense'] = data['dl_offense'].astype(np.float64)

    data['dl_defense'] = data['dl_defense'].astype(np.float64)

    data['lb_defense'] = data['lb_defense'].astype(np.float64)

    data['db_defense'] = data['db_defense'].astype(np.float64)

    data['ol_defense'] = data['ol_defense'].astype(np.float64)

    

    # Deleting unneccesary columns

    del_cols = ['Season','GameId', 'PlayId', 'S', 'NflId', 'DisplayName', 'JerseyNumber', 'NflIdRusher', 

                'TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'PlayerCollegeName', 'FieldPosition', 

                'Turf', 'Week', 'Stadium', 'Location', 'StadiumType', 'OffensePersonnel', 

                'DefensePersonnel', 'GameClock', 'PlayerHeight', 'GameWeather', 'WindSpeed', 

                'WindDirection']

    data.drop(columns = del_cols, axis = 1, inplace = True)

    

    # Filling the null values

    data['Orientation'].fillna(value = 0, inplace = True)

    data['Dir'].fillna(value = 0, inplace = True)

    data['OffenseFormation'].fillna(value = data['OffenseFormation'].mode(), inplace = True)

    data['DefendersInTheBox'].fillna(value = data['DefendersInTheBox'].mean(), inplace = True)

    data['Temperature'].fillna(value = data['Temperature'].mean(), inplace = True)

    data['Humidity'].fillna(value = data['Humidity'].mean(), inplace = True)

    if (data['RefineWindSpeed'].isnull().sum() == len(data)):

        data['RefineWindSpeed'] = 0

    else:

        data['RefineWindSpeed'].fillna(value = data['RefineWindSpeed'].mean(), inplace = True)

    

    # Creating dummies for the categorical columns

    cat_cols = ['Team', 'Quarter', 'Down', 'OffenseFormation', 'PlayDirection', 'Position', 

                'RefineGameWeather', 'RefineWindDirection', 'PossessionTeam', 'HomeTeamAbbr', 

                'VisitorTeamAbbr', 'RefineStadiumType', 'RefineTurf']

    data = pd.get_dummies(columns = cat_cols, data = data)

    

    return (data)
train_data = data_processing(train_data)
train_data_yards = train_data[['Yards']]

train_yards = train_data_yards['Yards']
y_train = np.zeros((train_yards.shape[0], 199))

for idx, target in enumerate(list(train_yards)):

    y_train[idx][99 + target] = 1
X_train = train_data.drop(['Yards'], axis = 1)
print (X_train.shape, y_train.shape)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
## accuracy metrics

def crps(y_true, y_pred):

    y_true = np.clip(np.cumsum(y_true, axis=1), 0, 1)

    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)

    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * y_true.shape[0]) 
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size = 0.25)
## dummy



print (X_train.shape, X_val.shape, y_train.shape, y_val.shape)
from tensorflow import keras

from tensorflow.keras import layers
def build_model():

    model = keras.Sequential([

    layers.Dense(1024, activation='relu', input_shape=[X_train.shape[1]]),

    layers.PReLU(),

    layers.BatchNormalization(),

    layers.Dropout(0.2),

    layers.Dense(1024, activation='relu'),

    layers.PReLU(),

    layers.BatchNormalization(),

    layers.Dropout(0.2),

    layers.Dense(199, activation='sigmoid')

    ])



    #optimizer = tf.keras.optimizers.RMSprop(0.001)



    model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

    return model



model = build_model()
model.summary()
history = model.fit(X_train,

                    y_train,

                    epochs=40,

                    batch_size=512,

                    validation_data=(X_val, y_val),

                    verbose=1)
test_predictions = model.predict(X_val)

accuracy = crps(y_val, test_predictions)

print(accuracy)
for (test_df, sample_submission_df) in env.iter_test():

    test_data = data_processing(test_df)

    excess_train_columns = [col for col in train_data.columns if not col in test_data.columns]

    excess_test_columns = [col for col in test_data.columns if not col in train_data.columns]



    for col in excess_train_columns:

        test_data[col] = 0



    for col in excess_test_columns:

        train_data[col] = 0



    test_data.drop(['Yards'], axis = 1, inplace = True)

    print (test_df.shape)



    y_pred = np.mean([model.predict(test_data)],axis=0)

    np.mean([model.predict(test_data)], axis=0)

    y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1).tolist()[0]

    

    preds_df = pd.DataFrame(data=[y_pred], columns=sample_submission_df.columns)

    preds_df.replace(r'\s+', np.nan, inplace = True)

    preds_df.fillna(value = 0, inplace= True)

    env.predict(preds_df)

    

    

env.write_submission_file()