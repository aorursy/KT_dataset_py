# Loading packages

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Statistic lib packages

from scipy import stats

from scipy.stats import skew, norm



# Utils packages

import pandasql as ps

import re 

import math, string, os

import datetime

from IPython.display import Image



# Options

import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_seq_items = 8000

pd.options.display.max_rows = 8000

pd.set_option('display.max_columns', None)

import gc

gc.enable()
# Loading train data

train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')

print ("Data is loaded !!")
Image(url = 'https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3258%2F820e86013d48faacf33b7a32a15e814c%2FIncreasing%20Dir%20and%20O.png?generation=1572285857588233&alt=media')
# Viewing first dataset rows

train.head()
# Viewing feature types

train.dtypes
# Viewing statistical data of numerical variables

train.describe().T
# Function to create missing feature dataset

def percent_missing(df):

    data = pd.DataFrame(df)

    df_cols = list(pd.DataFrame(data))

    dict_x = {}

    for i in range(0, len(df_cols)):

        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})

    

    return dict_x
# Checking columns with missing data

missing = percent_missing(train)

df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

print('Percent of missing data')

df_miss[0:50]
# Plot setup

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

sns.set_color_codes(palette='deep')



# Identifying missing values

missing = round(train.isnull().mean()*100,2)

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar(color="b")



# Visual presentation

ax.xaxis.grid(False)

ax.set(ylabel="Percent of missing values")

ax.set(xlabel="Features")

ax.set(title="Percent missing data by feature")

sns.despine(trim=True, left=True)
# Function to handle missing data of each feature

def fill_na(data):

    data['WindDirection'].fillna('unknown',inplace=True)

    data['OffenseFormation'].fillna('unknown',inplace=True)

    data['StadiumType'].fillna('unknown',inplace=True)

    data['GameWeather'].fillna('unknown',inplace=True)

    data['FieldPosition'].fillna('NA',inplace=True)

    data['Temperature'].fillna(data['Temperature'].mean(), inplace=True)

    data['Humidity'].fillna(data['Humidity'].mean(), inplace=True)

    data['DefendersInTheBox'].fillna(math.ceil(data['DefendersInTheBox'].mean()),inplace=True)

    

# Function to group descriptions of stadium types

def agrupar_tipo_estadio(StadiumType):

    outdoor       = ['Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 'Outdor', 'Ourdoor', 'Outside', 'Outddors', 'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl']

    indoor_closed = ['Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed']

    indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']

    dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']

    dome_open     = ['Domed, Open', 'Domed, open']

    

    if StadiumType in outdoor:

        return 'outdoor'

    elif StadiumType in indoor_closed:

        return 'indoor_closed'

    elif StadiumType in indoor_open:

        return 'indoor_open'

    elif StadiumType in dome_closed:

        return 'dome_closed'

    elif StadiumType in dome_open:

        return 'dome_open'

    else:

        return 'unknown'

    

# Function to group stadium descriptions

def agrupar_estadio(Stadium):

    if Stadium == 'Broncos Stadium at Mile High':

        return 'Broncos Stadium At Mile High'

    elif Stadium in ('CenturyField', 'CenturyLink'):

        return 'CenturyLink Field'

    elif Stadium == 'EverBank Field':

        return 'Everbank Field'

    elif Stadium in ('FirstEnergy', 'FirstEnergy Stadium', 'FirstEnergyStadium'):

        return 'First Energy Stadium'

    elif Stadium == 'Lambeau field':

        return 'Lambeau Field'

    elif Stadium == 'Los Angeles Memorial Coliesum':

        return 'Los Angeles Memorial Coliseum'

    elif Stadium in ('M & T Bank Stadium', 'M&T Stadium'):

        return 'M&T Bank Stadium'

    elif Stadium in ('Mercedes-Benz Dome', 'Mercedes-Benz Superdome'):

        return 'Mercedes-Benz SuperDome'

    elif Stadium in ('MetLife Stadium', 'Metlife Stadium', 'MetLife'):

        return 'MetLife Stadium' 

    elif Stadium == 'NRG':

        return 'NRG Stadium' 

    elif Stadium == 'Oakland-Alameda County Coliseum':

        return 'Oakland Alameda-County Coliseum' 

    elif Stadium == 'Paul Brown Stdium':

        return 'Paul Brown Stadium' 

    elif Stadium == 'Twickenham':

        return 'Twickenham Stadium' 

    else:

        return Stadium

    

# Function to group stadium and game location

def agrupar_local(Location):

    if Location == "Arlington, Texas":

        return "Arlington, TX"

    elif Location in ("Baltimore, Maryland","Baltimore, Md."):

        return "Baltimore, MD"

    elif Location == "Charlotte, North Carolina":

        return "Charlotte, NC"

    elif Location == "Chicago. IL":

        return "Chicago, IL"

    elif Location == "Cincinnati, Ohio":

        return "Cincinnati, OH"

    elif Location in ("Cleveland","Cleveland Ohio","Cleveland, Ohio","Cleveland,Ohio"):

        return "Cleveland, OH"

    elif Location == "Detroit":

        return "Detroit, MI"

    elif Location == "E. Rutherford, NJ" or Location == "East Rutherford, N.J.":

        return "East Rutherford, NJ"

    elif Location == "Foxborough, Ma":

        return "Foxborough, MA"

    elif Location == "Houston, Texas":

        return "Houston, TX"

    elif Location in ("Jacksonville Florida","Jacksonville, Fl","Jacksonville, Florida"):

        return "Jacksonville, FL"

    elif Location == "London":

        return "London, England"

    elif Location == "Los Angeles, Calif.":

        return "Los Angeles, CA"

    elif Location == "Miami Gardens, Fla.":

        return "Miami Gardens, FLA"

    elif Location in ("New Orleans","New Orleans, La."):

        return "New Orleans, LA"

    elif Location == "Orchard Park NY":

        return "Orchard Park, NY"

    elif Location == "Philadelphia, Pa.":

        return "Philadelphia, PA"

    elif Location == "Pittsburgh":

        return "Pittsburgh, PA"

    elif Location == "Seattle":

        return "Seattle, WA"

    else:

        return Location

    

# Function to group the stadium turf

def agrupar_gramado(Turf):

    if Turf == 'Artifical':

        return 'Artificial'

    

    elif Turf in ('FieldTurf', 'Field turf'):

        return 'Field Turf'



    elif Turf in ('FieldTurf360', 'FieldTurf 360'):

        return 'Field Turf 360'



    elif Turf in ('Natural', 'Natural grass', 'Naturall Grass', 'grass', 'natural grass', 'SISGrass', 'Natural Grass'):

        return "Grass"



    elif Turf == "UBU Sports Speed S5-M":

        return "UBU Speed Series-S5-M"



    else:

        return Turf



# Function to group wind direction

def agrupa_wind_direction(WindDirection):

    wd = str(WindDirection).upper()

    

    if wd == 'N' or 'FROM N' in wd:

        return 'north'

    if wd == 'S' or 'FROM S' in wd:

        return 'south'

    if wd == 'W' or 'FROM W' in wd:

        return 'west'

    if wd == 'E' or 'FROM E' in wd:

        return 'east'

    

    if 'FROM SW' in wd or 'FROM SSW' in wd or 'FROM WSW' in wd:

        return 'south west'

    if 'FROM SE' in wd or 'FROM SSE' in wd or 'FROM ESE' in wd:

        return 'south east'

    if 'FROM NW' in wd or 'FROM NNW' in wd or 'FROM WNW' in wd:

        return 'north west'

    if 'FROM NE' in wd or 'FROM NNE' in wd or 'FROM ENE' in wd:

        return 'north east'

    

    if 'NW' in wd or 'NORTHWEST' in wd:

        return 'north west'

    if 'NE' in wd or 'NORTH EAST' in wd:

        return 'north east'

    if 'SW' in wd or 'SOUTHWEST' in wd:

        return 'south west'

    if 'SE' in wd or 'SOUTHEAST' in wd:

        return 'south east'



    return 'unknown'



# Function to group climate descriptions

def agrupar_clima(GameWeather):

    chuva   = ['Rainy', 'Rain Chance 40%', 'Showers',

               'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

               'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain']

    nublado = ['Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',

               'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',

               'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',

               'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',

               'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',

               'Partly Cloudy', 'Cloudy']

    limpo   = ['Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',

               'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',

               'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',

               'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',

               'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',

               'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny']

    neve    = ['Heavy lake effect snow', 'Snow']

    none    = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']



    if GameWeather in chuva:

        return 'chuva'

    elif GameWeather in nublado:

        return 'nublado'

    elif GameWeather in limpo:

        return 'limpo'

    elif GameWeather in neve:

        return 'neve'

    elif GameWeather in none:

        return 'none'

    else:

        return 'none'

    

# Function to convert wind speed

def convert_wind_speed(WindSpeed):

    ws = str(WindSpeed)

    if ws.isdigit():

        return int(ws)

    if '-' in ws:

        return int(ws.split('-')[0])

    if ws.split(' ')[0].isdigit():

        return int(ws.split(' ')[0])

    if 'mph' in ws.lower():

        return int(ws.lower().split('mph')[0])

    else:

        return 0

    

# Function to convert height from feet-inches to centimeters

def convert_to_cm(ft_in):

    h_ft   = int(ft_in.split('-')[0])

    h_inch = int(ft_in.split('-')[1])

    h_inch += h_ft * 12

    h_cm = round(h_inch * 2.54, 1)   

    return h_cm



# Function to convert weight in lbs to kg

def convert_to_kg(lbs):

    kg = lbs * 0.45359237

    return kg



# Function to convert temperature Fahrenheit to Celsius

def convert_to_celsius(fah):

    celsius = (fah - 32) * 5.0/9.0

    return celsius



# Function to convert date features and extract day, month, year, hour, minute, second

def convert_data(data):

    data['PlayerBirthDate'] = data['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))

    data['PlayerBirthDate_day'] = data['PlayerBirthDate'].dt.day.astype(int)

    data['PlayerBirthDate_month'] = data['PlayerBirthDate'].dt.month.astype(int)

    data['PlayerBirthDate_year'] = data['PlayerBirthDate'].dt.year.astype(int)



    data['TimeSnap'] = data['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    data['TimeSnap_min'] = data['TimeSnap'].dt.minute.astype(int)

    data['TimeSnap_seg'] = data['TimeSnap'].dt.second.astype(int)

    

    data['TimeHandoff'] = data['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    data['TimeHandoff_min'] = data['TimeHandoff'].dt.minute.astype(int)

    data['TimeHandoff_seg'] = data['TimeHandoff'].dt.second.astype(int)

    

    

# Function to convert a time string in seconds

def str_to_seconds(time):

    time = time.split(':')

    sec = int(time[0])*60 + int(time[1]) + int(time[2])/60

    return sec

    

# Function to create a distribution plot for each feature

def plot_distribution(dataset, cols=5, width=20, height=25, hspace=0.4, wspace=0.5):

    """

    Plot distributions for each column in a dataset.

    Seaborn countplots are used for categorical data and distplots for numerical data



    args:

    ----

    dataset {dataframe} - the data that will be plotted

    cols {int} - how many distributions to plot for each row

    width {int} - how wide each plot should be

    height {int} - how tall each plot should be

    hspace {float} - horizontal space between plots

    wspace {float} - vertical space between plots 

    """

    # plot styling

    plt.style.use('fivethirtyeight')

    fig = plt.figure(figsize=(width, height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    # calculate rows needed

    rows = math.ceil(float(dataset.shape[1]) / cols)

    # create a countplot for top 20 categorical values

    # and a distplot for all numerical values

    for i, column in enumerate(dataset.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        if dataset.dtypes[column] == np.object:

            # grab the top 10 for each countplot

            g = sns.countplot(y=column, 

                              data=dataset,

                              order=dataset[column].value_counts().index[:10])

            # make labels only 20 characters long and rotate x labels for nicer displays

            substrings = [s.get_text()[:20] for s in g.get_yticklabels()]

            g.set(yticklabels=substrings)

            plt.xticks(rotation=25)

        else:

            g = sns.distplot(dataset[column])

            plt.xticks(rotation=25)
# Fix missing values

fill_na(train)



# Convert date features

convert_data(train)



# Convert player height to cm

train['PlayerHeight'] = train['PlayerHeight'].apply(convert_to_cm)



# Convert player height to cm

train['PlayerWeight'] = train['PlayerWeight'].apply(convert_to_kg)



# Convert temperature to Celsius

train['Temperature'] = train['Temperature'].apply(convert_to_celsius)



# Grouping the stadium types

train['StadiumType'] = train['StadiumType'].apply(agrupar_tipo_estadio)



# Grouping the stadium

train['Stadium'] = train['Stadium'].apply(agrupar_estadio)



# Grouping locations

train['Location'] = train['Location'].apply(agrupar_local)



# Grouping turf

train['Turf'] = train['Turf'].apply(agrupar_gramado)



# Grouping wind direction

train['WindDirection'] = train['WindDirection'].apply(agrupa_wind_direction)



# Converting wind speed to numeric

train['WindSpeed'] = train['WindSpeed'].apply(convert_wind_speed)



# grouping game weather

train['GameWeather'] = train['GameWeather'].apply(agrupar_clima)



# Convert game time to seconds

train['GameClock'] = train['GameClock'].apply(str_to_seconds)
# Pearson Correlations

cor_mat = train.corr(method = 'pearson')



# Heatmap plot

f, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(cor_mat,linewidths=.1,fmt= '.3f',ax=ax,square=True,cbar=True,annot=False)
# Definition: A unique play identifier

# There are 22 records by PlayId

# So there are 11 home team players and 11 home team players

train['PlayId'].value_counts().head()
# Linking NFLid and NFLidRusher

train_jog = train[train['NflId'] == train['NflIdRusher']]

print(train_jog.shape)
# When connecting NflId and NflIdRusher it's possible see that every play now has a unique record

train_jog['PlayId'].value_counts().head()
# Example the sequence of moves of a game that counted yard change (+ or -)

# It is possible to identify that this sequence is increasing, and registered in PlayId

# Using the 2017 GameId filter

train_jog[train_jog['GameId'] == 2017090700].head(10)
# Description: The number of yards won in play (variable to be predicted)

train_jog['Yards'].describe()
# Analyzing the target variable 'Yards'

sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(18, 8))



# Fit a Normal Distribution

mu, std = norm.fit(train_jog["Yards"])



# Checking the frequency distribution of the variable Yards

sns.distplot(train_jog["Yards"], color="b", fit = stats.norm)

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="Yards")

ax.set(title="Yards distribution: mu = %.2f,  std = %.2f" % (mu, std))

sns.despine(trim=True, left=True)



# Adding Skewness and Kurtosis

ax.text(x=1.1, y=1, transform=ax.transAxes, s="Skewness: %f" % train_jog["Yards"].skew(),\

        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\

        backgroundcolor='white', color='xkcd:poo brown')

ax.text(x=1.1, y=0.95, transform=ax.transAxes, s="Kurtosis: %f" % train_jog["Yards"].kurt(),\

        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\

        backgroundcolor='white', color='xkcd:dried blood')



plt.show()
# Distribution plot for each player feature

columns_to_plot = ['PlayerHeight', 'PlayerWeight', 'PlayerBirthDate_year', 'PlayerCollegeName', 'JerseyNumber', 'Position']

plot_distribution(train_jog[columns_to_plot], cols=3, width=30, height=20, hspace=0.45, wspace=0.5)
# Display scatterPlot between Player Height to Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='PlayerHeight', y='Yards', data=train_jog, color='b', x_jitter=1)

plt.xlabel('Player Height')

plt.ylabel('Yards')

plt.title('Player Height vs Yards', fontsize=20)

plt.show()
# Display scatterPlot between Player Weight to Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='PlayerWeight', y='Yards', data=train_jog, color='b', x_jitter=1)

plt.xlabel('Player Weight')

plt.ylabel('Yards')

plt.title('Player Weight vs Yards', fontsize=20)

plt.show()
# Display scatterPlot between Player Height to Player Weight

# It's a positive regression relationship where the taller the player, the heavier the player

plt.figure(figsize=(18, 8))

sns.regplot(x='PlayerHeight', y='PlayerWeight', data=train_jog, color='g', x_jitter=1)

plt.xlabel('Height')

plt.ylabel('Weight')

plt.title('Height vs Weight', fontsize=20)

plt.show()
# Checking players over 140kg

# Found only 1 player: Akiem Hicks

# The image below shows this player

train_jog[train_jog['PlayerWeight'] >= 140.00].head()
# Yeah, this guy is big

Image(url = 'https://radioimg.s3.amazonaws.com/wscram/styles/nts_image_cover_tall_775x425/s3/Akiem_Hicks_6.jpg?BOAlCf3Wc4ySX4yP9hqxL4r3e6OIKdL_&itok=-QTENpoi&c=ae8001fe2a7c6e1ccd639801892a4486',width=400,height=400)
# Checking players under 70kg

# Found only 1 player: JoJo Natson

# The image below shows this player

train_jog[train_jog['PlayerWeight'] <= 70.00].head()
Image(url = 'https://ssl.c.photoshelter.com/img-get2/I0000ajxYOXxxUQg/fit=1000x750/g=G0000TtQ9QQoRy9c/RAMS-CHARGERS-0923-0791.jpg',width=200,height=200)
# Display scatterPlot between Player Birth Date Year to Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='PlayerBirthDate_year', y='Yards', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Year')

plt.ylabel('Yards')

plt.title('Player Birth Date (Year) vs Yards', fontsize=20)

plt.show()
# Display catPlot between Season to Yards

g = sns.catplot(x='PlayerCollegeName', y='Yards', hue='Season', data=train_jog, height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display scatterPlot between Jersey Number to Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='JerseyNumber', y='Yards', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Jersey Number')

plt.ylabel('Yards')

plt.title('Jersey Number vs Yards', fontsize=20)

plt.show()
train_jog['Position'].value_counts()
# Display catPlot between Position to Yards

g = sns.catplot(x='Position', y='Yards', hue='Season', data=train_jog, kind="box", height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display linear plot between Player Weight, Position and Yards

g = sns.lmplot(x='PlayerWeight', y='Yards', data=train_jog, 

               x_jitter=.2, col="Position" , height=6, aspect=1, col_wrap=5)
# Distribution plot for each game feature

columns_to_plot = ['Season', 'Week', 'Team', 'StadiumType', 'Stadium', 

                   'Location', 'Turf', 'GameClock', 'HomeTeamAbbr', 'VisitorTeamAbbr']

plot_distribution(train_jog[columns_to_plot], cols=3, width=30, height=20, hspace=0.45, wspace=0.5)
# Display linear plot between Season and Yards

g = sns.lmplot(x='Season', y='Yards', hue='Season', data=train_jog, x_jitter=.2, col="Season" , height=10)
# Display linear plot between Season, Week and Yards

g = sns.lmplot(x='Week', y='Yards', hue='Season', data=train_jog, x_jitter=.2, col="Season" , height=10)
# Season has a correlation of 1.0 with GameId and PlayId. Example below with 2017 and 2018

# GameId and PlayId contain year and some more info

# Seems to be month and day plus one identifier

# You might need to leave the GameId and PlayId variables only with the individual indicator.

tmp = train_jog.set_index('Season')

print(tmp.loc[[2017], ['GameId','PlayId']].head(1))

print(tmp.loc[[2018], ['GameId','PlayId']].head(1))
# Checking which stadium are undefined type

# There are three stadium

# I'll have a look to see if you have a set type for them

q1 = """SELECT StadiumType, Stadium

          FROM train

        WHERE StadiumType = 'unknown'

        GROUP BY StadiumType, Stadium"""

ps.sqldf(q1, locals())
# To MetLife Stadium

# I'll try to fix

q1 = """SELECT StadiumType, Stadium

          FROM train

        WHERE Stadium IN ('MetLife Stadium', 'StubHub Center', 'TIAA Bank Field')

        GROUP BY StadiumType, Stadium

        ORDER BY Stadium, StadiumType"""

ps.sqldf(q1, locals())
# MetLife Stadium only has OUTDOOR

# I'll fix this, assigning outdoor

Image(url = 'https://upload.wikimedia.org/wikipedia/commons/0/04/Metlife_stadium_%28Aerial_view%29.jpg',width=400,height=400)
# StubHub Center only has OUTDOOR

# I'll fix this, assigning outdoor

Image(url = 'https://media.bizj.us/view/img/4273531/111813stubhubcenterga0014*750xx5184-2916-0-270.jpg',width=400,height=400)
# Fixing a feature Stadium

train_jog.loc[train_jog['Stadium'] == 'MetLife Stadium', 'StadiumType'] = 'outdoor'

train_jog.loc[train_jog['Stadium'] == 'StubHub Center', 'StadiumType'] = 'outdoor'
# View of Mercedes Benz stadium where there are several types of game modes

Image(url = 'https://cdn.vox-cdn.com/thumbor/FV1og2Kh9y8VwyrImv3tAM21vdM=/0x0:2740x1824/1200x800/filters:focal(1151x693:1589x1131)/cdn.vox-cdn.com/uploads/chorus_image/image/56280377/999476412.jpg.1504558796.jpg',width=400,height=400)
# Display linear plot between Season, Week and Yards

g = sns.lmplot(x='Week', y='Yards', hue='Season', data=train_jog, 

               x_jitter=.2, col="StadiumType" , height=8, col_wrap=3)
# Display catPlot between Stadium to Yards

g = sns.catplot(x='Stadium', y='Yards', hue='Season', data=train_jog, 

                kind="violin", split=True, height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display catPlot between Location to Yards

g = sns.catplot(x='Location', y='Yards', hue='Season', data=train_jog, 

                kind="violin", split=True, height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Description of the field surface

train_jog.groupby('Turf').size()
# Display catPlot between Turf to Yards

g = sns.catplot(x='Turf', y='Yards', hue='Season', data=train_jog, kind="box", height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display the distribution plot of the GameClock feature.

g = sns.distplot(train_jog['GameClock'])
# Distribution plot for each Environment feature

columns_to_plot = ['GameWeather', 'Temperature', 'Humidity', 'WindDirection', 'WindSpeed']

plot_distribution(train_jog[columns_to_plot], cols=3, width=30, height=20, hspace=0.45, wspace=0.5)
# Display catPlot between Weather to Yards

g = sns.catplot(x='GameWeather', y='Yards', hue='Season', data=train_jog, kind="box", height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display scatterPlot between Temperature to Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='Temperature', y='Yards', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Temperature')

plt.ylabel('Yards')

plt.title('Temperature(Celsius) vs Yards', fontsize=20)

plt.show()
# Display linear plot between Temperature, Weather and Yards

g = sns.lmplot(x='Temperature', y='Yards', data=train_jog, 

               x_jitter=.2, col="GameWeather" , height=6, aspect=1)
# Has too many records with moisture 0

# Maybe it's better to see this

train_jog['Humidity'].value_counts().head()
g = sns.distplot(train_jog['Humidity'])
# Display scatterPlot between Humidity to Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='Humidity', y='Yards', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Humidity')

plt.ylabel('Yards')

plt.title('Humidity vs Jardas', fontsize=20)

plt.show()
# Display linear graph with relationship between Humidity, Weather and Yards

# Most of humidity 0 is in NONE climate

g = sns.lmplot(x='Humidity', y='Yards', data=train_jog, 

               x_jitter=.2, col="GameWeather" , height=6, aspect=1)
train_jog['WindDirection'].value_counts()
# Display linear plot with relationship between Temperature, WindDirection and Yards

g = sns.lmplot(x='Temperature', y='Yards', data=train_jog, 

               x_jitter=.2, col="WindDirection" , height=6, aspect=1, col_wrap=3)
train_jog['WindSpeed'].value_counts().head()
# Display scatterPlot between Speed to Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='WindSpeed', y='Yards', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Wind Speed')

plt.ylabel('Yards')

plt.title('Wind Speed vs Yards', fontsize=20)

plt.show()
# Display linear plot with relationship between Wind Speed, Weather and Yards

# Most of humidity 0 is in NONE climate

g = sns.lmplot(x='WindSpeed', y='Yards', data=train_jog, 

               x_jitter=.2, col="GameWeather" , height=6, aspect=1)
# Display the linear plot with the relationship between Wind Speed, Wind Direction and Yards

g = sns.lmplot(x='WindSpeed', y='Yards', data=train_jog, 

               x_jitter=.2, col="WindDirection" , height=6, aspect=1, col_wrap=3)
# Distribution plot for each Play feature

columns_to_plot = ['HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'PossessionTeam', 'FieldPosition', 'Quarter',

                   'PlayDirection','OffensePersonnel','DefensePersonnel','OffenseFormation','DefendersInTheBox',

                   'Down','Distance','Dis','YardLine','X','A']

plot_distribution(train_jog[columns_to_plot], cols=4, width=30, height=20, hspace=0.8, wspace=0.5)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 8))



ax1.set_title('Visitor Score Before Play')

sns.regplot(x='VisitorScoreBeforePlay', y='Yards', data=train_jog, color='b', x_jitter=2, ax=ax1)



ax2.set_title('Home Score Before Play')

sns.regplot(x='HomeScoreBeforePlay', y='Yards', data=train_jog, color='g', x_jitter=2, ax=ax2)



plt.show()
# Display catPlot with the relationship between PossessionTeam and Yards

g = sns.catplot(x='PossessionTeam', y='Yards', data=train_jog, kind="box", height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display catPlot with the relationship between FieldPosition and Yards

g = sns.catplot(x='FieldPosition', y='Yards', data=train_jog, kind="box", height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display catPlot chart with the relationship between Distance and Yards (by Quarter)

g = sns.lmplot(x='Distance', y='Yards', data=train_jog, x_jitter=.2, col="Quarter" , height=6, aspect=1)
# Display catPlot with the relationship between Distance and Yards (by PlayDirection)

g = sns.lmplot(x='Distance', y='Yards', data=train_jog, x_jitter=.2, col="PlayDirection" , height=10)
# Display catPlot with the relationship between OffensePersonnel and Yards

g = sns.catplot(x='OffensePersonnel', y='Yards', data=train_jog, kind="violin", height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display catPlot with the relationship between DefensePersonnel and Yards

g = sns.catplot(x='DefensePersonnel', y='Yards', data=train_jog, kind="violin", height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display catPlot with the relationship between OffenseFormation and Yards

g = sns.catplot(x='OffenseFormation', y='Yards', data=train_jog, kind="violin", height=8, aspect=2)

g.set_xticklabels(rotation=90)
# Display scatterPlot plot with the relationship between DefendersInTheBox and Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='DefendersInTheBox', y='Yards', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Defenders In The Box')

plt.ylabel('Yards')

plt.title('Defenders In The Box vs Yards', fontsize=20)

plt.show()
# UTC time of the snap

train_jog['TimeSnap'].value_counts().head()
# UTC time of the handoff

train_jog['TimeHandoff'].value_counts().head()
# Display the linear plot with the relationship between Distance, Down and Yards

g = sns.lmplot(x='Distance', y='Yards', data=train_jog, x_jitter=.2, col="Down" , height=6, aspect=1)
# Display the scatterPlot between Distance and Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='Distance', y='Yards', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Distance')

plt.ylabel('Yards')

plt.title('Distance vs Yards', fontsize=20)

plt.show()
# Display the scatterPlot between Dis and Yards

plt.figure(figsize=(18, 8))

sns.regplot(x='Dis', y='Yards', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Dis')

plt.ylabel('Yards')

plt.title('Dis vs Yards', fontsize=20)

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 8))



ax1.set_title('Distance')

sns.regplot(x='Distance', y='Yards', data=train_jog, color='b', x_jitter=2, ax=ax1)



ax2.set_title('Dis')

sns.regplot(x='Dis', y='Yards', data=train_jog, color='g', x_jitter=2, ax=ax2)



plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 8))



ax1.set_title('Aceleração')

sns.regplot(x='A', y='Yards', data=train_jog, color='b', x_jitter=2, ax=ax1)



ax2.set_title('Velocidade')

sns.regplot(x='S', y='Yards', data=train_jog, color='g', x_jitter=2, ax=ax2)



plt.show()
# Display the scatterPlot between A and S

plt.figure(figsize=(18, 8))

sns.regplot(x='A', y='S', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Acceleration')

plt.ylabel('Speed')

plt.title('Acceleration vs Speed', fontsize=20)

plt.show()
# Distribution plot for each Position feature

columns_to_plot = ['X', 'Y', 'PossessionTeam', 'Orientation', 'Dir']

plot_distribution(train_jog[columns_to_plot], cols=3, width=30, height=20, hspace=0.8, wspace=0.5)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 8))



ax1.set_title('Angle X')

sns.regplot(x='X', y='Yards', data=train_jog, color='b', x_jitter=2, ax=ax1)



ax2.set_title('Angle Y')

sns.regplot(x='Y', y='Yards', data=train_jog, color='g', x_jitter=2, ax=ax2)



plt.show()
# Display the scatterPlot with the relationship between angle X and Y

plt.figure(figsize=(18, 8))

sns.regplot(x='X', y='Y', data=train_jog, color='b', x_jitter=2)

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Angle (X vs Y)', fontsize=20)

plt.show()
train_jog['Orientation'].value_counts().head()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 8))



ax1.set_title('Orientation')

sns.regplot(x='Orientation', y='Yards', data=train_jog, color='b', x_jitter=2, ax=ax1)



ax2.set_title('Dir')

sns.regplot(x='Dir', y='Yards', data=train_jog, color='g', x_jitter=2, ax=ax2)



plt.show()
# Display the scatterPlot with the relationship between Orientation and Dir

plt.figure(figsize=(18, 8))

sns.regplot(x='Orientation', y='Dir', data=train_jog, color='b', x_jitter=2)

plt.xlabel('Orientation')

plt.ylabel('Dir')

plt.title('Orientation x Dir', fontsize=20)

plt.show()