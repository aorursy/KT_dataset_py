# Competition Specific

from kaggle.competitions import nflrush



# Data Management

import numpy as np 

import pandas as pd 

pd.set_option('max_columns', 100)



# Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')



# Managing Warnings 

import warnings

warnings.filterwarnings('ignore')



# Plot Figures Inline

%matplotlib inline



# Extras

import math, string, os



# View Available Files

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

print(train_df.shape)

train_df.head()
train_df.columns
train_df.info()
train_df.isnull().sum()
def height_to_numerical(height):

    """

    Convert string representing height into total inches

    

    Ex. '5-11' --> 71

    Ex. '6-3'  --> 75

    """  

    feet   = height.split('-')[0]

    inches = height.split('-')[1]

    return int(feet)*12 + int(inches)
train_df['PlayerHeight'] = train_df['PlayerHeight'].apply(height_to_numerical)
train_df.drop(['GameId', 'PlayId', 'NflId', 'JerseyNumber', 'NflIdRusher'], axis=1).describe(include=['O']).T
train_df.drop(['GameId', 'PlayId', 'NflId', 'JerseyNumber', 'NflIdRusher'], axis=1).describe().T
train_df[ train_df['PlayerWeight'] == 153.00 ]
train_df['StadiumType'].value_counts()
def group_stadium_types(stadium):

    outdoor       = [

        'Outdoor', 'Outdoors', 'Cloudy', 'Heinz Field', 

        'Outdor', 'Ourdoor', 'Outside', 'Outddors', 

        'Outdoor Retr Roof-Open', 'Oudoor', 'Bowl'

    ]

    indoor_closed = [

        'Indoors', 'Indoor', 'Indoor, Roof Closed', 'Indoor, Roof Closed', 

        'Retractable Roof', 'Retr. Roof-Closed', 'Retr. Roof - Closed', 'Retr. Roof Closed',

    ]

    indoor_open   = ['Indoor, Open Roof', 'Open', 'Retr. Roof-Open', 'Retr. Roof - Open']

    dome_closed   = ['Dome', 'Domed, closed', 'Closed Dome', 'Domed', 'Dome, closed']

    dome_open     = ['Domed, Open', 'Domed, open']

    

    if stadium in outdoor:

        return 'outdoor'

    elif stadium in indoor_closed:

        return 'indoor closed'

    elif stadium in indoor_open:

        return 'indoor open'

    elif stadium in dome_closed:

        return 'dome closed'

    elif stadium in dome_open:

        return 'dome open'

    else:

        return 'unknown'
train_df['StadiumType'] = train_df['StadiumType'].apply(group_stadium_types)
weather = pd.DataFrame(train_df['GameWeather'].value_counts())

pd.options.display.max_rows=100

weather
def group_game_weather(weather):

    rain = [

        'Rainy', 'Rain Chance 40%', 'Showers',

        'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.',

        'Scattered Showers', 'Cloudy, Rain', 'Rain shower', 'Light Rain', 'Rain'

    ]

    overcast = [

        'Cloudy, light snow accumulating 1-3"', 'Party Cloudy', 'Cloudy, chance of rain',

        'Coudy', 'Cloudy, 50% change of rain', 'Rain likely, temps in low 40s.',

        'Cloudy and cold', 'Cloudy, fog started developing in 2nd quarter',

        'Partly Clouidy', '30% Chance of Rain', 'Mostly Coudy', 'Cloudy and Cool',

        'cloudy', 'Partly cloudy', 'Overcast', 'Hazy', 'Mostly cloudy', 'Mostly Cloudy',

        'Partly Cloudy', 'Cloudy'

    ]

    clear = [

        'Partly clear', 'Sunny and clear', 'Sun & clouds', 'Clear and Sunny',

        'Sunny and cold', 'Sunny Skies', 'Clear and Cool', 'Clear and sunny',

        'Sunny, highs to upper 80s', 'Mostly Sunny Skies', 'Cold',

        'Clear and warm', 'Sunny and warm', 'Clear and cold', 'Mostly sunny',

        'T: 51; H: 55; W: NW 10 mph', 'Clear Skies', 'Clear skies', 'Partly sunny',

        'Fair', 'Partly Sunny', 'Mostly Sunny', 'Clear', 'Sunny'

    ]

    snow  = ['Heavy lake effect snow', 'Snow']

    none  = ['N/A Indoor', 'Indoors', 'Indoor', 'N/A (Indoors)', 'Controlled Climate']

    

    if weather in rain:

        return 'rain'

    elif weather in overcast:

        return 'overcast'

    elif weather in clear:

        return 'clear'

    elif weather in snow:

        return 'snow'

    elif weather in none:

        return 'none'

    

    return 'none'
train_df['GameWeather'] = train_df['GameWeather'].apply(group_game_weather)
train_df['WindSpeed'].value_counts()
def clean_wind_speed(windspeed):

    """

    This is not a very robust function, 

    but it should do the job for this dataset.

    """

    ws = str(windspeed)

    # if it's already a number just return an int value

    if ws.isdigit():

        return int(ws)

    # if it's a range, just take the first value

    if '-' in ws:

        return int(ws.split('-')[0])

    # if there's a space between the number and mph

    if ws.split(' ')[0].isdigit():

        return int(ws.split(' ')[0])

    # if it looks like '10MPH' or '12mph' just take the first part

    if 'mph' in ws.lower():

        return int(ws.lower().split('mph')[0])

    else:

        return 0
train_df['WindSpeed'] = train_df['WindSpeed'].apply(clean_wind_speed)
train_df['WindDirection'].value_counts()
# This function has been updated to reflect what Subin An (https://www.kaggle.com/subinium) mentioned in comments below.

# WindDirection is indicated by the direction that wind is flowing FROM - https://en.wikipedia.org/wiki/Wind_direction



def clean_wind_direction(wind_direction):

    wd = str(wind_direction).upper()

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



    return 'none'
train_df['WindDirection'] = train_df['WindDirection'].apply(clean_wind_direction)
train_df['WindDirection'].value_counts()
train_df['Humidity'].fillna(train_df['Humidity'].mean(), inplace=True)

train_df['Temperature'].fillna(train_df['Temperature'].mean(), inplace=True)
train_df['FieldPosition'].value_counts()
train_df['FieldPosition'].isnull().sum()
train_df[ train_df['YardLine'] == 50 ].shape[0]
train_df['FieldPosition'] = np.where(train_df['YardLine'] == 50, train_df['PossessionTeam'], train_df['FieldPosition'])
na_map = {

    'Orientation': train_df['Orientation'].mean(),

    'Dir': train_df['Dir'].mean(),

    'DefendersInTheBox': math.ceil(train_df['DefendersInTheBox'].mean()),

    'OffenseFormation': 'UNKNOWN'

}



train_df.fillna(na_map, inplace=True)
train_df['DefendersInTheBox'].value_counts()
train_df.isnull().sum()
columns_to_plot = [

     'X', 'Y', 'S', 'A',

    'Dis', 

    'Orientation', 

    'Dir',

    'YardLine', 

    'HomeScoreBeforePlay',

    'VisitorScoreBeforePlay',

    'OffenseFormation',

    'DefendersInTheBox',

    'Yards',

    'PlayerHeight',

    'PlayerWeight',

    'PlayerBirthDate',     

    'PlayerCollegeName',

    'Position',

    'Week',

    'Stadium',

    'Location',

    'StadiumType',

    'Turf',

    'GameWeather',

    'Temperature',

    'Humidity',

    'WindSpeed',

    'WindDirection',

]



# Plot the distribution of each feature

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

    

plot_distribution(train_df[columns_to_plot], cols=3, width=30, height=50, hspace=0.45, wspace=0.5)
# reference: https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-nfl



rushing_df = train_df[ train_df['NflId'] == train_df['NflIdRusher']]

print(rushing_df.shape)

rushing_df.head()
rushing_df['Position'].value_counts()
rushing_df[rushing_df['Position'] == 'DE']
rushing_df[rushing_df['Position'] == 'DT']
rushing_df[rushing_df['Position'] == 'CB']
plt.figure(figsize=(12, 6))

sns.scatterplot(x='S', y='Yards', data=rushing_df, color='b')

plt.xlabel('Speed of Rusher')

plt.ylabel('Yards Gained')

plt.title('Running Speed vs Yards Gained', fontsize=24)

plt.show()
plt.figure(figsize=(12, 6))

sns.scatterplot(x='A', y='Yards', data=rushing_df, color='r')

plt.xlabel('Acceleration of Rusher')

plt.ylabel('Yards Gained')

plt.title('Rusher Acceleration vs Yards Gained', fontsize=24)

plt.show()
plt.figure(figsize=(12, 6))

sns.scatterplot(x='Dis', y='Yards', data=rushing_df, color='g')

plt.xlabel('Distance Traveled')

plt.ylabel('Yards Gained')

plt.title('Distance Traveled vs Yards Gained', fontsize=24)

plt.show()
plt.figure(figsize=(20, 6))

sns.boxplot(x='Distance', y='Yards', data=rushing_df, color='dodgerblue')

plt.xlabel('Yards Needed For First Down')

plt.ylabel('Yards Gained')

plt.title('Yards Needed for a First Down vs Yards Gained', fontsize=24)

plt.show()
plt.figure(figsize=(20, 10))

sns.boxplot(x='DefendersInTheBox', y='Yards', data=rushing_df[rushing_df['DefendersInTheBox'] > 3], color='dodgerblue')

plt.xlabel('Defenders in the Box')

plt.ylabel('Yards Gained')

plt.title('Defenders in the Box vs Yards Gained', fontsize=24)

plt.show()
plt.style.use('ggplot')



kws = dict(linewidth=.9)



g = sns.FacetGrid(train_df, col='OffenseFormation', col_wrap=3, size=8, aspect=.7, sharex=False)

g = (g.map(sns.boxplot, 'DefendersInTheBox', 'Yards', **kws)

     .set_titles("{col_name}")

     .fig.subplots_adjust(wspace=.1, hspace=.2))

# for ax in g.axes.flat:

#   ax.set_title(ax.get_title().split(' = ')[1])

#   for label in ax.get_xticklabels():

#     label.set_rotation(90)