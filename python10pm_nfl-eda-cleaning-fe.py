import re

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

import seaborn as sns

import gc

import warnings

warnings.filterwarnings("ignore")



%matplotlib inline

# This will help to have several prints in one cell in Jupyter.

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"





# don't truncate the pandas dataframe.

# so we can see all the columns

pd.set_option("display.max_columns", None)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Defining some useful functions and goodies for later preprocessing

int_types = ["int", "int8", "int16", "int32", "int64", "float"]
# Importing the DataFrame

train_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv")

rows = train_df.shape[0]

columns = train_df.shape[1]

print("Our DF has {} rows and {} columns.".format(rows, columns));

train_df.head()
# Let's check how many null values do we have in our DataFrame

for col in list(train_df.columns):

    null_values = train_df[col].isnull().sum()

    null_over_rows = round((null_values/rows)*100, 2)

    # print only columns with null values

    if null_over_rows > 0:

        print("The column {} has {} null values, this represents {}% of the total rows".format(col, null_values, null_over_rows))
# In general we see that there are very few empty values in the DataFrame
# let's see how many columns are numerical

int_columns = train_df.select_dtypes(include=int_types)

nr_int_columns = len(int_columns.columns)

int_over_columns = round((nr_int_columns/columns)*100, 2)

print("We have a total of {} numerical columns in the DF. Thi is roughly {}% of all columns.".format(nr_int_columns, \

                                                                                                       int_over_columns))
# let's see how many columns are categorical

cat_columns = train_df.select_dtypes(exclude=int_types)

nr_cat_columns = len(cat_columns.columns)

cat_over_columns = round((nr_cat_columns/columns)*100, 2)

print("We have a total of {} cathegorical columns in the DF. Thi is roughly {}% of all columns.".format(nr_cat_columns, \

                                                                                                       cat_over_columns))
# Almost half of all columns are categorical.
# Let's plot some distributions of the data to see how they behave.
# Plotting part

# Let's define some helper functions



def plot_distribution(x, y, title, xlabel, ylabel):

    '''

    Function that help to plot the distribution of some variables and expects 5 arguments:

        x: data for x axis.

        y: data for y axis.

        title: the title you want for our plot

        xlabel: the message you want to put to the x axis

        ylabel: the message you want to put to the y axis

    --------------------------------------------------------------------------------------

    Plots the data and annotates the max, min, median and mean.

    '''

    

    y_mean = [np.mean(y) for i in range(len(y))]

    y_median = [np.median(y) for i in range(len(y))]

    

    # beautiful color pallets

    # http://everyknightshoulddesign.blogspot.com/2013/08/beautiful-color-palettes-their-hex-codes.html

    colors = "#0F5959   #17A697   #638CA6   #8FD4D9   #D93240".split()

    

    # basic plot

    plt.figure(figsize=(15,7)) # define the size of the plot

    plt.plot(x, y, color = colors[0])

    plt.plot(x, y_mean, color = colors[1])

    plt.plot(x, y_median, color = colors[4])

    plt.ylim(np.min(y)*0.8 , np.max(y)*1.2)

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)



    # Using annotation to help understand and format a little the plot.



    plt.annotate('Mean value is {}'.format(np.mean(y)), # Message you want to put on the plot

                 xy=(150, np.mean(y)), # Coordinates were the arrows will point

                 xycoords='data', # 

                 xytext=(0.2, 0.90), # position of the text

                 textcoords='axes fraction', # specify the scale. If fraction the xytext has to be between 0 - 1

                 color = colors[1], # colors we have defined previously

                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = colors[1])) # parameters for the arrow



    plt.annotate('Median value is {}'.format(np.median(y)),

                 xy=(250, np.median(y)),

                 xycoords='data',

                 xytext=(0.7, 0.90), 

                 textcoords='axes fraction', 

                 color = colors[4],

                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color = colors[4])) 

    

    # calculate the max and min to determine the index of the value so that we can dinamically plot them

    # on different charts

    max_y = np.max(y)

    index_max = y.index(max_y)

    

    min_y = np.min(y)

    index_min = y.index(min_y)

    

    plt.annotate('Max value is {}'.format(np.max(y)),

                 xy=(index_max, max_y),

                 xycoords='data',

                 xytext=(25, -25), 

                 textcoords='offset points',

                 arrowprops=dict(arrowstyle="->"))



    plt.annotate('Min value is {}'.format(np.min(y)),

                 xy=(index_min, min_y),

                 xycoords='data',

                 xytext=(-25, 25), 

                 textcoords='offset points',

                 arrowprops=dict(arrowstyle="->"));

    



# More on annotations

# https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/annotation_demo.html

# Let's plot the number of interactions by game, SORTED by the number of interactions

groupby_df = pd.DataFrame(train_df[["GameId"]].groupby(by = "GameId").size()).sort_values(0, ascending=False)

groupby_df.rename({0:"InteractionsXGame"}, axis = 1, inplace = True)

games = groupby_df.shape[0]

# Same as rows.

total_interactions = groupby_df["InteractionsXGame"].sum()

print("We have a total of {} unique games. With a total of {} intercations.".format(games, total_interactions))



groupby_df.head(5)

x = [i +1 for i in range(len(list(groupby_df.index)))]

y = list(groupby_df["InteractionsXGame"])

title = "Interactions per game, mean, median, max and min"

xlabel = "Games"

ylabel = "Interactions per game"

# plot the data

plot_distribution(x, y, title, xlabel, ylabel)
# Let's do the same thing as before, but sort by the GameId.

# Since the game Id is like a datetime, we will be seeing the number of interactions as the season passes

# The results must be different.



groupby_df = pd.DataFrame(train_df[["GameId"]].groupby(by = "GameId").size()).sort_values("GameId", ascending=True)

groupby_df.rename({0:"InteractionsXGame"}, axis = 1, inplace = True)

games = groupby_df.shape[0]

# Same as rows.

total_interactions = groupby_df["InteractionsXGame"].sum()

print("We have a total of {} unique games. With a total of {} intercations.".format(games, total_interactions))



groupby_df.head(5)

x = [i +1 for i in range(len(list(groupby_df.index)))]

y = list(groupby_df["InteractionsXGame"])

title = "Interactions per game, mean, median, max and min"

xlabel = "Games"

ylabel = "Interactions per game"

# plot the data

plot_distribution(x, y, title, xlabel, ylabel)
# As we can see in the previous plot, the mean and the median of the interactions per game have almost the same value.
# Let's plot the number of unique players that participated in each game, SORTED by the count of unique players.



groupby_df  = pd.DataFrame(train_df.groupby("GameId")["NflId"].nunique()).sort_values("NflId", ascending=False)

groupby_df.head()



x = [i+1 for i in range(len(list(groupby_df.index)))]

y = list(groupby_df["NflId"])

title = "Count of unique players per game, mean, median, max and min"

xlabel = "Games"

ylabel = "Players per game"

# plot the data

plot_distribution(x, y, title, xlabel, ylabel)
# Let's do the same thing as before, but sort by the GameId.

# Since the game Id is like a datetime, we will be seeing the number of interactions as the season passes

# The results must be different.



groupby_df  = pd.DataFrame(train_df.groupby("GameId")["NflId"].nunique()).sort_values("GameId", ascending=False)

groupby_df.head()



x = [i+1 for i in range(len(groupby_df.index))]

y = list(groupby_df["NflId"])

title = "Count of unique players per game, mean, median, max and min"

xlabel = "Games"

ylabel = "Players per game"

# plot the data

plot_distribution(x, y, title, xlabel, ylabel)
# Let's plot the number of unique rushers that participated in each game, SORTED by the count of unique rushers.



groupby_df  = pd.DataFrame(train_df.groupby("GameId")["NflIdRusher"].nunique()).sort_values("NflIdRusher", ascending=False)

groupby_df.head()



x = [i+1 for i in range(len(groupby_df.index))]

y = list(groupby_df["NflIdRusher"])

title = "Count of unique rushers per game, mean, median, max and min"

xlabel = "Games"

ylabel = "Rushers per game"

# plot the data

plot_distribution(x, y, title, xlabel, ylabel)
# Let's do the same thing as before, but sort by the GameId.

# Since the game Id is like a datetime, we will be seeing the number of interactions as the season passes

# The results must be different.



groupby_df  = pd.DataFrame(train_df.groupby("GameId")["NflIdRusher"].nunique()).sort_values("GameId", ascending=True)

groupby_df.head()



x = [i+1 for i in range(len(groupby_df.index))]

y = list(groupby_df["NflIdRusher"])

title = "Count of unique rushers per game, mean, median, max and min"

xlabel = "Games"

ylabel = "Rushers per game"

# plot the data

plot_distribution(x, y, title, xlabel, ylabel)
train_df.groupby("Season")["GameId"].nunique()

# we can see that he vae 256 games for the season 2017 and 2018# Let's plot the interactions of the teams



groupby_df = pd.DataFrame(train_df.groupby(by = ["PossessionTeam", "Season"]).size())

groupby_df.rename({0:"Interactions"}, inplace = True, axis = 1)

groupby_df.unstack(level = 1).head()



# this way we can filter a multiindex dataframe and get the number of plays per team and season

data_2017 = groupby_df[np.in1d(groupby_df.index.get_level_values(1), 2017)].reset_index()

data_2017.sort_values("Interactions", ascending = False, inplace= True)





x = data_2017["PossessionTeam"]

y = data_2017["Interactions"]

plt.figure(figsize=(20,10))

plt.bar(x, y)

plt.title("Interactions by teams in season 2017")

plt.legend()



data_2018 = groupby_df[np.in1d(groupby_df.index.get_level_values(1), 2018)].reset_index()

data_2018.sort_values("Interactions", ascending = False, inplace= True)





x = data_2018["PossessionTeam"]

y = data_2018["Interactions"]

plt.figure(figsize=(20,10))

plt.bar(x, y)

plt.title("Interactions by teams in season 2018")

plt.legend()
# Let's take a deeper dive into one single game



gc.collect() # liberate some space in memory



one_play_df = train_df[(train_df["GameId"] == 2017090700) & (train_df["PlayId"] == 20170907000118)]



useful_columns = ["Team", "X", "Y", "NflId", "NflIdRusher", "GameClock"]



one_play_dfs = one_play_df[useful_columns]



# getting the data to plot

nlf_rusher_id = one_play_dfs["NflIdRusher"].iloc[0]

x_rusher = one_play_dfs[one_play_dfs["NflId"] == nlf_rusher_id]["X"].iloc[0]

y_rusher = one_play_dfs[one_play_dfs["NflId"] == nlf_rusher_id]["Y"].iloc[0]



x_away = one_play_dfs[one_play_dfs["Team"] == "away"]["X"]

y_away = one_play_dfs[one_play_dfs["Team"] == "away"]["Y"]



x_home = one_play_dfs[one_play_dfs["Team"] == "home"]["X"]

y_home = one_play_dfs[one_play_dfs["Team"] == "home"]["Y"]



# Setting the sizes of the pitch

# Thanks to this Kernel

# https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/111945#latest-647855



plt.xlim(0, 120)

plt.ylim(0, 53)

plt.scatter(x_away, y_away, color = "k", alpha=0.5, label = "Away/Visitor team", marker = "o")

plt.scatter(x_home, y_home, color = "b", alpha=0.5, label = "Home/Receiver team", marker = "^")

plt.scatter(x_rusher, y_rusher, color = "r", label = "Rusher nr {}".format(nlf_rusher_id), marker = "D")

plt.xlabel("Yars Long")

plt.ylabel("Yars Wide")

plt.legend(loc = "upper left");

train_df.head()



for quarter in list(train_df["Quarter"].unique()):

    

    x_away = train_df[(train_df["Team"] == "away") & (train_df["Quarter"] == quarter)]["X"]

    y_away = train_df[(train_df["Team"] == "away") & (train_df["Quarter"] == quarter)]["Y"]

    sns.jointplot(x_away, y_away, kind = "hex")

    plt.title("This is quarter {} and the data is for the away teams".format(quarter));



# we can see how the difference as we move from quarter 1 to the quarter 5
# Let's plot all the data for the home teams

for quarter in list(train_df["Quarter"].unique()):

    x_home = train_df[(train_df["Team"] == "home") & (train_df["Quarter"] == quarter)]["X"]

    y_home = train_df[(train_df["Team"] == "home") & (train_df["Quarter"] == quarter)]["Y"]

    sns.jointplot(x_home, y_home, kind = "hex")

    plt.title("This is quarter {} and the data is for the home teams".format(quarter));

    

# we can see how the difference as we move from quarter 1 to the quarter 5



# we can see almost identical distribution and a big concentration for the yars 35 aprox and 90 ayers (on the long side)

# and 23 and 33 on the wide side of the pitch
# Let's take a look one more time on our dataframe



# our complete dataframe

train_df.head()

# we will treat the college that have more than 5.000 entries as top college.

college_df = pd.DataFrame(train_df["PlayerCollegeName"].value_counts())

college_df.sample(20)

college_df.plot.hist(1000)
# Helper functions that we will use in the processing pipeline



# define this dict that will help normalize the data

mapping_team_dict = {"ARZ":"ARI",

                     "BLT":"BAL",

                     "CLV":"CLE",

                     "HST":"HOU"}



# let's clean the stadium type column

stadium_type_map = {"Outdoor":"Outdoor",

                    "Outdoors":"Outdoor",

                    "Open":"Outdoor",

                    "Oudoor":"Outdoor",

                    "Outddors":"Outdoor",

                    "Ourdoor":"Outdoor",

                    "Outdor":"Outdoor",

                    "Outside":"Outdoor",

                    "Retr. Roof-Open":"Outdoor",

                    "Outdoor Retr Roof-Open":"Outdoor",

                    "Retr. Roof - Open":"Outdoor",

                    "Indoor, Open Roof":"Outdoor",

                    "Domed, Open":"Outdoor",

                    "Heinz Field":"Outdoor",

                    "Bowl":"Outdoor",

                    "Retractable Roof":"Outdoor",

                    "Cloudy":"Outdoor",

                    "Indoors":"Indoor",

                    "Dome":"Indoor",

                    "Indoor":"Indoor",

                    "Domed, open":"Indoor",

                    "Retr. Roof-Closed":"Indoor",

                    "Retr. Roof - Closed":"Indoor",

                    "Domed, closed":"Indoor",

                    "Closed Dome":"Indoor",

                    "Domed":"Indoor",

                    "Dome, closed":"Indoor",

                    "Indoor, Roof Closed":"Indoor",

                    "Retr. Roof Closed":"Indoor",

                   "Other":"Indoor"} # any nans as indoor



# top colleges: with more than 5.000 entries

top_college = ['Alabama',

 'Ohio State',

 'Louisiana State',

 'Florida',

 'Georgia',

 'Florida State',

 'Notre Dame',

 'Clemson',

 'Oklahoma',

 'Stanford',

 'Wisconsin',

 'Michigan',

 'Southern California',

 'Penn State',

 'South Carolina',

 'California',

 'UCLA',

 'Iowa',

 'Oregon',

 'Miami',

 'Texas',

 'Washington',

 'North Carolina',

 'Texas A&M',

 'Mississippi',

 'Mississippi State',

 'Michigan State',

 'Utah',

 'North Carolina State',

 'Auburn',

 'Nebraska',

 'Louisville',

 'Boise State',

 'Tennessee',

 'Pittsburgh',

 'Missouri',

 'Central Florida',

 'Boston College',

 'USC',

 'Arkansas',

 'Kentucky',

 'Virginia Tech',

 'West Virginia',

 'Rutgers',

 'Colorado',

 'Oregon State',

 'Vanderbilt',

 'Temple',

 'Texas Christian',

 'Purdue',

 'Illinois',

 'Central Michigan',

 'LSU',

 'Utah State',

 'Maryland',

 'Oklahoma State',

 'Georgia Tech',

 'Cincinnati']



turf_dict = {\

'Grass':1, 

'Natural Grass':1, 

'Field Turf':-1, 

'Artificial':-1, 

'FieldTurf':-1,

'UBU Speed Series-S5-M':-1, 

'A-Turf Titan':-1, 

'UBU Sports Speed S5-M':-1,

'FieldTurf360':-1, 

'DD GrassMaster':-1, 

'Twenty-Four/Seven Turf':-1, 

'SISGrass':-1,

'FieldTurf 360':-1, 

'Natural grass':1, 

'Artifical':-1, 

'Natural':1, 

'Field turf':-1,

'Naturall Grass':1, 

'natural grass':1, 

'grass':1

}



def get_birth_year(birthdate):

    '''

    Get the year from the string column PlayerBirthDate of each player 

    '''

    

    year = int(birthdate.split("/")[2])

    

    return year



def label_encoder_teams(df):

    

    '''

    Encode the team values.

    '''

    

    columns = ["PossessionTeam", "FieldPosition", "HomeTeamAbbr", "VisitorTeamAbbr"]



    # first lets will any missing values with Other

    

    for col in columns:

        df.fillna({col:"Other"}, inplace=True)

        # cleaning any possible different names

        df[col].map(mapping_team_dict)

    

    le = LabelEncoder()

    # getting all the features so we don't miss anything

    unique_features = list(set(list(df["PossessionTeam"].unique()) + list(df["FieldPosition"].unique()) \

                        + list(df["HomeTeamAbbr"].unique()) + list(df["VisitorTeamAbbr"].unique())))

    le.fit(unique_features)

    

    for col in columns:

        df[col] = le.transform(df[col].values)

        

def label_encoder_team_formation(df):

    

    '''

    Encode the team formation values.

    '''

    columns = ["OffenseFormation", "OffensePersonnel", "DefensePersonnel"]

    

    for col in columns:

        df.fillna({col:"Other"}, inplace=True)

        le = LabelEncoder()    

        df[col] = le.fit_transform(df[col].values)

        

def label_encoder(df, column):

        

    '''

    Encode the stadiums.

    '''



    df.fillna({column:"Other"}, inplace=True)

    le = LabelEncoder()    

    df[column] = le.fit_transform(df[column].values)



def get_offense_personel(offense_scheme):

    '''

    Get's the number of persons from the OffensePersonnel column

    '''

    list_of_values = offense_scheme.split()

    counter = 0

    for val in list_of_values:

        try :

            counter += int(val)

        except:

            pass

    return counter



def input_defenders(df):

    si = SimpleImputer("most_frequent")

    df["DefendersInTheBox"] = si.fit_transform(df["DefendersInTheBox"].values).reshape(-1, 1)

    

def get_percentage(df, column):

    '''

    This function will return the percentage that a value represents from the total column

    '''

    # fill any posible nan

    if df[column].isnull().sum() > 0:

        df.fillna({column:"Other"}, inplace = True)

        

    list_of_values_ = list(df[column].unique())

    totals_ = df[column].value_counts()

    dict_to_return = {}

    for val in list_of_values_:

        dict_to_return[val] = totals_[val]/sum(totals_)

        

    return dict_to_return



def convert_to_int(value):

    value_to_return = ''

    

    try:

        return float(value)

    except:

        try:

            result = float(re.sub(r"[a-z]", "", value.lower()))

            return result

        except:

            # eliminate all text and eliminate the -

            result = re.sub(r"[a-z]", "", value.lower().replace("-", " "))

            # some might have multiple spaces, this will do the trick

            result = ' '.join(result.split())



            #result = list(map(int, result.split()))

            return result

        

def list_to_mean(value):

    if type(value) == str:

        list_values = value.split()

        list_values = list(map(float, list_values))

        return np.mean(list_values)

    else:

        return value
# create a custom pipeline to process data

drop_columns = ["GameId", "PlayId", "GameClock", "DisplayName", "JerseyNumber", "NflId", "Season", "NflIdRusher", "TimeHandoff", "TimeSnap", "PlayerBirthDate", "PlayerCollegeName"]



def pipeline(df):

    

    '''

    A custom pipeline to process our dataframe.

    '''

    

    # first fill in all the missing values

    # we will do in on a later stage

    

    # get the year of birth of each player

    df["Year_of_birth"] = df["PlayerBirthDate"].apply(get_birth_year)

    

    # now let's calculate how old is the player at the moment of the season

    df["Year_old"] = df["Season"] - df["Year_of_birth"]

    

    # Create a variable if the player is the rusher

    df["Is_rusher"] = df["NflId"] == df["NflIdRusher"]

    

    # modify in place

    label_encoder_teams(df)

    

    # encode data away or home

    df["Team"] = df["Team"].apply(lambda x: 0 if x == "away" else 1)

    

    # calculate the number offense personel in the 'box'

    df["OffenseInTheBox"] = df["OffensePersonnel"].apply(get_offense_personel)

    

    # input the nan from defenders in the box with the most common value

    #input_defenders(df)

    

    # more offense than deffense?

    df["MoreOffense"] = df["OffenseInTheBox"] > df["DefendersInTheBox"]

    

    # encode the teams formation and offence strategy

    label_encoder_team_formation(df)

    

    # clean stadium types

    df.fillna({"StadiumType":"Other"}, inplace = True)

    df["StadiumType"] = df["StadiumType"].map(stadium_type_map, na_action='ignore')

    df["StadiumType"] = df["StadiumType"].apply(lambda x: 1 if x == "Outdoor" else 0)

    

    # top college

    df["TopCollege"] = df["PlayerCollegeName"].apply(lambda x: 1 if x in top_college else 0)

    

    # play direction

    df["PlayDirection"] = df["PlayDirection"].map({"left":1, "right":0}, na_action="ignore")

    

    #list_with_grass = [name for name in list(copy_df["Turf"].unique()) if "grass" in name.lower()]

    #list_with_grass

    

    # map the turf

    df["Turf"] = df["Turf"].map(turf_dict)

    

    # encode the stadium

    label_encoder(df, "Stadium")

    

    # encode the Location

    label_encoder(df, "Location")

    

    # conver the height to float

    df["PlayerHeight"] = df["PlayerHeight"].apply(lambda x: float(x.replace("-", ".")))

    

    # convert the position to the percentage of most frequent

    dict_ = get_percentage(df, "Position")

    df["Position"] = df["Position"].map(dict_)

    

    # let's apply the same tecnique as for Position

    # I plan in the future to change and map them properly

    

    dict_ = get_percentage(df, "GameWeather")

    df["GameWeather"] = df["GameWeather"].map(dict_)

    

    dict_ = get_percentage(df, "WindDirection")

    df["WindDirection"] = df["WindDirection"].map(dict_)

    

    # sort and drop irrelevant columns

    df.sort_values(by = ["GameId", "PlayId"]).reset_index()

    df.drop(drop_columns, inplace = True, axis = 1)

    

    df["WindSpeed"] = df["WindSpeed"].apply(convert_to_int)

    df["WindSpeed"] = df["WindSpeed"].apply(list_to_mean)



    for col in list(df.columns):

        if df[col].isnull().sum() > 0:

            si = SimpleImputer(missing_values=np.nan, strategy = "mean")

            df[col] = si.fit_transform((df[col].values).reshape(-1, 1))

            

    return df

copy_df = train_df.copy(deep = True)

processed_df = pipeline(copy_df)

processed_df.sample(7)

processed_df["WindSpeed"].unique()

processed_df.isnull().sum()
# nice trick to filter string value in a column with different values

#small_df = processed_df[processed_df["WindSpeed1"].apply(lambda x: type(x) == str)]["WindSpeed1"].value_counts()




