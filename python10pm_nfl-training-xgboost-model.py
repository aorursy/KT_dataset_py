import re

import os

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import KFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import StratifiedShuffleSplit

from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

%matplotlib inline

import scipy

import scipy.stats as st

from xgboost import plot_importance



from datetime import datetime

        

# This will help to have several prints in one cell in Jupyter.

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



# don't truncate the pandas dataframe.

# so we can see all the columns

pd.set_option("display.max_columns", None)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Importing the DataFrame

train_df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2020/train.csv")

train_df.head()
# columns to drop after preprocesing

drop_columns = [\

"GameId", 

"PlayId", 

"GameClock", 

"DisplayName", 

"JerseyNumber", 

"NflId", 

"Season", 

"NflIdRusher", 

"TimeHandoff", 

"TimeSnap", 

"PlayerBirthDate", 

"PlayerCollegeName"

]



# define this dict that will help normalize the data

mapping_team_dict = {\

"ARZ":"ARI",

"BLT":"BAL",

"CLV":"CLE",

"HST":"HOU"

}



# let's clean the stadium type column

stadium_type_map = {\

"Outdoor":"Outdoor",

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

"Unknown":"Indoor" # any nans as indoor

} 



# top colleges: with more than 5.000 entries

top_college = [\

'Alabama',

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

'Cincinnati'

]



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

'grass':1,

'Unknown':0

}

# A beautiful solution for facing new labels

# https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values



class LabelEncoderExt(object):

    def __init__(self):

        """

        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]

        Unknown will be added in fit and transform will take care of new item. It gives unknown class id

        """

        self.label_encoder = LabelEncoder()



    def fit(self, data_list):

        """

        This will fit the encoder for all the unique values and introduce unknown value

        :param data_list: A list of string

        :return: self

        """

        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])

        self.classes_ = self.label_encoder.classes_



        return self



    def transform(self, data_list):

        """

        This will transform the data_list to id list where the new values get assigned to Unknown class

        :param data_list:

        :return:

        """

        new_data_list = list(data_list)

        for unique_item in np.unique(data_list):

            if unique_item not in self.label_encoder.classes_:

                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]



        return self.label_encoder.transform(new_data_list)


# Helper functions that we will use in the processing pipeline

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



    # first lets will any missing values with Unknown

    

    for col in columns:

        

        df.fillna({col:"Unknown"}, inplace=True)

        # cleaning any possible different names

        df[col].map(mapping_team_dict)

    

    le = LabelEncoderExt()

    # getting all the features so we don't miss anything

    unique_features = list(set(list(df["PossessionTeam"].unique()) + list(df["FieldPosition"].unique()) \

                        + list(df["HomeTeamAbbr"].unique()) + list(df["VisitorTeamAbbr"].unique())))

    le.fit(unique_features)

    

    for col in columns:

        df[col] = le.transform(df[col].values)

    

     # we will return the le so that we can use it when processing the data

     # and predicting the data

    return le



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

        

def label_encoder_team_formation(df):

    

    '''

    Encode the team formation values.

    '''

    columns = ["OffensePersonnel", "DefensePersonnel"]

    

    # first lets will any missing values with Unknown

    

    for col in columns:

        df.fillna({col:"Unknown"}, inplace=True)

        # cleaning any possible different names

        df[col].map(mapping_team_dict)

    

    le = LabelEncoderExt()

    # getting all the features so we don't miss anything

    unique_features = list(set(list(df["OffensePersonnel"].unique()) + list(df["DefensePersonnel"].unique())))

                           

    le.fit(unique_features)

    

    for col in columns:

        df[col] = le.transform(df[col].values)

    

     # we will return the le so that we can use it when processing the data

     # and predicting the data

    return le

        

def label_encoder(df, column):

    

    df.fillna({column:"Unknown"}, inplace=True)

    le = LabelEncoderExt()

    le.fit(df[column].values)

    df[column] = le.transform(df[column].values)

    # we will return the le so that we can use it when processing the data

     # and predicting the data

    return le



def get_percentage(df, column):

    '''

    This function will return the percentage that a value represents from the total column

    '''

    # fill any posible nan

    if df[column].isnull().sum() > 0:

        df.fillna({column:"Unknown"}, inplace = True)

        

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

            result = re.sub(r"[a-z]", "", value.lower().replace("-", " ")).replace(",", " ")

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

def pipeline(df):

    

    '''

    A custom pipeline to process our dataframe.

    '''

    

    df["Year_of_birth"] = df["PlayerBirthDate"].apply(get_birth_year) # get the year of birth of each player

    

    df["Year_old"] = df["Season"] - df["Year_of_birth"] # now let's calculate how old is the player at the moment of the season

    

    df["Is_rusher"] = df["NflId"] == df["NflIdRusher"] # Create a variable if the player is the rusher

    

    global le_teams # create a global variable to use in the predict part

    le_teams = label_encoder_teams(df) # modify in place the train data and get label encoder for teams



    df["Team"] = df["Team"].apply(lambda x: 0 if x == "away" else 1) # encode data away or home

    

    df["OffenseInTheBox"] = df["OffensePersonnel"].apply(get_offense_personel) # calculate the number offense personel in the 'box'

    

    df["MoreOffense"] = df["OffenseInTheBox"] > df["DefendersInTheBox"] # more offense than deffense?

    

    global le_team_formation # create a global variable to use in the predict part

    le_team_formation = label_encoder_team_formation(df) # encode the teams formation and offence strategy and get the label encoder for formation

    

    global le_offense_formation # create a global variable to use in the predict part

    le_offense_formation = label_encoder(df, "OffenseFormation") # encode the offense formation

    

    df.fillna({"StadiumType":"Unknown"}, inplace = True) # clean stadium types

    df["StadiumType"] = df["StadiumType"].map(stadium_type_map, na_action='ignore')

    df["StadiumType"] = df["StadiumType"].apply(lambda x: 1 if x == "Outdoor" else 0)

    

    df["TopCollege"] = df["PlayerCollegeName"].apply(lambda x: 1 if x in top_college else 0) # top college

    

    df["PlayDirection"] = df["PlayDirection"].map({"left":1, "right":0}, na_action="ignore") # play direction

    

    df["Turf"] = df["Turf"].map(turf_dict) # map the turf

    

    global le_stadium # create a global variable to use in the predict part

    le_stadium = label_encoder(df, "Stadium") # encode the stadium

    

    global le_location # create a global variable to use in the predict part

    le_location = label_encoder(df, "Location") # encode the Location

    

    df["PlayerHeight"] = df["PlayerHeight"].apply(lambda x: float(x.replace("-", "."))) # convert the height to float

    

    global dict_position # create a global variable to use in the predict part

    dict_position = get_percentage(df, "Position") # convert the position to the percentage of most frequent

    df["Position"] = df["Position"].map(dict_position)

    

    global dict_weather # create a global variable to use in the predict part

    dict_weather = get_percentage(df, "GameWeather") # convert the gameweather to the percentage of most frequent

    df["GameWeather"] = df["GameWeather"].map(dict_weather)

    

    global dict_wind_direction # create a global variable to use in the predict part

    dict_wind_direction = get_percentage(df, "WindDirection") # convert the wind direction to the percentage of most frequent

    df["WindDirection"] = df["WindDirection"].map(dict_wind_direction)

    

    df.sort_values(by = ["GameId", "PlayId"]).reset_index() # sort and drop irrelevant columns

    df.drop(drop_columns, inplace = True, axis = 1)

    

    df["WindSpeed"] = df["WindSpeed"].apply(convert_to_int)

    df["WindSpeed"] = df["WindSpeed"].apply(list_to_mean)



    for col in df.columns:

        if df[col].isnull().sum() > 0:

            si = SimpleImputer(missing_values=np.nan, strategy = "mean")

            df[col] = si.fit_transform((df[col].values).reshape(-1, 1))

    

    # if we have missed something

    df.fillna(-999, inplace = True)



    return df

copy_df = train_df.copy(deep = True)

processed_df = pipeline(copy_df)

processed_df.head()
columns = list(processed_df.columns)

columns.remove("Yards")



X_train = processed_df[columns]

y_train = processed_df["Yards"]



y = y_train.values

target = y[np.arange(0, len(X_train), 22)]

standard_deviation = np.std(target)
# model = XGBRegressor(\

#     n_estimators=500,

#     min_child_weight = 2,

#     max_depth=6,

#     verbosity = 1,

#     n_jobs=8,                                              

#     scale_pos_weight=1.025,

#     tree_method='exact',

#     objective = 'reg:squarederror',

#     predictor='cpu_predictor',

#     colsample_bytree = 0.66,

#     subsample = 1,

#     gamma = 0,

#     learning_rate=0.15,

#     num_parallel_tree = 1 

# )



# kfold = KFold(n_splits=10, shuffle=True, random_state=42)



# for train_index, test_index in kfold.split(X_train, y_train):

#     # train folds

#     X_train_fold = X_train.iloc[train_index]

#     y_train_fold = y_train.iloc[train_index]

    

#     # test folds

#     X_test_fold = X_train.iloc[test_index]

#     y_test_fold = y_train.iloc[test_index]

    

#     model.fit(X_train_fold, y_train_fold, eval_metric="rmse", early_stopping_rounds=50,

#                     eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],verbose=False)

model = XGBRegressor()

model.fit(X_train, y_train)
plt.rcParams["figure.figsize"] = (14, 7)

plot_importance(model, )

plt.show()
# create a custom pipeline to process data

def pipeline_for_predict(df):

    

    '''

    A custom pipeline to process our dataframe before we predict the data.

    '''

    

    df["Year_of_birth"] = df["PlayerBirthDate"].apply(get_birth_year) # get the year of birth of each player

    

    df["Year_old"] = df["Season"] - df["Year_of_birth"] # now let's calculate how old is the player at the moment of the season

    

    df["Is_rusher"] = df["NflId"] == df["NflIdRusher"] # Create a variable if the player is the rusher

    

    columns = ["PossessionTeam", "FieldPosition", "HomeTeamAbbr", "VisitorTeamAbbr"]

    for col in columns:

        df[col].fillna({col:"Unknown"}, inplace = True)

        for i in range(len(df[col])):

            if df[col].iloc[i] not in le_teams.classes_:

                df[col].iloc[i] = "Unknown"

        df[col] = le_teams.transform(df[col].values)



    df["Team"] = df["Team"].apply(lambda x: 0 if x == "away" else 1) # encode data away or home

    

    df["OffenseInTheBox"] = df["OffensePersonnel"].apply(get_offense_personel) # calculate the number offense personel in the 'box'

    

    df["MoreOffense"] = df["OffenseInTheBox"] > df["DefendersInTheBox"] # more offense than deffense?

    

    df.fillna({"OffenseFormation":"Unknown"}, inplace = True)

    for i in range(len(df["OffenseFormation"])):

        if df["OffenseFormation"].iloc[i] not in le_offense_formation.classes_:

            df["OffenseFormation"].iloc[i] = "Unknown"

    df["OffenseFormation"] = le_offense_formation.transform(df["OffenseFormation"].values)

    

    columns = ["OffensePersonnel", "DefensePersonnel"]

    

    for col in columns:

        df[col].fillna({col:"Unknown"}, inplace = True)

        for i in range(len(df[col])):

            if df[col].iloc[i] not in le_team_formation.classes_:

                df[col].iloc[i] = "Unknown"

        df[col] = le_team_formation.transform(df[col].values)

    

    df.fillna({"StadiumType":"Unknown"}, inplace = True) # clean stadium types

    

    for i in range(len(df["StadiumType"])):

        val = df["StadiumType"].iloc[i]

        if val not in stadium_type_map.keys():

            df["StadiumType"].iloc[i] = "Unknown"

    

    df["StadiumType"] = df["StadiumType"].map(stadium_type_map, na_action='ignore')

    df["StadiumType"] = df["StadiumType"].apply(lambda x: 1 if x == "Outdoor" else 0)

    

    df["TopCollege"] = df["PlayerCollegeName"].apply(lambda x: 1 if x in top_college else 0) # top college

    

    df["PlayDirection"] = df["PlayDirection"].map({"left":1, "right":0}, na_action="ignore") # play direction

    

    for i in range(len(df["Turf"])):

        val = df["Turf"].iloc[i]

        if val not in turf_dict.keys():

            df["Turf"].iloc[i] = "Unknown"

    

    df["Turf"] = df["Turf"].map(turf_dict) # map the turf

    

    df.fillna({"Stadium":"Unknown"}, inplace = True)

    for i in range(len(df["Stadium"])):

        if df["Stadium"].iloc[i] not in le_stadium.classes_:

            df["Stadium"].iloc[i] = "Unknown"

    df["Stadium"] = le_stadium.transform(df["Stadium"].values)

    

    df.fillna({"Location":"Unknown"}, inplace = True)

    for i in range(len(df["Location"])):

        if df["Location"].iloc[i] not in le_location.classes_:

            df["Location"].iloc[i] = "Unknown"

            

    # we have some problems with new

    df["Location"] = le_location.transform(df["Location"].values)    

    

    df["PlayerHeight"] = df["PlayerHeight"].apply(lambda x: float(x.replace("-", "."))) # convert the height to float

    

    df.fillna({"Position":"Unknown"}, inplace = True)

    for i in range(len(df["Position"])):

        val = df["Position"].iloc[i]

        if val not in dict_position.keys():

            df["Position"].iloc[i] = "Unknown"

            

    df["Position"] = df["Position"].map(dict_position)

    

    df.fillna({"GameWeather":"Unknown"}, inplace = True)

    for i in range(len(df["GameWeather"])):

        val = df["GameWeather"].iloc[i]

        if val not in dict_weather.keys():

            df["GameWeather"].iloc[i] = "Unknown"

            

    df["GameWeather"] = df["GameWeather"].map(dict_weather)



    df.fillna({"WindDirection":"Unknown"}, inplace = True)

    for i in range(len(df["WindDirection"])):

        val = df["WindDirection"].iloc[i]

        if val not in dict_wind_direction.keys():

            df["WindDirection"].iloc[i] = "Unknown" 

            

    df["WindDirection"] = df["WindDirection"].map(dict_wind_direction)

    

    df.sort_values(by = ["GameId", "PlayId"]).reset_index() # sort and drop irrelevant columns

    df.drop(drop_columns, inplace = True, axis = 1)

    

    df["WindSpeed"] = df["WindSpeed"].apply(convert_to_int)

    df["WindSpeed"] = df["WindSpeed"].apply(list_to_mean)



    # if we have missed something

    df.fillna(-999, inplace = True)

            

    return df



import tqdm

from kaggle.competitions import nflrush

env = nflrush.make_env()



counter = 0

for (test_df, sample_prediction_df) in tqdm.tqdm(env.iter_test()):

    try:

        test_df_processed = pipeline_for_predict(test_df)

        y_pred_p = model.predict(test_df_processed)

        y_pred_first = y_pred_p[0]

        

    except:

        test_df

        y_pred_first = 1

        

    pred_df = np.zeros((1, 199))



    for A in range(len(pred_df[0])):

        current_cdf = scipy.stats.norm(loc = y_pred_first, scale = standard_deviation).cdf(A-99)

        pred_df[0][A] = current_cdf



#     pred_df[0][:80] = 0



    final_pred_df = pd.DataFrame(data=pred_df, columns=sample_prediction_df.columns)

    env.predict(final_pred_df)

    

env.write_submission_file()
