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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb

from xgboost import XGBClassifier, XGBRegressor

from xgboost import plot_importance

from catboost import CatBoostRegressor

from matplotlib import pyplot

import shap



from time import time

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats

import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.model_selection import KFold, StratifiedKFold

import gc

import json



from pandas.io.json import json_normalize

import json
import numpy as np

import pandas as pd

from pandas.io.json import json_normalize

import json

from tqdm import tqdm_notebook as tqdm

from collections import Counter

from scipy import stats as sp

train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")

test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")

train_labels = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")

specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")
train = train[train.installation_id.isin(train_labels.installation_id.unique())]
def json_parser(dataframe, column):

    dataframe.reset_index(drop=True, inplace=True)

    parsed_set = dataframe[column].apply(json.loads)

    parsed_set = json_normalize(parsed_set)

    parsed_set.drop(columns=['event_count', 'event_code', 'game_time'], inplace=True)

    merged_set = pd.merge(

        dataframe,

        parsed_set,

        how='inner',

        left_index= True,

        right_index=True

    )



    del merged_set[column]



    return merged_set
def encode_title(train, test):



    train["title_event_code"] = list(map(lambda x, y : str(x) + '_' + str(y), train["title"], train["event_code"]))

    test["title_event_code"] = list(map(lambda x, y : str(x) + '_' + str(y), test["title"], test["event_code"]))

    unique_title_event_code = list(set(train["title_event_code"].unique()).union(set(test["title_event_code"].unique())))



    unique_titles = list(set(train["title"].unique()).union(set(test["title"].unique())))



    unique_event_codes = list(set(train["event_code"].unique()).union(set(test["event_code"].unique())))



    unique_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))



    unique_event_ids = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))



    unique_assessments = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))



    unique_games = list(set(train[train['type'] == 'Game']['title'].value_counts().index).union(set(test[test['type'] == 'Game']['title'].value_counts().index)))



    unique_clips = list(set(train[train['type'] == 'Clip']['title'].value_counts().index).union(set(test[test['type'] == 'Clip']['title'].value_counts().index)))



    unique_activitys = list(set(train[train['type'] == 'Activity']['title'].value_counts().index).union(set(test[test['type'] == 'Activity']['title'].value_counts().index)))



    # convert text into datetime

    train["timestamp"] = pd.to_datetime(train["timestamp"])

    test["timestamp"]  = pd.to_datetime(test["timestamp"])



    unique_data = {

        "unique_title_event_code" : unique_title_event_code,

        "unique_titles" : unique_titles,

        "unique_event_codes" : unique_event_codes,

        "unique_worlds" : unique_worlds,

        "unique_event_ids" : unique_event_ids,

        "unique_assessments" : unique_assessments,

        "unique_games" : unique_games,

        "unique_clips" : unique_clips,

        "unique_activitys" : unique_activitys

    }



    return train, test, unique_data
train, test, unique_data = encode_title(train, test)
def get_data(user_sample,unique_data ,test=False):



    final_features = []



    features = {}



    Assessments_count = {"count_"+ass : 0 for ass in unique_data["unique_assessments"]}

    Clips_count = {"count_"+clip : 0 for clip in unique_data["unique_clips"]}

    Games_count = {"count_"+game : 0 for game in unique_data["unique_games"]}

    Activitys_count = {"count_"+activity:0 for activity in unique_data["unique_activitys"]}

    Worlds_count = {"count_"+world:0 for world in unique_data["unique_worlds"]}

    Title_event_code_count = {etc:0 for etc in unique_data["unique_title_event_code"]}

    Event_ids_count = {uei:0 for uei in unique_data["unique_event_ids"]}

    Event_code_count = {code: 0 for code in unique_data["unique_event_codes"]}



    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    accuracy_groups_game = {'game_0':0, 'game_1':0, 'game_2':0, 'game_3':0}



    features["accumulated_false"] = 0

    features["accumulated_true"] = 0

    features["accumulated_false_ass"] = 0

    features["accumulated_true_ass"] = 0



    Clip_duration_accumulated = {"accu_duration_"+clip : 0 for clip in unique_data["unique_clips"]}

    Clip_duration = {"duration_"+clip : 0 for clip in unique_data["unique_clips"]}



    Games_duration_accumulated = {"accu_duration_"+game : 0 for game in unique_data["unique_games"]}

    Games_duration = {"duration_"+game : 0 for game in unique_data["unique_games"]}



    Activitys_duration_accumulated = {"accu_duration_"+activity:0 for activity in unique_data["unique_activitys"]}

    Activitys_duration = {"duration_"+activity:0 for activity in unique_data["unique_activitys"]}



    Assessments_duration_accumulated = {"accu_duration_"+ass : 0 for ass in unique_data["unique_assessments"]}

    Assessments_duration = {"duration_"+ass : 0 for ass in unique_data["unique_assessments"]}



    features.update(accuracy_groups)

    features.update(accuracy_groups_game)



    for i, session in user_sample.groupby("game_session", sort=False):

        

        # i = game_session_id



        session_type = session.type.iloc[0]

        session_title = session.title.iloc[0]

        session_world = session.world.iloc[0]



        Worlds_count["count_"+session_world] += 1



        if session_type == "Clip":

            # count

            Clips_count["count_"+session_title] += 1



            # duration

            try:

                index = session.index.values[0]

                duration = (user_sample.timestamp.loc[index+1] - user_sample.timestamp.loc[index]).seconds

                Clip_duration["duration_"+session_title] = duration

                Clip_duration_accumulated["accu_duration_"+session_title] += duration

            except:

                pass



            features["predicted_before_title"] = session_title



        if session_type == "Activity":

            # count

            Activitys_count["count_"+session_title] += 1



            # duration

            duration = round(session.game_time.iloc[-1] / 1000, 2)

            Activitys_duration["duration_"+session_title] = duration

            Activitys_duration_accumulated["accu_duration_"+session_title] += duration



            features["predicted_before_title"] = session_title





        if session_type == "Game":

            # count

            Games_count["count_"+session_title] += 1



            # duration

            duration = round(session.game_time.iloc[-1] / 1000, 2)

            Games_duration["duration_"+session_title] = duration

            Games_duration_accumulated["accu_duration_"+session_title] += duration



            features["predicted_before_title"] = session_title



        if (session_type == "Assessment") & (test or len(session) > 1):

            

            predicted_title = session["title"].iloc[0]

            predicted_game_session = session["game_session"].iloc[0]

            predicted_timestamp_session = session["timestamp"].iloc[0]



            features["predicted_title"] = predicted_title

            features["installation_id"] = session["installation_id"].iloc[0]

            features["game_session"] = predicted_game_session

            features["timestamp_session"] = predicted_timestamp_session



            pred_title_df = user_sample[user_sample.title == predicted_title]

            pred_title_df = pred_title_df[pred_title_df.timestamp < predicted_timestamp_session]



            predicted_assessment = {"pred_bef_attampt":0,

                            "pred_bef_true" : np.nan,

                            "pred_bef_false" : np.nan,

                            "pred_bef_acc_group": np.nan,

                            "pred_bef_accuracy": np.nan,

                            "pred_bef_timespent" : np.nan,

                            "pred_bef_time_diff":np.nan

                            }

            try:

                if len(pred_title_df) > 2:

                    for i, pred_session in pred_title_df.groupby("game_session", sort=False):

                        predicted_assessment["pred_bef_attampt"] += 1

                        predicted_assessment["pred_bef_timespent"] = round(pred_session.game_time.iloc[-1] / 1000, 2)

                        

                        if predicted_title == "Bird Measurer (Assessment)":

                            predicted_data = pred_session[pred_session.event_code == 4110]

                        else:

                            predicted_data = pred_session[pred_session.event_code == 4100]



                        true_attempts = predicted_data[predicted_data.correct == True]['correct'].count()

                        false_attempts = predicted_data[predicted_data.correct == False]['correct'].count()

                        accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

                        group = accuracy_groups_def(accuracy)



                        predicted_assessment["pred_bef_true"] = true_attempts

                        predicted_assessment["pred_bef_false"] = false_attempts

                        predicted_assessment["pred_bef_accuracy"] = accuracy

                        predicted_assessment["pred_bef_acc_group"] = group



                    predicted_assessment["pred_bef_time_diff"] = (predicted_timestamp_session - pred_title_df.timestamp.iloc[-1]).seconds

            except:

                pass



            

            counter_df = user_sample[user_sample.timestamp < predicted_timestamp_session]

            

            Title_event_code_count = update_counters(Title_event_code_count, "title_event_code", counter_df)

            Event_ids_count = update_counters(Event_ids_count, "event_id", counter_df)

            Event_code_count = update_counters(Event_code_count,"event_code", counter_df)



            features.update(Title_event_code_count.copy())

            features.update(Event_ids_count.copy())

            features.update(Event_code_count.copy())



            ed = AllEventDataFeatures(features, counter_df)

            try:

                ed.event_code_2000()

            except:

                pass

            try:

                ed.event_code_2010()

            except:

                pass

            try:

                ed.event_code_2020()

            except:

                pass



            try:

                ed.event_code_2030()

            except:

                pass

            try:

                ed.event_code_2025()

            except:

                pass

            try:

                ed.event_code_2035()

            except:

                pass

            try:

                ed.event_code_2040()

            except:

                pass

            try:

                ed.event_code_2050()

            except:

                pass

            try:

                ed.event_code_2060()

            except:

                pass

            try:

                ed.event_code_2070()

            except:

                pass

            try:

                ed.event_code_2075()

            except:

                pass

            try:

                ed.event_code_2080()

            except:

                pass

            try:

                ed.event_code_2081()

            except:

                pass

            try:

                ed.event_code_2083()

            except:

                pass

            try:

                ed.event_code_3010()

            except:

                pass

            try:

                ed.event_code_3020()

            except:

                pass

            try:

                ed.event_code_3021()

            except:

                pass

            try:

                ed.event_code_3110()

            except:

                pass

            try:

                ed.event_code_3120()

                ed.event_code_3121()

                ed.event_code_4010()

                ed.event_code_4020()

                ed.event_code_4021()

                ed.event_code_4022()

                ed.event_code_4025()

                ed.event_code_4030()

                ed.event_code_4031()

                ed.event_code_4035()

                ed.event_code_4040()

                ed.event_code_4045()

                ed.event_code_4050()

                ed.event_code_4070()

                ed.event_code_4080()

                ed.event_code_4090()

                ed.event_code_4095()

                ed.event_code_4100()

                ed.event_code_4110()

                ed.event_code_4220()

                ed.event_code_4230()

                ed.event_code_4235()

                ed.event_code_5000()

                ed.event_code_5010()

            except:

                pass



            edf = ed.Event_features

            features_ed = ed.features



            features.update(edf.copy())

            features.update(features_ed.copy())







            

            features.update(predicted_assessment.copy())



            features.update(Clips_count.copy())

            features.update(Clip_duration.copy())

            features.update(Clip_duration_accumulated.copy())

            features.update(Games_count.copy())

            features.update(Games_duration.copy())

            features.update(Games_duration_accumulated.copy())

            features.update(Activitys_count.copy())

            features.update(Activitys_duration.copy())

            features.update(Activitys_duration_accumulated.copy())

            features.update(Assessments_count.copy())

            features.update(Assessments_duration.copy())

            features.update(Assessments_duration_accumulated.copy())





            final_features.append(features.copy())



            try:

                # last Assessment

                last_assessment = {

                                "last_bef_true" : np.nan,

                                "last_bef_false" : np.nan,

                                "last_bef_acc_group": np.nan,

                                "last_bef_accuracy": np.nan,

                                "last_bef_timespent" : np.nan,

                                "last_bef_title" : np.nan

                                }



                last_assessment["last_bef_timespent"] = round(session.game_time.iloc[-1] / 1000, 2)



                if predicted_title == "Bird Measurer (Assessment)":

                    predicted_data = session[session.event_code == 4110]

                else:

                    predicted_data = session[session.event_code == 4100]

                

                true_attempts = predicted_data[predicted_data.correct == True]['correct'].count()

                false_attempts = predicted_data[predicted_data.correct == False]['correct'].count()

                accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

                group = accuracy_groups_def(accuracy)



                last_assessment["last_bef_true"] = true_attempts

                last_assessment["last_bef_false"] = false_attempts

                last_assessment["last_bef_accuracy"] = accuracy

                last_assessment["last_bef_acc_group"] = group

                last_assessment["last_bef_title"] = predicted_title





                features.update(last_assessment.copy())

        

            except:

                pass

            



            # count

            Assessments_count["count_"+session_title] += 1



            # duration

            duration = round(session.game_time.iloc[-1] / 1000, 2)

            Assessments_duration["duration_"+session_title] = duration

            Assessments_duration_accumulated["accu_duration_"+session_title] += duration



        ed = EventDataFeatures(features, session, user_sample, session_type, session_title)

        try:

            ed.event_code_2000()

        except:

            pass

        try:

            ed.event_code_2010()

        except:

            pass

        try:

            ed.event_code_2020()

        except:

            pass



        try:

            ed.event_code_2030()

        except:

            pass

        try:

            ed.event_code_2025()

        except:

            pass

        try:

            ed.event_code_2035()

        except:

            pass

        try:

            ed.event_code_2040()

        except:

            pass

        try:

            ed.event_code_2050()

        except:

            pass

        try:

            ed.event_code_2060()

        except:

            pass

        try:

            ed.event_code_2070()

        except:

            pass

        try:

            ed.event_code_2075()

        except:

            pass

        try:

            ed.event_code_2080()

        except:

            pass

        try:

            ed.event_code_2081()

        except:

            pass

        try:

            ed.event_code_2083()

        except:

            pass

        try:

            ed.event_code_3010()

        except:

            pass

        try:

            ed.event_code_3020()

        except:

            pass

        try:

            ed.event_code_3021()

        except:

            pass

        try:

            ed.event_code_3110()

        except:

            pass

        try:

            ed.event_code_3120()

            ed.event_code_3121()

            ed.event_code_4010()

            ed.event_code_4020()

            ed.event_code_4021()

            ed.event_code_4022()

            ed.event_code_4025()

            ed.event_code_4030()

            ed.event_code_4031()

            ed.event_code_4035()

            ed.event_code_4040()

            ed.event_code_4045()

            ed.event_code_4050()

            ed.event_code_4070()

            ed.event_code_4080()

            ed.event_code_4090()

            ed.event_code_4095()

            ed.event_code_4100()

            ed.event_code_4110()

            ed.event_code_4220()

            ed.event_code_4230()

            ed.event_code_4235()

            ed.event_code_5000()

            ed.event_code_5010()

        except:

            pass



        edf = ed.Event_features

        features_ed = ed.features



        features.update(edf.copy())

        features.update(features_ed.copy())







    if test:

        return final_features[-1]

    else:

        return final_features
def accuracy_groups_def(accuracy):

    if accuracy == 0:

        return 0

    elif accuracy == 1:

        return 3

    elif accuracy == 0.5:

        return 2

    else:

        return 1



def update_counters(counter: dict, col:str, counter_df):



    num_of_session_count = Counter(counter_df[col])



    for k in num_of_session_count.keys():

        x = k

        counter[x] += num_of_session_count[k]

    return counter









class EventDataFeatures(object):

    def __init__(self, features, session, user_sample, session_type, session_title):

        self.features = features

        self.session = session

        self.user_sample = user_sample

        self.session_type = session_type

        self.session_title = session_title

        self.Event_features = {}

        self.unique_event_codes = self.session.event_code.unique()



    def event_code_2000(self):

        pass



    def event_code_2010(self):

        """

        ['The exit game event is triggered when the game is quit. 

        This is used to compute things like time spent in game. 

        Depending on platform this may / may not be possible. 

        NOTE: “quit” also means navigating away from game.']

        """

        if 2010 in self.unique_event_codes:

            session_duration = self.session[self.session.event_code == 2010]["session_duration"].values[0]

            self.Event_features["session_duration_"+self.session_title] = round(session_duration / 1000, 2)



    def event_code_2020(self):

        """

        ['The start round event is triggered at the start of a round when 

        the player is prompted to weigh and arrange the chests. There is only one round per playthrough.

         This event provides information about the game characteristics of the round (i.e. resources, objectives, setup). 

         It is used in calculating things like time spent in a round (for speed and accuracy), attempts at 

        solving a round, and the number of rounds the player has visited (exposures).']

        """

        pass



    def event_code_2025(self):

        """

        ['The reset dinosaurs event is triggered when the player has placed the last dinosaur, 

        but not all dinosaurs are in the correct position. 

        This event provides information about the game characteristics of the round (i.e. resources, objectives, setup). 

        It is used to indicate a significant change in state during play.']

        

        This event is used for calculating time spent in a round and 

        the number of rounds the player has completed (completion).

        """

        pass



    def event_code_2030(self):

        """

        ['The beat round event is triggered when the player finishes a round by filling the jar.

         This event is used for calculating time spent in a round and

          the number of rounds the player has completed (completion).']



        """

        if 2030 in self.unique_event_codes:

            rounds = self.session[self.session.event_code == 2030]



            round_duration = rounds["duration"].values

            self.Event_features["round_duration_2030_sum_"+self.session_title] = round_duration.sum()

            self.Event_features["round_duration_2030_avg_"+self.session_title] = round_duration.mean()

            self.Event_features["round_duration_2030_std_"+self.session_title] = round_duration.std()

            self.Event_features["round_duration_2030_max_"+self.session_title] = round_duration.max()

            self.Event_features["round_duration_2030_min_"+self.session_title] = round_duration.min()

            self.Event_features["number_of_attempts_2030_"+self.session_title] = round_duration.count()



            try:

                round_rounds = rounds["round"].values

                self.Event_features["round_2030_max_"+self.session_title] = round_rounds.max()

            except:

                pass



            try:

                round_misses = rounds["misses"].values

                self.Event_features["misses_2030_sum_"+self.session_title] = round_misses.sum()

                self.Event_features["misses_2030_avg_"+self.session_title] = round_misses.mean()

                self.Event_features["misses_2030_max_"+self.session_title] = round_misses.max()

            except:

                pass

    

    def event_code_2035(self):

        """

        ['The finish filling tub event is triggered after the player finishes filling up the tub. 

        It is used to separate a section of gameplay that is different from the estimation section of the game.']

        """

        if 2035 in self.unique_event_codes:

            rounds = self.session[self.session.event_code == 2035]



            round_duration = rounds["duration"].values

            self.Event_features["round_duration_2035_sum_"+self.session_title] = round_duration.sum()

            self.Event_features["round_duration_2035_avg_"+self.session_title] = round_duration.mean()

    

    def event_code_2040(self):

        """

        ['The start level event is triggered when a new level begins 

        (at the same time as the start round event for the first round in the level). 

        This event is used for calculating time spent in a level (for speed and accuracy), 

        and the number of levels the player has completed (completion).']

        """

        pass



    def event_code_2050(self):

        """

        ['The beat level event is triggered when a level has been completed and 

        the player has cleared all rounds in the current layout (occurs at the same time as 

        the beat round event for the last round in the previous level). This event is used for 

        calculating time spent in a level (for speed and accuracy), 

        and the number of levels the player has completed (completion).']

        """

        if 2050 in self.unique_event_codes:

            level = self.session[self.session.event_code == 2050]



            level_duration = level["duration"].values

            self.Event_features["level_duration_2050_sum_"+self.session_title] = level_duration.sum()

            self.Event_features["level_duration_2050_avg_"+self.session_title] = level_duration.mean()

            self.Event_features["level_duration_2050_std_"+self.session_title] = level_duration.std()

            self.Event_features["level_duration_2050_max_"+self.session_title] = level_duration.max()

            self.Event_features["level_duration_2050_min_"+self.session_title] = level_duration.min()



            try:

                level_rounds = level["level"].values

                self.Event_features["level_2050_max_"+self.session_title] = level_rounds.max()

            except:

                pass



            try:

                level_misses = level["misses"].values

                self.Event_features["level_misses_2050_sum_"+self.session_title] = level_misses.sum()

                self.Event_features["level_misses_2050_avg_"+self.session_title] = level_misses.mean()

                self.Event_features["level_misses_2050_sum_"+self.session_title] = level_misses.std()

            except:

                pass



    def event_code_2060(self):

        """

        ['The start tutorial event is triggered at the start of the tutorial. 

        It is used in calculating time spent in the tutorial.']

        """

        pass



    def event_code_2070(self):

        """

        ['The beat round event is triggered when the player finishes the tutorial. 

        This event is used for calculating time spent in the tutorial.']

        """

        if 2070 in self.unique_event_codes:

            tutorial = self.session[self.session.event_code == 2070]



            tutorial_duration = tutorial["duration"].values

            self.Event_features["tutorial_duration_2070_sum_"+self.session_title] = tutorial_duration.sum()

            self.Event_features["tutorial_duration_2070_avg_"+self.session_title] = tutorial_duration.mean()

            self.Event_features["tutorial_duration_2070_std_"+self.session_title] = tutorial_duration.std()

            self.Event_features["tutorial_duration_2070_max_"+self.session_title] = tutorial_duration.max()

            self.Event_features["tutorial_duration_2070_min_"+self.session_title] = tutorial_duration.min()

    

    def event_code_2075(self):

        """

        ['The beat round event is triggered when the player skips the tutorial by clicking on the skip button.

         This event is used for calculating time spent in the tutorial.']

        """

        if 2075 in self.unique_event_codes:



            tutorial = self.session[self.session.event_code == 2075]



            self.Event_features["tutorial_skiping_count_2075_"+self.session_title] = tutorial["duration"].count()



    def event_code_2080(self):

        """

        ['The movie started event triggers when an intro or outro movie starts to play. 

        It identifies the movie being played. This is used to determine how long players 

        spend watching the movies (more relevant after the first play 

        through when the skip option is available).']

        """

        if 2080 in self.unique_event_codes:



            movie = self.session[self.session.event_code == 2080]



            movie_duration = movie["duration"].values

            self.Event_features["movie_duration_2080_sum_"+self.session_title] = movie_duration.sum()

            self.Event_features["movie_duration_2080_avg_"+self.session_title] = movie_duration.mean()

            self.Event_features["movie_duration_2080_std_"+self.session_title] = movie_duration.std()

            self.Event_features["movie_duration_2080_max_"+self.session_title] = movie_duration.max()

            self.Event_features["movie_duration_2080_min_"+self.session_title] = movie_duration.min()



    def event_code_2081(self):

        """

        ['The movie started event triggers when an intro or outro movie starts to play. 

        It identifies the movie being played. This is used to determine how long players 

        spend watching the movies (more relevant after the first play 

        through when the skip option is available).']

        """

        if 2081 in self.unique_event_codes:



            movie = self.session[self.session.event_code == 2081]



            self.Event_features["movie_skiping_count_2081_"+self.session_title] = movie["duration"].count()

    

    def event_code_2083(self):

        """

        ['The movie started event triggers when an intro or outro movie starts to play. 

        It identifies the movie being played. This is used to determine how long players 

        spend watching the movies (more relevant after the first play 

        through when the skip option is available).']

        """

        if 2083 in self.unique_event_codes:



            movie = self.session[self.session.event_code == 2083]



            movie_duration = movie["duration"].values

            self.Event_features["movie_duration_2083_sum_"+self.session_title] = movie_duration.sum()

            self.Event_features["movie_duration_2083_avg_"+self.session_title] = movie_duration.mean()

    

    def event_code_3010(self):

        """

        ['The system-initiated instruction event occurs when the game delivers instructions to the player.

         It contains information that describes the content of the instruction. This event differs from events 3020

          and 3021 as it captures instructions that are not given in response to player action. 

          These events are used to determine the effectiveness of the instructions. We can answer questions like,

         "did players who received instruction X do better than those who did not?"']

        """

        if 3010 in self.unique_event_codes:



            instruction = self.session[self.session.event_code == 3010]



            instruction_duration = instruction["total_duration"].values

            self.Event_features["instruction_duration_3010_sum_"+self.session_title] = instruction_duration.sum()

            self.Event_features["instruction_duration_3010_avg_"+self.session_title] = instruction_duration.mean()

        

            #self.Event_features["instruction_media_type_3010_"+self.session_title] = instruction["media_type"].values_count().index[0]

            

            self.Event_features["instruction_media_type_3010_count_"+self.session_title] = instruction["media_type"].count()



    def event_code_3020(self):

        """

        ['The system-initiated feedback (Incorrect) event occurs when the game starts delivering feedback 

        to the player in response to an incorrect round attempt (pressing the go button with the incorrect answer). 

        It contains information that describes the content of the instruction. These events are used to determine 

        the effectiveness of the feedback. We can answer questions like 

        "did players who received feedback X do better than those who did not?"']

        """

        if 3020 in self.unique_event_codes:



            Incorrect = self.session[self.session.event_code == 3020]



            Incorrect_duration = Incorrect["total_duration"].values

            self.Event_features["Incorrect_duration_3020_sum_"+self.session_title] = Incorrect_duration.sum()

            self.Event_features["Incorrect_duration_3020_avg_"+self.session_title] = Incorrect_duration.mean()

            #self.Event_features["Incorrect_duration_3020_std_"+self.session_title] = Incorrect_duration.std()

            #self.Event_features["Incorrect_duration_3020_max_"+self.session_title] = Incorrect_duration.max()

            #self.Event_features["Incorrect_duration_3020_min_"+self.session_title] = Incorrect_duration.min()

        

            #self.Event_features["Incorrect_media_type_3020_"+self.session_title] = Incorrect["media_type"].values[0]

            

            self.Event_features["Incorrect_media_type_3020_count_"+self.session_title] = Incorrect["media_type"].count()

    



    def event_code_3021(self):

        """

        ['The system-initiated feedback (Correct) event occurs when the game 

        starts delivering feedback to the player in response to a correct round attempt 

        (pressing the go button with the correct answer). It contains information that describes the

         content of the instruction, and will likely occur in conjunction with a beat round event. 

         These events are used to determine the effectiveness of the feedback. We can answer questions like, 

        "did players who received feedback X do better than those who did not?"']

        """

        if 3021 in self.unique_event_codes:



            Correct = self.session[self.session.event_code == 3021]



            Correct_duration = Correct["total_duration"].values

            self.Event_features["Correct_duration_3021_sum_"+self.session_title] = Correct_duration.sum()

            self.Event_features["Correct_duration_3021_avg_"+self.session_title] = Correct_duration.mean()

            #self.Event_features["Correct_duration_3021_std_"+self.session_title] = Correct_duration.std()

            #self.Event_features["Correct_duration_3021_max_"+self.session_title] = Correct_duration.max()

            #self.Event_features["Correct_duration_3021_min_"+self.session_title] = Correct_duration.min()

        

            #self.Event_features["Correct_media_type_3021_"+self.session_title] = Correct["media_type"].values[0]

            

            self.Event_features["Correct_media_type_3021_count_"+self.session_title] = Correct["media_type"].count()



    def event_code_3110(self):

        """

        ['The end of system-initiated instruction event occurs when the game finishes 

        delivering instructions to the player. It contains information that describes the

         content of the instruction including duration. These events are used to determine the 

         effectiveness of the instructions and the amount of time they consume. We can answer questions like, 

        "how much time elapsed while the game was presenting instruction?"']

        """

        if 3110 in self.unique_event_codes:



            Instuction = self.session[self.session.event_code == 3110]



            Instuction_duration = Instuction["duration"].values

            self.Event_features["Instuction_duration_3110_sum_"+self.session_title] = Instuction_duration.sum()

            self.Event_features["Instuction_duration_3110_avg_"+self.session_title] = Instuction_duration.mean()

            #self.Event_features["Instuction_duration_3110_std_"+self.session_title] = Instuction_duration.std()

            #self.Event_features["Instuction_duration_3110_max_"+self.session_title] = Instuction_duration.max()

            #self.Event_features["Instuction_duration_3110_min_"+self.session_title] = Instuction_duration.min()

        

            #self.Event_features["Instuction_media_type_3110_"+self.session_title] = Instuction["media_type"].values[0]

            

            self.Event_features["Instuction_media_type_3110_count_"+self.session_title] = Instuction["media_type"].count()



    def event_code_3120(self):

        """

        ['The end of system-initiated feedback (Incorrect) event 

        occurs when the game finishes delivering feedback to the player in response

         to an incorrect round attempt (pressing the go button with the incorrect answer). 

         It contains information that describes the content of the instruction. 

         These events are used to determine the effectiveness of the feedback. We can answer questions like,

         “how much time elapsed while the game was presenting feedback?”']

        """

        if 3120 in self.unique_event_codes:



            IncorrectInstruction = self.session[self.session.event_code == 3120]



            IncorrectInstruction_duration = IncorrectInstruction["duration"].values

            self.Event_features["IncorrectInstruction_duration_3120_sum_"+self.session_title] = IncorrectInstruction_duration.sum()

            self.Event_features["IncorrectInstruction_duration_3120_avg_"+self.session_title] = IncorrectInstruction_duration.mean()

            #self.Event_features["IncorrectInstruction_duration_3120_std_"+self.session_title] = IncorrectInstruction_duration.std()

            #self.Event_features["IncorrectInstruction_duration_3120_max_"+self.session_title] = IncorrectInstruction_duration.max()

            #self.Event_features["IncorrectInstruction_duration_3120_min_"+self.session_title] = IncorrectInstruction_duration.min()

        

            #self.Event_features["IncorrectInstruction_media_type_3120_"+self.session_title] = IncorrectInstruction["media_type"].values[0]

            

            self.Event_features["IncorrectInstruction_media_type_3120_count_"+self.session_title] = IncorrectInstruction["media_type"].count()



    def event_code_3121(self):

        """

        ['The end of system-initiated feedback (Correct) event 

        occurs when the game finishes delivering feedback to the player in response

         to an incorrect round attempt (pressing the go button with the incorrect answer). 

         It contains information that describes the content of the instruction. 

         These events are used to determine the effectiveness of the feedback. We can answer questions like,

         “how much time elapsed while the game was presenting feedback?”']

        """

        if 3121 in self.unique_event_codes:



            CorrectInstruction = self.session[self.session.event_code == 3121]



            CorrectInstruction_duration = CorrectInstruction["duration"].values

            self.Event_features["CorrectInstruction_duration_3121_sum_"+self.session_title] = CorrectInstruction_duration.sum()

            self.Event_features["CorrectInstruction_duration_3121_avg_"+self.session_title] = CorrectInstruction_duration.mean()

            #self.Event_features["CorrectInstruction_duration_3121_std_"+self.session_title] = CorrectInstruction_duration.std()

            #self.Event_features["CorrectInstruction_duration_3121_max_"+self.session_title] = CorrectInstruction_duration.max()

            #self.Event_features["CorrectInstruction_duration_3121_min_"+self.session_title] = CorrectInstruction_duration.min()

        

            #self.Event_features["CorrectInstruction_media_type_3121_"+self.session_title] = CorrectInstruction["media_type"].values[0]

            

            self.Event_features["CorrectInstruction_media_type_3121_count_"+self.session_title] = CorrectInstruction["media_type"].count()





    def event_code_4010(self):

        """



        ['This event occurs when the player clicks to start 

        the game from the starting screen.']

        

        """



        if 4010 in self.unique_event_codes:

            



            click_start = self.session[self.session.event_code == 4010]

            index = click_start.index.values[0]

            duration = (self.user_sample.timestamp.loc[index] - self.user_sample.timestamp.loc[index-1]).seconds



            self.Event_features["click_start_duration_4010_"+self.session_title] = duration

    

    def event_code_4020(self):

        """

        ['This event occurs when the player 

        clicks a group of objects. It contains information 

        about the group clicked, the state of the game, and the

         correctness of the action. This event is 

         to diagnose player strategies and understanding.']



         It contains information about the state of the game and the correctness of the action. This event is used 

         to diagnose player strategies and understanding.

        """

        

        if 4020 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4020]



            if self.session_title == "Bottle Filler (Activity)":

                true_attempts = event_data[event_data.jar_filled == True]['jar_filled'].count()

                false_attempts = event_data[event_data.jar_filled == False]['jar_filled'].count()

                accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



                self.Event_features["True_attempts_4020_"+self.session_title] = true_attempts

                self.Event_features["False_attempts_4020_"+self.session_title] = false_attempts

                self.Event_features["Accuracy_attempts_4020_"+self.session_title] = accuracy

                

                group = accuracy_groups_def(accuracy)



                self.features['game_'+str(group)] += 1

                self.features["accumulated_false"] += false_attempts

                self.features["accumulated_true"] += true_attempts



            elif self.session_title == 'Sandcastle Builder (Activity)':

                sandcastle_duration = event_data["duration"].values

                self.Event_features["sandcastle_duration_4020_sum_"+self.session_title] = sandcastle_duration.sum()

                self.Event_features["sandcastle_duration_4020_avg_"+self.session_title] = sandcastle_duration.mean()

                #self.Event_features["sandcastle_duration_4020_std_"+self.session_title] = sandcastle_duration.std()

                #self.Event_features["sandcastle_duration_4020_max_"+self.session_title] = sandcastle_duration.max()

                #self.Event_features["sandcastle_duration_4020_min_"+self.session_title] = sandcastle_duration.min()



            elif self.session_title == "Cart Balancer (Assessment)":

                try:

                    true_attempts = event_data[event_data.size == 'left']['size'].count()

                    false_attempts = event_data[event_data.size == 'right']['size'].count()

                    accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



                    self.Event_features["Left_attempts_4020_"+self.session_title] = true_attempts

                    self.Event_features["Right_attempts_4020_"+self.session_title] = false_attempts

                    self.Event_features["Accuracy_attempts_4020_"+self.session_title] = accuracy

                    

                    

                    

                    group = accuracy_groups_def(accuracy)

                    self.features['game_'+str(group)] += 1



                    self.features["accumulated_false"] += false_attempts

                    self.features["accumulated_true"] += true_attempts

                except:

                    pass



            elif self.session_title == "Fireworks (Activity)":

                true_attempts = event_data[event_data.launched == True]['launched'].count()

                false_attempts = event_data[event_data.launched == False]['launched'].count()

                accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



                self.Event_features["True_attempts_4020_"+self.session_title] = true_attempts

                self.Event_features["False_attempts_4020_"+self.session_title] = false_attempts

                self.Event_features["Accuracy_attempts_4020_"+self.session_title] = accuracy

                

                group = accuracy_groups_def(accuracy)

                self.features['game_'+str(group)] += 1

                

                

                self.features["accumulated_false"] += false_attempts

                self.features["accumulated_true"] += true_attempts



                rocket_duration = event_data["duration"].values

                self.Event_features["rocket_duration_4020_sum_"+self.session_title] = rocket_duration.sum()

                self.Event_features["rocket_duration_4020_avg_"+self.session_title] = rocket_duration.mean()

                self.Event_features["rocket_duration_4020_std_"+self.session_title] = rocket_duration.std()

                self.Event_features["rocket_duration_4020_max_"+self.session_title] = rocket_duration.max()

                self.Event_features["rocket_duration_4020_min_"+self.session_title] = rocket_duration.min()



                rocket_height = event_data["height"].values

                self.Event_features["rocket_height_4020_sum_"+self.session_title] = rocket_height.sum()

                self.Event_features["rocket_height_4020_avg_"+self.session_title] = rocket_height.mean()

                self.Event_features["rocket_height_4020_std_"+self.session_title] = rocket_height.std()

                self.Event_features["rocket_height_4020_max_"+self.session_title] = rocket_height.max()

                self.Event_features["rocket_height_4020_min_"+self.session_title] = rocket_height.min()



            elif self.session_title == "Watering Hole (Activity)":

                

                water_level = event_data["water_level"].values

                self.Event_features["water_level_4020_sum_"+self.session_title] = water_level.sum()

                self.Event_features["water_level_4020_avg_"+self.session_title] = water_level.mean()

                self.Event_features["water_level_4020_std_"+self.session_title] = water_level.std()

                self.Event_features["water_level_4020_max_"+self.session_title] = water_level.max()

                self.Event_features["water_level_4020_min_"+self.session_title] = water_level.min()



            elif self.session_title == "Chicken Balancer (Activity)":

                

                true_attempts = event_data[event_data["layout.right.pig"] == True]['layout.right.pig'].count()

                false_attempts = event_data[event_data["layout.right.pig"] == False]['layout.right.pig'].count()

                accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



                self.Event_features["True_attempts_4020_"+self.session_title] = true_attempts

                self.Event_features["False_attempts_4020_"+self.session_title] = false_attempts

                self.Event_features["Accuracy_attempts_4020_"+self.session_title] = accuracy



            elif self.session_title == 'Flower Waterer (Activity)':

                

                flower_duration = event_data["duration"].values

                self.Event_features["flower_duration_4020_sum_"+self.session_title] = flower_duration.sum()

                self.Event_features["flower_duration_4020_avg_"+self.session_title] = flower_duration.mean()

                #self.Event_features["flower_duration_4020_std_"+self.session_title] = flower_duration.std()

                #self.Event_features["flower_duration_4020_max_"+self.session_title] = flower_duration.max()

                #self.Event_features["flower_duration_4020_min_"+self.session_title] = flower_duration.min()

            

            elif self.session_title == "Egg Dropper (Activity)":

                

                true_attempts = event_data[event_data["gate.side"] == 'left']['gate.side'].count()

                false_attempts = event_data[event_data["gate.side"] == 'right']['gate.side'].count()

                accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



                self.Event_features["Left_attempts_4020_"+self.session_title] = true_attempts

                self.Event_features["Right_attempts_4020_"+self.session_title] = false_attempts

                self.Event_features["Accuracy_attempts_4020_"+self.session_title] = accuracy



            else:

                true_attempts = event_data[event_data.correct == True]['correct'].count()

                false_attempts = event_data[event_data.correct == False]['correct'].count()

                accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



                self.Event_features["True_attempts_4020_"+self.session_title] = true_attempts

                self.Event_features["False_attempts_4020_"+self.session_title] = false_attempts

                self.Event_features["Accuracy_attempts_4020_"+self.session_title] = accuracy

                

                

                

                group = accuracy_groups_def(accuracy)

                self.features['game_'+str(group)] += 1

                

                self.features["accumulated_false"] += false_attempts

                self.features["accumulated_true"] += true_attempts



    def event_code_4021(self):



        if 4021 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4021]



            if self.session_title == "Sandcastle Builder (Activity)":

                amount_sand = event_data["sand"].values

                self.Event_features["amount_sand_4020_sum_"+self.session_title] = amount_sand.sum()

                self.Event_features["amount_sand_4020_avg_"+self.session_title] = amount_sand.mean()

                #self.Event_features["amount_sand_4020_std_"+self.session_title] = amount_sand.std()

                self.Event_features["amount_sand_4020_max_"+self.session_title] = amount_sand.max()

                #self.Event_features["amount_sand_4020_min_"+self.session_title] = amount_sand.min()

            

            elif self.session_title == 'Watering Hole (Activity)':

                cloud_size = event_data["cloud_size"].values

                self.Event_features["cloud_size_4020_sum_"+self.session_title] = cloud_size.sum()

                self.Event_features["cloud_size_4020_avg_"+self.session_title] = cloud_size.mean()

                #self.Event_features["cloud_size_4020_std_"+self.session_title] = cloud_size.std()

                self.Event_features["cloud_size_4020_max_"+self.session_title] = cloud_size.max()

                #self.Event_features["cloud_size_4020_min_"+self.session_title] = cloud_size.min()

            else:

                pass

    

    def event_code_4022(self):

        pass



    def event_code_4025(self):

        if 4025 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4025]



            if self.session_title == "Cauldron Filler (Assessment)":

                true_attempts = event_data[event_data.correct == True]['correct'].count()

                false_attempts = event_data[event_data.correct == False]['correct'].count()

                accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



                self.Event_features["True_attempts_4025_"+self.session_title] = true_attempts

                self.Event_features["False_attempts_4025_"+self.session_title] = false_attempts

                self.Event_features["Accuracy_attempts_4025_"+self.session_title] = accuracy

                

                group = accuracy_groups_def(accuracy)

                self.features['game_'+str(group)] += 1

                

                self.features["accumulated_false"] += false_attempts

                self.features["accumulated_true"] += true_attempts

            

            elif self.session_title == "Bug Measurer (Activity)":



                self.Event_features["Bug_length_max_4025_"+self.session_title] = event_data["buglength"].max()

                self.Event_features["Number_of_Bugs_4025_"+self.session_title] = event_data["buglength"].count()

            

            else:

                pass



    def event_code_4030(self):

        pass



    def event_code_4031(self):

        pass



    def event_code_4035(self):



        if 4035 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4035]



            self.Event_features["wrong_place_count_4035_"+self.session_title] = len(event_data)



            if self.session_title == "All Star Sorting":



                wrong_place = event_data["duration"].values

                self.Event_features["wrong_place_duration_4035_sum_"+self.session_title] = wrong_place.sum()

                self.Event_features["wrong_place_duration_4035_avg_"+self.session_title] = wrong_place.mean()

                #self.Event_features["wrong_place_duration_4035_std_"+self.session_title] = wrong_place.std()

                #self.Event_features["wrong_place_duration_4035_max_"+self.session_title] = wrong_place.max()

                #self.Event_features["wrong_place_duration_4035_min_"+self.session_title] = wrong_place.min()



            elif self.session_title == "Bug Measurer (Activity)":



                wrong_place = event_data["duration"].values

                self.Event_features["wrong_place_duration_4035_sum_"+self.session_title] = wrong_place.sum()

                self.Event_features["wrong_place_duration_4035_avg_"+self.session_title] = wrong_place.mean()

                #self.Event_features["wrong_place_duration_4035_std_"+self.session_title] = wrong_place.std()

                #self.Event_features["wrong_place_duration_4035_max_"+self.session_title] = wrong_place.max()

                #self.Event_features["wrong_place_duration_4035_min_"+self.session_title] = wrong_place.min()



            elif self.session_title == "Pan Balance":

                pass



            elif self.session_title == "Chicken Balancer (Activity)":

                wrong_place = event_data["duration"].values

                self.Event_features["wrong_place_duration_4035_sum_"+self.session_title] = wrong_place.sum()

                self.Event_features["wrong_place_duration_4035_avg_"+self.session_title] = wrong_place.mean()

                #self.Event_features["wrong_place_duration_4035_std_"+self.session_title] = wrong_place.std()

                #self.Event_features["wrong_place_duration_4035_max_"+self.session_title] = wrong_place.max()

                #self.Event_features["wrong_place_duration_4035_min_"+self.session_title] = wrong_place.min()



            elif self.session_title == "Chest Sorter (Assessment)":



                wrong_place = event_data["duration"].values

                self.Event_features["wrong_place_duration_4035_sum_"+self.session_title] = wrong_place.sum()

                self.Event_features["wrong_place_duration_4035_avg_"+self.session_title] = wrong_place.mean()

                #self.Event_features["wrong_place_duration_4035_std_"+self.session_title] = wrong_place.std()

                #self.Event_features["wrong_place_duration_4035_max_"+self.session_title] = wrong_place.max()

                #self.Event_features["wrong_place_duration_4035_min_"+self.session_title] = wrong_place.min()



            else:



                try:

                    wrong_place = event_data["duration"].values

                    self.Event_features["wrong_place_duration_4035_sum_"+self.session_title] = wrong_place.sum()

                    self.Event_features["wrong_place_duration_4035_avg_"+self.session_title] = wrong_place.mean()

                    #self.Event_features["wrong_place_duration_4035_std_"+self.session_title] = wrong_place.std()

                    #self.Event_features["wrong_place_duration_4035_max_"+self.session_title] = wrong_place.max()

                    #self.Event_features["wrong_place_duration_4035_min_"+self.session_title] = wrong_place.min()

                except:

                    pass

    

    def event_code_4040(self):

        pass



    def event_code_4045(self):

        pass



    def event_code_4050(self):

        pass



    def event_code_4070(self):

        """

        

        ['This event occurs when the player clicks on

            something that isn’t covered elsewhere. 

            It can be useful in determining if there are

            attractive distractions (things the player think

            should do something, but don’t) in the game, or

            diagnosing players 

            who are having mechanical difficulties (near misses).']

        """

        if 4070 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4070]



            self.Event_features["something_not_covered_count_4070_"+self.session_title] = len(event_data)



    def event_code_4080(self):



        if 4080 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4080]



            self.Event_features["mouse_over_count_4080_"+self.session_title] = len(event_data)



            try:



                dwell_time = event_data["dwell_time"].values

                self.Event_features["dwell_time_duration_4080_sum_"+self.session_title] = dwell_time.sum()

                self.Event_features["dwell_time_duration_4080_avg_"+self.session_title] = dwell_time.mean()

                self.Event_features["dwell_time_duration_4080_std_"+self.session_title] = dwell_time.std()

                self.Event_features["dwell_time_duration_4080_max_"+self.session_title] = dwell_time.max()

                self.Event_features["dwell_time_duration_4080_min_"+self.session_title] = dwell_time.min()

            

            except:

                pass



    def event_code_4090(self):



        if 4090 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4090]



            self.Event_features["Player_help_count_4090_"+self.session_title] = len(event_data)



    def event_code_4095(self):



        if 4095 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4095]



            self.Event_features["Plage_again_4095_"+self.session_title] = len(event_data)

        

    def event_code_4100(self):

        

        if 4100 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4100]



            true_attempts = event_data[event_data.correct == True]['correct'].count()

            false_attempts = event_data[event_data.correct == False]['correct'].count()

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



            self.Event_features["True_attempts_4100_"+self.session_title] = true_attempts

            self.Event_features["False_attempts_4100_"+self.session_title] = false_attempts

            self.Event_features["Accuracy_attempts_4100_"+self.session_title] = accuracy

            

            group = accuracy_groups_def(accuracy)

            self.features[group] += 1

            

            self.features["accumulated_false_ass"] += false_attempts

            self.features["accumulated_true_ass"] += true_attempts



    def event_code_4110(self):

        



        if 4110 in self.unique_event_codes:



            event_data = self.session[self.session.event_code == 4110]



            true_attempts = event_data[event_data.correct == True]['correct'].count()

            false_attempts = event_data[event_data.correct == False]['correct'].count()

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



            self.Event_features["True_attempts_4110_"+self.session_title] = true_attempts

            self.Event_features["False_attempts_4110_"+self.session_title] = false_attempts

            self.Event_features["Accuracy_attempts_4110_"+self.session_title] = accuracy

            

            group = accuracy_groups_def(accuracy)

            self.features[group] += 1

            

            self.features["accumulated_false_ass"] += false_attempts

            self.features["accumulated_true_ass"] += true_attempts

            



    def event_code_4220(self):

        pass



    def event_code_4230(self):

        pass



    def event_code_4235(self):

        pass



    def event_code_5000(self):

        pass



    def event_code_5010(self):

        pass





def FeaturesGeneration(x, feature_name):

    feature_dict = dict()



    feature_dict['mean'+feature_name] = np.mean(x)

    feature_dict['max'+feature_name] = np.max(x)

    feature_dict['min'+feature_name] = np.min(x)

    feature_dict['std'+feature_name] = np.std(x)

    feature_dict['var'+feature_name] = np.var(x)

    feature_dict['ptp'+feature_name] = np.ptp(x)

    feature_dict['percentile_10'+feature_name] = np.percentile(x, 10)

    feature_dict['percentile_20'+feature_name] = np.percentile(x, 20)

    feature_dict['percentile_30'+feature_name] = np.percentile(x, 30)

    feature_dict['percentile_40'+feature_name] = np.percentile(x, 40)

    feature_dict['percentile_50'+feature_name] = np.percentile(x, 50)

    feature_dict['percentile_60'+feature_name] = np.percentile(x, 60)

    feature_dict['percentile_70'+feature_name] = np.percentile(x, 70)

    feature_dict['percentile_80'+feature_name] = np.percentile(x, 80)

    feature_dict['percentile_90'+feature_name] = np.percentile(x, 90)



    # scipy

    feature_dict['skew'+feature_name] = sp.stats.skew(x)

    feature_dict['kurtosis'+feature_name] = sp.stats.kurtosis(x)

    feature_dict['kstat_1'+feature_name] = sp.kstat(x, 1)

    feature_dict['kstat_2'+feature_name] = sp.kstat(x, 2)

    feature_dict['kstat_3'+feature_name] = sp.kstat(x, 3)

    feature_dict['kstat_4'+feature_name] = sp.kstat(x, 4)

    feature_dict['moment_1'+feature_name] = sp.stats.moment(x, 1)

    feature_dict['moment_2'+feature_name] = sp.stats.moment(x, 2)

    feature_dict['moment_3'+feature_name] = sp.stats.moment(x, 3)

    feature_dict['moment_4'+feature_name] = sp.stats.moment(x, 4)



    return feature_dict





class AllEventDataFeatures(object):

    def __init__(self, features, user_sample):

        self.features = features

        self.user_sample = user_sample

        self.Event_features = {}

        self.unique_event_codes = self.user_sample.event_code.unique()



    def event_code_2000(self):

        pass



    def event_code_2010(self):

        """

        ['The exit game event is triggered when the game is quit. 

        This is used to compute things like time spent in game. 

        Depending on platform this may / may not be possible. 

        NOTE: “quit” also means navigating away from game.']

        """

        if 2010 in self.unique_event_codes:

            session_duration = self.user_sample[self.user_sample.event_code == 2010]["session_duration"].values

            features_2010 = FeaturesGeneration(session_duration, "_2010")

            self.Event_features.update(features_2010.copy())



    def event_code_2020(self):

        """

        ['The start round event is triggered at the start of a round when 

        the player is prompted to weigh and arrange the chests. There is only one round per playthrough.

         This event provides information about the game characteristics of the round (i.e. resources, objectives, setup). 

         It is used in calculating things like time spent in a round (for speed and accuracy), attempts at 

        solving a round, and the number of rounds the player has visited (exposures).']

        """

        pass



    def event_code_2025(self):

        """

        ['The reset dinosaurs event is triggered when the player has placed the last dinosaur, 

        but not all dinosaurs are in the correct position. 

        This event provides information about the game characteristics of the round (i.e. resources, objectives, setup). 

        It is used to indicate a significant change in state during play.']

        

        This event is used for calculating time spent in a round and 

        the number of rounds the player has completed (completion).

        """

        pass



    def event_code_2030(self):

        """

        ['The beat round event is triggered when the player finishes a round by filling the jar.

         This event is used for calculating time spent in a round and

          the number of rounds the player has completed (completion).']



        """

        if 2030 in self.unique_event_codes:

            rounds = self.user_sample[self.user_sample.event_code == 2030]



            round_duration = rounds["duration"].values

            

            features_2030 = FeaturesGeneration(round_duration, "_2030")

            self.Event_features.update(features_2030.copy())

            



            try:

                round_misses = rounds["misses"].values



                features_2030 = FeaturesGeneration(round_misses, "_2030_misses")

                self.Event_features.update(features_2030.copy())

            except:

                pass

    

    def event_code_2035(self):

        """

        ['The finish filling tub event is triggered after the player finishes filling up the tub. 

        It is used to separate a section of gameplay that is different from the estimation section of the game.']

        """

        if 2035 in self.unique_event_codes:

            rounds = self.user_sample[self.user_sample.event_code == 2035]



            round_duration = rounds["duration"].values

            features_2035 = FeaturesGeneration(round_duration, "_2035")

            self.Event_features.update(features_2035.copy())

    

    def event_code_2040(self):

        """

        ['The start level event is triggered when a new level begins 

        (at the same time as the start round event for the first round in the level). 

        This event is used for calculating time spent in a level (for speed and accuracy), 

        and the number of levels the player has completed (completion).']

        """

        pass



    def event_code_2050(self):

        """

        ['The beat level event is triggered when a level has been completed and 

        the player has cleared all rounds in the current layout (occurs at the same time as 

        the beat round event for the last round in the previous level). This event is used for 

        calculating time spent in a level (for speed and accuracy), 

        and the number of levels the player has completed (completion).']

        """

        if 2050 in self.unique_event_codes:

            level = self.user_sample[self.user_sample.event_code == 2050]



            level_duration = level["duration"].values

            features_2050 = FeaturesGeneration(level_duration, "_2050")

            self.Event_features.update(features_2050.copy())





    def event_code_2060(self):

        """

        ['The start tutorial event is triggered at the start of the tutorial. 

        It is used in calculating time spent in the tutorial.']

        """

        pass



    def event_code_2070(self):

        """

        ['The beat round event is triggered when the player finishes the tutorial. 

        This event is used for calculating time spent in the tutorial.']

        """

        if 2070 in self.unique_event_codes:

            tutorial = self.user_sample[self.user_sample.event_code == 2070]



            tutorial_duration = tutorial["duration"].values

            

            features_2070 = FeaturesGeneration(tutorial_duration, "_2070")

            self.Event_features.update(features_2070.copy())

    

    def event_code_2075(self):

        """

        ['The beat round event is triggered when the player skips the tutorial by clicking on the skip button.

         This event is used for calculating time spent in the tutorial.']

        """

        pass



    def event_code_2080(self):

        """

        ['The movie started event triggers when an intro or outro movie starts to play. 

        It identifies the movie being played. This is used to determine how long players 

        spend watching the movies (more relevant after the first play 

        through when the skip option is available).']

        """

        if 2080 in self.unique_event_codes:



            movie = self.user_sample[self.user_sample.event_code == 2080]



            movie_duration = movie["duration"].values

            

            features_2080 = FeaturesGeneration(movie_duration, "_2080")

            self.Event_features.update(features_2080.copy())



    def event_code_2081(self):

        """

        ['The movie started event triggers when an intro or outro movie starts to play. 

        It identifies the movie being played. This is used to determine how long players 

        spend watching the movies (more relevant after the first play 

        through when the skip option is available).']

        """

        if 2081 in self.unique_event_codes:



            movie = self.user_sample[self.user_sample.event_code == 2081]



            self.Event_features["accu_movie_skiping_count_2081"] = movie["duration"].count()

    

    def event_code_2083(self):

        """

        ['The movie started event triggers when an intro or outro movie starts to play. 

        It identifies the movie being played. This is used to determine how long players 

        spend watching the movies (more relevant after the first play 

        through when the skip option is available).']

        """

        if 2083 in self.unique_event_codes:



            movie = self.user_sample[self.user_sample.event_code == 2083]



            movie_duration = movie["duration"].values

            features_2083 = FeaturesGeneration(movie_duration, "_2083")

            self.Event_features.update(features_2083.copy())

    

    def event_code_3010(self):

        """

        ['The system-initiated instruction event occurs when the game delivers instructions to the player.

         It contains information that describes the content of the instruction. This event differs from events 3020

          and 3021 as it captures instructions that are not given in response to player action. 

          These events are used to determine the effectiveness of the instructions. We can answer questions like,

         "did players who received instruction X do better than those who did not?"']

        """

        if 3010 in self.unique_event_codes:



            instruction = self.user_sample[self.user_sample.event_code == 3010]



            instruction_duration = instruction["total_duration"].values

            

            features_3010 = FeaturesGeneration(instruction_duration, "_3010")

            self.Event_features.update(features_3010.copy())



    def event_code_3020(self):

        """

        ['The system-initiated feedback (Incorrect) event occurs when the game starts delivering feedback 

        to the player in response to an incorrect round attempt (pressing the go button with the incorrect answer). 

        It contains information that describes the content of the instruction. These events are used to determine 

        the effectiveness of the feedback. We can answer questions like 

        "did players who received feedback X do better than those who did not?"']

        """

        if 3020 in self.unique_event_codes:



            Incorrect = self.user_sample[self.user_sample.event_code == 3020]



            Incorrect_duration = Incorrect["total_duration"].values

            

            features_3020 = FeaturesGeneration(Incorrect_duration, "_3020")

            self.Event_features.update(features_3020.copy())

            

            self.Event_features["accu_Incorrect_media_type_3020_count_"] = Incorrect["media_type"].count()

    



    def event_code_3021(self):

        """

        ['The system-initiated feedback (Correct) event occurs when the game 

        starts delivering feedback to the player in response to a correct round attempt 

        (pressing the go button with the correct answer). It contains information that describes the

         content of the instruction, and will likely occur in conjunction with a beat round event. 

         These events are used to determine the effectiveness of the feedback. We can answer questions like, 

        "did players who received feedback X do better than those who did not?"']

        """

        if 3021 in self.unique_event_codes:



            Correct = self.user_sample[self.user_sample.event_code == 3021]



            Correct_duration = Correct["total_duration"].values

            features_3021 = FeaturesGeneration(Correct_duration, "_3021")

            self.Event_features.update(features_3021.copy())

            

            self.Event_features["accu_Correct_media_type_3021_count_"] = Correct["media_type"].count()



    def event_code_3110(self):

        """

        ['The end of system-initiated instruction event occurs when the game finishes 

        delivering instructions to the player. It contains information that describes the

         content of the instruction including duration. These events are used to determine the 

         effectiveness of the instructions and the amount of time they consume. We can answer questions like, 

        "how much time elapsed while the game was presenting instruction?"']

        """

        if 3110 in self.unique_event_codes:



            Instuction = self.user_sample[self.user_sample.event_code == 3110]



            Instuction_duration = Instuction["duration"].values

            features_3110 = FeaturesGeneration(Instuction_duration, "_3110")

            self.Event_features.update(features_3110.copy())

            

            self.Event_features["accu_Instuction_media_type_3110_count_"] = Instuction["media_type"].count()



    def event_code_3120(self):

        """

        ['The end of system-initiated feedback (Incorrect) event 

        occurs when the game finishes delivering feedback to the player in response

         to an incorrect round attempt (pressing the go button with the incorrect answer). 

         It contains information that describes the content of the instruction. 

         These events are used to determine the effectiveness of the feedback. We can answer questions like,

         “how much time elapsed while the game was presenting feedback?”']

        """

        if 3120 in self.unique_event_codes:



            IncorrectInstruction = self.user_sample[self.user_sample.event_code == 3120]



            IncorrectInstruction_duration = IncorrectInstruction["duration"].values

            

            features_3120 = FeaturesGeneration(IncorrectInstruction_duration, "_3120")

            self.Event_features.update(features_3120.copy())

            

            self.Event_features["accu_IncorrectInstruction_media_type_3120_count_"] = IncorrectInstruction["media_type"].count()



    def event_code_3121(self):

        """

        ['The end of system-initiated feedback (Correct) event 

        occurs when the game finishes delivering feedback to the player in response

         to an incorrect round attempt (pressing the go button with the incorrect answer). 

         It contains information that describes the content of the instruction. 

         These events are used to determine the effectiveness of the feedback. We can answer questions like,

         “how much time elapsed while the game was presenting feedback?”']

        """

        if 3121 in self.unique_event_codes:



            CorrectInstruction = self.user_sample[self.user_sample.event_code == 3121]



            CorrectInstruction_duration = CorrectInstruction["duration"].values



            features_3121 = FeaturesGeneration(CorrectInstruction_duration, "_3121")

            self.Event_features.update(features_3121.copy())

            

            self.Event_features["accu_CorrectInstruction_media_type_3121_count_"] = CorrectInstruction["media_type"].count()





    def event_code_4010(self):

        """



        ['This event occurs when the player clicks to start 

        the game from the starting screen.']

        

        """



        pass

    

    def event_code_4020(self):

        """

        ['This event occurs when the player 

        clicks a group of objects. It contains information 

        about the group clicked, the state of the game, and the

         correctness of the action. This event is 

         to diagnose player strategies and understanding.']



         It contains information about the state of the game and the correctness of the action. This event is used 

         to diagnose player strategies and understanding.

        """

        

        if 4020 in self.unique_event_codes:



            event_data = self.user_sample[self.user_sample.event_code == 4020]



            true_attempts = event_data[event_data.correct == True]['correct'].count()

            false_attempts = event_data[event_data.correct == False]['correct'].count()

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



            self.Event_features["accu_True_attempts_4020_"] = true_attempts

            self.Event_features["accu_False_attempts_4020_"] = false_attempts

            self.Event_features["accu_Accuracy_attempts_4020_"] = accuracy



    def event_code_4021(self):

        pass

    

    def event_code_4022(self):

        pass



    def event_code_4025(self):

        pass



    def event_code_4030(self):

        pass



    def event_code_4031(self):

        pass



    def event_code_4035(self):



        if 4035 in self.unique_event_codes:



            event_data = self.user_sample[self.user_sample.event_code == 4035]



            self.Event_features["accu_wrong_place_count_4035_"] = len(event_data)



            try:

                wrong_place = event_data["duration"].values



                features_4035 = FeaturesGeneration(wrong_place, "_4035")

                self.Event_features.update(features_4035.copy())

            except:

                pass



    

    def event_code_4040(self):

        pass



    def event_code_4045(self):

        pass



    def event_code_4050(self):

        pass



    def event_code_4070(self):

        """

        

        ['This event occurs when the player clicks on

            something that isn’t covered elsewhere. 

            It can be useful in determining if there are

            attractive distractions (things the player think

            should do something, but don’t) in the game, or

            diagnosing players 

            who are having mechanical difficulties (near misses).']

        """

        if 4070 in self.unique_event_codes:



            event_data = self.user_sample[self.user_sample.event_code == 4070]

            self.Event_features["accu_something_not_covered_count_4070_"] = len(event_data)



    def event_code_4080(self):



        if 4080 in self.unique_event_codes:



            event_data = self.user_sample[self.user_sample.event_code == 4080]



            self.Event_features["accu_mouse_over_count_4080_"] = len(event_data)



            try:



                dwell_time = event_data["dwell_time"].values

                

                features_4080 = FeaturesGeneration(dwell_time, "_4080")

                self.Event_features.update(features_4080.copy())

            

            except:

                pass



    def event_code_4090(self):

        pass



    def event_code_4095(self):

        pass

        

    def event_code_4100(self):

        

        if 4100 in self.unique_event_codes:



            event_data = self.user_sample[self.user_sample.event_code == 4100]



            true_attempts = event_data[event_data.correct == True]['correct'].count()

            false_attempts = event_data[event_data.correct == False]['correct'].count()

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



            self.Event_features["accu_True_attempts_4100_"] = true_attempts

            self.Event_features["accu_False_attempts_4100_"] = false_attempts

            self.Event_features["accu_Accuracy_attempts_4100_"] = accuracy



    def event_code_4110(self):

        



        if 4110 in self.unique_event_codes:



            event_data = self.user_sample[self.user_sample.event_code == 4110]



            true_attempts = event_data[event_data.correct == True]['correct'].count()

            false_attempts = event_data[event_data.correct == False]['correct'].count()

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0



            self.Event_features["accu_True_attempts_4110_"+self.user_sample_title] = true_attempts

            self.Event_features["accu_False_attempts_4110_"+self.user_sample_title] = false_attempts

            self.Event_features["accu_Accuracy_attempts_4110_"+self.user_sample_title] = accuracy

            



    def event_code_4220(self):

        pass



    def event_code_4230(self):

        pass



    def event_code_4235(self):

        pass



    def event_code_5000(self):

        pass



    def event_code_5010(self):

        pass
def get_train_test(train, test, unique_data):

    compiled_train = []

    compiled_test = []

    

    

    if os.path.exists("../input/amma-reduce/amma_train.csv"):

        reduce_train_file = True

        reduce_train = pd.read_csv("../input/amma-reduce/amma_train.csv")

    else:

        for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby("installation_id", sort=False)), total=len(train.installation_id.unique())):



            if "Assessment" in user_sample.type.unique():

                temp_df = json_parser(user_sample, "event_data")

                temp_df.sort_values("timestamp", inplace=True)

                temp_df.reset_index(inplace=True, drop=True)

                temp_df["index"] = temp_df.index.values

                compiled_train.extend(get_data(temp_df, unique_data))



        reduce_train = pd.DataFrame(compiled_train)



    for i, (ins_id, user_sample) in tqdm(enumerate(test.groupby("installation_id", sort=False)), total=len(test.installation_id.unique())):



        if "Assessment" in user_sample.type.unique():

            temp_df = json_parser(user_sample, "event_data")

            temp_df.sort_values("timestamp", inplace=True)

            temp_df.reset_index(inplace=True, drop=True)

            temp_df["index"] = temp_df.index.values

            compiled_test.append(get_data(temp_df,unique_data, test=True))



    reduce_test = pd.DataFrame(compiled_test)



    return reduce_train, reduce_test
#reduce_train.to_csv("amma_train.csv", index=False)

#reduce_test.to_csv("amma_test.csv", index=False)
reduce_train, reduce_test = get_train_test(train, test, unique_data)
reduce_train.shape , reduce_test.shape
reduce_train = reduce_train[reduce_train.game_session.isin(train_labels.game_session.unique())]

reduce_train.shape
reduce_train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_train.columns]

reduce_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in reduce_test.columns]
rem = list(set(reduce_train.columns).intersection(set(reduce_test)))
reduce_train = reduce_train[rem]

reduce_test = reduce_test[rem]
reduce_train.shape, reduce_test.shape
import numpy as np

import pandas as pd

from functools import partial

from sklearn import metrics

import scipy as sp



from sklearn.preprocessing import OneHotEncoder

from scipy.stats import boxcox, skew, randint, uniform

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.linear_model import Lasso, ElasticNet, Ridge, LinearRegression

from sklearn.kernel_ridge import KernelRidge

from sklearn.feature_selection import RFECV

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import mean_squared_error



import warnings

warnings.filterwarnings('ignore')
categorical_cols = []

for col in reduce_train.columns:

    if reduce_train[col].dtype == 'object':

        categorical_cols.append(col)
categorical_cols
def tree_based_models(train,test,columns_map):

    for col in columns_map:

        list_of_values = list(set(train[col].unique()).union(set(test[col].unique())))

        list_of_values_map = dict(zip(list_of_values, np.arange(len(list_of_values))))

        train[col] = train[col].map(list_of_values_map)

        test[col] = test[col].map(list_of_values_map)

    return train, test
def merge_with_labels(train, train_labels):

    train = train[train.game_session.isin(train_labels.game_session.unique())]

    tld = train_labels[['game_session', 'installation_id', 'num_correct','num_incorrect', 'accuracy', 'accuracy_group']]

    final_train = pd.merge(tld, train, left_on=['game_session', 'installation_id'], right_on=['game_session','installation_id'], how='inner')

    final_train.sort_values('timestamp_session', inplace=True)

    col_drop = tld.columns.values

    col_drop = np.append(col_drop, 'timestamp_session')

    return final_train, col_drop
final_train, col_drop = merge_with_labels(reduce_train, train_labels)
cat_cols = []

for col in categorical_cols:

    if col not in col_drop:

        cat_cols.append(col)
len(cat_cols)
cat_drop_com = cat_cols + col_drop.tolist()

numaric_cols = list(set(final_train.columns.values) - set(cat_drop_com))
final_train, final_test = tree_based_models(final_train, reduce_test, cat_cols)
final_train.shape , final_test.shape
def eval_qwk_lgb_regr2(y_true, y_pred, train):

    """

    Fast cappa eval function for lgb.

    """

    dist = Counter(train['accuracy_group'])

    for k in dist:

        dist[k] /= len(train)

    train['accuracy_group'].hist()

    

    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(y_pred, acum * 100)



    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)



    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True



def qwk(a1, a2):

    """

    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168



    :param a1:

    :param a2:

    :param max_rat:

    :return:

    """

    max_rat = 3

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e



def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    y_pred[y_pred <= 1.12232214] = 0

    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2

    y_pred[y_pred > 2.22506454] = 3



    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)



    return 'cappa', qwk(y_true, y_pred), True



from sklearn.metrics import confusion_matrix, accuracy_score



def confusion_matrix_reg(y_true, y_pred):

    y_pred[y_pred <= 1.12232214] = 0

    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2

    y_pred[y_pred > 2.22506454] = 3

    

    print("Accuracy : ", accuracy_score(y_true=y_true, y_pred=y_pred))

    print("Confussion_matrix \n", confusion_matrix(y_true, y_pred))

    print("\n\n")
col_drop
col_drop1 = col_drop
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            else:

                X_p[i] = 3



        ll = quadratic_weighted_kappa(y, X_p)

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [1.1, 1.7, 2.2]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            else:

                X_p[i] = 3

        return X_p



    def coefficients(self):

        return self.coef_['x']
# The following 3 functions have been taken from Ben Hamner's github repository

# https://github.com/benhamner/Metrics

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(rater_a + rater_b)

    if max_rating is None:

        max_rating = max(rater_a + rater_b)

    num_ratings = int(max_rating - min_rating + 1)

    conf_mat = [[0 for i in range(num_ratings)]

                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):

        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat





def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """

    if min_rating is None:

        min_rating = min(ratings)

    if max_rating is None:

        max_rating = max(ratings)

    num_ratings = int(max_rating - min_rating + 1)

    hist_ratings = [0 for x in range(num_ratings)]

    for r in ratings:

        hist_ratings[r - min_rating] += 1

    return hist_ratings





def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return (1.0 - numerator / denominator)
def qwk(a1, a2):

    """

    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168



    :param a1:

    :param a2:

    :param max_rat:

    :return:

    """

    max_rat = 3

    a1 = np.asarray(a1, dtype=int)

    a2 = np.asarray(a2, dtype=int)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e



def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    y_pred[y_pred <= 1.12232214] = 0

    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2

    y_pred[y_pred > 2.22506454] = 3



    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)



    return 'cappa', qwk(y_true, y_pred), True



from sklearn.metrics import confusion_matrix, accuracy_score



def confusion_matrix_reg(y_true, y_pred):

    y_pred[y_pred <= 1.12232214] = 0

    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1

    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2

    y_pred[y_pred > 2.22506454] = 3

    

    print("Accuracy : ", accuracy_score(y_true=y_true, y_pred=y_pred))

    print("Confussion_matrix \n", confusion_matrix(y_true, y_pred))

    print("\n\n")

def eval_qwk_lgb_regr2(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    dist = Counter(final_train['accuracy_group'])

    for k in dist:

        dist[k] /= len(final_train)

    final_train['accuracy_group'].hist()

    

    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(y_pred, acum * 100)



    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)



    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True



def cohenkappa(ypred, y):

    y = y.get_label().astype("int")

    ypred = ypred.reshape((4, -1)).argmax(axis = 0)

    loss = cohenkappascore(y, y_pred, weights = 'quadratic')

    return "cappa", loss, True

# The following 3 functions have been taken from Ben Hamner's github repository

# https://github.com/benhamner/Metrics

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(rater_a + rater_b)

    if max_rating is None:

        max_rating = max(rater_a + rater_b)

    num_ratings = int(max_rating - min_rating + 1)

    conf_mat = [[0 for i in range(num_ratings)]

                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):

        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat





def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """

    if min_rating is None:

        min_rating = min(ratings)

    if max_rating is None:

        max_rating = max(ratings)

    num_ratings = int(max_rating - min_rating + 1)

    hist_ratings = [0 for x in range(num_ratings)]

    for r in ratings:

        hist_ratings[r - min_rating] += 1

    return hist_ratings





def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return (1.0 - numerator / denominator)
from sklearn.model_selection import KFold, StratifiedKFold

train_predictions = np.zeros((len(final_train),1))

train_predictions_y = np.zeros((len(final_train),1))

test_predictions = np.zeros((final_test.shape[0], 1))

zero_test_predictions = np.zeros((final_test.shape[0], 1))



FOLDS = 5



print("stratified k-folds")



train = final_train

y = final_train.accuracy_group



skf = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)

skf.get_n_splits(train, y)

cv_scores = []

fold = 1

coefficients = np.zeros((FOLDS, 3))

for train_ids_indx, test_ids_indx in skf.split(train, y):

    #train_ids = unique_installation_ids[train_ids_indx]

    #test_ids = unique_installation_ids[test_ids_indx]

    #print(train_ids.shape, test_ids.shape)



    train_X = train.iloc[train_ids_indx]

    test_X = train.iloc[test_ids_indx]



    x_train = train_X.drop(columns=col_drop1)

    y_train = train_X["accuracy_group"]

    x_test = test_X.drop(columns=col_drop1)

    y_test = test_X["accuracy_group"]



    w = y_test.value_counts()

    weights = {i : np.sum(w) / w[i] for i in w.index}

    print(weights)



    lgb_params = {

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'learning_rate': 0.005,

        'subsample': .8,

        'colsample_bytree': 0.8,

        'min_split_gain': 0.006,

        'min_child_samples': 150,

        'min_child_weight': 0.1,

        'max_depth': 17,

        'n_estimators': 10000,

        'num_leaves': 80,

        'silent': -1,

        'verbose': -1,

        'max_depth': 15,

        'random_state': 2018

    }



    model = lgb.LGBMRegressor(**lgb_params)

    model.fit(

        x_train, y_train,

        eval_set=[(x_test, y_test)],

        eval_metric='rmse',

        verbose=100,

        early_stopping_rounds=150

    )



    valid_preds = model.predict(x_test, num_iteration=model.best_iteration_)

    

    train_predictions[test_ids_indx] = valid_preds.reshape(-1,1)

    train_predictions_y[test_ids_indx] = y_test.values.reshape(-1,1)

    

    optR = OptimizedRounder()

    optR.fit(valid_preds, y_test.values)

    coefficients[fold-1, :] = optR.coefficients()

    print("Coefficients : ", optR.coefficients())

    valid_p = optR.predict(valid_preds, coefficients[fold-1, :])

    valid_preds1 = valid_preds.copy()

    print("non optimized qwk : ", eval_qwk_lgb_regr(y_test, valid_preds1))

    print("optimized qwk : ", qwk(y_test, valid_p))

    print("Valid Counts = ", Counter(y_test.values))

    print("Predicted Counts = ", Counter(valid_p))



    test_preds = model.predict(final_test[x_train.columns], num_iteration=model.best_iteration_)



    scr = quadratic_weighted_kappa(y_test.values, valid_p)



    cv_scores.append(scr)

    print("Fold = {}. QWK = {}. Coef = {}".format(fold, scr, coefficients[fold-1, :]))

    print("\n")



    

    test_predictions += test_preds.reshape(-1,1)

    fold += 1



test_predictions = test_predictions * 1./FOLDS

print("Mean Score: {}. Std Dev: {}. Mean Coeff: {}".format(np.mean(cv_scores), np.std(cv_scores), np.mean(coefficients, axis=0)))

optR = OptimizedRounder()

train_predictions1 = np.array([item for sublist in train_predictions for item in sublist])

y1 = np.array([item for sublist in train_predictions_y for item in sublist])

optR.fit(train_predictions1, y1)

coefficients = optR.coefficients()

print(quadratic_weighted_kappa(y1, optR.predict(train_predictions1, coefficients)))



predictions = optR.predict(test_predictions, coefficients).astype(int)

predictions = [item for sublist in predictions for item in sublist]
n = pd.Series(optR.predict(train_predictions1, coefficients))
n.value_counts(normalize=True)
sample_submission = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")

sample_submission["accuracy_group"] = predictions

sample_submission.to_csv('submission.csv', index=False)
sample_submission['accuracy_group'].value_counts(normalize=True)