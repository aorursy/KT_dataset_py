import pandas as pd

from tqdm import tqdm
# load in our data

tracking_df = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")

injury_df = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

play_df = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")
players = play_df.PlayerKey.unique()
tracking_df.head()
play_df.head()
cat_names = ['StadiumType', 'FieldType', 'Weather', 'PlayType', 'PositionGroup', 'RosterPosition']



model_data_df = play_df[cat_names + ['PlayerKey', 'PlayKey']]

model_data_df.head()
# convert all categories to integers and normalize such that the minimum value is 0

from fastai.tabular.transform import Categorify



tfm = Categorify(cat_names, [])

tfm(model_data_df)

for cat_name in cat_names:

    codes = model_data_df[cat_name].cat.codes

    model_data_df[cat_name] = codes - min(codes)

model_data_df.head()
def get_plays_for_player(player):

    """

    Returns data for all of the plays associated with the player

    """

    return model_data_df[model_data_df.PlayerKey == player].drop(['PlayerKey'], axis=1)
import os

PLAYER_DATA_CACHE_PATH = "../input/tracking-cache/player_data_tracking"



# makes sure the local directory for the cache exists

if not os.path.exists(PLAYER_DATA_CACHE_PATH):

    os.makedirs(PLAYER_DATA_CACHE_PATH)



def get_tracking_for_player(player):

    """

    Returns tracking data stored in a local json file. If the file

    doesn't exist the function creates it

    """

    file_cache_path = f"{PLAYER_DATA_CACHE_PATH}/{player}.csv"

    if not os.path.exists(file_cache_path):

        small_tracking = tracking_df[tracking_df.PlayKey.str.startswith(str(player))]

        small_tracking.to_csv(file_cache_path)

    return pd.read_csv(file_cache_path)
get_tracking_for_player(players[0]).head()
def get_data_for_player(player):

    """

    Combines all of the data associated with the plays for an individual player

    """

    plays = get_plays_for_player(player)

    tracking = get_tracking_for_player(player)

    

    group = tracking.groupby("PlayKey")

    avg_speed = group.mean()['s'] * 100

    total_distance = group.sum()['dis']

    

    data = plays.merge(avg_speed, on="PlayKey").merge(total_distance, on="PlayKey").drop('PlayKey', axis=1)

    data['id'] = player

    data['time'] = data.index

    

    return data



get_data_for_player(players[1])
fresh_data = pd.concat([get_data_for_player(player) for player in tqdm(players)])

fresh_data
from tsfresh import extract_relevant_features

from tsfresh.feature_extraction import MinimalFCParameters
injured_players = set(injury_df.PlayerKey)

y = pd.DataFrame(index=players)

y['target'] = [int(player in injured_players) for player in players]

y.head()
extracted_features = extract_relevant_features(

    fresh_data,

    y.target,

    column_id="id",

    column_sort="time",

    default_fc_parameters=MinimalFCParameters()

)

extracted_features['target'] = y.target

extracted_features.head()
from fastai.tabular import *

from fastai.callbacks import SaveModelCallback

from sklearn.model_selection import train_test_split
_, valid = train_test_split(range(len(players)))

data = TabularDataBunch.from_df("./models", extracted_features, 'target', valid_idx=valid)
# this choice of layer sizes comes from experimental success in similar tabular deep learning problems

LAYERS = [200, 100]



def create_learner():

    """

    We define this as a function as it allows us to later load models from disk easily

    later in the notebook

    """

    return tabular_learner(data, layers=LAYERS, metrics=accuracy)
learn = create_learner()
learn.lr_find()

learn.recorder.plot()
lr = 1e-2

N_CYCLES = 10
# clear the existing model cache

!rm -rf ./models/models
# train our model and save the best result

learn.fit_one_cycle(

    N_CYCLES,

    lr,

    callbacks=[SaveModelCallback(learn, every="improvement", monitor="accuracy")]

)
best_model = create_learner()

best_model.load("bestmodel")
interpretation = best_model.interpret()

interpretation.plot_confusion_matrix()