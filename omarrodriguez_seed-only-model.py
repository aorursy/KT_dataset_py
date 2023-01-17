from pathlib import Path

import os

import datetime

import numpy as np

import pandas as pd



from lightgbm import LGBMClassifier

from sklearn.metrics import log_loss
PATH_DATA = Path("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/")

path_sample = PATH_DATA / "MSampleSubmissionStage1_2020.csv"

path_seeds = PATH_DATA / "MDataFiles_Stage1" / "MNCAATourneySeeds.csv"

path_regular = PATH_DATA / "MDataFiles_Stage1" / "MRegularSeasonCompactResults.csv"

path_tourney = PATH_DATA / "MDataFiles_Stage1" / "MNCAATourneyCompactResults.csv"
def read_seeds():

    seeds = pd.read_csv(path_seeds)

    seeds["SeedValue"] = seeds.Seed.apply(lambda x: float("".join(c for c in x if c.isnumeric())) + (0.5 if "b" in x else 0))

    seeds = seeds[["Season", "TeamID", "SeedValue"]]

    return seeds
def read_regular(seeds, test_season=2019):

    regular = pd.read_csv(path_regular)

    regular = (regular

        .merge(seeds, left_on=["WTeamID", "Season"], right_on=["TeamID", "Season"], how="inner")

        .rename(columns={"SeedValue": "WSeed"})

        .drop(columns="TeamID")

        .merge(seeds, left_on=["LTeamID", "Season"], right_on=["TeamID", "Season"], how="inner")

        .rename(columns={"SeedValue": "LSeed"})

        .drop(columns="TeamID")

        .assign(

            WLoc=regular.WLoc.map({"H": 1, "N": 0, "A": -1}),

            LLoc=regular.WLoc.map({"H": -1, "N": 0, "A": 1}),

        )

        .loc[regular.Season <= test_season]

    )

    return regular
def read_tourney(seeds, test_season=2019):

    tourney = pd.read_csv(path_tourney)

    tourney = (tourney

        .merge(seeds, left_on=["WTeamID", "Season"], right_on=["TeamID", "Season"], how="left")

        .rename(columns={"SeedValue": "WSeed"})

        .drop(columns="TeamID")

        .merge(seeds, left_on=["LTeamID", "Season"], right_on=["TeamID", "Season"], how="left")

        .rename(columns={"SeedValue": "LSeed"})

        .drop(columns="TeamID")

        .assign(WLoc=0, LLoc=0)

        .loc[tourney.Season < test_season]

    )

    return tourney
def stack_WL_columns(dataframe, include_target=True):

    wcols = dataframe.columns[dataframe.columns.str.startswith("W")]

    lcols = dataframe.columns[dataframe.columns.str.startswith("L")]

    

    renaming_mapping = {c: c[1:]+"1" for c in wcols}

    renaming_mapping.update({c: c[1:]+"2" for c in lcols})

    f1 = dataframe.rename(columns=renaming_mapping)

    

    renaming_mapping = {c: c[1:]+"1" for c in lcols}

    renaming_mapping.update({c: c[1:]+"2" for c in wcols})

    f2 = dataframe.rename(columns=renaming_mapping)

    

    if include_target:

        f1 = f1.assign(y=1)

        f2 = f2.assign(y=0)

    

    return pd.concat([f1, f2]).sample(frac=1).reset_index(drop=True)
def read_sample_submission(seeds):

    sample = pd.read_csv(path_sample)

    sample[["Season", "TeamID1", "TeamID2"]] = sample.ID.str.split("_", expand=True).astype(int)

    sample = (sample

        .merge(seeds, left_on=["TeamID1", "Season"], right_on=["TeamID", "Season"], how="left")

        .rename(columns={"SeedValue": "Seed1"})

        .drop(columns="TeamID")

        .merge(seeds, left_on=["TeamID2", "Season"], right_on=["TeamID", "Season"], how="left")

        .rename(columns={"SeedValue": "Seed2"})

        .drop(columns="TeamID")

    )

    return sample
def submit_to_kaggle(submission_dataframe, message="easy"):

    submission_dataframe = submission_dataframe[["ID", "Pred"]]

    assert all(submission_dataframe.isnull().sum() == 0)

    timestamp = datetime.datetime.now().strftime("D%Y%m%dT%H%M%S")

    command = (

        "kaggle competitions submit "

        "-c google-cloud-ncaa-march-madness-2020-division-1-mens-tournament "

        f"-f submission_{timestamp}.csv "

        f"-m {message}"

    )

    submission_dataframe.to_csv(f"submission_{timestamp}.csv", index=False)

    os.system(command)

    print(f"File submission_{timestamp}.csv has save to the output folder!")
selected_features = ["Seed1", "Seed2"]
seeds = read_seeds()
submission_file = read_sample_submission(seeds)
lgb = LGBMClassifier()



for season in range(2015, 2020):

    X_train = pd.concat(

        [

            stack_WL_columns(read_regular(seeds, test_season=season)),

            stack_WL_columns(read_tourney(seeds, test_season=season)),

        ],

        axis=0,

        ignore_index=True,

    )



    y_train = X_train.pop("y")

    

    lgb.fit(X_train[selected_features], y_train)

    

    # Predict:

    X_test = submission_file.loc[submission_file.Season == season, selected_features]

    submission_file.loc[submission_file.Season == season, "Pred"] = lgb.predict_proba(X_test)[:, 1]
submit_to_kaggle(submission_file, message="No time for fancy model.")