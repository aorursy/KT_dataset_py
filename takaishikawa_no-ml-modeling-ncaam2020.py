import os

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.metrics import log_loss

from tqdm import tqdm

import warnings
config = {

    "mode": {

        "stage": 1

    },

    "const": {

        "score_diff": 5,

        "this_season": 2020,

        "total_season": 10,

        "seed_num": 16,

        "clip_min": 0.01,

        "clip_max": 0.99,

    },

    "path": {

        "prefix": "/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament",

        "stage1_prefix": f"/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1",

    },

    "plot": {

        "palette": "viridis_r"

    }

}
%matplotlib inline

sns.set()

warnings.filterwarnings("ignore")



path_prefix = config["path"]["prefix"]

stage1_prefix = config["path"]["stage1_prefix"]

print(os.listdir(f"{path_prefix}"))

print(os.listdir(f"{stage1_prefix}"))
display(pd.read_csv(f"{path_prefix}/MSampleSubmissionStage1_2020.csv").shape)

display(pd.read_csv(f"{path_prefix}/MSampleSubmissionStage1_2020.csv").head())

display(pd.read_csv(f"{path_prefix}/MSampleSubmissionStage1_2020.csv").tail())
def load_data():

    df_seed = pd.read_csv(os.path.join(stage1_prefix, "MNCAATourneySeeds.csv"))

    df_result = pd.read_csv(

        os.path.join(stage1_prefix, "MNCAATourneyCompactResults.csv")

    )

    return df_seed, df_result





def _seed_to_int(seed):

    s_int = int(seed[1:3])

    return s_int





def clean_df(df_seed, df_result):

    df_seed["seed_int"] = df_seed["Seed"].apply(_seed_to_int)

    df_seed.drop(["Seed"], axis=1, inplace=True)

    df_result.drop(["DayNum", "WLoc", "NumOT"], axis=1, inplace=True)

    return df_seed, df_result





# Merge seed for each team

def merge_seed_result(df_seed, df_result):

    df_win_seed = df_seed.rename(columns={"TeamID": "WTeamID", "seed_int": "WSeed"})

    df_loss_seed = df_seed.rename(columns={"TeamID": "LTeamID", "seed_int": "LSeed"})

    df_result = df_result.merge(df_win_seed, how="left", on=["Season", "WTeamID"])

    df_result = df_result.merge(df_loss_seed, how="left", on=["Season", "LTeamID"])

    df_result["SeedDiff"] = np.abs(df_result["WSeed"] - df_result["LSeed"])

    df_result["ScoreDiff"] = np.abs(df_result["WScore"] - df_result["LScore"])

    return df_result
df_seed, df_result = load_data()

df_seed, df_result = clean_df(df_seed, df_result)

df_result = merge_seed_result(df_seed, df_result)

df_result["upset"] = [

    1 if ws > ls else 0 for ws, ls, in zip(df_result["WSeed"], df_result["LSeed"])

]



# Remove the games that end within 3 points difference, which are likely to be the other results

df_result = df_result[df_result["ScoreDiff"] > config["const"]["score_diff"]]

df_result.head()
def check_target(df_result):

    upset_proba = df_result['upset'].value_counts() / len(df_result) * 100

    print(f"upset probability:\n{upset_proba}")



check_target(df_result)
# Use only last 10 seasons, since some trends are likely to be changed

this_season = 2015

total_season = config["const"]["total_season"]

seed_num = config["const"]["seed_num"]
"""

The probability of the occurrence of the upset is likely to be different between

a game 1st seed vs. 6th seed and a game 11th seed vs. 16th seed, 

so I want to include the information

"""

def aggregation(df_result, plot=True):

    df_result["Seed_combi"] = [

        str(ws) + "_" + str(ls) if ws < ls else str(ls) + "_" + str(ws)

        for ws, ls in zip(df_result["WSeed"], df_result["LSeed"])

    ]



    df_result_aggs = pd.DataFrame()

    df_result_filter_aggs = pd.DataFrame()

    df_result_season = df_result[

        (df_result["Season"] >= (this_season - total_season))

        & (df_result["Season"] < (this_season - 1))

    ]

    for s_num in range(seed_num):

        df_result_agg = (

            df_result_season[df_result_season["SeedDiff"] == s_num]

            .groupby("SeedDiff")

            .agg({"upset": ["mean", "count"]})

        )

        df_result_agg.columns = [

            f"{col[0]}_{col[1]}_all" for col in df_result_agg.columns

        ]

        df_result_filter_agg = (

            df_result_season[df_result_season["SeedDiff"] == s_num]

            .groupby("Seed_combi")

            .agg({"upset": ["mean", "count"]})

        )

        df_result_filter_agg.columns = [

            f"{col[0]}_{col[1]}" for col in df_result_filter_agg.columns

        ]

        if s_num == 0:

            df_result_agg["upset_mean_all"] = 0.5

            df_result_filter_agg["upset_mean"] = 0.5

        df_result_aggs = pd.concat([df_result_aggs, df_result_agg])

        df_result_filter_aggs = pd.concat([df_result_filter_aggs, df_result_filter_agg])



    if plot:

        sns.barplot(df_result_aggs.index, df_result_aggs.upset_mean_all, palette=config["plot"]["palette"])

        plt.title("probability of upset based on past result aggretation")

        plt.tight_layout()

        plt.show()



    return df_result_aggs, df_result_filter_aggs





# Merge upset probability

def merge(df_result, df_result_aggs, df_result_filter_aggs):

    df_result = df_result.join(df_result_aggs, how="left", on="SeedDiff").join(

        df_result_filter_aggs, how="left", on="Seed_combi"

    )

    df_result["upset_prob"] = [

        m if c > 20 else a

        for a, m, c in zip(

            df_result["upset_mean_all"],

            df_result["upset_mean"],

            df_result["upset_count"],

        )

    ]

    valid = df_result[(df_result["Season"] == (this_season - 1))]

    return valid





# heuristic smoothing

def smoothing(df_result_aggs, plot=True):

    for i in range(config["const"]["seed_num"]):

        if i == 0:

            df_result_aggs.loc[i, "upset_mean_all"] = 0.5

        else:

            try:

                df_result_aggs.loc[i, "upset_mean_all"]

                if df_result_aggs.loc[i, "upset_mean_all"] == 0:

                    raise Exception

                elif df_result_aggs.loc[i, "upset_mean_all"] > 0.5:

                    df_result_aggs.loc[i, "upset_mean_all"] = 0.5

            except Exception:

                df_result_aggs.loc[i, "upset_mean_all"] = (

                    df_result_aggs.loc[(i - 1), "upset_mean_all"] / 4

                    + df_result_aggs.loc[(i - 2), "upset_mean_all"] / 4

                )



    if plot:

        sns.barplot(df_result_aggs.index, df_result_aggs.upset_mean_all, palette=config["plot"]["palette"])

        plt.title("probability of upset based on past result aggretation")

        plt.tight_layout()

        plt.show()



    return df_result_aggs





def merge_smooting(df_result, df_result_aggs_smooth, df_result_filter_aggs):

    df_result = df_result.join(df_result_aggs_smooth, how="left", on="SeedDiff").join(

        df_result_filter_aggs, how="left", on="Seed_combi"

    )

    df_result["upset_prob"] = [

        m if c > 20 else a

        for a, m, c in zip(

            df_result["upset_mean_all"],

            df_result["upset_mean"],

            df_result["upset_count"],

        )

    ]



    valid = df_result[(df_result["Season"] == (this_season - 1))]

    return valid





def clipping(array, a_min=config["const"]["clip_min"], a_max=config["const"]["clip_max"]):

    return np.clip(array, a_min, a_max)





def scoring(valid, clip=False):

    if clip:

        return log_loss(valid["upset"], clipping(valid["upset_prob"]))

    else:

        return log_loss(valid["upset"], valid["upset_prob"])
# df_result_aggs, df_result_filter_aggs = aggregation(df_result, plot=True)

# valid = merge(df_result, df_result_aggs, df_result_filter_aggs)

# print(scoring(valid))
df_result_aggs, df_result_filter_aggs = aggregation(df_result, plot=True)

df_result_aggs_smooth = smoothing(df_result_aggs, plot=True)

valid = merge_smooting(df_result, df_result_aggs_smooth, df_result_filter_aggs)

print(scoring(valid))
df_result_aggs, df_result_filter_aggs = aggregation(df_result, plot=True)

df_result_aggs_smooth = smoothing(df_result_aggs, plot=True)

valid = merge_smooting(df_result, df_result_aggs_smooth, df_result_filter_aggs)

print(scoring(valid, clip=True))
df_seed_test = df_seed[df_seed["Season"]==this_season]

df_result_aggs, df_result_filter_aggs = aggregation(df_result, plot=True)

df_result_aggs_smooth = smoothing(df_result_aggs)
def load_test(df_seed_this_season, df_result_aggs, df_result_filter_aggs):

    test = pd.read_csv(os.path.join(path_prefix, "MSampleSubmissionStage1_2020.csv"))

    test = pd.DataFrame(

        np.array([ID.split("_") for ID in test["ID"]]),

        columns=["Season", "TeamA", "TeamB"],

        dtype=int,

    )



    test = test.merge(

        df_seed_this_season,

        how="left",

        left_on=["Season", "TeamA"],

        right_on=["Season", "TeamID"],

    )

    test = test.rename(columns={"seed_int": "TeamA_seed"}).drop("TeamID", axis=1)



    test = test.merge(

        df_seed_this_season,

        how="left",

        left_on=["Season", "TeamB"],

        right_on=["Season", "TeamID"],

    )

    test = test.rename(columns={"seed_int": "TeamB_seed"}).drop("TeamID", axis=1)



    test["SeedDiff"] = np.abs(test.TeamA_seed - test.TeamB_seed)

    test["Seed_combi"] = [

        str(a) + "_" + str(b) if a < b else str(b) + "_" + str(a)

        for a, b in zip(test["TeamA_seed"], test["TeamB_seed"])

    ]



    test = (

        test.join(df_result_aggs, how="left", on="SeedDiff")

        .join(df_result_filter_aggs, how="left", on="Seed_combi")

        .fillna(-1)

    )

    test["upset_prob"] = [

        m if c > 20 else a

        for a, m, c in zip(

            test["upset_mean_all"], test["upset_mean"], test["upset_count"]

        )

    ]



    # convert upset_prob to win_prob

    test["win_prob"] = [

        (1 - upset_prob) if teamA < teamB else upset_prob if teamA > teamB else 0.5

        for teamA, teamB, upset_prob in zip(

            test["TeamA_seed"], test["TeamB_seed"], test["upset_prob"]

        )

    ]



    return test





def make_submit(test, clip):

    if clip:

        sub = clipping(test["win_prob"].values)

    else:

        sub = test["win_prob"].values

    submit = pd.read_csv(os.path.join(path_prefix, "MSampleSubmissionStage1_2020.csv"))

    submit["Pred"] = sub

    start_index = int((len(submit) / 5) * (this_season - 2015))

    end_index = int((len(submit) / 5) * (this_season - 2015 + 1))

    submit_this_season = submit.iloc[start_index:end_index, :]

    return submit_this_season
test = load_test(df_seed_test, df_result_aggs_smooth, df_result_filter_aggs)

make_submit(test, clip=False)

make_submit(test, clip=True)
seasons = [2015, 2016, 2017, 2018, 2019]

scores = []

scores_clip = []

submits = []

submits_clip = []

for this_season in tqdm(seasons):

    df_seed, df_result = load_data()

    df_seed, df_result = clean_df(df_seed, df_result)

    df_result = merge_seed_result(df_seed, df_result)

    df_result['upset'] = [1 if ws > ls else 0 for ws, ls, in zip(df_result["WSeed"], df_result["LSeed"])]

    df_result = df_result[df_result['ScoreDiff'] > config["const"]["score_diff"]]



    df_result_aggs, df_result_filter_aggs = aggregation(df_result, plot=False)

    df_result_aggs_smooth = smoothing(df_result_aggs, plot=False)

    valid = merge_smooting(df_result, df_result_aggs_smooth, df_result_filter_aggs)

    score = scoring(valid, clip=False)

    scores.append(score)

    print(f"{this_season}: {score} without clipping")

    score_clip = scoring(valid, clip=True)

    scores_clip.append(score_clip)

    print(f"{this_season}: {score_clip} with clipping")

    

    df_seed_test = df_seed[df_seed["Season"]==this_season]

    df_result_aggs, df_result_filter_aggs = aggregation(df_result, plot=False)

    df_result_aggs_smooth = smoothing(df_result_aggs, plot=False)

    test = load_test(df_seed_test, df_result_aggs_smooth, df_result_filter_aggs)

    submit_this_season = make_submit(test, clip=False)

    submits.append(submit_this_season)

    submit_this_season = make_submit(test, clip=True)

    submits_clip.append(submit_this_season)



print(f"cv all without clipping: {round(np.mean(scores), 6)}")

print(f"cv all with clipping: {round(np.mean(scores_clip), 6)}")



submit = pd.concat(submits, axis=0)

filename = "submission_agg_all_manually_nocliped.csv"

submit.to_csv(filename, index=False)



submit_clip = pd.concat(submits_clip, axis=0)

filename = "submission_agg_all_manually_cliped.csv"

submit_clip.to_csv(filename, index=False)