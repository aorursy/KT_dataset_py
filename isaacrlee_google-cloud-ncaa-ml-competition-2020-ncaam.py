# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



!pip install fastai2



# from fastai.tabular import *

from fastai2.tabular.all import *

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Seed Feature

seeds = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv")



seeds = seeds.assign(

    SeedNum = lambda df: df.apply(lambda row: int(row.Seed[1:3]), axis = 1)

)



seeds.head()
# Team Stats Feature

reg_season_results = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv")



w_team_stats = (

    reg_season_results.groupby(["Season", "WTeamID"])

    .agg(

        W=("WTeamID","count"),

        NumOT=("NumOT", "sum"),

        Score=("WScore", "sum"),

        FGM=("WFGM", "sum"),

        FGA=("WFGA", "sum"),

        FGM3=("WFGM3", "sum"),

        FGA3=("WFGA3", "sum"),

        FTM=("WFTM", "sum"),

        FTA=("WFTA", "sum"),

        OR=("WOR", "sum"),

        DR=("WDR", "sum"),

        Ast=("WAst", "sum"),

        TO=("WTO", "sum"),

        Stl=("WStl", "sum"),

        Blk=("WBlk", "sum"),

        PF=("WPF", "sum"),

        OppScore=("LScore", "sum"),

        OppFGM=("LFGM", "sum"),

        OppFGA=("LFGA", "sum"),

        OppFGM3=("LFGM3", "sum"),

        OppFGA3=("LFGA3", "sum"),

        OppFTM=("LFTM", "sum"),

        OppFTA=("LFTA", "sum"),

        OppOR=("LOR", "sum"),

        OppDR=("LDR", "sum"),

        OppAst=("LAst", "sum"),

        OppTO=("LTO", "sum"),

        OppStl=("LStl", "sum"),

        OppBlk=("LBlk", "sum"),

        OppPF=("LPF", "sum"),

    )

    .reset_index()

    .rename(columns={"WTeamID": "TeamID"})

)



l_team_stats = (

    reg_season_results.groupby(["Season", "LTeamID"])

    .agg(

        L=("LTeamID","count"),

        NumOT=("NumOT", "sum"),

        Score=("LScore", "sum"),

        FGM=("LFGM", "sum"),

        FGA=("LFGA", "sum"),

        FGM3=("LFGM3", "sum"),

        FGA3=("LFGA3", "sum"),

        FTM=("LFTM", "sum"),

        FTA=("LFTA", "sum"),

        OR=("LOR", "sum"),

        DR=("LDR", "sum"),

        Ast=("LAst", "sum"),

        TO=("LTO", "sum"),

        Stl=("LStl", "sum"),

        Blk=("LBlk", "sum"),

        PF=("LPF", "sum"),

        OppScore=("WScore", "sum"),

        OppFGM=("WFGM", "sum"),

        OppFGA=("WFGA", "sum"),

        OppFGM3=("WFGM3", "sum"),

        OppFGA3=("WFGA3", "sum"),

        OppFTM=("WFTM", "sum"),

        OppFTA=("WFTA", "sum"),

        OppOR=("WOR", "sum"),

        OppDR=("WDR", "sum"),

        OppAst=("WAst", "sum"),

        OppTO=("WTO", "sum"),

        OppStl=("WStl", "sum"),

        OppBlk=("WBlk", "sum"),

        OppPF=("WPF", "sum"),

    )

    .reset_index()

    .rename(columns={"LTeamID": "TeamID"})

)



team_stats = (

    pd.merge(w_team_stats, l_team_stats, how="outer", on=["Season", "TeamID"], suffixes=("_W", "_L"))

    .fillna(0)

    .astype(int)

    .assign(

        G = lambda df: df.W + df.L,

        NumOT = lambda df: df.NumOT_W + df.NumOT_L,

        Pts = lambda df: df.Score_W + df.Score_L,

        FGM = lambda df: df.FGM_W + df.FGM_L,

        FGA = lambda df: df.FGA_W + df.FGA_L,

        FGM3 = lambda df: df.FGM3_W + df.FGM3_L,

        FGA3 = lambda df: df.FGA3_W + df.FGA3_L,

        FTM = lambda df: df.FTM_W + df.FTM_L,

        FTA = lambda df: df.FTA_W + df.FTA_L,

        ORB = lambda df: df.OR_W + df.OR_L,

        DRB = lambda df: df.DR_W + df.DR_L,

        Ast = lambda df: df.Ast_W + df.Ast_L,

        TO = lambda df: df.TO_W + df.TO_L,

        Stl = lambda df: df.Stl_W + df.Stl_L,

        Blk = lambda df: df.Blk_W + df.Blk_L,

        PF = lambda df: df.PF_W + df.PF_L,

        OppPts = lambda df: df.OppScore_W + df.OppScore_L,

        OppFGM = lambda df: df.OppFGM_W + df.OppFGM_L,

        OppFGA = lambda df: df.OppFGA_W + df.OppFGA_L,

        OppFGM3 = lambda df: df.OppFGM3_W + df.OppFGM3_L,

        OppFGA3 = lambda df: df.OppFGA3_W + df.OppFGA3_L,

        OppFTM = lambda df: df.OppFTM_W + df.OppFTM_L,

        OppFTA = lambda df: df.OppFTA_W + df.OppFTA_L,

        OppORB = lambda df: df.OppOR_W + df.OppOR_L,

        OppDRB = lambda df: df.OppDR_W + df.OppDR_L,

        OppAst = lambda df: df.OppAst_W + df.OppAst_L,

        OppTO = lambda df: df.OppTO_W + df.OppTO_L,

        OppStl = lambda df: df.OppStl_W + df.OppStl_L,

        OppBlk = lambda df: df.OppBlk_W + df.OppBlk_L,

        OppPF = lambda df: df.OppPF_W + df.OppPF_L,

    )

    .assign(

        WinPerc = lambda df: df.W / df.G,

        Minutes = lambda df: df.G * 40 + df.NumOT * 5,

        PtsPG = lambda df: df.Pts / df.G,

        FGPerc = lambda df: df.FGM / df.FGA,

        FG3Perc = lambda df: df.FGM3 / df.FGA3,

        FTPerc = lambda df: df.FTM / df.FTA,

        ORBPG = lambda df: df.ORB / df.G,

        DRBPG = lambda df: df.DRB / df.G,

        AstPG = lambda df: df.Ast / df.G,

        TOPG = lambda df: df.TO / df.G,

        AstTO = lambda df: df.Ast / df.TO,

        StlPG = lambda df: df.Stl / df.G,

        BlkPG = lambda df: df.Blk / df.G,

        PFPG = lambda df: df.PF / df.G,

        OppPtsPG = lambda df: df.OppPts / df.G,

        OppFGPerc = lambda df: df.OppFGM / df.OppFGA,

        OppFG3Perc = lambda df: df.OppFGM3 / df.OppFGA3,

        OppFTPerc = lambda df: df.OppFTM / df.OppFTA,

        OppORBPG = lambda df: df.OppORB / df.G,

        OppDRBPG = lambda df: df.OppDRB / df.G,

        OppAstPG = lambda df: df.OppAst / df.G,

        OppTOPG = lambda df: df.OppTO / df.G,

        OppAstTO = lambda df: df.OppAst / df.OppTO,

        OppStlPG = lambda df: df.OppStl / df.G,

        OppBlkPG = lambda df: df.OppBlk / df.G,

        OppPFPG = lambda df: df.OppPF / df.G,

    )

    .assign(

        Possessions = lambda df: df.FGA - df.ORB + df.TO + .475 * df.FTA,

        OppPossessions = lambda df: df.OppFGA - df.OppORB + df.OppTO + .475 * df.OppFTA,

        OffEff = lambda df: 100 * df.Pts / df.Possessions,

        DefEff = lambda df: 100 * df.OppPts / df.OppPossessions,

        EFGPerc = lambda df: (df.FGM + 0.5 * df.FGM3) / df.FGA,

        ORBPerc = lambda df: df.ORB / (df.ORB + df.OppDRB),

        TOPerc = lambda df: df.TO / df.Possessions,

        FTRate = lambda df: df.FTA / df.FGA,

        OppEFGPerc = lambda df: (df.OppFGM + 0.5 * df.OppFGM3) / df.OppFGA,

        OppORBPerc = lambda df: df.OppORB / (df.OppORB + df.DRB),

        OppTOPerc = lambda df: df.OppTO / df.OppPossessions,

        OppFTRate = lambda df: df.OppFTA / df.OppFGA,

    )

    .loc[:, [

        "Season",

        "TeamID",

        "W",

        "L",

        "G",

        "WinPerc",

        "Minutes",

        "PtsPG",

        "FGPerc",

        "FG3Perc",

        "FTPerc",

        "ORBPG",

        "DRBPG",

        "AstPG",

        "TOPG",

        "AstTO",

        "StlPG",

        "BlkPG",

        "PFPG",

        "OppPtsPG",

        "OppFGPerc",

        "OppFG3Perc",

        "OppFTPerc",

        "OppORBPG",

        "OppDRBPG",

        "OppAstPG",

        "OppTOPG",

        "OppAstTO",

        "OppStlPG",

        "OppBlkPG",

        "OppPFPG",

        "OffEff",

        "DefEff",

        "EFGPerc",

        "ORBPerc",

        "TOPerc",

        "FTRate",

        "OppEFGPerc",

        "OppORBPerc",

        "OppTOPerc",

        "OppFTRate"

    ]])



team_stats.head()
# Team Conferences Feature

team_conferences = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamConferences.csv")

team_conferences.head()
# Regular Season Results



regular_season_compact_results = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv")



regular_season_compact_results = (

    regular_season_compact_results

    .assign(

        LowerTeamID = regular_season_compact_results[["WTeamID", "LTeamID"]].apply(min, axis = 1),

        HigherTeamID = regular_season_compact_results[["WTeamID", "LTeamID"]].apply(max, axis = 1),

    )

)



# Data Augmentation

regular_season_compact_results = (

    regular_season_compact_results

    .append(

        regular_season_compact_results.copy()

        .assign(

            LowerTeamID = regular_season_compact_results.HigherTeamID,

            HigherTeamID = regular_season_compact_results.LowerTeamID,

        )

    )

)



# Merge Team Stats Feature

regular_season_compact_results = (

    regular_season_compact_results

    .merge(team_stats, left_on=["Season", "LowerTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .merge(team_stats, left_on=["Season", "HigherTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .drop("TeamID_LowerTeamID", axis=1)

    .drop("TeamID_HigherTeamID", axis=1)

)



# Merge Team Conferences Feature

regular_season_compact_results = (

    regular_season_compact_results

    .merge(team_conferences, left_on=["Season", "LowerTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .merge(team_conferences, left_on=["Season", "HigherTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .drop("TeamID_LowerTeamID", axis=1)

    .drop("TeamID_HigherTeamID", axis=1)

)



regular_season_compact_results = regular_season_compact_results.assign(

    LowerTeamIDWin = lambda df: (df.LowerTeamID == df.WTeamID).astype(int)

)



regular_season_compact_results.head()
# Regular Season Model



eff_vars = [

    "OffEff_LowerTeamID",

    "DefEff_LowerTeamID",

    "EFGPerc_LowerTeamID",

    "ORBPerc_LowerTeamID",

    "TOPerc_LowerTeamID",

    "FTRate_LowerTeamID",

    "OppEFGPerc_LowerTeamID",

    "OppORBPerc_LowerTeamID",

    "OppTOPerc_LowerTeamID",

    "OppFTRate_LowerTeamID",

    "OffEff_HigherTeamID",

    "DefEff_HigherTeamID",

    "EFGPerc_HigherTeamID",

    "ORBPerc_HigherTeamID",

    "TOPerc_HigherTeamID",

    "FTRate_HigherTeamID",

    "OppEFGPerc_HigherTeamID",

    "OppORBPerc_HigherTeamID",

    "OppTOPerc_HigherTeamID",

    "OppFTRate_HigherTeamID"

]



box_vars = [

    "WinPerc_LowerTeamID",

#     "W_LowerTeamID",

#     "L_LowerTeamID",

#     "G_LowerTeamID",

#     "PtsPG_LowerTeamID",

#     "FGPerc_LowerTeamID",

#     "FG3Perc_LowerTeamID",

#     "FTPerc_LowerTeamID",

#     "ORBPG_LowerTeamID",

#     "DRBPG_LowerTeamID",

#     "AstPG_LowerTeamID",

#     "TOPG_LowerTeamID",

#     "AstTO_LowerTeamID",

#     "StlPG_LowerTeamID",

#     "BlkPG_LowerTeamID",

#     "PFPG_LowerTeamID",

#     "OppPtsPG_LowerTeamID",

#     "OppFGPerc_LowerTeamID",

#     "OppFG3Perc_LowerTeamID",

#     "OppFTPerc_LowerTeamID",

#     "OppORBPG_LowerTeamID",

#     "OppDRBPG_LowerTeamID",

#     "OppAstPG_LowerTeamID",

#     "OppTOPG_LowerTeamID",

#     "OppAstTO_LowerTeamID",

#     "OppStlPG_LowerTeamID",

#     "OppBlkPG_LowerTeamID",

#     "OppPFPG_LowerTeamID",

    "WinPerc_HigherTeamID",

#     "W_HigherTeamID",

#     "L_HigherTeamID",

#     "G_HigherTeamID",

#     "PtsPG_HigherTeamID",

#     "FGPerc_HigherTeamID",

#     "FG3Perc_HigherTeamID",

#     "FTPerc_HigherTeamID",

#     "ORBPG_HigherTeamID",

#     "DRBPG_HigherTeamID",

#     "AstPG_HigherTeamID",

#     "TOPG_HigherTeamID",

#     "AstTO_HigherTeamID",

#     "StlPG_HigherTeamID",

#     "BlkPG_HigherTeamID",

#     "PFPG_HigherTeamID",

#     "OppPtsPG_HigherTeamID",

#     "OppFGPerc_HigherTeamID",

#     "OppFG3Perc_HigherTeamID",

#     "OppFTPerc_HigherTeamID",

#     "OppORBPG_HigherTeamID",

#     "OppDRBPG_HigherTeamID",

#     "OppAstPG_HigherTeamID",

#     "OppTOPG_HigherTeamID",

#     "OppAstTO_HigherTeamID",

#     "OppStlPG_HigherTeamID",

#     "OppBlkPG_HigherTeamID",

#     "OppPFPG_HigherTeamID"

]



conf_vars = [

    "ConfAbbrev_LowerTeamID",

    "ConfAbbrev_HigherTeamID",

]



ind_vars = eff_vars + conf_vars

dep_var = "LowerTeamIDWin"



df = regular_season_compact_results.loc[:, ind_vars + [dep_var]]



procs = [Categorify]



cond = regular_season_compact_results.Season < 2015

train_idx = np.where(cond)[0]

valid_idx = np.where(~cond)[0]



# FastAI v1

# cat = []

# cont = eff_vars + [dep_var]

# tl = (

#     TabularList.from_df(

#         df,

#         cat_names=cat,

#         cont_names=cont,

#         procs=procs

#     )

#     .split_by_idx(list(valid_idx))

#     .label_from_df(cols=dep_var)

# )

# tl.train.inner_df.head()



# FastAI v2

splits = (list(train_idx), list(valid_idx))

cont, cat = cont_cat_split(df, 1, dep_var=dep_var)

data = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)

data.show()
def pred(season, lowerTeam, higherTeam):

    return 0.5



def kaggle_clip_log(x):

    return np.log(np.clip(x, 1.0e-15, 1.0 - 1.0e-15))



def kaggle_log_loss(pred, result):

    return -(result * kaggle_clip_log(pred) + (1 - result) * kaggle_clip_log(1.0 - pred))



def model_kaggle_log_loss(m, xs, y): return kaggle_log_loss(m.predict(xs), y).mean()



regular_season_compact_results = regular_season_compact_results.assign(

    LowerTeamIDWinProb = lambda df: df.apply(lambda row: pred(row.Season, row.LowerTeamID, row.HigherTeamID), axis = 1),

    LogLoss = lambda df: kaggle_log_loss(df.LowerTeamIDWinProb, df.LowerTeamIDWin)

)



regular_season_compact_results.LogLoss.mean()
# FastAI v1

# xs, y = tl.train.inner_df[ind_vars], tl.train.inner_df[dep_var]

# valid_xs, valid_y = tl.valid.inner_df[ind_vars], tl.valid.inner_df[dep_var]



# FastAI v2

xs, y = data.train.xs, data.train.y

valid_xs, valid_y = data.valid.xs, data.valid.y



m = DecisionTreeRegressor(max_leaf_nodes = 64)

m.fit(xs, y)

model_kaggle_log_loss(m, xs, y), model_kaggle_log_loss(m, valid_xs, valid_y)
def rf(xs, y, n_estimators=40, max_features=0.5, min_samples_leaf=5, **kwargs):

    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,

        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)



m = rf(xs, y)

model_kaggle_log_loss(m, xs, y), model_kaggle_log_loss(m, valid_xs, valid_y)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)



def plot_fi(fi):

    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)



fi = rf_feat_importance(m, xs)

plot_fi(fi[:20])
procs = [Categorify, Normalize]

data = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)

data.show()
regular_season_learn = tabular_learner(

    data.dataloaders(1024),

    layers=[500, 250],

    n_out=1,

    y_range=(0, 1),

    loss_func=lambda i, t: F.binary_cross_entropy(i, t.type(torch.FloatTensor))

)



# regular_season_learn.lr_find()



regular_season_learn.fit_one_cycle(5, 1e-2)
regular_season_learn.show_results()
# Tourney Results

results = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")

results = results.loc[(results.DayNum != 134) & (results.DayNum != 135)]



results = results.assign(

    LowerTeamID = results[["WTeamID", "LTeamID"]].apply(min, axis = 1),

    HigherTeamID = results[["WTeamID", "LTeamID"]].apply(max, axis = 1),

)



# Data Augmentation

results = (

    results

    .append(

        results.copy()

        .assign(

            LowerTeamID = results.HigherTeamID,

            HigherTeamID = results.LowerTeamID,

        )

    )

)



# Merge Team Stats Feature

results = (results

    .merge(team_stats, left_on=["Season", "LowerTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .merge(team_stats, left_on=["Season", "HigherTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .drop("TeamID_LowerTeamID", axis=1)

    .drop("TeamID_HigherTeamID", axis=1)

)



# Merge Team Conferences Feature

results = (

    results

    .merge(team_conferences, left_on=["Season", "LowerTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .merge(team_conferences, left_on=["Season", "HigherTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .drop("TeamID_LowerTeamID", axis=1)

    .drop("TeamID_HigherTeamID", axis=1)

)



# Merge Seeds Feature

results = (results

    .merge(seeds[["Season","TeamID", "SeedNum"]], left_on=["Season", "LowerTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .merge(seeds[["Season","TeamID", "SeedNum"]], left_on=["Season", "HigherTeamID"], right_on=["Season", "TeamID"], suffixes=("_LowerTeamID", "_HigherTeamID"))

    .drop("TeamID_LowerTeamID", axis=1)

    .drop("TeamID_HigherTeamID", axis=1)

    .assign(

        SeedDiff = lambda df: df.SeedNum_LowerTeamID - df.SeedNum_HigherTeamID,

    )

)



results = results.assign(

    LowerTeamIDWin = lambda df: (df.LowerTeamID == df.WTeamID).astype(int)

)



results.head()
# Tourney Model

seed_vars = ["SeedNum_LowerTeamID", "SeedNum_HigherTeamID"]



ind_vars = eff_vars + conf_vars + box_vars + seed_vars

dep_var = "LowerTeamIDWin"



df = results.loc[:, ind_vars + [dep_var]]



procs = [Categorify]



cond = results.Season < 2015

train_idx = np.where(cond)[0]

valid_idx = np.where(~cond)[0]



# FastAI v1

# cat = seed_vars

# cont = eff_vars + [dep_var]

# tl = (

#     TabularList.from_df(

#         df,

#         cat_names=cat,

#         cont_names=cont,

#         procs=procs

#     )

#     .split_by_idx(list(valid_idx))

#     .label_from_df(cols=dep_var)

# )

# tl.train.inner_df.head()



# FastAI v2

splits = (list(train_idx), list(valid_idx))

cont, cat = cont_cat_split(df, 1, dep_var=dep_var)

data = (

    TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)

)



data.show()
# FastAI v1

# xs, y = tl.train.inner_df[ind_vars], tl.train.inner_df[dep_var]

# valid_xs, valid_y = tl.valid.inner_df[ind_vars], tl.valid.inner_df[dep_var]



# FastAI v2

xs, y = data.train.xs, data.train.y

valid_xs, valid_y = data.valid.xs, data.valid.y



m = DecisionTreeRegressor(max_leaf_nodes = 16)

m.fit(xs, y)

model_kaggle_log_loss(m, xs, y), model_kaggle_log_loss(m, valid_xs, valid_y)
def rf(xs, y, n_estimators=40, max_features=0.5, min_samples_leaf=5, **kwargs):

    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_features=max_features,

        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)



m = rf(xs, y)

model_kaggle_log_loss(m, xs, y), model_kaggle_log_loss(m, valid_xs, valid_y)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)



def plot_fi(fi):

    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)



fi = rf_feat_importance(m, xs)

plot_fi(fi)
procs = [Categorify, Normalize]

data = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)

data.show()
tourney_learn = tabular_learner(

    data.dataloaders(1024),

    layers=[500, 250],

    n_out=1,

    y_range=(0, 1),

    loss_func=lambda i, t: F.binary_cross_entropy(i, t.type(torch.FloatTensor))

)



# tourney_learn.lr_find()



tourney_learn.fit_one_cycle(5, 1e-2)
teams = pd.read_csv("/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv")
teams.head()