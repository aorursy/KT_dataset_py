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
from pandas import Series, DataFrame
#original data

qb_stats = pd.read_csv("/kaggle/input/quarterback-stats/QBs.csv")

z_p = pd.read_csv("/kaggle/input/zscores/z_scores.csv")

team_offense = pd.read_csv("/kaggle/input/nfl-20182019-team-offenses/NFL_Offense_Data.csv")

rushing_stats = pd.read_csv("/kaggle/input/rushing/Rushing.csv")

receiving_stats = pd.read_csv("/kaggle/input/receiving/Receiving.csv")
#quarterbacks
#cleaned qb stats

def clean_qb(df):

    index = 0

    bad_indexes = []

    while index < len(df):

        if df.loc[index]["Pos"] != "QB":

            bad_indexes.append(index)

        index += 1

    return bad_indexes

cleaned_qb_stats = qb_stats.drop(clean_qb(qb_stats))
#qb names

qb_names = []

for name in cleaned_qb_stats["Player"].tolist():

    if "*" in name:

        qb_name = name.split("*")

    elif "\\" in name:

        qb_name = name.split("\\")

    qb_names.append(qb_name[0])
#sorted data

sorted_cleaned_qb_stats = cleaned_qb_stats.sort_values("Tm")

sorted_team_offense = team_offense.sort_values("Tm")
#correlation

def correlate(player_data):

    ppg = sorted_team_offense["PPG"].tolist()

    columns = []

    correlations = []

    for col in player_data:

        if col == "Rk" or col == "Player" or col == "Tm" or col == "Pos" or col == "QBrec":

            continue

        else: 

            correlation = np.corrcoef(ppg,player_data[col].tolist())

            columns.append(col)

            correlations.append(correlation[0][1])

    correlate_df = DataFrame(

    {

    "stat": columns,

    "correlation": correlations

    }

    )

    return correlate_df

ppg_correlation = correlate(sorted_cleaned_qb_stats).sort_values("correlation",ascending=False)
#cleaning qb correlation (correlation > 0.5 only)

def clean_qb_correlation(ppg_correlation):

    bad_indexes = []

    i = 0

    while i < len(ppg_correlation):

        if ppg_correlation["stat"][i] == "QBR" or ppg_correlation["stat"][i] == "TD" or ppg_correlation["stat"][i] == "Rate" or ppg_correlation["stat"][i] == "Yds" or ppg_correlation["stat"][i] == "Cmp" or ppg_correlation["stat"][i] == "4QC" or ppg_correlation["stat"][i] == "GWD":

            bad_indexes.append(i)

        if ppg_correlation["correlation"][i] < 0.5 and i not in bad_indexes:

            bad_indexes.append(i)

        i += 1

    cleaned_df = ppg_correlation.drop(bad_indexes)

    return cleaned_df

cleaned_ppg_correlation = clean_qb_correlation(ppg_correlation)

correlation_stats = cleaned_ppg_correlation["stat"].tolist()
#qb overall

qb_overall = DataFrame(

{

"Name": qb_names,

"Team": cleaned_qb_stats["Tm"].tolist()

}

)

for stat in correlation_stats:

    p = []

    p_values = z_p.set_index("z-score")["p-value"]

    for val in cleaned_qb_stats[stat]:

        z = round(((val-cleaned_qb_stats[stat].mean())/cleaned_qb_stats[stat].std()),2)

        p_val = round(p_values[z]*100,1)

        p.append(p_val)

    qb_overall[stat] = p
#qb weighted overall

def qb_score():

    i = 0

    correlations = cleaned_ppg_correlation.set_index("stat")

    qb_weighted_overall = []

    while i < len(qb_overall):

        total = 0

        for stat in correlation_stats:

            val = qb_overall[i:(i+1)][stat].item()*correlations["correlation"][stat].item()

            total += val

        overall = total/correlations["correlation"].sum()

        qb_weighted_overall.append(overall)

        i += 1

    return qb_weighted_overall

qb_overall["weighted overall"] = qb_score()

qb_overall = qb_overall.sort_values("weighted overall",ascending=False)

qb_overall.head()
#receivers
def team_receiving():

    teams = []

    Tgt = []

    Rec = []

    Catch = []

    Yds = []

    Td = []

    YTgt = []

    RG = []

    YG = []

    YR = []

    TDP = []

    for team in receiving_stats["Tm"]:

        if team != "2TM" and team not in teams:

            team_tgt = receiving_stats.loc[receiving_stats["Tm"] == team]["Tgt"].sum()

            team_rec = receiving_stats.loc[receiving_stats["Tm"] == team]["Rec"].sum()

            team_yds = receiving_stats.loc[receiving_stats["Tm"] == team]["Yds"].sum()

            team_tds = receiving_stats.loc[receiving_stats["Tm"] == team]["TD"].sum()

            teams.append(team)

            Tgt.append(team_tgt)

            Rec.append(team_rec)

            Catch.append(team_rec/team_tgt)

            Yds.append(team_yds)

            Td.append(team_tds)

            YTgt.append(team_yds/team_tgt)

            RG.append(team_rec/16)

            YG.append(team_yds/16)

            YR.append(team_yds/team_rec)

            TDP.append(team_tds/16)

    df = DataFrame(

    {

    "Tm": teams,

    "Tgt": Tgt,

    "Rec": Rec,

    "Catch %": Catch,

    "Yds": Yds,

    "Td": Td,

    "Y/Tgt": YTgt,

    "Y/R": YR,

    "R/G": RG,

    "Y/G": YG,

    "TD/G": TDP

    }

    )

    return df

team_receiving = team_receiving().sort_values("Tm")

receiving_correlation = correlate(team_receiving).sort_values("correlation",ascending = False)
#cleaning wr correlation (correlation > 0.5 only)

def clean_wr_correlation(ppg_correlation):

    bad_indexes = []

    i = 0

    while i < len(ppg_correlation):

        if ppg_correlation["stat"][i] == "Td" or ppg_correlation["stat"][i] == "Yds" or ppg_correlation["stat"][i] == "Rec":

            bad_indexes.append(i)

        if ppg_correlation["correlation"][i] < 0.5 and i not in bad_indexes:

            bad_indexes.append(i)

        i += 1

    cleaned_df = ppg_correlation.drop(bad_indexes)

    return cleaned_df

cleaned_wr_correlation = clean_wr_correlation(receiving_correlation)

wr_correlation_stats = cleaned_wr_correlation["stat"].tolist()
receiving_stats["TD/G"] = receiving_stats["TD"]/receiving_stats["G"]
#cleaned wr stats

def clean_wr(df):

    index = 0

    bad_indexes = []

    while index < len(df):

        if df.loc[index]["Rec"] < 25: 

            bad_indexes.append(index)

        if  df.loc[index]["Pos"] != "WR" and df.loc[index]["Pos"] != "wr" and index not in bad_indexes:

            bad_indexes.append(index)

        index += 1

    return bad_indexes

wr_stats = receiving_stats.drop(clean_wr(receiving_stats))
#wr names

wr_names = []

for name in wr_stats["Player"].tolist():

    if "*" in name:

        wr_name = name.split("*")

    elif "\\" in name:

        wr_name = name.split("\\")

    wr_names.append(wr_name[0])
#wr overall

wr_overall = DataFrame(

{

"Name": wr_names,

"Team": wr_stats["Tm"].tolist()

}

)

for stat in wr_correlation_stats:

    p = []

    p_values = z_p.set_index("z-score")["p-value"]

    for val in wr_stats[stat]:

        z = round(((val-wr_stats[stat].mean())/wr_stats[stat].std()),2)

        p_val = round(p_values[z]*100,1)

        p.append(p_val)

    wr_overall[stat] = p
#qb weighted overall

def wr_score():

    i = 0

    correlations = cleaned_wr_correlation.set_index("stat")

    wr_weighted_overall = []

    while i < len(wr_overall):

        total = 0

        for stat in wr_correlation_stats:

            val = wr_overall[i:(i+1)][stat].item()*correlations["correlation"][stat].item()

            total += val

        overall = total/correlations["correlation"].sum()

        wr_weighted_overall.append(overall)

        i += 1

    return wr_weighted_overall

wr_overall["weighted overall"] = wr_score()

wr_overall = wr_overall.sort_values("weighted overall",ascending=False)

wr_overall.head()
#tight ends
#cleaned te stats

def clean_te(df):

    index = 0

    bad_indexes = []

    while index < len(df):

        if df.loc[index]["Rec"] < 20: 

            bad_indexes.append(index)

        if  df.loc[index]["Pos"] != "TE" and df.loc[index]["Pos"] != "te" and index not in bad_indexes:

            bad_indexes.append(index)

        index += 1

    return bad_indexes

te_stats = receiving_stats.drop(clean_te(receiving_stats))
#te names

te_names = []

for name in te_stats["Player"].tolist():

    if "*" in name:

        te_name = name.split("*")

    elif "\\" in name:

        te_name = name.split("\\")

    te_names.append(te_name[0])
#tight end overall

te_overall = DataFrame(

{

"Name": te_names,

"Team": te_stats["Tm"].tolist()

}

)

for stat in wr_correlation_stats:

    p = []

    p_values = z_p.set_index("z-score")["p-value"]

    for val in te_stats[stat]:

        z = round(((val-te_stats[stat].mean())/te_stats[stat].std()),2)

        p_val = round(p_values[z]*100,1)

        p.append(p_val)

    te_overall[stat] = p
#te weighted overall

def te_score():

    i = 0

    correlations = cleaned_wr_correlation.set_index("stat")

    te_weighted_overall = []

    while i < len(te_overall):

        total = 0

        for stat in wr_correlation_stats:

            val = te_overall[i:(i+1)][stat].item()*correlations["correlation"][stat].item()

            total += val

        overall = total/correlations["correlation"].sum()

        te_weighted_overall.append(overall)

        i += 1

    return te_weighted_overall

te_overall["weighted overall"] = te_score()

te_overall = te_overall.sort_values("weighted overall",ascending=False)

te_overall.head()
#running backs
#rb receiving

def rb_receiving(df):

    index = 0

    bad_indexes = []

    while index < len(df):

        if df.loc[index]["Pos"] != "RB" and df.loc[index]["Pos"] != "rb":

            bad_indexes.append(index)

        index += 1

    return bad_indexes

rb_receiving_stats = receiving_stats.drop(rb_receiving(receiving_stats))

rb_receiving_stats = rb_receiving_stats.sort_values("Player")
#rb rushing

def clean_rb_rushing(df):

    index = 0

    bad_indexes = []

    while index < len(df):

        if df.loc[index]["Player"] not in rb_receiving_stats["Player"].tolist():

            bad_indexes.append(index)

        index += 1

    return bad_indexes

rb_rushing_stats = rushing_stats.drop(clean_rb_rushing(rushing_stats)).sort_values("Player")

rb_rushing_stats["TD/G"] = rb_rushing_stats["TD"]/rb_rushing_stats["G"]
#rb total stats

rb_total_stats = DataFrame(

{

"Player": rb_rushing_stats["Player"].tolist(),

"Tm":rb_rushing_stats["Tm"].tolist(),

"Att": rb_rushing_stats["Att"].tolist(),

"Rush Yds": rb_rushing_stats["Yds"].tolist(),

"Rush TD": rb_rushing_stats["TD"].tolist(),

"Rush Y/A": rb_rushing_stats["Y/A"].tolist(),

"Rush Y/G": rb_rushing_stats["Y/G"].tolist(),

"Rush TD/G": rb_rushing_stats["TD/G"].tolist(),

"Tgt": rb_receiving_stats["Tgt"].tolist(),

"Rec": rb_receiving_stats["Rec"].tolist(),

"Rec Yds": rb_receiving_stats["Yds"].tolist(),

"Y/R": rb_receiving_stats["Y/R"].tolist(),

"Rec TD": rb_receiving_stats["TD"].tolist(),

"Y/Tgt": rb_receiving_stats["Y/Tgt"].tolist(),

"R/G": rb_receiving_stats["R/G"].tolist(),

"Rec Y/G": rb_receiving_stats["Y/G"].tolist(),

"Rec TD/G": rb_receiving_stats["TD/G"].tolist()

}

)
#running back stats by team

def team_rbs():

    teams = []

    rush_YA = []

    rush_YG = []

    rush_TDG = []

    YR = []

    YTgt = []

    RG = []

    YR = []

    rec_YG = []

    rec_TDG = []

    for team in rb_total_stats["Tm"]:

        if team != "2TM" and team not in teams:

            team_att = rb_total_stats.loc[rb_total_stats["Tm"] == team]["Att"].sum()

            team_rush_yds = rb_total_stats.loc[rb_total_stats["Tm"] == team]["Rush Yds"].sum()

            team_rush_tds = rb_total_stats.loc[rb_total_stats["Tm"] == team]["Rush TD"].sum()

            team_tgt = rb_total_stats.loc[rb_total_stats["Tm"] == team]["Tgt"].sum()

            team_rec = rb_total_stats.loc[rb_total_stats["Tm"] == team]["Rec"].sum()

            team_rec_yds = rb_total_stats.loc[rb_total_stats["Tm"] == team]["Rec Yds"].sum()

            team_rec_tds = rb_total_stats.loc[rb_total_stats["Tm"] == team]["Rec TD"].sum()

            teams.append(team)

            rush_YA.append(team_rush_yds/team_att)

            rush_YG.append(team_rush_yds/16)

            rush_TDG.append(team_rush_tds/16)

            YR.append(team_rec_yds/team_rec)

            YTgt.append(team_rec_yds/team_tgt)

            RG.append(team_rec/16)

            rec_YG.append(team_rec_yds/16)

            rec_TDG.append(team_rec_tds/16)

    df = DataFrame(

    {

    "Tm": teams,

    "Rush Y/A": rush_YA,

    "Rush Y/G": rush_YG,

    "Rush TD/G": rush_TDG,

    "Y/R": YR,

    "Y/Tgt": YTgt,

    "R/G": RG,

    "Y/R": YR,

    "Rec Y/G": rec_YG,

    "Rec TD/G": rec_TDG

    }

    )

    return df

team_rbs = team_rbs().sort_values("Tm")

rb_correlation = correlate(team_rbs).sort_values("correlation",ascending = False)
#cleaning rb correlation (correlation > 0.5 only)

def clean_rb_correlation(ppg_correlation):

    bad_indexes = []

    i = 0

    while i < len(ppg_correlation):

        if ppg_correlation["correlation"][i] < 0.2 and i not in bad_indexes:

            bad_indexes.append(i)

        i += 1

    cleaned_df = ppg_correlation.drop(bad_indexes)

    return cleaned_df

cleaned_rb_correlation = clean_rb_correlation(rb_correlation)

rb_correlation_stats = cleaned_rb_correlation["stat"].tolist()
#rb names

rb_names = []

for name in rb_total_stats["Player"].tolist():

    if "*" in name:

        rb_name = name.split("*")

    elif "\\" in name:

        rb_name = name.split("\\")

    rb_names.append(rb_name[0])
#rb overall

rb_overall = DataFrame(

{

"Name": rb_names,

"Team": rb_total_stats["Tm"].tolist()

}

)

for stat in rb_correlation_stats:

    p = []

    p_values = z_p.set_index("z-score")["p-value"]

    for val in rb_total_stats[stat]:

        z = round(((val-rb_total_stats[stat].mean())/rb_total_stats[stat].std()),2)

        p_val = round(p_values[z]*100,1)

        p.append(p_val)

    rb_overall[stat] = p
#rb weighted overall

def rb_score():

    i = 0

    correlations = cleaned_rb_correlation.set_index("stat")

    rb_weighted_overall = []

    while i < len(rb_overall):

        total = 0

        for stat in rb_correlation_stats:

            val = rb_overall[i:(i+1)][stat].item()*correlations["correlation"][stat].item()

            total += val

        overall = total/correlations["correlation"].sum()

        rb_weighted_overall.append(overall)

        i += 1

    return rb_weighted_overall

rb_overall["weighted overall"] = rb_score()

rb_overall = rb_overall.sort_values("weighted overall",ascending=False)

rb_overall.head()
pd.set_option('display.max_rows', 1000)
qb_overall
wr_overall
rb_overall
te_overall
qb_overall.to_csv('qb_overall.csv', index = False)
rb_overall.to_csv('rb_overall.csv', index = False)
wr_overall.to_csv('wr_overall.csv', index = False)
te_overall.to_csv('te_overall.csv', index = False)