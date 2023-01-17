# Load video_review.csv data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

video_review = pd.read_csv("../input/video_review.csv")
video_review.head(3)
# Count injuries by year
injuries_by_year = pd.DataFrame(video_review["Season_Year"].value_counts())
injuries_by_year.columns = ["Concussions"]
injuries_by_year.index.name = "Year"

total_injuries = injuries_by_year.sum()[0]
injuries_by_year["Share"] = injuries_by_year["Concussions"] / total_injuries
injuries_by_year["Share"] = injuries_by_year["Share"].map(lambda s: round(s * 100, 2))
injuries_by_year
# Count activity derived of injured player
player_activity_derived = pd.DataFrame(video_review["Player_Activity_Derived"].value_counts())
player_activity_derived.columns = ["Concussions"]
player_activity_derived.index.name = "Player Activity"

total_injuries = player_activity_derived.sum()[0]
player_activity_derived["Share"] = player_activity_derived["Concussions"] / total_injuries
player_activity_derived["Share"] = player_activity_derived["Share"].map(lambda s: round(s * 100, 2))
player_activity_derived
# Count activity derived of primary player "causing" the injury
primary_partner_ad = pd.DataFrame(video_review["Primary_Partner_Activity_Derived"].value_counts())
primary_partner_ad.columns = ["Concussions"]
primary_partner_ad.index.name = "Partner Activity"

total_injuries = primary_partner_ad.sum()[0]
primary_partner_ad["Share"] = primary_partner_ad["Concussions"] / total_injuries
primary_partner_ad["Share"] = primary_partner_ad["Share"].map(lambda s: round(s * 100, 2))
primary_partner_ad
# Count whether impact was with teammate
friendly_fire = pd.DataFrame(video_review["Friendly_Fire"].value_counts())
friendly_fire.columns = ["Concussions"]
friendly_fire.index.name = "Friendly Fire"

total_injuries = friendly_fire.sum()[0]
friendly_fire["Share"] = friendly_fire["Concussions"] / total_injuries
friendly_fire["Share"] = friendly_fire["Share"].map(lambda s: round(s * 100, 2))
friendly_fire
# Count injuries by the primary type of impact
primary_impact_type = pd.DataFrame(video_review["Primary_Impact_Type"].value_counts())
primary_impact_type.columns = ["Concussions"]
primary_impact_type.index.name = "Impact Type"

total_injuries = primary_impact_type.sum()[0]
primary_impact_type["Share"] = primary_impact_type["Concussions"] / total_injuries
primary_impact_type["Share"] = primary_impact_type["Share"].map(lambda s: round(s * 100, 2))
primary_impact_type
# Load related data files
game_data = pd.read_csv("../input/game_data.csv")
play_information = pd.read_csv("../input/play_information.csv")
player_punt_data = pd.read_csv("../input/player_punt_data.csv")
play_player_role_data = pd.read_csv("../input/play_player_role_data.csv")
video_footage_injury = pd.read_csv("../input/video_footage-injury.csv")

# Prepare tables for merging
video_review["Primary_Partner_GSISID"] = video_review["Primary_Partner_GSISID"].fillna(0).replace("Unclear", 0).astype("int64")
player_punt_data = player_punt_data.groupby("GSISID").agg({"Number": lambda x: ",".join(x), "Position": lambda x: ",".join(x.unique()) })
video_footage_injury = video_footage_injury.rename(columns = {"season": "Season_Year", "Type": "Season_Type", "Home_team": "Home_Team", "Qtr": "Quarter", "gamekey": "GameKey", "playid": "PlayID"})

# Merge tables
shared_columns = ["Season_Year", "GameKey"] # between video_review and game_data
concussion_data = video_review.merge(game_data, how = "left", left_on = shared_columns, right_on = shared_columns)

shared_columns = ["Season_Year", "GameKey", "PlayID", "Season_Type", "Week"] # between concussion_data and play_information (also "Game_Date" but in different formats)
concussion_data = concussion_data.merge(play_information, how = "left", left_on = shared_columns, right_on = shared_columns, suffixes = ("", "_play_inf"))

shared_columns = ["GSISID"] # between concussion_data and player_punt_data
concussion_data = concussion_data.merge(player_punt_data, how = "left", left_on = shared_columns, right_on = shared_columns)
concussion_data = concussion_data.merge(player_punt_data, how = "left", left_on = ["Primary_Partner_GSISID"], right_on = shared_columns, suffixes = ("_Injured", "_Partner"))

shared_columns = ["Season_Year", "GameKey", "PlayID", "GSISID"] # between concussion_data and play_player_role_data
concussion_data = concussion_data.merge(play_player_role_data, how = "left", left_on = shared_columns, right_on = shared_columns)
concussion_data = concussion_data.merge(play_player_role_data, how = "left", left_on = ["Season_Year", "GameKey", "PlayID", "Primary_Partner_GSISID"], right_on = shared_columns, suffixes = ("_Injured", "_Partner"))

shared_columns = ["Season_Year", "GameKey", "PlayID", "Season_Type", "Week", "Home_Team", "Visit_Team", "Quarter"] # between concussion_data and video_footage_injury , (also "PlayDescription")
concussion_data = concussion_data.merge(video_footage_injury, how = "left", left_on = shared_columns, right_on = shared_columns, suffixes = ("", "_Other"))
concussion_data.head(3)
# We don´t know the team of the players, so let's add that column
kicking_team_positions = ['PLS', 'PLG', 'PRG', 'PLT', 'PRT', 'PLW', 'PRW', 'GL', 'GR', 'PPR', 'P'] # from Data, Appendix 1
returning_team_positions = ['PDL1', 'PDL2', 'PDL3', 'PDR1', 'PDR2', 'PDR3', 'PLL', 'PLM', 'PLR', 'PLL1', 'VLo', 'VR', 'PFB', 'PR'] # from Data, Appendix 2

def role_in_kicking_team(role):
    if role in kicking_team_positions:
        return "Kicking"
    elif role in returning_team_positions:
        return "Returning"

def team_of_player(is_kicking, poss_team, home_visit_teams):
    if is_kicking == "Kicking":
        return poss_team
    elif is_kicking == "Returning":
        return home_visit_teams.replace(poss_team, "").replace("-", "")
    
def add_information(row):
    home_visit = row["Home_Team_Visit_Team"]
    poss = row["Poss_Team"]
    is_player_punting = role_in_kicking_team(row["Role_Injured"])
    is_partner_punting = role_in_kicking_team(row["Role_Partner"])

    row["Player_Kicking/Return"] = is_player_punting
    row["Player_Team"] = team_of_player(is_player_punting, poss, home_visit)
    row["Partner_Kicking/Return"] = is_partner_punting
    row["Partner_Team"] = team_of_player(is_partner_punting, poss, home_visit)
    
    return row

concussion_data = concussion_data.apply(add_information, axis = "columns")

# Now let's clean up the columns that we don't need and format the information
# Game - Type, Year, Week, Teams Home-Visit, GameKey, PlayID
# Impact_Type
# Player Injured - Team, Numbers, Role, Activity, Kicking/Return Team
# Primary Partner - Team, Numbers, Role, Activity, Kicking/Return Team 
# Video Link
def get_summary(row):
    return row[["Season_Type", "Season_Year", "Week", "Home_Team_Visit_Team", "GameKey", "PlayID",
                "Primary_Impact_Type", "Player_Team", "Number_Injured", "Role_Injured", "Player_Activity_Derived", "Player_Kicking/Return",
                "Partner_Team", "Number_Partner", "Role_Partner", "Primary_Partner_Activity_Derived", "Partner_Kicking/Return",
                "PREVIEW LINK (5000K)"]]

summarized_concussion_data = concussion_data.apply(get_summary, axis = "columns")
summarized_concussion_data = summarized_concussion_data.rename(columns = {"Season_Type": "Type", "Season_Year": "Year", "Home_Team_Visit_Team": "Game (Home-Visit)", "Number_Injured": "Player_Numbers", "Player_Activity_Derived": "Player_Activity", "Role_Injured": "Player_Role", "Primary_Impact_Type": "Impact_Type",  "Number_Partner": "Partner_Numbers", "Primary_Partner_Activity_Derived": "Partner_Activity", "Role_Partner": "Partner_Role", "PREVIEW LINK (5000K)": "Video Link"})
summarized_concussion_data.head(3)
# Let's add the categories to the table
plays_cat_list = ["Poor Tackling", "Dirty Hit", "Poor Tackling", "Poor Tackling", "Accidental", "Poor Blocking", 
             "Poor Blocking", "Poor Blocking", "Poor Tackling", "Ground", "Dirty Hit", "Poor Tackling", 
             "Dirty Hit", "Poor Tackling", "Poor Blocking", "Poor Tackling", "Unclear", "Dirty Hit", 
             "Accidental", "Poor Tackling", "Dirty Hit", "Dirty Hit", "Accidental", "Dirty Hit", 
             "Poor Tackling", "Poor Tackling", "Ground", "Poor Blocking", "Accidental", "Poor Tackling",
             "Poor Blocking", "Poor Blocking", "Poor Tackling", "Dirty Hit", "Poor Tackling", "Poor Blocking", "Accidental"]
plays_cat_serie = pd.Series(plays_cat_list, index = list(range(0, 37)))
concussion_data["Cat_Play"] = plays_cat_serie

# Count injuries by category
cat_play = pd.DataFrame(concussion_data["Cat_Play"].value_counts())
cat_play.columns = ["Concussions"]
cat_play.index.name = "Category"

total_injuries = cat_play.sum()[0]
cat_play["Share"] = cat_play["Concussions"] / total_injuries
cat_play["Share"] = cat_play["Share"].map(lambda s: round(s * 100, 2))
cat_play
# Count injuries by game type
game_type_injuries = pd.DataFrame(summarized_concussion_data["Type"].value_counts())
game_type_injuries = game_type_injuries.append(pd.Series({"Type": 0}, name = "Post"))

# Number of total games by type
game_type_totals = game_data["Season_Type"].value_counts()
game_type_injuries = game_type_injuries.join(game_type_totals)

game_type_injuries.index.name = "Game Type"
game_type_injuries.columns = ["Concussions", "Total Games"]

# Ratio between number of games and injuries
game_type_injuries["Ratio"] = game_type_injuries["Concussions"] / game_type_injuries["Total Games"]
game_type_injuries["Ratio"] = game_type_injuries["Ratio"].map(lambda s: round(s, 3))
game_type_injuries.sort_values(by = "Ratio", ascending = False)
# Count injuries by kicking/returning punt
kick_return_team = pd.DataFrame(summarized_concussion_data["Player_Kicking/Return"].value_counts())
kick_return_team.columns = ["Concussions"]
kick_return_team.index.name = "Team"

total_injuries = kick_return_team.sum()[0]
kick_return_team["Share"] = kick_return_team["Concussions"] / total_injuries
kick_return_team["Share"] = kick_return_team["Share"].map(lambda s: round(s * 100, 2))
kick_return_team
# Count injuries by day of the game
game_day = pd.DataFrame(concussion_data.loc[concussion_data.Season_Type == "Reg"]["Game_Day"].value_counts())
game_day.index.name = "Game_Day"
game_day.columns = ["Concussions"]

# Number of total games by type
game_day_totals = game_data.loc[game_data.Season_Type == "Reg"]["Game_Day"].value_counts()
game_day = game_day.join(game_day_totals)

game_day.index.name = "Regular Season"
game_day.columns = ["Concussions", "Total Games"]

# Ratio between number of games and injuries
game_day["Ratio"] = game_day["Concussions"] / game_day["Total Games"]
game_day["Ratio"] = game_day["Ratio"].map(lambda s: round(s, 3))
game_day.sort_values(by = "Ratio", ascending = False)
# Count injuries by role of injured player
role_injured = pd.DataFrame(concussion_data["Role_Injured"].value_counts())
role_injured.columns = ["Concussions"]
role_injured.index.name = "Position"

total_injuries = role_injured.sum()[0]
role_injured["Share"] = role_injured["Concussions"] / total_injuries
role_injured["Share"] = role_injured["Share"].map(lambda s: round(s * 100, 2))
role_injured
# Count injuries by quarter
quarter_inj = pd.DataFrame(concussion_data["Quarter"].value_counts())
quarter_inj.columns = ["Concussions"]
quarter_inj.index.name = "Quarter"

total_injuries = quarter_inj.sum()[0]
quarter_inj["Share"] = quarter_inj["Concussions"] / total_injuries
quarter_inj["Share"] = quarter_inj["Share"].map(lambda s: round(s * 100, 2))
quarter_inj
def get_score_diff(row):
    teams = row["Home_Team_Visit_Team"].split("-")
    home_team = teams[0]
    visit_team = teams[1]
    
    scores = row["Score_Home_Visiting"].split("-")
    home_score = int(scores[0].rstrip())
    visit_score = int(scores[1].lstrip())
    
    if row["Player_Team"] == home_team:
        score_def = home_score - visit_score
    elif row["Player_Team"] == visit_team:
        score_def = visit_score - home_score
        
    if score_def == 0:
        status_score = "Tied"
    elif score_def < 0:
        status_score = "Losing"
    else:
        status_score = "Winning"
    
    row["Score_Diff"] = score_def
    row["Status_Score"] = status_score
    
    return row

score_diff = concussion_data[["Home_Team_Visit_Team", "Score_Home_Visiting", "Player_Team"]]
score_diff = score_diff.apply(get_score_diff, axis = "columns")
score_diff.head(3)
# Count injuries by game status
game_status = pd.DataFrame(score_diff["Status_Score"].value_counts())
game_status.columns = ["Concussions"]
game_status.index.name = "Game Status"

total_injuries = game_status.sum()[0]
game_status["Share"] = game_status["Concussions"] / total_injuries
game_status["Share"] = game_status["Share"].map(lambda s: round(s * 100, 2))
game_status
# Load all data
ngs_16_pre = pd.read_csv("../input/NGS-2016-pre.csv")
ngs_16_reg_1_6 = pd.read_csv("../input/NGS-2016-reg-wk1-6.csv")
ngs_16_reg_7_12 = pd.read_csv("../input/NGS-2016-reg-wk7-12.csv")
ngs_16_reg_13_17 = pd.read_csv("../input/NGS-2016-reg-wk13-17.csv")
ngs_17_pre = pd.read_csv("../input/NGS-2017-pre.csv")
ngs_17_reg_1_6 = pd.read_csv("../input/NGS-2017-reg-wk1-6.csv")
ngs_17_reg_7_12 = pd.read_csv("../input/NGS-2017-reg-wk7-12.csv")
ngs_17_reg_13_17 = pd.read_csv("../input/NGS-2017-reg-wk13-17.csv")
# Load only concussion specific plays
ngs_all_data = pd.DataFrame()

def get_play_ngs(row):
    year = row.Season_Year
    season_type = row.Season_Type
    week = row.Week
    gamekey = row.GameKey
    play_id = row.PlayID
    player_id = row.GSISID_Injured
    partner_id = row.Primary_Partner_GSISID

    if year == 2016:
        if season_type == "Pre":
            ngs_data = ngs_16_pre.loc[(ngs_16_pre["GameKey"] == gamekey) & (ngs_16_pre["PlayID"] == play_id) & ((ngs_16_pre["GSISID"] == player_id) | (ngs_16_pre["GSISID"] == partner_id))].sort_values(by = "Time")
        elif season_type == "Reg":
            if (week >= 1) & (week <= 6):
                ngs_data = ngs_16_reg_1_6.loc[(ngs_16_reg_1_6["GameKey"] == gamekey) & (ngs_16_reg_1_6["PlayID"] == play_id) & ((ngs_16_reg_1_6["GSISID"] == player_id) | (ngs_16_reg_1_6["GSISID"] == partner_id))].sort_values(by = "Time")
            elif (week >= 7) & (week <= 12):
                ngs_data = ngs_16_reg_7_12.loc[(ngs_16_reg_7_12["GameKey"] == gamekey) & (ngs_16_reg_7_12["PlayID"] == play_id) & ((ngs_16_reg_7_12["GSISID"] == player_id) | (ngs_16_reg_7_12["GSISID"] == partner_id))].sort_values(by = "Time")
            elif (week >= 13) & (week <= 17):
                ngs_data = ngs_16_reg_13_17.loc[(ngs_16_reg_13_17["GameKey"] == gamekey) & (ngs_16_reg_13_17["PlayID"] == play_id) & ((ngs_16_reg_13_17["GSISID"] == player_id) | (ngs_16_reg_13_17["GSISID"] == partner_id))].sort_values(by = "Time")
    elif year == 2017:
        if season_type == "Pre":
            ngs_data = ngs_17_pre.loc[(ngs_17_pre["GameKey"] == gamekey) & (ngs_17_pre["PlayID"] == play_id) & ((ngs_17_pre["GSISID"] == player_id) | (ngs_17_pre["GSISID"] == partner_id))].sort_values(by = "Time")
        elif season_type == "Reg":
            if (week >= 1) & (week <= 6):
                ngs_data = ngs_17_reg_1_6.loc[(ngs_17_reg_1_6["GameKey"] == gamekey) & (ngs_17_reg_1_6["PlayID"] == play_id) & ((ngs_17_reg_1_6["GSISID"] == player_id) | (ngs_17_reg_1_6["GSISID"] == partner_id))].sort_values(by = "Time")
            elif (week >= 7) & (week <= 12):
                ngs_data = ngs_17_reg_7_12.loc[(ngs_17_reg_7_12["GameKey"] == gamekey) & (ngs_17_reg_7_12["PlayID"] == play_id) & ((ngs_17_reg_7_12["GSISID"] == player_id) | (ngs_17_reg_7_12["GSISID"] == partner_id))].sort_values(by = "Time")
            elif (week >= 13) & (week <= 17):
                ngs_data = ngs_17_reg_13_17.loc[(ngs_17_reg_13_17["GameKey"] == gamekey) & (ngs_17_reg_13_17["PlayID"] == play_id) & ((ngs_17_reg_13_17["GSISID"] == player_id) | (ngs_17_reg_13_17["GSISID"] == partner_id))].sort_values(by = "Time")
                
    ngs_data.Time = ngs_data.Time.astype("datetime64")
    ball_snap = ngs_data.loc[(ngs_data.Event == "ball_snap")]["Time"].iloc[0]
    
    return ngs_data.loc[ngs_data.Time >= ball_snap]

for row in concussion_data.itertuples():
    ngs_all_data = ngs_all_data.append(get_play_ngs(row))

ngs_all_data.head(3)
# Calculate speed
def add_speed(row):
    # Multiply by 0.9144 to convert yards to meters
    row["Speed m/s"] = row.dis * 0.9144 / 0.1
    return row

ngs_all_data = ngs_all_data.apply(add_speed, axis = "columns")

# Draw the graphics
for row in concussion_data.itertuples():
    game_key = row.GameKey
    play_id = row.PlayID
    player_id = row.GSISID_Injured
    partner_id = row.Primary_Partner_GSISID
        
    player_positions = ngs_all_data.loc[(ngs_all_data["GameKey"] == game_key) & (ngs_all_data["PlayID"] == play_id) & (ngs_all_data["GSISID"] == player_id)].sort_values(by = "Time").reset_index()
    partner_positions = ngs_all_data.loc[(ngs_all_data["GameKey"] == game_key) & (ngs_all_data["PlayID"] == play_id) & (ngs_all_data["GSISID"] == partner_id)].sort_values(by = "Time").reset_index()
    tackle_time = player_positions[(player_positions.Event == "tackle")].Time

    if tackle_time.empty:
        continue
    else:
        player_positions = player_positions.loc[player_positions["Time"] <=  tackle_time.iloc[0]]
        player_x = player_positions["x"]
        player_y = player_positions["y"]
        player_speed = player_positions["Speed m/s"]
        
        # Configure graph
        sns.set()
        fig = plt.figure(figsize = (7, 3))
        ax = fig.add_subplot(111)
        ax.patch.set_facecolor("green")

        if not partner_positions.empty:
            partner_positions = partner_positions.loc[partner_positions["Time"] <=  tackle_time.iloc[0]]
            partner_x = partner_positions["x"]
            partner_y = partner_positions["y"]
            partner_speed = partner_positions["Speed m/s"]
            cmap = plt.get_cmap('summer')
            plt.scatter(partner_x, partner_y, c = partner_speed, cmap = cmap)
        
        cmap = plt.get_cmap('PuRd')
        plt.scatter(player_x, player_y, c = player_speed, cmap = cmap)
        
        plt.clim(0, 12)
        plt.colorbar(label = "Speed (m/s)")
        plt.xlim(0, 120)
        plt.xticks(np.arange(0, 120, step = 10), ["", "G", "10", "20", "30", "40", "50", "40", "30", "20", "10", "G"])
        plt.ylim(-5, 59) # short side of the field is 53.3
        plt.yticks(np.arange(0, 59, 53.3))
        ax.set_title(row[-5] + "  " + row.Role_Injured + " - " + row.Player_Team + "-" + row.Number_Injured, fontsize = 12)
        plt.show()
#Let's get the max speed in the last second leading to the tackle
def get_max_speed(row):
    game_key = row.GameKey
    play_id = row.PlayID
    player_id = row.GSISID_Injured
    partner_id = row.Primary_Partner_GSISID
    max_speed_player = 0
    max_speed_partner = 0

    player_positions = ngs_all_data.loc[(ngs_all_data.GameKey == game_key) & (ngs_all_data.PlayID == play_id) & (ngs_all_data.GSISID == player_id)]
    partner_positions = ngs_all_data.loc[(ngs_all_data.GameKey == game_key) & (ngs_all_data.PlayID == play_id) & (ngs_all_data.GSISID == partner_id)]
    tackle_time = player_positions[(player_positions.Event == "tackle")].Time

    if not tackle_time.empty:
        tackle_time = tackle_time.iloc[0]
        player_positions = player_positions.loc[(player_positions.Time <= tackle_time) & (player_positions.Time >= tackle_time + datetime.timedelta(seconds = -1))]
        max_speed_player = player_positions["Speed m/s"].max()
        
        if not partner_positions.empty:
            partner_positions = partner_positions.loc[(partner_positions.Time <= tackle_time) & (partner_positions.Time >= tackle_time + datetime.timedelta(seconds = -1))]
            max_speed_partner = partner_positions["Speed m/s"].max()    
    
    row["Speed_Player"] = max_speed_player
    row["Speed_Partner"] = max_speed_partner
    
    return row

concussion_data = concussion_data.apply(get_max_speed, axis = "columns")

speed_df = concussion_data.loc[concussion_data.Speed_Player > 0].loc[:, ["GSISID_Injured", "Speed_Player", "Primary_Partner_GSISID", "Speed_Partner"]].sort_values(by = "Speed_Player", ascending = False)
speed_df.columns = ["Injured_Player_GSISID", "Speed_Player", "Partner_GSISID", "Speed_Partner"]
speed_df