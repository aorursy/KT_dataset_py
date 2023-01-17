import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence
data = pd.read_csv("../input/shot_logs.csv")
data.head()
data.info()
data.drop(["CLOSEST_DEFENDER_PLAYER_ID", "GAME_ID", "player_name", "player_id"], axis=1, inplace=True)
match_info = data["MATCHUP"].str.split(" - ", n=1, expand=True)
data["MONTH"] = pd.to_datetime(match_info[0]).dt.month
teams = match_info[1].str.split(" ", expand=True)
teams[0], teams[2] = np.where(teams[1] == "@", [teams[2], teams[0]], [teams[0], teams[2]])
data["HOME_TEAM"] = teams[0]
data["AWAY_TEAM"] = teams[2]
data["TEAM"] = np.where(data["LOCATION"] == "H", data["HOME_TEAM"], data["AWAY_TEAM"])
data["OPPOSING_TEAM"] = np.where(data["LOCATION"] == "H", data["AWAY_TEAM"], data["HOME_TEAM"])
data.drop(["MATCHUP", "HOME_TEAM", "AWAY_TEAM"], axis=1, inplace=True)
period_time = pd.to_datetime(data["GAME_CLOCK"], format="%M:%S")
data.drop("GAME_CLOCK", axis=1, inplace=True)
data["PERIOD_CLOCK"] = period_time.dt.minute*60 + period_time.dt.second
pd.get_dummies(data["SHOT_RESULT"])["made"].astype("bool").equals(data["FGM"].astype("bool"))
data.drop("SHOT_RESULT", axis=1, inplace=True)
data.drop("PTS", axis=1, inplace=True)
data.drop("W", axis=1, inplace=True)
data.describe(include="all")
ax = sns.countplot(x="FGM", data=data)
ax.set_xticklabels(["Missed", "Made"])
ax.set_xlabel("Shot")
ax.set_ylabel("Total")
fig, axarr = plt.subplots(1, 2, figsize=(24,8))
sns.countplot(x="LOCATION", data=data, ax=axarr[0])
axarr[0].set_xticklabels(["Away", "Home"])
axarr[0].set_xlabel("Game Location")
axarr[0].set_ylabel("Total")
location = pd.crosstab(data["LOCATION"], data["FGM"]).reset_index()
location["Success_Rate"] = location[1] / (location[0] + location[1])
sns.barplot(location["LOCATION"], location["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[0].set_xticklabels(["Away", "Home"])
axarr[1].set_xlabel("Game Location")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24,8))
sns.distplot(data["FINAL_MARGIN"], kde=False, ax=axarr[0])
axarr[0].set_xlabel("Final Score Margin")
axarr[0].set_ylabel("Total")
final_margin = pd.crosstab(data["FINAL_MARGIN"], data["FGM"]).reset_index()
final_margin["Success_Rate"] = final_margin[1] / (final_margin[0] + final_margin[1])
sns.regplot(final_margin["FINAL_MARGIN"], final_margin["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Final Score Margin")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.countplot(x="SHOT_NUMBER", data=data, ax=axarr[0])
for label in axarr[0].xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
axarr[0].set_xlabel("In-Game Shot Number")
axarr[0].set_ylabel("Total")
shot_number = pd.crosstab(data["SHOT_NUMBER"], data["FGM"]).reset_index()
shot_number["Success_Rate"] = shot_number[1] / (shot_number[0] + shot_number[1])
sns.regplot(shot_number["SHOT_NUMBER"], shot_number["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("In-Game Shot Number")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.countplot(x="PERIOD", data=data, ax=axarr[0])
axarr[0].set_xticklabels(["1", "2", "3", "4", "OT1", "OT2", "OT3"])
axarr[0].set_xlabel("Quarter")
axarr[0].set_ylabel("Total")
quarter = pd.crosstab(data["PERIOD"], data["FGM"]).reset_index()
quarter["Success_Rate"] = quarter[1] / (quarter[0] + quarter[1])
sns.barplot(quarter["PERIOD"], quarter["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Quarter")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.distplot(data["SHOT_CLOCK"].dropna(), kde=False, ax=axarr[0])
axarr[0].set_xlabel("Time Remaining on Shot Clock (s)")
axarr[0].set_ylabel("Total")
axarr[0].set_xlim((0, 24))
shot_clock = pd.crosstab(data["SHOT_CLOCK"], data["FGM"]).reset_index()
shot_clock["Success_Rate"] = shot_clock[1] / (shot_clock[0] + shot_clock[1])
sns.regplot(shot_clock["SHOT_CLOCK"], shot_clock["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].invert_xaxis()
axarr[1].set_xlabel("Time Remaining on Shot Clock (s)")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.distplot(data["DRIBBLES"], kde=False, ax=axarr[0])
axarr[0].set_xlabel("Number of Dribbles Prior to Shot")
axarr[0].set_ylabel("Total")
axarr[0].set_xlim((0, axarr[0].get_xlim()[1]))
dribbles = pd.crosstab(data["DRIBBLES"], data["FGM"]).reset_index()
dribbles["Success_Rate"] = dribbles[1] / (dribbles[0] + dribbles[1])
sns.regplot(dribbles["DRIBBLES"], dribbles["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Number of Dribbles Prior to Shot")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
data["TOUCH_TIME"] = data["TOUCH_TIME"].clip_lower(0)
sns.distplot(data["TOUCH_TIME"], kde=False, ax=axarr[0])
axarr[0].set_xlabel("Ball Possession Time Prior to Shot (s)")
axarr[0].set_ylabel("Total")
axarr[0].set_xlim((0, axarr[0].get_xlim()[1]))
touch_time = pd.crosstab(data["TOUCH_TIME"], data["FGM"]).reset_index()
touch_time["Success_Rate"] = touch_time[1] / (touch_time[0] + touch_time[1])
sns.regplot(touch_time["TOUCH_TIME"], touch_time["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Ball Possession Time Prior to Shot (s)")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.distplot(data["SHOT_DIST"], kde=False, ax=axarr[0])
axarr[0].set_xlabel("Shot Distance")
axarr[0].set_ylabel("Total")
axarr[0].set_xlim((0, axarr[0].get_xlim()[1]))
shot_distance = pd.crosstab(data["SHOT_DIST"], data["FGM"]).reset_index()
shot_distance["Success_Rate"] = shot_distance[1] / (shot_distance[0] + shot_distance[1])
sns.regplot(shot_distance["SHOT_DIST"], shot_distance["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Shot Distance")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
data["PTS_TYPE"] = data["PTS_TYPE"].clip_lower(0)
sns.countplot(x="PTS_TYPE", data=data, ax=axarr[0])
axarr[0].set_xticklabels(["Two Point", "Three Point"])
axarr[0].set_xlabel("Shot Type")
axarr[0].set_ylabel("Total")
shot_type = pd.crosstab(data["PTS_TYPE"], data["FGM"]).reset_index()
shot_type["Success_Rate"] = shot_type[1] / (shot_type[0] + shot_type[1])
sns.barplot(shot_type["PTS_TYPE"], shot_type["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Shot Type")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.countplot(x="CLOSEST_DEFENDER", data=data, ax=axarr[0])
axarr[0].get_xaxis().set_ticks([])
axarr[0].set_xticklabels("")
axarr[0].set_xlabel("Defender")
axarr[0].set_ylabel("Total")
defender = pd.crosstab(data["CLOSEST_DEFENDER"], data["FGM"]).reset_index()
defender["Success_Rate"] = defender[1] / (defender[0] + defender[1])
defender.sort_values("Success_Rate", inplace=True)
sns.barplot(defender["CLOSEST_DEFENDER"], 
            defender["Success_Rate"], 
            ax=axarr[1])
axarr[1].get_xaxis().set_ticks([])
axarr[1].set_xticklabels("")
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Defender")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.distplot(data["CLOSE_DEF_DIST"], kde=False, ax=axarr[0])
axarr[0].set_xlabel("Defender Distance from Shooter")
axarr[0].set_ylabel("Total")
axarr[0].set_xlim((0, axarr[0].get_xlim()[1]))
defender_distance = pd.crosstab(data["CLOSE_DEF_DIST"], data["FGM"]).reset_index()
defender_distance["Success_Rate"] = (defender_distance[1] / 
                                     (defender_distance[0] + 
                                      defender_distance[1]))
sns.regplot(defender_distance["CLOSE_DEF_DIST"], 
            defender_distance["Success_Rate"], 
            ax=axarr[1])
axarr[1].invert_xaxis()
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Defender Distance from Shooter")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.countplot(x="MONTH", data=data, order=[10, 11, 12, 1, 2, 3], ax=axarr[0])
axarr[0].set_xticklabels(["October", 
                          "November", 
                          "December", 
                          "January", 
                          "February", 
                          "March"])
axarr[0].set_xlabel("Month of Match")
axarr[0].set_ylabel("Total")
month = pd.crosstab(data["MONTH"], data["FGM"])
month["Success_Rate"] = month[1] / (month[0] + month[1])
month = month.reindex([10, 11, 12, 1, 2, 3]).reset_index().reset_index()
sns.barplot(month["index"], 
            month["Success_Rate"], 
            ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xticklabels(["October", 
                          "November", 
                          "December", 
                          "January", 
                          "February", 
                          "March"])
axarr[1].set_xlabel("Month of Match")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.countplot(x="TEAM", 
              data=data, 
              order=data["TEAM"].value_counts().sort_index().index, 
              ax=axarr[0])
axarr[0].set_xticklabels(labels=axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel("Team")
axarr[0].set_ylabel("Total")
team = pd.crosstab(data["TEAM"], data["FGM"]).reset_index()
team["Success_Rate"] = team[1] / (team[0] + team[1])
team.sort_values(by="Success_Rate", inplace=True)
sns.barplot(team["TEAM"], team["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xticklabels(labels=axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel("Team")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.countplot(x="OPPOSING_TEAM", 
              data=data, 
              order=data["OPPOSING_TEAM"].value_counts().sort_index().index, 
              ax=axarr[0])
axarr[0].set_xticklabels(labels=axarr[0].get_xticklabels(), rotation=90)
axarr[0].set_xlabel("Opposing Team")
axarr[0].set_ylabel("Total")
opponent = pd.crosstab(data["OPPOSING_TEAM"], data["FGM"]).reset_index()
opponent["Success_Rate"] = opponent[1] / (opponent[0] + opponent[1])
opponent.sort_values(by="Success_Rate", inplace=True)
sns.barplot(opponent["OPPOSING_TEAM"], opponent["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xticklabels(labels=axarr[1].get_xticklabels(), rotation=90)
axarr[1].set_xlabel("Opposing Team")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
fig, axarr = plt.subplots(1, 2, figsize=(24, 8))
sns.distplot(data["PERIOD_CLOCK"], kde=False, ax=axarr[0])
axarr[0].set_xlim((0, axarr[0].get_xlim()[1]))
axarr[0].set_xlabel("Time Remaining in Quarter (s)")
axarr[0].set_ylabel("Total")
quarter_clock = pd.crosstab(data["PERIOD_CLOCK"], data["FGM"]).reset_index()
quarter_clock["Success_Rate"] = quarter_clock[1] / (quarter_clock[0] + quarter_clock[1])
sns.regplot(quarter_clock["PERIOD_CLOCK"], quarter_clock["Success_Rate"], ax=axarr[1])
axarr[1].set_ylim((0, 1))
axarr[1].set_xlabel("Time Remaining in Quarter (s)")
axarr[1].set_ylabel("% of Shots Made")
fig.tight_layout()
data["SHOT_CLOCK"].fillna(data["SHOT_CLOCK"].mean(), inplace=True)
ml_data = data[["FGM", 
                "FINAL_MARGIN", 
                "SHOT_NUMBER", 
                "PERIOD", 
                "SHOT_CLOCK", 
                "DRIBBLES", 
                "TOUCH_TIME", 
                "SHOT_DIST", 
                "PTS_TYPE", 
                "CLOSE_DEF_DIST"]]
X_data = ml_data.drop("FGM", axis=1)
y_data = ml_data["FGM"]
X_train, X_test, y_train, y_test = train_test_split(X_data, 
                                                    y_data, 
                                                    test_size=0.2, 
                                                    random_state=0)
cv_model = GridSearchCV(GradientBoostingClassifier(random_state=0), 
                        param_grid={"learning_rate": np.logspace(-3, -1, 3)}).fit(X_train, 
                                                                                  y_train)
cv_model.best_params_
model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
model.score(X_test, y_test)
features = X_train.columns
feature_significance = pd.DataFrame({"Feature": features, 
                                     "Importance": model.feature_importances_})
feature_significance.sort_values("Importance", 
                                 ascending=False, 
                                 inplace=True)
feature_significance.reset_index(drop=True, inplace=True)
feature_significance.index += 1
feature_significance.at[1, "Feature"] = "Shot Distance"
feature_significance.at[2, "Feature"] = "Defender Distance from Shooter"
feature_significance.at[3, "Feature"] = "Ball Possession Time Prior to Shot (s)"
feature_significance.at[4, "Feature"] = "Final Margin"
feature_significance.at[5, "Feature"] = "Time Remaining on Shot Clock (s)"
feature_significance.at[6, "Feature"] = "Number of Dribbles Prior to Shot"
feature_significance.at[7, "Feature"] = "Shot Type"
feature_significance.at[8, "Feature"] = "Shot Number"
feature_significance.at[9, "Feature"] = "Quarter"
ax = sns.barplot(feature_significance["Importance"], feature_significance["Feature"], orient="h")
fig, axarr =plt.subplots(3, 3, figsize=(24, 16))
for feature in range(len(features)):
    if feature < 3:
        axis = axarr[0][feature]
    elif feature < 6:
        axis = axarr[1][feature-3]
    else:
        axis = axarr[2][feature-6]
    sns.regplot(partial_dependence(model, 
                                   [feature], 
                                   X=X_train)[1][0], 
                partial_dependence(model, 
                                   [feature], 
                                   X=X_train)[0][0], 
                ax=axis)
    axis.set_ylim((-1, 1))
axarr[0][0].set_xlabel("Final Margin")
axarr[0][1].set_xlabel("In-Game Shot Number")
axarr[0][2].set_xlabel("Quarter")
axarr[1][0].invert_xaxis()
axarr[1][0].set_xlabel("Time Remaining on Shot Clock (s)")
axarr[1][1].set_xlabel("Number of Dribbles Prior to Shot")
axarr[1][2].set_xlabel("Ball Possession Time Prior to Shot (s)")
axarr[2][0].set_xlabel("Shot Distance")
axarr[2][1].set_xlabel("Shot Type")
axarr[2][2].set_xlabel("Defender Distance from Shooter")
