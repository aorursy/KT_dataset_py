# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../input/nba-shot-logs/shot_logs.csv")
df.describe()
df.columns
df.head(30)
#check for 1:1 assignments of player_name and player_id

df[['player_name', 'player_id']].nunique()
df[['player_name', 'player_id']].value_counts()
df = df.sort_values(by=["GAME_ID","player_id","SHOT_NUMBER"])
df.tail(30)
#check for 1:1 assignments of GAME_ID and MATCHUP

df[['GAME_ID', 'MATCHUP']].nunique()
df[['GAME_ID', 'MATCHUP']].groupby("GAME_ID").nunique().describe()
# look at one sample

df[df["GAME_ID" ] == 21400899]['MATCHUP'].unique()
df = df.drop(columns=['MATCHUP', 'FINAL_MARGIN', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSEST_DEFENDER',

       'CLOSEST_DEFENDER_PLAYER_ID', 'CLOSE_DEF_DIST', 'FGM', 'PTS'])
df.head(50)
df["SHOT_RESULT_BINARY"] = df["SHOT_RESULT"].map({"missed": 0, "made": 1})

df["SHOT_RESULT"] = df["SHOT_RESULT"].map({"missed": -1, "made": 1})
streak = [0,-1]

for i in range(2, df.shape[0]-2):

    # each player starts with a 0 streak

    if df.at[i, "SHOT_NUMBER"] == 1:

        streak.append(0)

    # start new series when shot result changes

    elif df.at[i-2, "SHOT_RESULT"] != df.at[i-1, "SHOT_RESULT"]:

        streak.append(df.at[i-1, "SHOT_RESULT"]) 

    # increment streak (because shot result is the same)

    else:

        streak.append(df.at[i-1, "SHOT_RESULT"] + streak[-1])



df["streak"] = pd.Series(streak)
print(df[["streak"]].describe())

print(df.groupby("streak")["SHOT_RESULT_BINARY"].describe())

plt.bar(df["streak"].value_counts().index, df["streak"].value_counts())

plt.legend()

plt.show()
streak_pct = pd.Series(df.groupby("streak")["SHOT_RESULT_BINARY"].mean()).drop(index=range(-13,-9)).drop(index=range(9,14))
plt.scatter(streak_pct.index, streak_pct, label="Shooting percentage with streaks")

plt.xticks(streak_pct.index)

plt.xlabel("Streak")

plt.ylabel("Shooting Percentage")

plt.legend()
plt.scatter(streak_pct.index, streak_pct, label="Shooting percentage with streaks")

plt.xticks(streak_pct.index)

plt.xlabel("Streak")

plt.ylabel("Shooting Percentage")

plt.ylim(0,1)

plt.legend()
from scipy import stats

import numpy as np



slope, intercept, r_value, p_value, std_err = stats.linregress(streak_pct.index, streak_pct)



x_reg = np.linspace(-9, 9)

y_reg = x_reg*slope+intercept



plt.scatter(streak_pct.index, streak_pct, label="Shooting percentage with streaks")

plt.plot(x_reg, y_reg, label="regression line", color="r")

plt.xticks(streak_pct.index)

plt.xlabel("Streak")

plt.ylabel("Shooting Percentage")

plt.legend()



print(f"slope:\t\t{slope},\nintercept:\t{intercept}\nr_value:\t{r_value}\np_value:\t{p_value}\nstd_err:\t{std_err}")
plt.scatter(streak_pct.index, streak_pct, label="Shooting percentage with streaks")

plt.plot(x_reg, y_reg, label="regression line", color="r")

plt.xticks(streak_pct.index)

plt.ylim(0,1)

plt.xlabel("Streak")

plt.ylabel("Shooting Percentage")

plt.legend()
df.head(30)