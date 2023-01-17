#%matplotlib inline



import matplotlib.pyplot as plt

from matplotlib.widgets import CheckButtons

import numpy as np

import pandas as pd

import seaborn as sns



from plotly.subplots import make_subplots

import plotly.express as px

from plotly.offline import iplot



#import warnings

#warnings.filterwarnings('ignore')
!ls ../input/nfl-big-data-bowl-2021/
df_week1 = pd.read_csv("../input/nfl-big-data-bowl-2021/week1.csv")

df_week2 = pd.read_csv("../input/nfl-big-data-bowl-2021/week2.csv")

df_week3 = pd.read_csv("../input/nfl-big-data-bowl-2021/week3.csv")

df_week4 = pd.read_csv("../input/nfl-big-data-bowl-2021/week4.csv")

df_week5 = pd.read_csv("../input/nfl-big-data-bowl-2021/week5.csv")

df_week6 = pd.read_csv("../input/nfl-big-data-bowl-2021/week6.csv")

df_week7 = pd.read_csv("../input/nfl-big-data-bowl-2021/week7.csv")

df_week8 = pd.read_csv("../input/nfl-big-data-bowl-2021/week8.csv")

df_week9 = pd.read_csv("../input/nfl-big-data-bowl-2021/week9.csv")

df_week10 = pd.read_csv("../input/nfl-big-data-bowl-2021/week10.csv")

df_week11 = pd.read_csv("../input/nfl-big-data-bowl-2021/week11.csv")

df_week12 = pd.read_csv("../input/nfl-big-data-bowl-2021/week12.csv")

df_week13 = pd.read_csv("../input/nfl-big-data-bowl-2021/week13.csv")

df_week14 = pd.read_csv("../input/nfl-big-data-bowl-2021/week14.csv")

df_week15 = pd.read_csv("../input/nfl-big-data-bowl-2021/week15.csv")

df_week16 = pd.read_csv("../input/nfl-big-data-bowl-2021/week16.csv")

df_week17 = pd.read_csv("../input/nfl-big-data-bowl-2021/week17.csv")
df_weeks = [df_week1,df_week2,df_week3,df_week4,df_week5,df_week6,df_week7,df_week8,df_week9,

            df_week10,df_week11,df_week12,df_week13,df_week14,df_week15,df_week16,df_week17]
df_week1.head()
def visualize_coordinate(df_weeks, displayName):

    df = pd.DataFrame(columns=df_weeks[0].columns)

    

    for i, df_week in enumerate(df_weeks):

        df_tmp = df_week[df_week["displayName"] == displayName].copy()

        df_tmp["week"] = str(i + 1)

        df = pd.concat([df, df_tmp])

    

    fig = px.scatter(df, x="x", y="y", color="week")

    fig.update_layout(height=600, width=800, title_text=f"Coordinate of {displayName} in the game")

    iplot(fig)#fig.show()
def visualize_angle_and_numeric_by_polar(df_weeks, displayName, angle_col, r_col):

    

    df = pd.DataFrame(columns=df_weeks[0].columns)

    

    for i, df_week in enumerate(df_weeks):

        df_tmp = df_week[df_week["displayName"] == displayName].copy()

        df_tmp["week"] = str(i + 1)

        df = pd.concat([df, df_tmp])

        

    fig = px.scatter_polar(df, r=r_col, theta=angle_col ,symbol="week", color="week",

                       color_discrete_sequence=px.colors.sequential.Plasma_r)



    fig.update_layout(height=600, width=800, title_text=f"{angle_col} and {r_col} polar plot of {displayName}")

    iplot(fig)#.show()
df_week1.info()
sns.scatterplot(data=df_week1[df_week1["displayName"] == "Leonard Williams"], x="x", y="y")
ax = plt.subplot(111, polar=True)

ax.set_theta_zero_location('N')

ax.set_theta_direction(-1)

plt.show()
set(df_week1["position"])
list(set(df_week1[df_week1["position"].isin(["DL", "DE"])]["displayName"]))[:10]
df_week1[df_week1["displayName"] == "Patrick Ricard"].head(1)
visualize_coordinate(df_weeks, "Patrick Ricard")
#visualize_angle_and_numeric_by_polar(df_weeks, "Patrick Ricard", "o", "s")
#visualize_angle_and_numeric_by_polar(df_weeks, "Patrick Ricard", "o", "a")
#visualize_angle_and_numeric_by_polar(df_weeks, "Patrick Ricard", "dir", "s")
visualize_angle_and_numeric_by_polar(df_weeks, "Patrick Ricard", "dir", "a")
list(set(df_week1[df_week1["position"].isin(["LB", "OLB", "MLB"])]["displayName"]))[:10]
df_week1[df_week1["displayName"] == "Nate Gerry"].head(1)
visualize_coordinate(df_weeks, "Nate Gerry")
#visualize_angle_and_numeric_by_polar(df_weeks, "Nate Gerry", "o", "s")
#visualize_angle_and_numeric_by_polar(df_weeks, "Nate Gerry", "o", "a")
#visualize_angle_and_numeric_by_polar(df_weeks, "Nate Gerry", "dir", "s")
visualize_angle_and_numeric_by_polar(df_weeks, "Nate Gerry", "dir", "a")
db_players = list(set(df_week1[df_week1["position"].isin(["CB", "FS", "SS", "S", "DB"])]["displayName"]))

db_players[:10]
print(f"Number of players in DB position is {len(db_players)}")
df_week1[df_week1["displayName"] == "T.J. Carrie"].head(1)
visualize_coordinate(df_weeks, "T.J. Carrie")
#visualize_angle_and_numeric_by_polar(df_weeks, "T.J. Carrie", "o", "s")
visualize_angle_and_numeric_by_polar(df_weeks, "T.J. Carrie", "o", "a")
#visualize_angle_and_numeric_by_polar(df_weeks, "T.J. Carrie", "dir", "s")
visualize_angle_and_numeric_by_polar(df_weeks, "T.J. Carrie", "dir", "a")