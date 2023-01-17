from typing import Any, List, Callable, Union



# Data Management

import numpy as np 

import pandas as pd 

import scipy



pd.set_option('max_columns', 100)

pd.set_option('max_rows', 50)



# Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.express as px

from IPython.display import HTML, Image





# Managing Warnings 

import warnings

warnings.filterwarnings('ignore')



# Plot Figures Inline

%matplotlib inline



# Extras

import math, string, os, datetime, dateutil



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_players = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/players.csv')

print(df_players.shape)

df_players.tail()
def height_to_numerical(height):

    """

    Convert string representing height into total inches

    

    Ex. '5-11' --> 71

    Ex. '6-3'  --> 75

    """  

    feet   = height.split('-')[0]

    inches = height.split('-')[1]

    return int(feet)*12 + int(inches)





def clean_height(val):

    try:

        # height is already in inches

        height = int(val)

    except:

        # convert it from string

        height = height_to_numerical(val)



    return height
def calculate_age(birthDate):

    today = datetime.date.today()

    age = dateutil.relativedelta.relativedelta(today, birthDate)

    return age.years + (age.months / 12)
def clean_players_data(df):

    df_players["height"] = df_players["height"].apply(clean_height)

    df_players["birthDate"] = pd.to_datetime(df_players["birthDate"])

    df_players["age"] = df_players["birthDate"].apply(calculate_age)

    return df_players.set_index("nflId")

    

df_players_cleaned = clean_players_data(df_players)
df_players_cleaned.info()
df_players_cleaned.describe(include=["O"])
df_players_cleaned.describe()
def get_bar_trace(

    *,

    df: pd.DataFrame, 

    column: str, 

    num_entries: int = 20,

    colorscale: str = "Portland",

    orientation: str = "h",

) -> go.Bar:

    data = df[column].value_counts()[:num_entries][::-1]    

    x = data.values if orientation == "h" else data.index

    y = data.index if orientation == "h" else data.values



    return go.Bar(

        x=x,

        y=y,

        name=column,

        marker=dict(

            color=data.values,

            line=dict(color="black", width=1.5),

            colorscale=px.colors.diverging.Portland,      

        ),

        text=data.values,

        textposition="auto",

        orientation="h",

        showlegend=False,

    )





def get_hist_trace(

    *, 

    df: pd.DataFrame, 

    column: str,

    color: str = "dodgerblue",

) -> go.Histogram:

    return go.Histogram(

        x=df[column],

        opacity=0.75,

        name=column,

        marker=dict(

            color=color,

            line=dict(color="black", width=1.5),   

        ),

        text=df[column].values, 

#         histnorm="probability"

    )





def get_scatter_trace(

    *, 

    df: pd.DataFrame, 

    column: str,

    color: str = "dodgerblue",

) -> go.Scatter:

    data = df[column]

    kde = scipy.stats.kde.gaussian_kde(data.values)    

    x = np.linspace(min(data.values), max(data.values), len(data.values))

    y = [val * len(data.values) for val in kde(x)]  # denormalize

    return go.Scatter(

        x=x, 

        y=y,

        marker=dict(

            size=6,

            color=color,

        ),

        showlegend=False

    )

def plotly_distributions(

    df: pd.DataFrame, 

    height: int = 1000, 

    width: int = 1500,    

    cols: int = 3,

    horizontal_spacing: float = 0.2,

    vertical_spacing: float = 0.3,

    colorscale: List[str] = px.colors.diverging.Portland,

) -> None: 

    rows = math.ceil(float(df.shape[1]) / cols)

    fig = plotly.subplots.make_subplots(

        rows=rows, 

        cols=cols,

        horizontal_spacing=horizontal_spacing,

        vertical_spacing=vertical_spacing,

        subplot_titles=df.columns,

    )

    

    for i, column in enumerate(df.columns):

        row = math.ceil((i + 1) / cols)

        col = (i % cols) + 1

        if df.dtypes[column] == np.object:

            fig.add_trace(

                get_bar_trace(

                    df=df, 

                    column=column, 

                    colorscale=colorscale

                ), 

                row=row, 

                col=col

            )

            fig.update_xaxes(title_text="Count", row=row, col=col)

        else:

#             distplfig = ff.create_distplot(

#                 [df[column]], 

#                 group_labels=[column], 

#                 colors=colorscale,

#                 bin_size=.2, 

#                 show_rug=False,

#             )



#             for k in range(len(distplfig.data)):

#                 fig.add_trace(

#                     distplfig.data[k],

#                     row=row, 

#                     col=col,

#                 )            

            fig.add_trace(

                get_hist_trace(

                    df=df, 

                    column=column, 

                    color=colorscale[i % len(colorscale)]

                ), 

                row=row, 

                col=col

            )

            fig.add_trace(

                get_scatter_trace(

                    df=df, 

                    column=column, 

                    color=colorscale[(i + 1) % len(colorscale)]                    

                ), 

                row=row, 

                col=col,

            )

            fig.update_xaxes(title_text=column, row=row, col=col)

            fig.update_yaxes(title_text="Count", row=row, col=col)

            

    fig.update_layout(

        height=height, 

        width=width

    )



    iplot(fig)
columns_to_plot = [col for col in df_players.columns if col not in ["nflId", "birthDate", "displayName"]]



plotly_distributions(df_players[columns_to_plot], horizontal_spacing=0.10, vertical_spacing=0.15)
df_wr = df_players[df_players["position"] == "WR"]

columns_to_plot = [col for col in df_players.columns if col not in ["nflId", "birthDate", "displayName", "position"]]

plotly_distributions(df_wr[columns_to_plot], cols=2, horizontal_spacing=0.10, vertical_spacing=0.15)
# df_wr = df_players[(df_players["position"] == "RB") | (df_players["position"] == "FB")]

# columns_to_plot = [col for col in df_players.columns if col not in ["nflId", "birthDate", "displayName", "position"]]

# plotly_distributions(df_wr[columns_to_plot], cols=2, horizontal_spacing=0.10, vertical_spacing=0.15)
# df_wr = df_players[df_players["position"] == "QB"]

# columns_to_plot = [col for col in df_players.columns if col not in ["nflId", "birthDate", "displayName", "position"]]

# plotly_distributions(df_wr[columns_to_plot], cols=2, horizontal_spacing=0.10, vertical_spacing=0.15)
# df_wr = df_players[df_players["position"] == "TE"]

# columns_to_plot = [col for col in df_players.columns if col not in ["nflId", "birthDate", "displayName", "position"]]

# plotly_distributions(df_wr[columns_to_plot], cols=2, horizontal_spacing=0.10, vertical_spacing=0.15)
# df_wr = df_players[df_players["position"].isin(["DT", "DE", "NT"])]

# columns_to_plot = [col for col in df_players.columns if col not in ["nflId", "birthDate", "displayName", "position"]]

# plotly_distributions(df_wr[columns_to_plot], cols=2, horizontal_spacing=0.10, vertical_spacing=0.15)
# df_wr = df_players[df_players["position"].isin(["LB", "ILB", "OLB", "MLB"])]

# columns_to_plot = [col for col in df_players.columns if col not in ["nflId", "birthDate", "displayName", "position"]]

# plotly_distributions(df_wr[columns_to_plot], cols=2, horizontal_spacing=0.10, vertical_spacing=0.15)
# df_wr = df_players[df_players["position"].isin(["SS", "FS", "CB", "DB"])]

# columns_to_plot = [col for col in df_players.columns if col not in ["nflId", "birthDate", "displayName", "position"]]

# plotly_distributions(df_wr[columns_to_plot], cols=2, horizontal_spacing=0.10, vertical_spacing=0.15)
df_games = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/games.csv')

print(df_games.shape)

df_games.head()
def clean_games_df(df):

    df["gameDate"] = pd.to_datetime(df["gameDate"])

    return df.set_index("gameId")

df_games_cleaned = clean_games_df(df_games)
df_games_cleaned.info()
def plot_bar_chart(

    *,

    x: List[Any],

    y: List[Any],

    name: str,

    title: str,

    xaxis_title: str,

    yaxis_title: str,    

    colorscale: List[str] = px.colors.diverging.Portland,

) -> None:

    trace = go.Bar(

        x=x,

        y=y,

        name=name,

        marker=dict(

            color=y,

            line=dict(color="black", width=1.5),

            colorscale=px.colors.diverging.Portland,

        ),

        text=y,

        textposition="auto",

        orientation="v",

    )

    layout = go.Layout(

        title=title, 

        xaxis=dict(title=xaxis_title), 

        yaxis=dict(title=yaxis_title)

    )

    fig = go.Figure(data=[trace], layout=layout)

    fig.update_xaxes(type='category')    

    iplot(fig)    
df_grouped_by_week = df_games_cleaned.groupby(by=["week"]).count()

data = df_grouped_by_week["gameDate"]



plot_bar_chart(

    x=data.index, 

    y=data.values, 

    name="Weekly Game Count", 

    title="Weekly Game Count", 

    xaxis_title="Week",

    yaxis_title="Number of Games",

)
data = df_games_cleaned["gameTimeEastern"].value_counts()



plot_bar_chart(

    x=data.index, 

    y=data.values, 

    name="Number of Games Per Start Time", 

    title="Number of Games Per Start Time",

    xaxis_title="Game Time",

    yaxis_title="Number of Games",

)
# teamAbbrevs = list(set(df_games["homeTeamAbbr"].tolist() + df_games["visitorTeamAbbr"].values.tolist()))



# df_grouped_by_week = df_games.groupby(by=["week"])[["homeTeamAbbr", "visitorTeamAbbr"]].agg({

#     "homeTeamAbbr": list,

#     "visitorTeamAbbr": list,

# })
df_plays = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/plays.csv')

print(df_plays.shape)

df_plays.head()
df_plays.info()
def convert_game_clock_to_seconds(gameClock: str) -> int:

    # handle NaN values

    try:

        [minutes, seconds, ms] = str(gameClock).split(':')

        total_seconds = int(minutes) * 60 + int(seconds)

        return total_seconds

    except:

        return np.nan
def clean_and_merge_plays_df(df):

    df["gameClock"] = df["gameClock"].apply(convert_game_clock_to_seconds)

    merged_with_games = df.merge(df_games_cleaned, left_on="gameId", right_on=df_games_cleaned.index)

    return merged_with_games
df_merged = clean_and_merge_plays_df(df_plays)

df_merged.head(10)
data = df_merged["down"].value_counts()



trace = go.Bar(

    x=data.index,

    y=data.values,

    marker=dict(

        color=data.values,

        line=dict(color="black", width=1.5),

        colorscale=px.colors.diverging.Portland,

    ),

    text=data.values,

    textposition="auto",

    orientation="v",

)

layout = go.Layout(

    title="Number Of Plays Run By Down",

    yaxis=dict(title="Number of Plays"),

    xaxis=dict(title="Down")

)

fig = go.Figure(data=[trace], layout=layout)

fig.update_xaxes(type='category')

iplot(fig) 


def grouped_bar_chart(

    *, 

    title: str,

    x_values: List[Any], 

    y_values: List[Any],

    labels: List[str],

    xaxis_title: str = "",

    yaxis_title: str = "",

) -> None:

    traces = []

    

    for i, label in enumerate(labels):

        x = x_values

        y = y_values[i]

        

        trace = go.Bar(

            x=x,

            y=y,

            name=label,

            marker=dict(

                color=y,

                line=dict(color="black", width=1.5),

                colorscale=px.colors.diverging.Portland,

            ),

            text=label,

            textposition="auto",

            orientation="v",

            offsetgroup=i,

        )

        traces.append(trace)        



    layout = go.Layout(

        title=title, 

        xaxis=dict(title=xaxis_title), 

        yaxis=dict(title=yaxis_title)

    )

    fig = go.Figure(data=traces, layout=layout)

    fig.update_xaxes(type='category')

    iplot(fig)

    
weekly_down_df = pd.DataFrame(df_merged.groupby(by=["week"])["down"].value_counts()).unstack()

grouped_bar_chart(

    labels=["1st Down", "2nd Down", "3rd Down", "4th Down"],

    x_values=weekly_down_df.index.tolist(), 

    y_values=weekly_down_df.values.transpose().tolist(), 

    title="Plays By Down Week over Week",

    xaxis_title="Week",

    yaxis_title="Number of Plays Run",

    

)
def convert_to_percent(df):

    ret = df.copy()

    for col in ret.columns:

        total = ret[col].sum()

        ret[col] = ret[col] / total

    return ret



weekly_down_percentage = convert_to_percent(

    pd.DataFrame(

        df_merged.groupby(by=["week"])["down"].value_counts()

    ).unstack(level=0)

)
grouped_bar_chart(

    labels=["1st Down", "2nd Down", "3rd Down", "4th Down"],

    x_values=[idx[1] for idx in weekly_down_percentage.transpose().index], 

    y_values=weekly_down_percentage.values.tolist(), 

    title="Percentage of Plays Run By Down Week over Week",

    xaxis_title="Week",

    yaxis_title="Percentage of Total Plays Run",

    

)
def plot_weekly_categorical_values(

    *,

    df: pd.DataFrame,

    column: str,

    title: str = "Formation Count By Week",

    horizontal_spacing=0.10,

    vertical_spacing=0.10,   

) -> None:

    traces = []

    fig = plotly.subplots.make_subplots(

        rows=2, 

        cols=1,

        horizontal_spacing=horizontal_spacing,

        vertical_spacing=vertical_spacing,

        shared_xaxes=True,

        row_heights=[0.4, 0.6],        

    )

    

    uniqs = df[~(pd.isna(df[column]))][column].unique().tolist()

    

    for i, category in enumerate(uniqs):

        df_categorical = df[df[column] == category]

        weekly_counts = df_categorical.groupby(by=["week"]).count()

        x = weekly_counts[column].index

        y = weekly_counts[column].values

        

        fig.add_trace(

            go.Scatter(

                x=x,

                y=y,

                name=category,

            ),

            row=1,

            col=1,

        )        

        

        fig.add_trace(

            go.Scatter(

                x=x,

                y=y,

                mode="none",

                fill="tozeroy" if i == 1 else "tonexty",

                name=category,

                stackgroup="one",

            ),

            row=2,

            col=1,

        )



    fig.update_layout(title=title, xaxis=dict(title="Week"), height=800)

    fig.update_xaxes(type='category')

    fig.show()    

    
plot_weekly_categorical_values(df=df_merged, column="offenseFormation")
plot_weekly_categorical_values(df=df_merged, column="passResult", title="Pass Result By Week")
def plot_weekly_numerical_values(

    *,

    df: pd.DataFrame,

#     columns: List[str],

    title: str,

    horizontal_spacing=0.10,

    vertical_spacing=0.10, 

    aggregator: Union[Callable, str] = np.sum,

    agg_mapping: dict = {},

) -> None:

    traces = []

    fig = plotly.subplots.make_subplots(

        rows=2, 

        cols=1,

        horizontal_spacing=horizontal_spacing,

        vertical_spacing=vertical_spacing,

        shared_xaxes=True,

        row_heights=[0.4, 0.6],        

    )

    

    weekly_data = df.groupby(by=["week"])[list(agg_mapping.keys())].agg(agg_mapping)

    

    for i, col in enumerate(weekly_data.columns):

        data = weekly_data[col]

        x = data.index

        y = data.values        

#         data = df[col]

#         weekly_data = df.groupby(by=["week"])[col].agg({ col: aggregator })

#         x = data.index

#         y = data.values

    

#     for i, category in enumerate(uniqs):

#         df_categorical = df[df[column] == category]

#         weekly_counts = df_categorical.groupby(by=["week"]).agg({ column: aggregator })

#         x = weekly_counts[column].index

#         y = weekly_counts[column].values

        

        fig.add_trace(

            go.Scatter(

                x=x,

                y=y,

                name=col,

            ),

            row=1,

            col=1,

        )        

        

        fig.add_trace(

            go.Scatter(

                x=x,

                y=y,

                mode="none",

#                 fill="tozeroy" if i == 1 else "tonexty",

                fill="tonexty",

                name=col,

                stackgroup='one'

            ),

            row=2,

            col=1,

        )



    fig.update_layout(title=title, xaxis=dict(title="Week"), height=800)

    fig.update_xaxes(type='category')

    fig.show()  
plot_weekly_numerical_values(

    df=df_merged,

    agg_mapping={

#         "offensePlayResult": np.sum,

#         "epa": np.sum,

        "preSnapHomeScore": np.sum,

        "preSnapVisitorScore": np.sum,

    },

    title="Results"

)


# def plot_scatter_matrix(df):

#     data = df.loc[:, ["offensePlayResult", "preSnapHomeScore", "preSnapVisitorScore", "epa"]]

#     data.index = np.arange(1, len(data)+1)



#     fig = ff.create_scatterplotmatrix(

#         data,

#         diag='box', 

#         colormap='Portland',

#         colormap_type='cat',

#         height=700, 

#         width=700,

#     )



#     iplot(fig)

    

# plot_scatter_matrix(df_merged)
data = df_plays.groupby(by=["offenseFormation"])["playId"].count()



trace = go.Bar(

    x=data.index,

    y=data.values,

    marker=dict(

        color=data.values,

        line=dict(color="black", width=1.5),

        colorscale=px.colors.diverging.Portland,      

    ),

    text=data.values,

    textposition="auto",

    orientation="v",        

)

layout = go.Layout(title="Offensive Formation Count")

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
# df_plays.groupby(by=["personnelO"])["playId"].count().sort_values()[::-1]
# df_plays.groupby(by=["personnelD"])["playId"].count().sort_values()[::-1]
df_week_1 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2021/week1.csv")

print(df_week_1.shape)

df_week_1.head()
df_week_1.info()
def clean_weekly_df(df):

    df["time"] = pd.to_datetime(df["time"])

    

    return df.merge(df_merged)
weekly_data_filenames = [

    '/kaggle/input/nfl-big-data-bowl-2021/week1.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week2.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week3.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week4.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week5.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week6.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week7.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week8.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week9.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week10.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week11.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week12.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week13.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week14.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week15.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week16.csv',

    '/kaggle/input/nfl-big-data-bowl-2021/week17.csv'

]
weekly_data = [pd.read_csv(filename) for filename in weekly_data_filenames]



weekly_qb_data = clean_weekly_df(

    pd.concat(

        [df[df["position"] == "QB"] for df in weekly_data],

        axis="index",

    )

)

print(weekly_qb_data.shape)
weekly_qb_data.head()
weekly_qb_data["epa"].describe()
def plot_categorical_bar_chart(

    *,

    x_values: List[Any],

    y_values: List[Any],

    name: str,

    title: str,

    xaxis_title: str,

    yaxis_title: str,    

    colorscale: List[str] = px.colors.diverging.Portland,

) -> None:

    layout = go.Layout(

        title=title, 

        xaxis=dict(title=xaxis_title), 

        yaxis=dict(title=yaxis_title)

    )        

    fig = go.Figure(

        data=[

            go.Bar(

                x=x_values,

                y=y_values,

                text=y_values,

                textposition="auto",

                orientation="v",  

                marker=dict(

                    color=y_values,

                    line=dict(color="black", width=1.5),

                    colorscale=px.colors.diverging.Portland,

                )                

            )

        ], 

        layout=layout,

    )    



    fig.update_xaxes(type='category')    

    iplot(fig)
touchdown_data = weekly_qb_data[(weekly_qb_data["event"] == "touchdown") | (weekly_qb_data["event"] == "pass_outcome_touchdown")]



plot_categorical_bar_chart(

    x_values=touchdown_data.groupby(by=["offenseFormation"])["event"].count().index, 

    y_values=touchdown_data.groupby(by=["offenseFormation"])["event"].count().values, 

    name="Number of Touchdowns Per Formation", 

    title="Number of Touchdowns Per Formation", 

    xaxis_title="Formation",

    yaxis_title="Touchdowns",

)
sack_data = weekly_qb_data[(weekly_qb_data["event"] == "qb_strip_sack") | (weekly_qb_data["event"] == "qb_sack")]



plot_categorical_bar_chart(

    x_values=sack_data.groupby(by=["offenseFormation"])["event"].count().index, 

    y_values=[

        round(val, 3) for val in 

        (

            sack_data.groupby(by=["offenseFormation"])["event"].count().values /

            weekly_qb_data.groupby(by=["offenseFormation"])["event"].count().values * 100

        )

    ],

    name="Percentage of Play Resulting in a Sack By Formation", 

    title="Percentage of Play Resulting in a Sack By Formation", 

    xaxis_title="Formation",

    yaxis_title="Percentage of Plays Resulting in a Sack",



)
interception_data = weekly_qb_data[(weekly_qb_data["passResult"] == "IN")]



plot_bar_chart(

    x=interception_data.groupby(by=["offenseFormation"])["passResult"].count().index, 

    y=[

        round(val, 3) for val in 

        (

            interception_data.groupby(by=["offenseFormation"])["passResult"].count().values /

            weekly_qb_data.groupby(by=["offenseFormation"])["passResult"].count().values * 100

        )

    ],

    name="Percentage of Plays Resulting in an Interception by Formation",

    title="Percentage of Plays Resulting in an Interception by Formation",

    xaxis_title="Formation",

    yaxis_title="Percentage of Plays Resulting in an Interception",

)
def plot_scatterplot(

    *,

    df: pd.DataFrame, 

    x_column: str, 

    y_column: str,

    title: str,

) -> None:

    trace = go.Scattergl(

        x=df[x_column],

        y=df[y_column],

        mode="markers",

    )

    

    layout = go.Layout(

        title=title,

        xaxis=dict(title=x_column),

        yaxis=dict(title=y_column),

    )

    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig)

        
def plot_boxplot(

    *,

    df: pd.DataFrame, 

    x_column: str, 

    y_column: str,

    title: str,

) -> None:    

    data = []

    categorical_labels = df[x_column].unique()

    

    # sort x values

    try:

        are_integers = all([float(l) for l in categorical_labels])

        if are_integers:

            sorted_labels = sorted(categorical_labels, key=lambda x: float(x))

        else:

            sorted_labels = sorted(categorical_labels, key=lambda x: str(x))

    except:

        sorted_labels = sorted(categorical_labels, key=lambda x: str(x))

    

    for i in range(len(categorical_labels)):

        label = sorted_labels[i]

        data.append(

            dict(

                x=label,

                y=df[df[x_column] == label][y_column],

            )

        )

        

    fig = go.Figure()

    for item in data:

        fig.add_trace(

            go.Box(

                y=item["y"],

                name=item["x"],

                line_width=1,

                whiskerwidth=0.2,

            )

        )

        

    fig.update_layout(

        title=title,

        xaxis=dict(title=x_column),

        yaxis=dict(

            title=y_column,

            autorange=True,

            showgrid=True,

            zeroline=True,

            dtick=5,

            gridcolor='rgb(255, 255, 255)',

            gridwidth=1,

            zerolinecolor='rgb(255, 255, 255)',

            zerolinewidth=2,            

        ),

        showlegend=False,

        margin=dict(

            l=40,

            r=30,

            b=80,

            t=100,

        ),        

    )

    fig.update_xaxes(type='category')    

    iplot(fig)

        
plot_boxplot(

    df=weekly_qb_data,

    x_column="numberOfPassRushers",

    y_column="offensePlayResult",

    title="QB Play Results Based on Number of Pass Rushers",

)
plot_boxplot(

    df=weekly_qb_data,

    x_column="offenseFormation",

    y_column="offensePlayResult",

    title="QB Play Results Based on Offensive Formation",

)
plot_boxplot(

    df=weekly_qb_data,

    x_column="defendersInTheBox",

    y_column="offensePlayResult",

    title="QB Play Results Based on Defenders in the Box",

)
plot_boxplot(

    df=weekly_qb_data,

    x_column="offenseFormation",

    y_column="epa",

    title="EPA Based on Offensive Formation",

)