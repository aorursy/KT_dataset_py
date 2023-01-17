import os

import zipfile

import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import plotly

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn.linear_model import LinearRegression

import datetime

import colorlover as cl



plt.style.use('ggplot')

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
# Format the data

df = pd.read_csv('../input/riiid-leaderboad/leaderboard.csv')

df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])

df = df.set_index('SubmissionDate')

df.columns = [name for name in df.columns]

df.drop(columns=['Unnamed: 0', 'Unnamed: 2'], inplace=True)

df.drop_duplicates(inplace=True)



FIFTEENTH_SCORE = df.max().sort_values(ascending=False)[15]

FIFTYTH_SCORE = df.max().sort_values(ascending=False)[50]

TOP_SCORE = df.max().sort_values(ascending=False)[0]
# Interative Plotly

mypal = cl.scales['9']['div']['Spectral']

colors = cl.interp( mypal, 15 )

annotations = []

init_notebook_mode(connected=True)

TOP_TEAMS = df.max().loc[df.max() > FIFTEENTH_SCORE].index.values

df_filtered = df[TOP_TEAMS].ffill()

df_filtered = df_filtered.iloc[df_filtered.index >= df.index.min()]

team_ordered = df_filtered.max(axis=0).sort_values(ascending=False).index.tolist()



data = []

i = 0

for col in df_filtered[team_ordered].columns:

    data.append(go.Scatter(x = df_filtered.index, y = df_filtered[col], name=col, line=dict(color=colors[i], width=2),))

    i += 1



annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05, xanchor='left', yanchor='bottom', 

                        text='Top Teams Public Leaderboard',

                        font=dict(family='Arial', size=30, color='rgb(37,37,37)'), showarrow=False))



layout = go.Layout(yaxis=dict(range=[FIFTEENTH_SCORE-0.0001, TOP_SCORE+0.0001]), hovermode='x', plot_bgcolor='white', annotations=annotations)

fig = go.Figure(data=data, layout=layout)

fig.update_layout(

    legend=go.layout.Legend(

        traceorder="normal",

        font=dict(family="sans-serif", size=12, color="black"),

        bgcolor="LightSteelBlue",

        bordercolor="Black",

        borderwidth=2,

    )

)



fig.update_layout(legend_orientation="h")

fig.update_layout(template="plotly_white")

fig.update_xaxes(showgrid=False)



iplot(fig)
# Scores of top teams over time

plt.rcParams["font.size"] = "12"

ALL_TEAMS = df.columns.values[1:]

df_ffill = df[ALL_TEAMS].ffill()



df_ffill.plot(figsize=(20, 10), color=color_pal[0], legend=False, alpha=0.05, 

              xlim=('10/06/2020', df_ffill.index.max()),

              ylim=(0.495, TOP_SCORE+0.01), 

              title='All Teams Public Leaderboard Scores over Time')



df_ffill.max(axis=1).plot(color=color_pal[1], label='1st Place Public LB', legend=True)



df['sample_submission.csv'] = 0.5

df['sample_submission.csv'].plot(color='k', label='Sample Submission', legend=True)



plt.show()
plt.rcParams["font.size"] = "13"

ax = df.ffill().count(axis=1).plot(figsize=(20, 8), title='Number of Teams in the Competition by Date', color=color_pal[5], lw=5)

ax.set_ylabel('Number of Teams')

plt.show()
plt.rcParams["font.size"] = "12"

# Create Top Teams List

TOP_TEAMS = df.max().loc[df.max() > FIFTYTH_SCORE].index.values

df[TOP_TEAMS].max().sort_values(ascending=True).plot(kind='barh',

                                                     xlim=(FIFTYTH_SCORE-0.0005, TOP_SCORE+0.0005),

                                                     title='Top 50 Public LB Teams',

                                                     figsize=(12, 15),

                                                     color=color_pal[1])

plt.show()
plt.rcParams["font.size"] = "7"

n_weeks = (datetime.date.today() - datetime.date(2020, 10, 6)).days #/ 7 # Num days of the comp

n_weeks = int(n_weeks)

fig, axes = plt.subplots(n_weeks, 1, figsize=(15, 25), sharex=True)

#plt.subplots_adjust(top=8, bottom=2)

for x in range(n_weeks):

    date2 = df.loc[df.index.date == datetime.date(2020, 10, 6) + datetime.timedelta(x+1)].index.min()

    num_teams = len(df.ffill().loc[date2].dropna())

    max_cutoff = df.ffill().loc[date2] > 0.5

    df.ffill().loc[date2].loc[max_cutoff].plot(kind='hist',

                               bins=50,

                               ax=axes[x],

                               title='{} ({} Teams)'.format(date2.date().isoformat(), num_teams), xlim=(0.5, TOP_SCORE + 0.005))

    y_axis = axes[x].yaxis

    y_axis.set_label_text('')

    y_axis.label.set_visible(False)
%%capture

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

import matplotlib.pylab as plt



import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML



import matplotlib.colors as mcolors



import seaborn as sns
mypal = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Grab the color pal

cm = plt.get_cmap('tab20')



NUM_COLORS = 20

mypal = [mcolors.to_hex(cm(1.*i/NUM_COLORS)) for i in range(NUM_COLORS)]



my_df = df.T



min_sub_dict = {}

for c in df.columns:

    min_sub_dict[c] =  df[c].dropna().index.min()

    



my_df['colors'] = [np.random.choice(mypal) for c in range(len(my_df))]

color_map = my_df['colors'].to_dict()
def draw_barchart(mydate):

    mydate = pd.to_datetime(mydate)

    dff = df_ffill.loc[df_ffill.index <= mydate].iloc[-1].sort_values(ascending=True).dropna().tail(25)



    last_sub_date = {}

    df2 = df.loc[df.index <= mydate]

    for c in df2.columns:

        last_sub_date[c] = df2[c].dropna().index.max()



    ax.clear()

    ax.barh(dff.index, dff.values, color=[color_map[x] for x in dff.index])

    ax.set_xlim(dff.min()-0.01, dff.max()+0.0005)

    dx = dff.values.max() / 10000

    for i, (value, name) in enumerate(zip(dff.values, dff.index)):

        ax.text(dff.min()-0.0099,

                i,

                abs(i-25),

                size=14, weight=600, ha='left', va='center')

        ax.text(value-dx,

                i,

                name,

                size=14, weight=600, ha='right', va='bottom')

        ax.text(value-dx,

                i-.25,

                f'first sub: {min_sub_dict[name]:%d-%b-%Y} / last sub {last_sub_date[name]:%d-%b-%Y}',

                size=10,

                color='#444444',

                ha='right',

                va='baseline')

        ax.text(value+dx, i,     f'{value:,.3f}',  size=14, ha='left',  va='center')

        

    # ... polished styles

    ax.text(1.0, 1.05, mydate.strftime('%d-%b-%Y'), transform=ax.transAxes, color='#777777', size=32, ha='right', weight=800)

    ax.text(0, 1.06, 'Score', transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.3f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.12, 'Top 25 Public Leaderboard Animation', transform=ax.transAxes, size=24, weight=600, ha='left')

    plt.box(False)
fig, ax = plt.subplots(figsize=(15, 18))

draw_barchart('09-Oct-2020')
dates = [pd.to_datetime(x) for x in pd.Series(df.index.date).unique() if x > pd.to_datetime('08-Oct-2020')]

dates = dates + [dates[-1] + pd.Timedelta('1 day')]

fig, ax = plt.subplots(figsize=(15, 18))

animator = animation.FuncAnimation(fig,

                                   draw_barchart,

                                   frames=dates,

                                   interval=750)

ani = HTML(animator.to_jshtml())
ani