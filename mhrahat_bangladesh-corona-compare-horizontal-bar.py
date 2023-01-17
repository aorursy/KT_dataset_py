#import libraries

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/corona-bangladesh-database-wide/wide_corona_19May.csv') 

df.set_index('date',inplace = True)

del(df['Unnamed: 0'])

df.head()
#selecting a custom date range

df=df.loc['1/5/2020':'19-05-2020']

df.tail()
#deleting the less confirmed cases districts

del(df['Nawabganj'])

del(df['Khagrachhari'])

del(df['Satkhira'])

del(df['Rangamati'])

del(df['Bagerhat'])

del(df['Meherpur'])

del(df['Bandarban'])

del(df['Sirajganj'])

del(df['Pirojpur'])

del(df['Bhola'])

del(df['Panchagarh'])

del(df['Natore'])

del(df['Feni'])

del(df['Jhalokati'])

del(df['Narail'])

del(df['Lalmonirhat'])

del(df['Bogra'])

del(df['Pabna'])

del(df['Magura'])

del(df['Khulna'])

del(df['Kushtia'])

del(df['Rajbari'])

del(df['Thakurgaon'])

del(df['Gaibandha'])

del(df['Kurigram'])

del(df['Chuadanga'])

del(df['Faridpur'])

del(df['Maulvibazar'])

del(df['Naogaon'])

del(df['Nilphamari'])

del(df['Sherpur'])

df.head()
def prepare_data(df, steps=15):

    df = df.reset_index()

    df.index = df.index * steps

    last_idx = df.index[-1] + 1

    df_expanded = df.reindex(range(last_idx))

    df_expanded['date'] = df_expanded['date'].fillna(method='ffill')

    df_expanded = df_expanded.set_index('date')

    df_rank_expanded = df_expanded.rank(axis=1, method='first')

    df_expanded = df_expanded.interpolate()

    df_rank_expanded = df_rank_expanded.interpolate()

    return df_expanded, df_rank_expanded



df_expanded, df_rank_expanded = prepare_data(df)
#We defined a color range

colors = plt.cm.Dark2(range(9))

labels = df_expanded.columns
def nice_axes(ax):

    ax.set_facecolor('.8')

    ax.tick_params(labelsize=8, length=0)

    ax.grid(True, axis='x', color='white')

    ax.set_axisbelow(True)

    [spine.set_visible(False) for spine in ax.spines.values()]
from matplotlib.animation import FuncAnimation

def init():

    ax.clear()

    nice_axes(ax)

    ax.set_ylim(.9, 8)



def update(i):

    for bar in ax.containers:

        bar.remove()

    y = df_rank_expanded.iloc[i]

    width = df_expanded.iloc[i]

    ax.barh(y=y, width=width, color=colors, tick_label=labels)

    date_str = df_expanded.index[i]

    ax.set_title(f'COVID-19 Cases by Zilla - {date_str}', fontsize='smaller')
fig = plt.Figure(figsize=(6,9 ), dpi=144)

ax = fig.add_subplot()

anim = FuncAnimation(fig=fig, func=update, init_func=init, frames=len(df_expanded), 

                     interval=200, repeat=False)
from IPython.display import HTML

html=anim.to_html5_video()

HTML(html)