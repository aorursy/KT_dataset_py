import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/covid19-malaysia-by-region/Cases_ByState.csv")
#there is comma "," character and need to remove this in order to convert to numeric for computation

df.replace(to_replace=[r','],

           value=[''],

           regex=True, 

           inplace=True)
#copy dataframe to new dataframe for numbers in bracket

df_inc = df.copy()



#extract numbers in bracket for new dataframe

cols=[i for i in df_inc.columns if i not in ["Date"]]



for col in cols:

    df_inc[col]=df_inc[col].str.extract(r"\((.*?)\)", expand=False)
#replace Nan to 0 for df

df = df.replace(np.nan,0)
#replace NaN to 0 for df_inc

df_inc = df_inc.replace(np.nan,0)
#/\(([^)]+)\)/

#remove numbers in bracket



df.replace(to_replace=[r'\(([^)]+)\)'],

           value=[''],

           regex=True, 

           inplace=True)
# all column dtype are object

df.info()
#change all dtype to int64 except date

cols=[i for i in df.columns if i not in ["Date"]]

for col in cols:

    df[col]=pd.to_numeric(df[col])
df.info()
#add  year

df.Date=df.Date+'/2020'

df.head(15)
df.tail(15)
# make date as index

df.index = df['Date']

df.index
df['Date'].dtype
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df['Date'].head(15)
df.Date.min()
df.Date.max()
#theres's gap in the date, thus fill date with 0 cases to complete time series

r = pd.date_range(start=df.Date.min(), end=df.Date.max())

df=df.set_index('Date').reindex(r).fillna(0).rename_axis('Date').reset_index()
df.info()
df1 = df.copy()
df1['Date']= np.asarray(df1['Date']).astype(float)
#reference http://maxberggren.se/2016/11/21/right-labels/

%matplotlib inline

%config InlineBackend.figure_formats = {'png', 'retina'}

import matplotlib.pyplot as plt  

import seaborn as sns

# I like my plots on a white background

sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 1.2})

sns.set_style("whitegrid")

custom_style = {

            'grid.color': '0.8',

            'grid.linestyle': '--',

            'grid.linewidth': 0.5,

}

sns.set_style(custom_style)    
def legend_positions(df1, y):

    """ Calculate position of labels to the right in plot... """

    positions = {}

    for column in y:    

        positions[column] = df[column].values[-1] - 0.5    



    def push():

        """

        ...by puting them to the last y value and

        pushing until no overlap

        """

        collisions = 0

        for column1, value1 in positions.items():

            for column2, value2 in positions.items():

                if column1 != column2:

                    dist = abs(value1-value2)

                    if dist < 2.5:

                        collisions += 1

                        if value1 < value2:

                            positions[column1] -= .1

                            positions[column2] += .1

                        else:

                            positions[column1] += .1

                            positions[column2] -= .1

                        return True

    while True:

        pushed = push()

        if not pushed:

            break



    return positions
x = 'Date'

y = ['JH', 'KD', 'KE',

     'ML', 'NS', 'PH',    

     'PG', 'PK', 'PR',    

     'SB', 'SR', 'SE',    

     'TR', 'KL', 'PT','LB']  

positions = legend_positions(df, y)



f, ax = plt.subplots(figsize=(20,15))        

cmap = plt.cm.get_cmap('Set1', len(y))



for i, (column, position) in enumerate(positions.items()):



    # Get a color

    color = cmap(float(i)/len(positions))

    # Plot each line separatly so we can be explicit about color

    ax = df1.plot(x=x, y=column, legend=False, ax=ax, color=color, linewidth=2)



    # Add the text to the right

    plt.text(

        df1[x][df1[column].last_valid_index()] + 0.5,

        position, column, fontsize=12,

        color=color # Same color as line

    )

ax.set_ylabel('Cumulative cases')

ax.set_xlabel('Time series')

t = [df1['Date'].min(), df1['Date'].max()]

plt.xticks(t,t)

a=ax.get_xticks().tolist()

a[0]='25-Jan-2020'

a[1]='08-Aug-2020'

ax.set_xticklabels(a)



y_ticks = np.arange(0, 3100, 500)

plt.yticks(y_ticks)

sns.despine()
#change all dtype to int64 except date

cols=[i for i in df_inc.columns if i not in ["Date"]]

for col in cols:

    df_inc[col]=pd.to_numeric(df_inc[col])
df_inc.info()
df_inc["total"] = df_inc.sum(axis=1)

df_inc.head()
#add  year

df_inc.Date=df_inc.Date+'/2020'

df_inc.head(15)
# make date as index

df_inc.index = df_inc['Date']

df_inc['Date'] = pd.to_datetime(df_inc['Date'], format="%d/%m/%Y")
import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter



plt.figure(figsize=(16,14))



plt.plot(df_inc['Date'].values, df_inc['total'], lw=2)

plt.xlabel('Dates',)

plt.ylabel('Amount of Cases')





plt.title('Malaysia COVID-19 Number of Cases over Time')

plt.show()
years = mdates.YearLocator()   # every year

months = mdates.MonthLocator()  # every month

years_fmt = mdates.DateFormatter('%Y')



df_inc['Date'] = pd.to_datetime(df_inc.Date)



fig, ax = plt.subplots(figsize=(16,9))

ax.plot('Date', 'total', data=df_inc)



# format the ticks

ax.xaxis.set_major_locator(months)

#ax.xaxis.set_major_formatter(years_fmt)

#ax.xaxis.set_minor_locator(month)



# round to nearest years.

#datemin = np.datetime64(df_inc['Date'][0], 'Y')

#datemax = np.datetime64(df_inc['Date'][-1], 'Y') + np.timedelta64(1, 'Y')

#ax.set_xlim(datemin, datemax)



# format the coords message box

ax.format_xdata = mdates.DateFormatter('%d-%m-%Y')

ax.grid(True)



# rotates and right aligns the x labels, and moves the bottom of the

# axes up to make room for them

fig.autofmt_xdate()



plt.show()