import pandas as pd

import numpy as np



!ls ../input
raw_data = '../input/cardioActivities.csv'

raw_df = pd.read_csv(raw_data, parse_dates=True, index_col='Date')



# Deleting unnecessary columns

del raw_df['Friend\'s Tagged']

del raw_df['Route Name']

del raw_df['GPX File']

del raw_df['Activity Id']



# Dealing with NaN values and types

raw_df['Notes'].fillna('Missing', inplace=True)

raw_df['Average Heart Rate (bpm)'].fillna(raw_df['Average Heart Rate (bpm)'].mean(), inplace=True)

raw_df.astype({'Average Heart Rate (bpm)': 'int64'}, inplace=True)

raw_df['Average Speed (km/h)'].fillna(raw_df['Average Speed (km/h)'].mean(), inplace=True)



df1 = raw_df.dropna()

df1.info()
import datetime

import timeit





def validate_time(t_str):

    separator = ':'

    tstamp = t_str.split(separator)



    while len(tstamp) < 3:

        tstamp.insert(0, '00')



    iso_time_string = ':'.join(tstamp)

    #iso_time_obj = datetime.datetime.strptime(iso_time_string, '%H:%M:%S').isoformat()



    return iso_time_string



"""

# Slow: runtime = 0.0019679

start1 = timeit.timeit()

df1['Duration'] = df1['Duration'].apply(validate_time)

df.loc[:, 'Average Pace'] = df1['Average Pace'].apply(validate_time)

end1 = timeit.timeit()

print(end1 - start1)

"""



# Faster: runtime = 0.0010434

vfunc = np.vectorize(validate_time)

df = df1.copy()

df.loc[:,'Duration'] = vfunc(df1['Duration'])

df.loc[:, 'Average Pace'] = vfunc(df1['Average Pace'])



df[['Average Pace', 'Duration']].sample(10)
df1 = df.copy()

df1.loc[:,'Duration'] = pd.to_datetime(df['Duration'], format='%H:%M:%S')

df1.loc[:,'Average Pace'] = pd.to_datetime(df['Average Pace'], format='%H:%M:%S')

df1.info()
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

#import seaborn as sns

#sns.set()



# Dates filter. Dataset has dates in ramge from 2012-08-22 till 2018-11-11

till_date = '2018'

from_date = '2012'



params = [

 'Distance (km)',

 'Duration',

 'Average Pace',

 'Average Speed (km/h)',

 'Climb (m)',

 'Average Heart Rate (bpm)',   

]

filtrs = list(df1['Type'].unique())

colors = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']

markers = ['h', 'o', '^', '*', '+']



for i, filtr in enumerate(filtrs):

    col = colors[i]

    mkr = markers[i]

    for param in params:

        s = df1.loc[till_date:from_date][param][df1['Type'] == filtr]

        x = s.index

        y = s.values

        #totals = s.values



        if len(y) > 0:

            fig, ax = plt.subplots(figsize=(14,4), facecolor='#a1a1a1')

            ax.plot(x, y,

                         color=col,

                         label=param, 

                         marker=mkr,

                         markeredgewidth=2,

                         markerfacecolor='black',           

                   )

            # Extra text to include in plot legend. Varies for different parameters 

            if param in ['Distance (km)', 'Climb (m)'] :

                atext = 'Total: {:.2f}\n Max: {:.2f}'.format(float(y.sum()), float(y.max()))

            elif param in ['Duration', 'Average Pace'] :

                atext = 'Max: {:%H:%M:%S}'.format(pd.to_datetime(y.max()))

            else:

                atext = 'Max: {}'.format(str(y.max()))

                

            # Set suitable formatting for datetime y-axes based on parameter

            if param == 'Duration':

                ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            elif param == 'Average Pace':

                ax.yaxis.set_major_formatter(mdates.DateFormatter("%M:%S"))



            t = filtr + ": " + str(len(y)) + " event(s) in " + from_date + ":" + till_date

            plt.title(t, fontsize=20)

            plt.xlabel('Date', fontsize=17)

            plt.xticks(rotation='vertical')

            plt.ylabel(param, fontsize=17)          

            plt.grid(True)

            ax.plot([], [], ' ', label=atext)

            plt.legend()

            

        plt.show()