# import libraries which are necessary in this notebook

import numpy as np

import pandas as pd

from os import path

import datetime

import matplotlib

# import folium library

from folium import plugins

import folium

# use Waffle from pywaffle library for waffle plot

!pip install pywaffle

from pywaffle import Waffle

# Start with loading all necessary libraries

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from matplotlib import cm # color map
# Read the whole dataset into a Pandas' DataFrame

df=pd.read_csv('../input/us-accidents/US_Accidents_June20.csv')
# quick overview of the data

df.head()
# Number of total entries

df.shape
# Lets check the nan values with in each column / feature

# percentage of missing values in each column

print((100*df.isnull().sum()/df.shape[0]).round(2))
# drop these columns

df_new = df.drop(['TMC', 'End_Lat', 'End_Lng', 'Number', 'Wind_Chill(F)', 'Wind_Speed(mph)', 'Precipitation(in)'], axis = 1)

df_new.dropna(axis = 0, how = 'any', inplace = True)
# check the shape again

df_new.shape
# check again the possible nan / missing values

df_new.isnull().sum()
# Number of accidents by each state

df_state=df_new.groupby(['State'], as_index=False).count().iloc[:,:2]

# Rename the column that make more sence

df_state=df_state.rename(columns={"ID":"NrAccidents"})

# sort by number of accidents

df_state.sort_values(by=['NrAccidents'], ascending=False, inplace=True)

df_state.head()
# plot the map by using folium with corresponding distribution of accidents

# geojson file without AK, Alaska

us_states_geo = r'../input/geojson/us_states_49.json' 





# set the size of the plotting canvas / figsize

f = folium.Figure(width=900, height=500)

# create a plain USA map object

us_accident_distribution_map = folium.Map(location=[40, -100], zoom_start=4).add_to(f)



# threshold scaling

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration

threshold_scale = np.linspace(df_state['NrAccidents'].min(),

                              df_state['NrAccidents'].max(),

                              6, dtype=int)

threshold_scale = threshold_scale.tolist() # change the numpy array to a list

threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration

# Apply the corresponding dataset to the map

folium.Choropleth(

    geo_data=us_states_geo,

    name='choropleth',

    data=df_state,

    columns=['State','NrAccidents'],

    key_on='feature.id',

    threshold_scale=threshold_scale,

    fill_color='OrRd',

    fill_opacity=0.8,

    line_opacity=0.2,

    legend_name='Overview of the number of accidents across US (Alaska is not included)',

    reset=True

).add_to(us_accident_distribution_map)

folium.LayerControl().add_to(us_accident_distribution_map)

us_accident_distribution_map
# set the state names as the index

df_state.set_index('State', inplace=True)


# plot data in bar chart

df_state.plot(kind='bar', width=0.8, figsize=(15, 8), legend=False)

plt.xlabel('State', fontsize=14) # add to x-label to the plot

plt.ylabel('Number of Accidents', fontsize=14) # add y-label to the plot

plt.title('Number of accidents by each state', fontsize=14) # add title to the plot

plt.show()
# lets look at the first six states with highest number of accidents

state6 = df_state['NrAccidents'].iloc[:6].sum(axis=0)

state_rest = df_new.shape[0] - df_state['NrAccidents'].iloc[:6].sum(axis=0)

# plot as waffle

data = {'CA, TX, FL, SC, NC, NY': (100*state6/df_new.shape[0]).round(1), 'Other states': (100*state_rest/df_new.shape[0]).round(1)}

fig = plt.figure(

    figsize=(15, 20),

    FigureClass=Waffle, 

    rows=10, 

    columns=50,

    values=data, 

    labels=["{0} ({1}%)".format(k, v) for k, v in data.items()],

    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)}

    )

plt.show()
bins=300

plt.figure(figsize=(10, 6))





for st in ['CA', 'TX', 'FL', 'SC', 'NC', 'NY']:    

    # set s filter

    stfilt = (df_new['State'] == st)

    plt.hist(df_new.loc[stfilt,'Visibility(mi)'], bins, density=False)

plt.xlabel('Visibility(mi)', fontsize=14)

plt.ylabel('Number of accidents', fontsize=14)

plt.xlim(0,15)

plt.grid()

plt.show()
def which_day(date_time):

    '''

    To find out which weekday according to given timestamp with the format 'yyyy-mm-dd hh:mm:ss'

        input: datetime string with the format of 'yyyy-mm-dd hh:mm:ss'

        return: nth day of the week

    '''

    # import time and date modules

    from datetime import datetime

    # import calendae modules to extract the exact weekday

    import calendar

    try:

        if type(date_time) is str:

            my_string=date_time.split(' ')[0]

            my_date = datetime.strptime(my_string, "%Y-%m-%d")

            return my_date.weekday()

        else:

            raise Exception("'date_time' has unexpected data type, it is expected to be a sting")



    except Exception as e:

        print(e)

# use above function to find which weekday 

nth_day=[]

date_time=[dt for dt in df_new['Start_Time']]

for i in range(len(date_time)):

    nth_day.append(which_day(date_time[i]))

# add four new columns 'year', 'month', 'hour', 'weekday'

df_new['year'] = pd.DatetimeIndex(df_new['Start_Time']).year

df_new['month'] = pd.DatetimeIndex(df_new['Start_Time']).month

df_new['hour'] = pd.DatetimeIndex(df_new['Start_Time']).hour

df_new['weekday']=nth_day
df_new.shape
df_new.loc[:,['year', 'month', 'hour', 'weekday', 'Start_Time']].head()
df_month=df_new[df_new['year'].isin(['2016','2017', '2018', '2019', '2020'])].groupby(['month'], as_index=False).count().iloc[:,:2]

# by changing the argument in 'isin()' one can look at quite directly the change of the accidents during the years,

# which I did not do it here.

df_month.head()
# plot data in bar chart

ax=df_month.plot(kind='bar', width=0.8, figsize=(10, 6), legend=None)

xtick_labels=['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']

ax.set_xticks(list(df_month.index))

ax.set_xticklabels(xtick_labels)

ax.set_xlabel('Month', fontsize=14) # add to x-label to the plot

ax.set_ylabel('Number of Accidents', fontsize=14) # add y-label to the plot

ax.set_title('Number of accidents by each month', fontsize=14) # add title to the plot

plt.show()
wday_filt = (df_new['weekday'].isin([0, 1, 2, 3, 4]))#.to_frame()

weekend_filt = (df_new['weekday'].isin([5, 6]))#.to_frame()

df_wday = (df_new.loc[wday_filt])[['hour']]#.count().iloc[:, :2]

df_weekend = (df_new.loc[weekend_filt])[['hour']]#.count().iloc[:, :2]
# plot the distribution of accidents during the day

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 12), sharex=True)

ax0, ax1, ax2 = axes.flatten()

bins=24

kwargs = dict(bins=24, density=False, histtype='stepfilled', linewidth=3)

# ax0

ax0.hist(list(df_new['hour']),  **kwargs, color='orange', label='Whole week')

ax0.set_ylabel('Number of accidents', fontsize=14)

# ax1

ax1.hist(list(df_wday['hour']), **kwargs, color='blue', label='Work days')

ax1.set_ylabel('Number of accidents', fontsize=14)

# ax2

ax2.hist(list(df_weekend['hour']),  **kwargs, color='Red', label='Only weekend')

ax2.set_ylabel('Number of accidents', fontsize=14)

ax2.set_xlabel('Hour', fontsize=14)

ax0.legend(); ax1.legend(); ax2.legend()

plt.xlim(0, 23)

#plt.ylim(0, 2.5e5)

plt.show()

df_weekday=df_new.groupby(['weekday'], as_index=False).count().iloc[:,:2]

# set the month as the index

df_weekday.set_index('weekday', inplace=True)
# plot data in bar chart

labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']

x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots(figsize=(10, 6))

ax1 = ax.bar(x, df_weekday['ID'], width=0.5)

#ax1 = ax.plot(x, df_weekday['ID'],marker='o', lw=2)

# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Number of accidents', fontsize=14)

ax.set_xlabel('Weekday', fontsize=14)

ax.set_title('Distribution of accidents along the weekdays', fontsize=14)

ax.set_xticks(x)

ax.set_xticklabels(labels)



#df_weekday.plot(kind='line', figsize=(10, 6), legend=None)



#plt.xlabel('Weekday', fontsize=14) # add to x-label to the plot

#plt.ylabel('Number of Accidents', fontsize=14) # add y-label to the plot

#plt.title('Number of accidents by each state', fontsize=14) # add title to the plot

plt.show()
!pip install Pillow

!pip install wordcloud
# join all descriptions from all accidents

dsc=df_new['Description'].astype(str)

# remove non-words

#sanitized_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()) 

text = " ".join(desc for desc in dsc)

print ("There are {} words in the combination of all description.".format(len(text)))
more_stopwords=["accident", "due", "blocked", "Right", "hand"]

for more in more_stopwords:

    STOPWORDS.add(more)

# Generate a word cloud image

# lower max_font_size

wordcloud = WordCloud(stopwords=STOPWORDS, max_font_size=40, background_color="white").generate(text)

plt.figure(figsize=(18, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show

# Save the image in the img folder:

wordcloud.to_file("us_accidents_description.png")
df_T=df_new['Temperature(F)'].values
'''

# lambda function 

ftoc=lambda f:5/9*(f-32)

# function call

c=[]

for fi in f:

    c.append(round(ftoc(ni), 1))

c=np.array(c)

c

'''

num_bins = 50



fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_subplot(111)



# the histogram of the data

n, bins, patches = ax1.hist(df_T, num_bins, density=0) # set density=1 to normalize

# find bincenters

# bincenters = 0.5*(bins[1:]+bins[:-1])





ax1.set_xlabel(r"Temperature(°F)", fontsize=14, color='red')

ax1.set_ylabel('Number of accidents', fontsize=14, color='red')

ax1.set_xlim(-25, 125) # set xlim 

# Set the temperature in celisius

ax2 = ax1.twiny()

ax2.set_xlabel(r"Temperature(°C)", fontsize=14, color='red')

ax2.set_xlim(ax1.get_xlim())

ax2.set_xticks([-58, -13, 32, 77, 122])

ax2.set_xticklabels(['-50', '-25', '0','25', '50'])

plt.grid()

plt.show()
100*df.Severity.value_counts()/df.shape[0]
df.Stop.value_counts()
df['Sunrise_Sunset'].value_counts()
df['Traffic_Signal'].value_counts()
df['Give_Way'].value_counts()