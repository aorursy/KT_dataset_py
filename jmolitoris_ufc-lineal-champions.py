# Import relevant libraries

import pandas as pd

from pandas.plotting import register_matplotlib_converters

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib.colors import ListedColormap

import matplotlib as mpl

import matplotlib.cm as cm

import numpy as np

register_matplotlib_converters()



# Load the data

raw_data = pd.read_csv("../input/ufcdata/raw_total_fight_data.csv", sep = ";")



#Keeping the only columns necessary for constructing the lineages

keep_cols = ['date', 'Winner', 'Fight_type']



# Create a copy to work with

df = raw_data[keep_cols].copy()

all_fights = len(df)

df.head()
# Only keeping fights that were weightclass title bouts, but not The Ultimate Fighter title bouts, Tournament bouts, or Interim title bouts 

df = df[df['Fight_type'].str.endswith('Title Bout')]

df = df[(~df['Fight_type'].str.startswith('Ultimate Fighter') & ~df['Fight_type'].str.startswith('TUF') & ~df['Fight_type'].str.contains('Interim') & ~df['Fight_type'].str.contains('Tournament'))]



# Creates a list of fight types, weight classes, and genders

type_list = []

wc_list = []

gender_list = []



for val in list(set(df['Fight_type'].values)):

    type_list.append(val) # appending fight name to list

    if "Women's" in val: # appending the gender of the weight class to list

        gender_list.append('(w)')

    else:

        gender_list.append('(m)')

    if 'weight' not in val: # appending weight classes to list

        wc_list.append('Open weight')

    elif 'Light Heavyweight' in val:

        wc_list.append('Light Heavyweight')

    else:

        split_types = val.split(" ")

        for j in split_types:

            if 'weight' in j:

                wc_list.append(j)



# Create a dictionary of weight classes and fight types

wc_dict = dict(zip(type_list, wc_list))

# Create a dicitonary of genders and fight types

gender_dict = dict(zip(type_list, gender_list))



# Create new columns by mapping dictionaries to fight types

df['weightclass'] = df['Fight_type'].map(wc_dict)

df['gender'] = df['Fight_type'].map(gender_dict)

# Updates female weight classes to differentiate them from male ones

df.loc[df['gender'] == '(w)', 'weightclass'] = df['weightclass'] + " (w)"

df = df.drop('gender', axis = 1)

df = df.drop('Fight_type', axis = 1)



# Converts date to datetime

df['date'] = pd.to_datetime(df['date'])

df = df.sort_values(by = ['weightclass', 'date'])

title_fights = len(df)



print("Total number of UFC fights: {}".format(all_fights), ", Total number of title fights: {}".format(title_fights))
# Updating the winner if a title fight ended in a draw or no contest

df.loc[1246, 'Winner'] = 'Tyron Woodley'

df.loc[893, 'Winner'] = 'Daniel Cormier'

df.loc[3633, 'Winner'] = 'Frankie Edgar'

df.loc[4796, 'Winner'] = 'Vacant'



# Adding a final column to represent the most recent date of data collection

wc_list = list(df['weightclass'].unique())

for i in wc_list:

    df = df.append({'date' : "2019-11-28 00:00:00",

                      'Winner' : "",

                      'weightclass' : i}, ignore_index = True)



# Updating some champions because the dataset was last collected in June 2019

## Heavyweight

df1 = pd.DataFrame({'date' : "2019-08-17 00:00:00",

                'Winner' : "Stipe Miocic",

                'weightclass' : 'Heavyweight'}, index = [0])



## Middleweight 

## Need to add two here for Whittaker and Adesanya

df1 = df1.append({'date' : "2017-12-07 00:00:00",

                'Winner' : "Robert Whittaker",

                'weightclass' : 'Middleweight'}, ignore_index = True)



df1 = df1.append({'date' : "2019-10-06 00:00:00",

                'Winner' : "Israel Adesanya",

                'weightclass' : 'Middleweight'}, ignore_index = True)



## Womens Strawweight

df1 = df1.append({'date' : "2019-08-31 00:00:00",

                'Winner' : "Weili Zhang",

                'weightclass' : 'Strawweight (w)'}, ignore_index = True)



# Title Vacancies

## Heavyweight

df2 = pd.DataFrame({'date' : ["1998-01-15 00:00:00", "1999-06-15 00:00:00", "2002-07-26 00:00:00", "2003-10-15 00:00:00", "2005-08-12 00:00:00"],

                'Winner' : ["Vacant", "Vacant", "Vacant", "Vacant", "Vacant"],

                'weightclass' : ['Heavyweight', 'Heavyweight', 'Heavyweight', 'Heavyweight', 'Heavyweight']})



df1 = df1.append(df2)



## Light Heavyweight

df2 = pd.DataFrame({'date' : ["1999-11-24 00:00:00", "2015-04-28 00:00:00", "2018-12-28 00:00:00"],

                'Winner' : ["Vacant", "Vacant", "Vacant"],

                'weightclass' : ['Light Heavyweight', 'Light Heavyweight', 'Light Heavyweight']})

df1 = df1.append(df2)



## Middleweight

df2 = pd.DataFrame({'date' : ["2002-10-05 00:00:00"],

                'Winner' : ["Vacant"],

                'weightclass' : ['Middleweight']})

df1 = df1.append(df2)



## Welterweight

df2 = pd.DataFrame({'date' : ["2004-05-17 00:00:00", "2013-12-13 00:00:00"],

                'Winner' : ["Vacant", "Vacant"],

                'weightclass' : ['Welterweight', 'Welterweight']})

df1 = df1.append(df2)



## Lightweight

df2 = pd.DataFrame({'date' : ["2002-03-23 00:00:00", "2007-12-08 00:00:00"],

                'Winner' : ["Vacant", "Vacant"],

                'weightclass' : ['Lightweight', 'Lightweight']})

df1 = df1.append(df2)



## Featherweight

df2 = pd.DataFrame({'date' : ["2016-11-26 00:00:00"],

                'Winner' : ["Jose Aldo"],

                'weightclass' : ['Featherweight']})

df1 = df1.append(df2)



## Bantamweight

df2 = pd.DataFrame({'date' : ["2014-01-06 00:00:00", "2019-03-20 00:00:00"],

                'Winner' : ["Vacant", "Vacant"],

                'weightclass' : ['Bantamweight', 'Bantamweight']})

df1 = df1.append(df2)



## Womens featherweight

df2 = pd.DataFrame({'date' : ["2017-06-19 00:00:00"],

                'Winner' : ["Vacant"],

                'weightclass' : ['Featherweight (w)']})

df1 = df1.append(df2)



## Womens flyweight

df2 = pd.DataFrame({'date' : ["2018-10-07 00:00:00"],

                'Winner' : ["Vacant"],

                'weightclass' : ['Flyweight (w)']})

df1 = df1.append(df2)



df = df.append(df1)

df = df.sort_values(by = ['weightclass', 'date'])
# Fills empty added column value for Winner with the name of previous winner

df['previous_winner'] = df.groupby('weightclass')['Winner'].shift()

df['Winner'].replace('', df['previous_winner'], inplace = True)

df = df.drop('previous_winner', axis = 1)



# Identify date of subsequent fight and fills last column with date

df['next_fight'] = df.groupby('weightclass')['date'].shift(-1)

df = df.reset_index(drop = True)

df['next_fight'] = df['next_fight'].fillna(df['date'])



# Had to reconvert date to datetime after adding those new rows

df['date'] = pd.to_datetime(df['date'])

df['next_fight'] = pd.to_datetime(df['next_fight'])

# Calculate difference in days between fights

df['diff'] = (df['next_fight'] - df['date']).dt.days.astype(int)



# Calculate the cumulative number of consecutive days a fighter was champion 

grouper = (df['Winner'] != df['Winner'].shift()).cumsum()

df['cum_days'] = df.groupby(['Winner', grouper]).cumsum()



# Assigns an id to each unique title reign. Some fighters have had multiple

# non consecutive reigns, thus we need to have a unique ID for each one.

df['group'] = grouper



# Now, I identify the length of the title reign for each unique ID

df['streak'] = df.groupby('group')['cum_days'].transform('max')



# Calculates the number of fights during the title reign

df['fights'] = df.groupby('group')['group'].transform(lambda x: len(x))



# All that is needed here is one row per title reign, so I only keep the first

# observation per reign. Consequently, the date column now reflects the date

# that a title reign began. I will rename the column to reflect that.

df = df.groupby('group').first()

df = df.reset_index()

df = df.drop(['diff', 'next_fight', 'cum_days'], axis = 1)

df = df.rename(columns = {'date': 'start_date'})



# Calculates the end of the reign as the start date plus the streak

df['end_date'] = df['start_date'] + pd.to_timedelta(df['streak'], unit = 'd')



df['group_start'] = (df['weightclass'] != df['weightclass'].shift(1)).astype(int)

df['group_end'] = (df['weightclass'] != df['weightclass'].shift(-1)).astype(int)



# Drop the vacant time gaps

df = df[df['Winner'] != 'Vacant']

df = df.reset_index()



print("There have been {} distinct title reigns.".format(len(df)))

print("The mean title reign lasted {} days.".format(round(df['streak'].mean(), 1)))
# Normalizing total number of title defenses

df['min_fights'] = df.groupby('weightclass')['fights'].transform('min')

df['max_fights'] = df.groupby('weightclass')['fights'].transform('max')

df['fights_norm'] = (df['fights'] - df['min_fights'])/(df['max_fights'] - df['min_fights'])



# Reorders the list by weight class

wc_list = ['Strawweight (w)',

           'Flyweight (w)',

           'Bantamweight (w)',

           'Featherweight (w)',

           'Flyweight',

           'Bantamweight',

           'Featherweight',

           'Lightweight',

           'Welterweight',

           'Middleweight',

           'Light Heavyweight',

           'Heavyweight']



fem_wc = wc_list[:4]

male_wc = wc_list[4:] 

# Making the plots

## Creating a custom colormap

brg = plt.cm.get_cmap('CMRmap', 256)

new_colors = brg(np.linspace(0, 1, 256))

new_colors = new_colors[25:125]

newcmp = ListedColormap(new_colors)



## Female weight classes

fig, ax = plt.subplots(2, 2, sharex = True, figsize = (10, 8), gridspec_kw={'width_ratios': [0.8, 1]})

fig.tight_layout()

fig.subplots_adjust(top = 0.9, wspace=0.31, hspace = 0.1)

fig.suptitle("UFC Women's Lineal Champions", fontsize = 16)



j = 0

k = 0

for x in fem_wc:

    start = df[(df['weightclass'] == x) & (df['group_start'] == 1)].index.astype(int)[0]

    stop = df[(df['weightclass'] == x) & (df['group_end'] == 1)].index.astype(int)[0]

    names = df['Winner'].values

    labs, ticklock = [], []

    for i in range(start, stop+1):   

        color = newcmp(df['fights_norm'][i])

        im = ax[j, k].hlines(i+1, df['start_date'][i], df['end_date'][i], label = df['Winner'][i], linewidth = 5, color = color, cmap = newcmp)

        firstname = names[i].find(" ")

        lastname = firstname + 1

        newname = names[i][lastname:]

        labs.append(newname)

        ticklock.append(i+1)

        #set ticks every year

        ax[j, k].xaxis.set_major_locator(mdates.YearLocator())

        #set major ticks format

        ax[j, k].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax[j, k].set_title(x[:-4])

    ax[j, k].set_ylim(start, stop+2)

    ax[j, k].set_yticks(ticklock)

    ax[j, k].set_yticklabels(labs)

    

    if k == 0:

        k += 1

    elif k == 1:

        norm = mpl.colors.Normalize(vmin=0,vmax=1)

        sm = plt.cm.ScalarMappable(cmap=newcmp, norm=norm)

        sm.set_array([])

        cbar = fig.colorbar(sm, cmap = newcmp, ax = ax[j,k], ticks = [0, 1.0])

        cbar.ax.set_yticklabels(['Fewest', 'Most'])

        cbar.ax.set_ylabel('Relative Number of Title Defenses', rotation = 270, labelpad = -30)

        j += 1

        k -=1
## Male weight classes

fig, ax = plt.subplots(4, 2, sharex = True, figsize = (10, 16), gridspec_kw={'width_ratios': [0.8, 1]})

fig.tight_layout()

fig.subplots_adjust(top = 0.95, wspace=0.32, hspace = 0.1)

fig.suptitle("UFC Men's Lineal Champions", fontsize = 16)

j = 0

k = 0

for x in male_wc:

    start = df[(df['weightclass'] == x) & (df['group_start'] == 1)].index.astype(int)[0]

    stop = df[(df['weightclass'] == x) & (df['group_end'] == 1)].index.astype(int)[0]

    names = df['Winner'].values

    labs, ticklock = [], []

    for i in range(start, stop+1):   

        color = newcmp(df['fights_norm'][i])

        im = ax[j, k].hlines(i+1, df['start_date'][i], df['end_date'][i], label = df['Winner'][i], linewidth = 5, color = color, cmap = newcmp)

        lastname = names[i].find(" ") + 1

        newname = names[i][lastname:]

        labs.append(newname)

        ticklock.append(i+1)

        #set ticks every year

        ax[j, k].xaxis.set_major_locator(mdates.YearLocator(5))

        #set major ticks format

        ax[j, k].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax[j, k].set_title(x)

    ax[j, k].set_ylim(start, stop+2)

    ax[j, k].set_yticks(ticklock)

    ax[j, k].set_yticklabels(labs)

    if k == 0:

        k += 1

    elif k == 1:

        norm = mpl.colors.Normalize(vmin=0,vmax=1)

        sm = plt.cm.ScalarMappable(cmap=newcmp, norm=norm)

        sm.set_array([])

        cbar = fig.colorbar(sm, cmap = newcmp, ax = ax[j,k], ticks = [0, 1.0])

        cbar.ax.set_yticklabels(['Fewest', 'Most'])

        cbar.ax.set_ylabel('Relative Number of Title Defenses', rotation = 270, labelpad = -30)

        j += 1

        k -=1