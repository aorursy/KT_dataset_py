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

file_path = '../input/police-violence-in-the-us/police_killings.csv'

my_data = pd.read_csv(file_path)

state_sum = my_data.groupby(['State', "Victim's race"]).count()

state_sum["Victim's name"].head()
race_AK = state_sum.loc['AK']["Victim's name"].index

import matplotlib.pyplot as plt

labels = race_AK

sizes = state_sum.loc['AK']["Victim's name"]

explode = (0.3, 0, 0, 0, 0, 0) 



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode = explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=150)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title('Police Killings as Percentage')

import matplotlib.pyplot as plt

def create_pie_chart(input_df, state):

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

    labels = input_df.loc[state]["Victim's name"].index

    sizes = input_df.loc[state]["Victim's name"]

    

    explode_len = len(input_df.loc[state]["Victim's name"].index)

    zero_list = [0]*explode_len

    if input_df.loc[state]["Victim's name"].index[0] == 'Black':

        zero_list[0] = 0.2

    elif input_df.loc[state]["Victim's name"].index[1] == 'Black':

        zero_list[1] = 0.2

    elif input_df.loc[state]["Victim's name"].index[2] == 'Black':

        zero_list[2] = 0.2

        

    explode = tuple(zero_list)



    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, explode = explode, labels=labels, autopct='%1.1f%%',

            shadow=True, startangle=180)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax1.set_title('Police Killings by Race as Percentage')

    return plt.show()
create_pie_chart(state_sum, 'VA')
import pandas as pd

demo_filepath = '../input/us-state-population-with-demographic-info-2017/Demographic_by_state.xlsx'

my_data2 = pd.read_excel(demo_filepath)

my_data2 = my_data2.set_index('State')

my_data2 = my_data2.rename(columns = {'Hispanic (of any race)' : 'Hispanic', 'Non-Hispanic White' : 'White', 'Non-Hispanic Black' : 'Black', 'Non-Hispanic Asian' : 'Asian', 'Non-Hispanic American Indian' : 'Native American'})

my_data2 = my_data2.sort_index()

my_data2.head()
my_data.columns

my_data["Victim's race"].unique()

hispanic_data = my_data[my_data["Victim's race"] == 'Hispanic']

hispanic_data.head()

sorted_hm = hispanic_data[["Victim's name", 'State']]

hispanic_group = sorted_hm.groupby('State')["Victim's name"].nunique()

hispanic_df = hispanic_group.to_frame()

hispanic_df = hispanic_df.rename(columns = {"Victim's name" : 'Hispanic Police Killings'})

hispanic_df.head()
def pull_race_data(data, race):

    new_data = data[data["Victim's race"] == race]

    sort_data = new_data[["Victim's name", "State"]]

    data_grouped = sort_data.groupby('State')["Victim's name"].nunique()

    data_df = data_grouped.to_frame()

    data_df = data_df.rename(columns = {"Victim's name" : race + ' Police Killings'})

    return data_df
black_df = pull_race_data(my_data, 'Black')

white_df = pull_race_data(my_data, 'White')

native_df = pull_race_data(my_data, 'Native American')

other_df = pull_race_data(my_data, 'Unknown race')
police_killing_total = my_data[["Victim's name", 'State']]

murder_state_total = police_killing_total.groupby('State')["Victim's name"].nunique()

murder_total_df = murder_state_total.to_frame()

murder_total_df = murder_total_df.rename(columns = {"Victim's name" : 'Total Police Killings'})

murder_total_df.head()
hispanic_df['Total Police Killings'] = murder_total_df['Total Police Killings']

hispanic_df[['Hispanic Population', 'Total State Pop']] = my_data2[['Hispanic', 'Total population']]



black_df['Total Police Killings'] = murder_total_df['Total Police Killings']

black_df[['Black Population', 'Total State Pop']] = my_data2[['Black', 'Total population']]



white_df['Total Police Killings'] = murder_total_df['Total Police Killings']

white_df[['White Population', 'Total State Pop']] = my_data2[['White', 'Total population']]



native_df['Total Police Killings'] = murder_total_df['Total Police Killings']

native_df[['Native American Population', 'Total State Pop']] = my_data2[['Native American', 'Total population']]



native_df.head()
hispanic_df['Hispanic PK as Percentage'] = 100*(hispanic_df['Hispanic Police Killings'] / hispanic_df['Total Police Killings'])

white_df['White PK as Percentage'] = 100*(white_df['White Police Killings'] / white_df['Total Police Killings'])

black_df['Black PK as Percentage'] = 100*(black_df['Black Police Killings'] / black_df['Total Police Killings'])

native_df['Native PK as Percentage'] = 100*(native_df['Native American Police Killings'] / native_df['Total Police Killings'])
native_df.head()
hispanic_df['Hispanic Pop as Percentage'] = 100*(hispanic_df['Hispanic Population'] / hispanic_df['Total State Pop'])

white_df['White Pop as Percentage'] = 100*(white_df['White Population'] / white_df['Total State Pop'])

black_df['Black Pop as Percentage'] = 100*(black_df['Black Population'] / black_df['Total State Pop'])

native_df['Native Pop as Percentage'] = 100*(native_df['Native American Population'] / native_df['Total State Pop'])
native_df.head()
Compare_perc_df = pd.DataFrame([hispanic_df['Hispanic PK as Percentage'], hispanic_df['Hispanic Pop as Percentage'], white_df['White PK as Percentage'], white_df['White Pop as Percentage'], black_df['Black PK as Percentage'], black_df['Black Pop as Percentage'], native_df['Native PK as Percentage'],  native_df['Native Pop as Percentage']])

Compare_perc_df = Compare_perc_df.transpose()

Compare_perc_df = Compare_perc_df.fillna(0)
Compare_perc_df.head()
highest_diff_b = pd.DataFrame([Compare_perc_df['Black PK as Percentage'] - Compare_perc_df['Black Pop as Percentage']]).transpose()
highest_diff_b.sort_values(0, ascending = False).head()
RI_compare = Compare_perc_df.loc['RI']

DC_compare = Compare_perc_df.loc['DC']

IL_compare = Compare_perc_df.loc['IL']

NJ_compare = Compare_perc_df.loc['NJ']

MD_compare = Compare_perc_df.loc['MD']

import matplotlib.pyplot as plt

def create_pie_charts(input_list, state):

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

    labels = 'Hispanic', 'White', 'Black', 'Native American'

    sizes1 = input_list[[0, 2, 4, 6]]

    sizes2 = input_list[[1, 3, 5, 7]]

    explode = (0, 0, 0.3, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.pie(sizes1, explode = explode, labels=labels, autopct='%1.1f%%',

            shadow=True, startangle=150)

    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ax2.pie(sizes2, explode = explode, labels=labels, autopct='%1.1f%%',

            shadow=True, startangle=150)

    ax2.axis('equal')

    ax1.set_title('Police Murder Rate as Percentage', fontsize = 10)

    ax2.set_title('Population as Percentage', fontsize = 10)

    fig.suptitle(state)

    

    fig.savefig(state + '_Police.png')

    return plt.show()
create_pie_charts(RI_compare, 'Rhode Island')
create_pie_charts(DC_compare, 'District of Columbia')
create_pie_charts(IL_compare, 'Illinois')
create_pie_charts(NJ_compare, 'New Jersey')
create_pie_charts(MD_compare, 'Maryland')
def per_100000(population_cleaned, race):

    _per_100000 = 100000*(population_cleaned[race + ' Police Killings'] / population_cleaned[race + ' Population'])

    return _per_100000
native_per_100k = per_100000(native_df, 'Native American')

black_per_100k = per_100000(black_df, 'Black')

white_per_100k = per_100000(white_df, 'White')

hispanic_per_100k = per_100000(hispanic_df, 'Hispanic')

compare_per_100k = pd.DataFrame([native_per_100k, white_per_100k, black_per_100k, hispanic_per_100k]).transpose()

compare_per_100k = compare_per_100k.rename(columns = {0: 'Native American', 1: 'White', 2:'Black', 3:'Hispanic'})

compare_per_100k = compare_per_100k.fillna(0)

compare_per_100k.head()
compare_per_100k = compare_per_100k.sort_index()

compare_per_100k.head()
import numpy as np

per_ten = compare_per_100k[['Native American',

                      'White', 'Black', 'Hispanic']]



x = np.arange(len(per_ten))  # the label locations

width = 0.2  # the width of the bars



fig, ax = plt.subplots(figsize=(25,10))

i = 0

for elt in per_ten.columns:

    barplot = ax.bar(x + width/2 + (i-3)*width, per_ten[elt], width)

    i+=1



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Police Killings')

ax.set_title('Police Killings by Race per 100,000')

ax.set_xticks(x)

ax.set_xticklabels(per_ten.index, rotation=30, horizontalalignment='right')

ax.legend(['Native American', 'White Populaton', 'Black Population', 'Hispanic Population'])



fig.tight_layout()



plt.show()
import numpy as np

per_ten = compare_per_100k[['White', 'Black', 'Hispanic']]



x = np.arange(len(per_ten))  # the label locations

width = 0.2  # the width of the bars



fig, ax = plt.subplots(figsize=(25,10))

i = 0

for elt in per_ten.columns:

    barplot = ax.bar(x + width/2 + (i-3)*width, per_ten[elt], width)

    i+=1



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Police Killings')

ax.set_title('Police Killings by Race per 100,000')

ax.set_xticks(x)

ax.set_xticklabels(per_ten.index, rotation=30, horizontalalignment='right')

ax.legend(['White Populaton', 'Black Population', 'Hispanic Population'])



fig.tight_layout()



plt.show()