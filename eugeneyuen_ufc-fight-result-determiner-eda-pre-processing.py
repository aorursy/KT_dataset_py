# Import packages

import numpy as np

import pandas as pd

import matplotlib. pyplot as plt

import seaborn as sns



plt.style.use('seaborn-white')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/ufcdata/data.csv")

data_fighter_ratios = data.copy()

data.sample(5)
# Check the number of rows and columns

data.info()
# Summary statistics of the numerical columns excluding the non-numerical columns

data.describe()
# Summarizes the statistics for values that are recorded as strings in the dataframe

data.describe(include = np.object)
#Count the number of unique R_fighters

data.R_fighter.value_counts()
#Count the number of unique B_fighters

data.B_fighter.value_counts()
#Number of red wins

red_wins = np.sum(data['Winner'] == "Red")

#Number of blue wins

blue_wins = np.sum(data['Winner'] == "Blue")



print("Red wins: " + str(red_wins) + ", Blue wins: " + str(blue_wins))
# check the number of unique weight class and the frequency for each

data.weight_class.value_counts()
# Visualize the distribution of the weightclass

plt.figure(figsize = (12,12))

sns.set(font_scale = 1)

sns.countplot(data = data, x = data.weight_class).set_xticklabels(labels = data.weight_class ,rotation = 45)

plt.show()
# Convert the True/False values into binary values

data.title_bout = data.title_bout.astype(int)

data.title_bout.head(10)
# Visualize the distribution of title bouts

plt.figure(figsize = (12,12))

sns.set(font_scale = 1)

sns.countplot(data = data, x = data.title_bout)

plt.show()
# Display all unique values of R_Stance

data.loc[:,"R_Stance"].unique()
# Display all unique values of B_Stance

data.loc[:,"B_Stance"].unique()
# Create a new dataframe object that contains all the stances used by B_Fighter and R_Fighter

stance_df = pd.DataFrame(pd.concat([data.R_Stance, data.B_Stance]))

stance_df.head()
# Check the number of NaN values for stance column

stance_df.info()
stance_df.describe()
# shows the different unique values and the frequency of each unique value

stance_df[0].value_counts()
# count the number of null values

stance_df.isna().sum()
# Visualize the distribution of the stances used for all the fights

plt.figure(figsize = (12,8))

sns.set(font_scale = 1)

sns.countplot(data = stance_df, x = stance_df[0])

plt.xlabel("Stance", fontsize = 12)

plt.ylabel("Frequency",fontsize = 12)
# Create a new dataframe that contains B_Stance and B_age only

bstance_age = data.loc[:, ("B_Stance", "B_age")]

bstance_age.head()
#renaming the columns to join it with rstance_age

bstance_age.rename(axis=1, mapper={"B_Stance" : "Stance", "B_age" : "Age"}, inplace=True)

bstance_age.head()
# creating a dataframe that contains R_Stance and R_age only

rstance_age = data.loc[:, ("R_Stance", "R_age")]

rstance_age.head()
#renaming the columns to join it with bstance_age

rstance_age.rename(axis=1, mapper={"R_Stance" : "Stance", "R_age" : "Age"}, inplace=True)

rstance_age.head()
# Create a new dataframe that contains all the information of the relevant age and the stance used

stance_age_df = bstance_age.append(rstance_age)
#drop the null values present

stance_age_df.dropna(inplace = True)
# Visualize the correlation between age of fighters and their stance

plt.figure(figsize=(12,8))

sns.set(font_scale=1.5)

sns.swarmplot(data = stance_age_df, x = "Stance", y = "Age", dodge = True)

plt.show()
r_age_streak = data.loc[:, ("R_age", "R_longest_win_streak")]

r_age_streak.head()
b_age_streak = data.loc[:, ("B_age", "B_longest_win_streak")]

b_age_streak.head()
# rename the column headers to combine with b_age_streak

r_age_streak.rename(axis= 1, mapper = {'R_age':"Age", "R_longest_win_streak" : "Win_streak"}, inplace=True)

r_age_streak.head()
# rename the column headers to combine with r_age_streak

b_age_streak.rename(axis= 1, mapper = {'B_age':"Age", "B_longest_win_streak" : "Win_streak"}, inplace=True)

b_age_streak.head()
# to get the combined statistics of age and winstreak

age_streak_df = pd.concat([r_age_streak, b_age_streak])

age_streak_df.head()
#drop the NaN values

age_streak_df.dropna(inplace = True)
age_streak_df.info()
# Visualise the correlation between age and win streak

plt.figure(figsize=(12,8))

sns.set(font_scale=1.4)

sns.heatmap(age_streak_df.corr(), annot= True, cmap = "Reds")

plt.show()
# Create a new dataframe of all the heights

height_df = pd.concat([data.R_Height_cms, data.B_Height_cms])
# to see the distribution of heights

plt.figure(figsize = (12,8))

sns.set(font_scale = 1)

sns.distplot(height_df.dropna())
# see which are the hottest places to hold fights

plt.figure(figsize=(12,8))

plt.pie(x=data.location.value_counts(), autopct='%1.1f%%', labels=data.location.unique(), rotatelabels=True)

plt.show()
# create a Pandas Series containing the ages of all the fighters

age_series = pd.concat([data.B_age, data.R_age])

age_series.head(10)
# discretize the ages to below 35 and above 35

age_bins = pd.cut(x=age_series, bins=[0, 35, 100], labels=["Below 35", "Above 35"])
# visualize the distribution of the ages

plt.figure(figsize = (12,8))

sns.set(font_scale = 1)

sns.countplot(x = age_bins)

plt.show()
# select the b_fighter's significant strikes made and significant strikes landed

b_sign_stats = data.loc[:, ("B_avg_SIG_STR_att", "B_avg_SIG_STR_landed")]

b_sign_stats.head()
# select the r_fighter's significant strikes made and significant strikes landed

r_sign_stats = data.loc[:, ("R_avg_SIG_STR_att", "R_avg_SIG_STR_landed")]

r_sign_stats.head()
#rename the column headers 

b_sign_stats.rename(axis=1, mapper={"B_avg_SIG_STR_att" : "Strikes_made", "B_avg_SIG_STR_landed" : "Strikes_landed"}, inplace=True)

b_sign_stats.head()
#rename the column headers

r_sign_stats.rename(axis=1, mapper={"R_avg_SIG_STR_att" : "Strikes_made", "R_avg_SIG_STR_landed" : "Strikes_landed"}, inplace=True)

r_sign_stats.head()
# create a dataframe that contains all significant statistics of all the fighters' matches

sign_stats_df = pd.DataFrame(pd.concat([b_sign_stats, r_sign_stats]))

sign_stats_df.head()
# Visualize the relationship between significant strikes made and significant strikes landed

plt.figure(figsize = (12,8))

sns.set(font_scale = 1)

sns.lineplot(x=sign_stats_df.Strikes_made,y=sign_stats_df.Strikes_landed)

plt.show()
# this is to check whether is age is static,

# turns out that age is not static

data.loc[data.R_fighter == "Valentina Shevchenko", ("date", "R_age")]
data.corr()
# create a new dataframe called bluewinner_df

bluewinner_df = data
# converts True and False to binary values

(data.Winner=="Blue").astype(int)
# then convert the "Winner" column to binary values to show if the blue fighter is a winner or a loser

bluewinner_df["Winner"] = (data.Winner=="Blue").astype(int)
# create a correlation dataframe that performs correlation analaysis on all the variables

# and fill the NaN values with 0

corr_bluewinner = bluewinner_df.corr().fillna(0)

corr_bluewinner
# select only the winner column of the correlation matrix and create a copy of it(if not will have an error)

corr_bluewinner_copy = corr_bluewinner['Winner'].copy()

corr_bluewinner_copy.sort_values(inplace=True, ascending=False)

corr_bluewinner_copy
data_fighter_ratios = data_fighter_ratios[data_fighter_ratios['B_Stance'].notna()]

data_fighter_ratios = data_fighter_ratios[data_fighter_ratios['R_Stance'].notna()]

#Win-lose ratio for each fighter

#identify unique fighters first

unique_fig = data_fighter_ratios['R_fighter'].append(data_fighter_ratios['B_fighter'])

unique_fig = unique_fig.drop_duplicates()

#Total number of unique fighters

counter = range(unique_fig.count())

#results dictionary will use fighter name and his fight stats will be values

results = {}

for rows in counter:    

    #Store results as win-lose-draw

    fighter_results = {"win":0,"lose":0,"draw":0,"total":0, "win-lose ratio":0}

    fighter = unique_fig.iloc[rows]

    

    #Stats as a Red Fighter

    as_red_fighter = data_fighter_ratios.loc[data_fighter_ratios['R_fighter'].isin([fighter])]

    values_red = as_red_fighter['Winner'].value_counts()

    #Declaring variables in case they don't get created in the try-except e.g. no draws

    win = 0

    lose = 0

    draw = 0

    try:

        win = values_red['Red']

    except Exception: 

        pass

    try:

        lose = values_red['Blue']

    except Exception: 

        pass

    try:

        draw = values_red['Draw']

    except Exception: 

        pass



    #Stats as a Blue Fighter

    as_blue_fighter = data_fighter_ratios.loc[data_fighter_ratios['B_fighter'].isin([fighter])]

    values_blue = as_blue_fighter['Winner'].value_counts()

    try:

        win += values_blue['Blue']

    except Exception: 

        pass

    try:

        lose += values_blue['Red']

    except Exception: 

        pass

    try:

        draw += values_blue['Draw']

    except Exception: 

        pass



    #add the data to fighter's dictionary

    total = win+lose+draw

    

    #Win lose in percentage as a decimal figure out of 1, excludes draws

    if (lose == 0 and win > 0):

        win_lose = 1.0

    elif (win==0):

        win_lose = 0.0

    else:

        win_lose = (win/(win+lose))

        win_lose = float("{0:.2f}".format(win_lose))

        

    fighter_results["draw"] = draw

    fighter_results["win"] = win

    fighter_results["lose"] = lose

    fighter_results["win-lose ratio"] = win_lose

    fighter_results["total"] = total

    #Putting this entry into the main dictionary with all fighter match stats

    results[fighter] = fighter_results





print(results) 

#unique_fighters=df.loc[:,('R_fighter','B_fighter')].nunique()

#print(unique_fighters)