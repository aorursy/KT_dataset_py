# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Lets read the data in using pandas. 



df = pd.read_csv("../input/FootballEurope.csv")
#I want to look at how a teams performance changes with respect to months and years. 



years = []

months = []

days = []



for i in range(len(df['date'])):

    years.append(df['date'][i].split('-')[0])

    months.append(df['date'][i].split('-')[1])

    days.append(df['date'][i].split('-')[2])

    

df["Year"] = years

df["Month"] = months

df["Day"] = days



df['Year'] = df['Year'].apply(pd.to_numeric)

df['Month'] = df['Month'].apply(pd.to_numeric)

df['Day'] = df['Day'].apply(pd.to_numeric)
# Check the data types within the dataset. 



df.dtypes
# Drop unnecessary column. 

df = df.drop('Unnamed: 0', axis=1)
# Let's take a quick look at the numeric variables within the dataset.



df.drop('id', axis=1).describe()
# Collect the numeric variable names 

num_columns = df.drop('id', axis=1).describe().columns.tolist()
num_columns
# Let's separate the home numeric variables from the away numeric variables. 



home_num_columns = []

away_num_columns = []

standard_num_columns = []



for i in range(len(num_columns)):

    if num_columns[i].startswith('home'):

        home_num_columns.append(num_columns[i])

    elif num_columns[i].startswith('away'):

        away_num_columns.append(num_columns[i])

    else:

        standard_num_columns.append(num_columns[i])
home_num_columns
away_num_columns
# Numeric variables that are not defined by home or away. 



standard_num_columns
# Lets quickly check I haven't left any variables behind. 

print(len(num_columns))

print(len(home_num_columns))

print(len(away_num_columns))

print(len(standard_num_columns))
# Now lets split the full time home team numeric variables from the half time home numeric variables. 



FT_home_num_columns = []

HT_home_num_columns = []



for i in range(len(home_num_columns)):

    if home_num_columns[i].endswith('HT'):

        HT_home_num_columns.append(home_num_columns[i])

    else:

        FT_home_num_columns.append(home_num_columns[i])
HT_home_num_columns
# Collect the categoric variables within the data frame 

cat_columns = list(set(df.drop('id', axis=1).columns) - set(df.drop('id', axis=1).describe().columns))
cat_columns
# Again let's separate the home variables from the away variables but this time for only categoric variables.

home_cat_columns = []

away_cat_columns = []

standard_cat_columns = ['id']



for i in range(len(cat_columns)):

    if cat_columns[i].startswith('home'):

        home_cat_columns.append(cat_columns[i])

    elif cat_columns[i].startswith('away'):

        away_cat_columns.append(cat_columns[i])

    else:

        standard_cat_columns.append(cat_columns[i])
home_cat_columns
away_cat_columns
standard_cat_columns
#Check for Null Values within the Dataset



for i in cat_columns:

    

    if np.any(df[i].unique() == 'null'):

        for j in range(len(df[i].values)):

            if df[i].values[j] == 'null':

                df[i].values[j] = None

 

print('Number of Missing Values in Each Column as a Percentage')

print(((df.isnull().sum() / df.isnull().sum().sum())*100).sort_values(axis=0, ascending=False))
#Lets create a correlation heat map for the home full-time numeric variables 



import seaborn as sns

import matplotlib.pyplot as plt

sns.set(context="paper", font="monospace")



# Load the datset of correlations between cortical brain networks



corrmat = df[FT_home_num_columns + standard_num_columns].corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)



plt.show()
# Why not the half-time version for good measure.



# Load the datset of correlations between cortical brain networks



corrmat = df[HT_home_num_columns + standard_num_columns].corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)



plt.show()
# There is clearly a difference in patterns for both plots. 

# But I think that's more to do with the fact the equivalent variables are not in the same positions for both graphs. 
# Lets look at the more correlated pairs by considering the absolute value of correlation 

# between any two variables that belong to the full-time home dataset.



corr_matrix = df[FT_home_num_columns + standard_num_columns].corr()



def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations")

print(get_top_abs_correlations(corr_matrix, 30))
# Again the half-time version for comparison



corr_matrix = df[HT_home_num_columns + standard_num_columns].corr()



print("Top Absolute Correlations")

print(get_top_abs_correlations(corr_matrix, 30))
# BoxPlot of homeGoalFT variable with respect to the division. 



var_name = "division"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='homeGoalFT', data=df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('homeGoalFT', fontsize=12)

plt.title("Distribution of homeGoalFT variable with "+var_name, fontsize=15)

plt.show()
# Basic statistics for the homeGoalFT variable with respect to the division. 

df.groupby('division')['homeGoalFT'].describe()
# The most correlated variables with respect to homeGoalFT

corr_matrix = df[FT_home_num_columns + standard_num_columns].corr()

corr_matrix['homeGoalFT'].sort_values(axis=0, ascending=False)[1:]
# Simple line plot for homeGoalFT with respect to Month

sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeGoalFT", data=df)

plt.show()
# Now by division

sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeGoalFT", hue='division', data=df)

plt.show()
#Lets remove the month of June 

sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeGoalFT", hue='division', data=df[df['Month'] != 6])

plt.show()
# Each division appears to follow its on pattern regardless of the other divisions. 
# Now lets follow the same process but for the homeRatingsFT variable.
var_name = "division"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='homeRatingsFT', data=df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('homeRatingsFT', fontsize=12)

plt.title("Distribution of homeRatingsFT variable with "+var_name, fontsize=15)

plt.show()
df.groupby('division')['homeRatingsFT'].describe()
corr_matrix = df[FT_home_num_columns + standard_num_columns].corr()

corr_matrix['homeRatingsFT'].sort_values(axis=0, ascending=False)[1:]
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeRatingsFT", data=df)

plt.show()
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeRatingsFT", hue='division', data=df)

plt.show()
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeRatingsFT", hue='division', data=df[df['Month'] != 6])

plt.show()
# Now lets consider the homeShotsTotalFT variable.
var_name = "division"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='homeShotsTotalFT', data=df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('homeShotsTotalFT', fontsize=12)

plt.title("Distribution of homeShotsTotalFT variable with "+var_name, fontsize=15)

plt.show()
df.groupby('division')['homeShotsTotalFT'].describe()
corr_matrix = df[FT_home_num_columns + standard_num_columns].corr()

corr_matrix['homeShotsTotalFT'].sort_values(axis=0, ascending=False)[1:]
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeShotsTotalFT", data=df)

plt.show()
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeShotsTotalFT", hue='division', data=df)

plt.show()
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeShotsTotalFT", hue='division', data=df[df['Month'] != 6])

plt.show()
# Let's create a new variable that divides the number of 

# home goals scored at full time by the total number of shots a team makes by full time. 

df['homeGoalPerShotsTotalFT'] = df["homeGoalFT"]/df["homeShotsTotalFT"]
# Finally lets follow the same procedure but for the new variable homeGoalPerShotsTotalFT 

# (I really need to get better at naming variables)
var_name = "division"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='homeGoalPerShotsTotalFT', data=df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('homeGoalPerShotsTotalFT', fontsize=12)

plt.title("Distribution of homeGoalPerShotsTotalFT variable with "+var_name, fontsize=15)

plt.show()
df.groupby('division')['homeGoalPerShotsTotalFT'].describe()
corr_matrix = df[FT_home_num_columns + standard_num_columns + ['homeGoalPerShotsTotalFT']].corr()

corr_matrix['homeGoalPerShotsTotalFT'].sort_values(axis=0, ascending=False)[1:]
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeGoalPerShotsTotalFT", data=df)

plt.show()
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeGoalPerShotsTotalFT", hue='division', data=df)

plt.show()
sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeGoalPerShotsTotalFT", hue='division', data=df[df['Month'] != 6])

plt.show()
#Enough about division let's look at specific team performance. 

df.groupby('homeTeam')['homeGoalPerShotsTotalFT'].describe().sort_values('mean',ascending=False)
#Lets take a quick look at how Man Utd have been shaping up this past couple of years at home 

# (I'm a Man Utd I couldn't help myself).
#Lets mark entries as for whether they are Man Utd entires or not. 



def isManUtd(x):

    if x == 'Man Utd':

        return 'Man Utd'

    else:

        return 'Rest of Europe'

    

df['ManUtdHome'] = df['homeTeam'].apply(isManUtd)
# Compare Utd's homeGoalFT performance against the rest of Europe's 

var_name = "ManUtdHome"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='homeGoalFT', data=df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('homeGoalFT', fontsize=12)

plt.title("Distribution of homeGoalFT variable with "+var_name, fontsize=15)

plt.show()
# Now let's compare Utd's performance with rest to the average homeGoalFT performance of each division

df.groupby(['ManUtdHome', 'division'])['homeGoalFT'].describe().sort_values('mean', ascending=False)
# Compare Utd's homeShotsTotalFT performance against the rest of Europe's 

var_name = "ManUtdHome"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='homeShotsTotalFT', data=df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('homeShotsTotalFT', fontsize=12)

plt.title("Distribution of homeShotsTotalFT variable with "+ var_name, fontsize=15)

plt.show()
# Now let's compare Utd's performance with rest to the average homeShotsTotalFT performance of each division

df.groupby(['ManUtdHome', 'division'])['homeShotsTotalFT'].describe().sort_values('mean', ascending=False)
# Compare Utd's homeRatingsFT performance against the rest of Europe's 

var_name = "ManUtdHome"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='homeRatingsFT', data=df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('homeRatingsFT', fontsize=12)

plt.title("Distribution of homeRatingsFT variable with "+var_name, fontsize=15)

plt.show()
# Now let's compare Utd's performance with rest to the average homeRatingsFT performance of each division

df.groupby(['ManUtdHome', 'division'])['homeRatingsFT'].describe().sort_values('mean', ascending=False)
# Compare Utd's homeGoalPerShotsTotalFT performance against the rest of Europe's 

var_name = "ManUtdHome"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='homeGoalPerShotsTotalFT', data=df, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('homeGoalPerShotsTotalFT', fontsize=12)

plt.title("Distribution of homeGoalPerShotsTotalFT variable with " + var_name, fontsize=15)

plt.show()
# Now let's compare Utd's performance with rest to the average homeGoalPerShotsTotalFT performance of each division

df.groupby(['ManUtdHome', 'division'])['homeGoalPerShotsTotalFT'].describe().sort_values('mean', ascending=False)
# Comparison by year

sns.set_style("darkgrid")

ax = sns.pointplot(x="Year", y="homeGoalFT", hue='ManUtdHome', data=df)

plt.show()
# by month.

sns.set_style("darkgrid")

ax = sns.pointplot(x="Month", y="homeGoalFT", hue='ManUtdHome', data=df)

plt.show()
# Average rating by homeTeam and division.

df.groupby(['homeTeam', 'division'])['homeRatingsFT'].mean().sort_values(axis=0, ascending=False)
#Find Man Utd's position in this list

[i for i,x in enumerate(df.groupby(['homeTeam'])['division', 'homeRatingsFT'].mean().sort_values('homeRatingsFT', ascending=False).index.tolist()) if x == 'Man Utd']
# 36th in Europe for average homeRatingsFT
#Find Man Utd's position in the EPL subset list

[i for i,x in enumerate(df[df['division'] == 'EPL'].groupby(['homeTeam'])['division', 'homeRatingsFT'].mean().sort_values('homeRatingsFT', ascending=False).index.tolist()) if x == 'Man Utd']
# 5th in EPL for average homeRatingsFT
# Now let's compare Utd's performance with 10 best home teams in this dataset.

# We will use homeRatingsFT as the obvious decider variable of performance.

Top10Teams = df.groupby(['homeTeam'])['homeRatingsFT'].mean().sort_values(axis=0, ascending=False).index.tolist()[:10]
def inTopTen(x):

    if x in Top10Teams:

        return 'Yes'

    else:

        return 'No'

    

df['TopTenHome'] = df['homeTeam'].apply(inTopTen)    
df.groupby(['ManUtdHome', 'TopTenHome', 'Year'])['homeGoalFT'].describe().sort_values('mean', ascending=False)
df.groupby(['ManUtdHome', 'TopTenHome', 'Year'])['homeRatingsFT'].describe().sort_values('mean', ascending=False)
df.groupby(['ManUtdHome', 'TopTenHome', 'Year'])['homeGoalPerShotsTotalFT'].describe().sort_values('mean', ascending=False)
# 2012 was Utd's year with a high mean and small standard deviation for home goals scored at full time. 

# Out-performing the top 10 European teams. 

# Although not having that great of an average home rating score at full time. 

# Utd's performance at home appears to be on the decline from year to year even though it is the wealthiest club in the world.

# It's true what they say, money doesn't buy you happiness. 