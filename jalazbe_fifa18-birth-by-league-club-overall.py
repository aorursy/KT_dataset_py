# load libraries and set styles options
import warnings; warnings.simplefilter('ignore')

# libraries for data manipulation
import numpy as np 
import pandas as pd 

# Libraries for data visualization
import seaborn as sns #Data visualization
import matplotlib.pyplot as plt
import matplotlib; matplotlib.style.use('ggplot')

# Configuration for 'pandas' library
from pandas import set_option
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

from ipywidgets import interact

# Seaborn option settings
sns.set_context("notebook")
sns.set_palette('husl')
# read csv dataset and load into pandas dataframe object "df"
df = pd.read_csv('../input/complete.csv', encoding='utf_8')
# check first few lines
df.head()
# verify data frame shape
df.shape
# Visualice the name of each columns and 
# the corresponding number for selection
for e, col in enumerate(df.columns):
    print(e,"->",col)
# Dicard unnecesary rows
# There are 185 however the load of data
# will only be done with the selected columns

selectedColumns = [0,2,3,6,7,8,9,10,14,19]

# load data into a DataFrame object named 'df'
df = pd.read_csv('../input/complete.csv', encoding='utf_8', usecols=selectedColumns)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(df.head(5))

# verify dataframe shape
df.shape
# set dataframe row index to ID
df = df.set_index('ID')
# check first few lines
df.head()
# review dataframe info
df.info()
# This will display information about each column
# Look for duplicates
df.duplicated().sum()
# look for Nan values (if any)
pd.isnull(df).sum()
# Display % clubs with NaN values
df.club.isnull().value_counts() / len(df)
# Display % leagues with NaN values
df.club.isnull().value_counts() / len(df)
# club and league NaN values are asigned "Unknown"
df.club.fillna("Unknown", inplace=True)
df.league.fillna("Unknown", inplace=True)
# look for Nan values (if any)
pd.isnull(df).sum()
# Extraction of the month information from the 'birth_date' information
df['month'] = pd.DatetimeIndex(df['birth_date']).month
df['year'] = pd.DatetimeIndex(df['birth_date']).year

# Set the type of column 'month' and 'year' as int
df.month = df.month.astype(int)
df.year = df.year.astype(int)
#Lets create a column with the overall category of the player
df.overall.describe()
# Prepare a new column "overallCat" to save data about
# overall categories will be:
# 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

# First we create and empty column:
df['overallCat'] = np.nan

# Then the column value is changed depending on the given conditions
df['overallCat'].loc[(df['overall'] > 40) & (df['overall'] <=50 )] = 40
df['overallCat'].loc[(df['overall'] > 50) & (df['overall'] <=60 )] = 50
df['overallCat'].loc[(df['overall'] > 60) & (df['overall'] <=70 )] = 60
df['overallCat'].loc[(df['overall'] > 70) & (df['overall'] <=80 )] = 70
df['overallCat'].loc[(df['overall'] > 80) & (df['overall'] <=90 )] = 80
df['overallCat'].loc[(df['overall'] > 90) & (df['overall'] <=100 )] = 90

# display the total values for each category
df.overallCat.value_counts()
df.shape
df.info()
# lets select players form the league and count for each month how many there are. 

numPlayers = df[df['league']== "Campeonato Brasileiro Série A"]['month'].value_counts()
print("Number of players on Campeonato Brasileiro Série A by month of birth:\n",numPlayers)
print("Before deleting there are: \t",(len(df.league.value_counts())), " leagues")

# Deletion of these players is done by:
df = df[df['league'] != 'Campeonato Brasileiro Série A']

print("After deleting there are: \t",(len(df.league.value_counts())), " leagues")
# Create the image
g = sns.factorplot(x='month', 
                   data=df, 
                   kind="count",
                   size=5, aspect=3, 
                   edgecolor='black');

# Configure image visualization
# This gives space to plot the figure title
g.fig.subplots_adjust(top=.9)

# Set the x-axis labels
monthList = ['Jan','Feb','Mar','Apr',
             'May','Jun','Jul','Ago',
             'Sep','Oct','Nov','Dic']
g.set_xticklabels(monthList)

# Set figure title, x-label and y-label
g.fig.suptitle("Quantity of soccer player's by month")
g.set_ylabels("Number of ocurrences")
g.set_xlabels("Month of birth")

# Get current axis on current figure
ax = plt.gca()

# Modify each column value
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., 
            p.get_height(), 
            '%d' % int(p.get_height()), 
            fontsize=10, 
            color='black', ha='center', va='bottom');
sns.kdeplot(df['month'], 
            label='Month of birth', 
            shade=True)
plt.title("Density of month of birth");
df_overall = df.groupby(['month'])['overall'].mean()

axis = df_overall.plot()
axis.set_ylabel('Overall Mean Points')
axis.set_title("Player's overall mean by month of birth");
g = sns.FacetGrid(df, col="year",col_wrap=3) 
g.map(sns.kdeplot, "month");
label2000 = str(len(df[df.year==2000]))
sns.kdeplot(df[df.year==2000]['month'], 
            label='2000, '+str(len(df[df.year==2000]))+' players', 
            shade=True)
sns.kdeplot(df[df.year==1999]['month'], 
            label='1999, '+str(len(df[df.year==2000]))+' players', 
            shade=True)
plt.xlabel('Month of birth');
# Create a list of the leagues to explore:
Top5Leagues = ["Spanish Primera División",
                "English Premier League",
               "Italian Serie A",
                "French Ligue 1", 
               "Colombian Primera A"
               ]
# Create a version of the dataframe to plot
dfPlot = df.copy()

dfPlot = dfPlot[df['league'].isin(Top5Leagues)]

g = sns.factorplot(x='month', 
                   data=dfPlot, 
                   kind="count", 
                   hue='league',
                   size=8, aspect=2, 
                   edgecolor='black');
g.fig.subplots_adjust(top=.9)
monthList = ['Jan','Feb','Mar','Apr',
             'May','Jun','Jul','Ago',
             'Sep','Oct','Nov','Dic']
g.set_xticklabels(monthList)

g.fig.suptitle('Quantity of soccer players by month for the selected leagues')
g.set_ylabels("Number of ocurrences")
g.set_xlabels("Month of birth");

# Lets see the same information on different charts. 
g = sns.factorplot(x="month", kind="count", 
               col="league", col_wrap=3, 
               data=dfPlot, legend=True, 
               palette='deep');

g.fig.subplots_adjust(top=.9)
g.fig.suptitle('Quantity of soccer players by month for the selected leagues');

g = sns.FacetGrid(dfPlot, col="league",col_wrap=3) 
g.map(sns.kdeplot, "month");
sns.kdeplot(df[df.league=='Italian Serie A']['month'], 
            label='Italian Serie A', 
            shade=True)
sns.kdeplot(df[df.league=='English Premier League']['month'], 
            label='English Premier League', 
            shade=True)
plt.xlabel('Month of birth')
plt.title("Density of month of birth comparison for leagues: Italian Serie A and English Premier");
TopClubs = ["Real Madrid CF",
           "Grêmio",
            "Manchester United",
            "FC Barcelona",
            "Paris Saint-Germain",
            "Flamengo",
            "Juventus",
            "FC Bayern Munich",
            "Manchester City",
            "FC Red Bull Salzburg"
            ]

# Create a version of the dataframe to plot
dfPlot = df.copy()

dfPlot = dfPlot[df['club'].isin(TopClubs)]

g = sns.factorplot(x='month', data=dfPlot, 
                   kind="count", hue='club',
                   size=5, aspect=2, edgecolor='black', 
                   palette='bright');

monthList = ['Jan','Feb','Mar','Apr',
             'May','Jun','Jul','Ago',
             'Sep','Oct','Nov','Dic']
g.set_xticklabels(monthList)

g.fig.subplots_adjust(top=.9)
g.fig.suptitle('Quantity of soccer players by month for the selected Clubs')
g.set_ylabels("Number of ocurrences")
g.set_xlabels("Month of birth");
# Lets see the same information on different charts. 
sns.factorplot(x="month", kind="count", 
               col="club", col_wrap=3, 
               data=dfPlot, legend=True, 
               palette='bright');
for club in TopClubs:
    ax = sns.kdeplot(dfPlot[dfPlot['club']==club]['month'],  label=club, legend=True)
g.fig.subplots_adjust(top=.9)
ax.set_title("Density of month of birth for Top clubs",fontsize=10);
g = sns.FacetGrid(dfPlot, col="club",col_wrap=3) 
g.map(sns.kdeplot, "month");
g = sns.factorplot(x="month", y="club", 
                   data=dfPlot, kind='box', 
                   size=4, aspect=2, 
                   palette='bright')

#monthList = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
#g.set_xticklabels(monthList)
g.fig.subplots_adjust(top=.9)
g.fig.suptitle('Month of birth distribution for Top Clubs')
g.set_ylabels("Club Name")
g.set_xlabels("Month of birth");

g = sns.factorplot(x='month', data=dfPlot, 
                   kind="count", hue='overallCat',
                   size=5, aspect=2, edgecolor='black', 
                   palette='bright');

monthList = ['Jan','Feb','Mar','Apr',
             'May','Jun','Jul','Ago',
             'Sep','Oct','Nov','Dic']
g.set_xticklabels(monthList)

g.fig.subplots_adjust(top=.9)
g.fig.suptitle('Quantity of soccer players by month and Overall Category')
g.set_ylabels("Number of ocurrences")
g.set_xlabels("Month of birth");
sns.set_palette('husl')
overallCategories = df.groupby("overallCat").full_name.count().sort_values(ascending=True)
overallCategories = list(overallCategories.index)
for category in overallCategories:
    ax = sns.kdeplot(dfPlot[dfPlot['overallCat']==category]['month'],  
                     label=category, 
                     legend=True)
g.fig.subplots_adjust(top=.9)
ax.set_title("Density of month of birth by overall category",fontsize=10);

dfPlot = df.copy()
g = sns.FacetGrid(dfPlot, col="overallCat",col_wrap=3) 
g.map(sns.kdeplot, "month");
sns.kdeplot(df[df.overallCat==40.0]['month'], 
            label='overall 40', 
            shade=True)
sns.kdeplot(df[df.overallCat==90.0]['month'], 
            label='overall 90', 
            shade=True)
plt.xlabel('Month of birth');
plt.title("Density of month of birth comparison for overall category 40 and overall category 90");
# Selection of top 10 nationalities by number of players
Top10Nationality = df.groupby("nationality").full_name.count().sort_values(ascending=True).tail(10)
Top10Nationality = list(Top10Nationality.index)
Top10Nationality
dfPlot = df[df['nationality'].isin(Top10Nationality)]

g = sns.factorplot(x='month', data=dfPlot, 
                   kind="count", hue='nationality',
                   size=5, aspect=2, edgecolor='black', 
                   palette='bright');

g.fig.subplots_adjust(top=.9)
g.fig.suptitle('Month of birth by Top 10 nationality')
g.set_ylabels("Number of occurrences")
g.set_xlabels("Month of birth");
g.set_xticklabels(monthList);
sns.set_palette('husl')
for nationality in Top10Nationality:
    ax = sns.kdeplot(dfPlot[dfPlot['nationality']==nationality]['month'],  
                     label=nationality, legend=True)

g.fig.subplots_adjust(top=.9)
ax.set_title("Month of birth by nationality of the player",fontsize=10);
g = sns.FacetGrid(dfPlot, col="nationality", col_wrap=3) 
g.map(sns.kdeplot, "month");
sns.kdeplot(df[df.nationality=='Colombia']['month'], 
            label='Colombia', 
            shade=True)
sns.kdeplot(df[df.nationality=='Argentina']['month'], 
            label='Argentina', 
            shade=True)
sns.kdeplot(df[df.nationality=='England']['month'], 
            label='England', 
            shade=True)
sns.kdeplot(df[df.nationality=='Spain']['month'], 
            label='Spain', 
            shade=True)
plt.xlabel('Month of birth');
plt.title("Density of month of birth comparison for 4 nationalities \n"+
        "(Colombia, Argentina, England, Spain)");
