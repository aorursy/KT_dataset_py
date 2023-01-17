# importing libraries to be used
import numpy as np                  # Scientific Computing
import pandas as pd                 # Data Analysis
import matplotlib.pyplot as plt     # Plotting
import seaborn as sns               # Statistical Data Visualization
# loading data from a CSV file into a Pandas DataFrame
data_g = pd.read_csv("../input/WorldCupMatches.csv") 
data_p = pd.read_csv("../input/WorldCupPlayers.csv")
data_c = pd.read_csv("../input/WorldCups.csv")
# printing a concise summary of created dataframes
data_g.info()  
# data_p.info()  -> not used but here just in case you may work on it
# data_c.info()  -> not used but here just in case you may work on it
# correlation table for WC games columns
# One important note here; Correlation can be created between integer values, so columns come with string values will not be included.
data_g.corr()
# You may consider dropping some columns that not make sense for your analysis.
data_g_d = data_g.drop(['RoundID', 'MatchID'], axis = 1)
# data_g_d       -> new data frame after drop
# data_g         -> old data frame for WC Games
# Syntax must be precise, Round Brackets, Square Brackets etc.
# axis = 1       -> dropping index use "0", dropping column/s "1"
# New correlation table after drop of ID columns.
data_g_d.corr()
fig, ax = plt.subplots(1, 1, figsize = (12, 12))
# fig                 -> figure
# ax                  -> array of axes objects
# "first" 1           -> it's the i in an array of [i]x[j] (to see effect, please change it)
# "second" 1          -> it's the j in an array of [i]x[j] (to see effect, please change it)
# figsize = (4, 4)    -> footprint of table (to see effect, please change it)

sns.heatmap(data_g_d.corr(), vmin = -1.50, vmax = 1.50, cmap = "YlGnBu", annot = True, linewidths = 1.5, linecolor = "red", fmt = ".2f")
# data_g_d.corr()     -> correlation excel data table above to be used into annotated type table
# vmin                -> min value of color bar, not really possible but just to show it can be changed.
# vmax                -> max value of color bar, Correlation can't be higher than 1 but just to show it's changable.
# cmap = "YlGnBu"     -> selection of heatmap color 
# annot = True        -> correlation rates on above table will be visible on new table.
# linewidths = 1.5    -> table border thickness
# linecolor = "red"   -> table border color
# fmt = ".2f"         -> Number of digits after dot for each correlated data

plt.xticks(rotation = 45) # making labels readable otherwise "overlapping text" problem occurs (You may consider changing figsize at first place, then this action won't be necessary)
plt.yticks(rotation = 45) # making labels readable otherwise "overlapping text" problem occurs (You may consider changing figsize at first place, then this action won't be necessary)

# Bring first ten rows, it brings first five rows, if you leave pharantesis empty.
data_g_d.head(10)
# correlation ascending order, also duplicated correlation value will be dropped with drop_duplicates()
data_g_d.corr().unstack().sort_values().drop_duplicates()
# just columns
data_g_d.columns
# Generates descriptive statistics excluding NaN values, Analyzes both numeric and object series
data_g_d.describe()
# Some of columns are with space, let's make them workable by replacing "spaces" with "underscores" 
data_g_d.columns = [c.replace(' ', '_') for c in data_g_d.columns]
# Now it will be ready to work
data_g_d.columns
## LINE PLOT - Please click twice on graphics to increase visibility for regions

data_g_d.Home_Team_Goals.plot(kind = 'line', color = 'b', label = 'Home', linewidth = 2.5, alpha = 0.5, grid = True, linestyle = '-', figsize = (36,10))
data_g_d.Away_Team_Goals.plot(color = 'r',label = 'Away',linewidth=2.5, alpha = 0.5,grid = True,linestyle = '-.')
# you may find complete syntax and legends here, https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html
# kind = 'line'           -> various type possible, we will use line plot, if you choose another type, some syntax parameters must be taken out.
# color                   -> color of graphic line
# label                   -> name of the legend
# linewidth               -> thickness of graphic lines.
# alpha                   -> transparency
# grid                    -> to be able to see and read the graphic better.
# linestyle = '-'         -> graphic line style, you may choose others as | '-' | '--' | '-.' | ':' | 'None' | ' ' | '' |
# figsize = (36,10)       -> You know, size matters a lot. The bigger is better as Freddie said.
plt.legend(loc = 'upper right',fontsize = 20)        # showing place for legend
plt.xlabel('games', fontsize = 20)                   # label = name of label
plt.ylabel('goals', fontsize = 20)
plt.title('Line Plot', fontsize = 20)                # title = title of plot

plt.show()
data_c.info()
# I will use WC datas for scatter plot
data_c.head()
## SCATTER PLOT

plt.scatter(data_c.Year, data_c.MatchesPlayed, color = 'b', alpha = 0.5, label = 'Number of Games')
plt.scatter(data_c.Year, data_c.GoalsScored, color = 'r', alpha = 0.5, label = 'Number of Goals' )
plt.xlabel('Year')
plt.ylabel('Goals & Games')
plt.title('World Cup Matches & Goals')
plt.legend(loc = 'upper left')
plt.show()
# HISTOGRAM
data_g_d.Away_Team_Goals.plot(kind = 'hist', bins = 8, figsize = (10,10))
plt.xlabel('Away Team Goals', fontsize = 20)
plt.ylabel('Frequency', fontsize = 20)
plt.title('Frequency based Number of Goals scored by Away Teams')
plt.show()
# plt.clf() for clean up
# Filtering Pandas data frame
x = data_c['GoalsScored'] > 100
data_c[x]   # We filter table and bring WC where more than 100 goals scored.
# Filtering Pandas data frame with logical_and
data_c[np.logical_and(data_c['GoalsScored'] > 100, data_c['QualifiedTeams'] == 24)]
# WC with 24 participants and more than 100 goals scored

# another method
# data_c[(data_c['GoalsScored'] > 100) & (data_c['QualifiedTeams'] == 24)]