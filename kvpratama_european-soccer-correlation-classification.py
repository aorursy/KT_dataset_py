# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 # for interacting with a local relational database
import matplotlib.pyplot as plt # data visualization
from sklearn.cluster import KMeans # KMeans clustering from sklearn
from sklearn.preprocessing import scale
#from customplot import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cnx = sqlite3.connect('../input/database.sqlite')
print(cnx)
# df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.columns
# Simple Statistice of the Dataset
df.describe().transpose()
# Null of missing values
# Show if any of the 183978 rows have null value in one of the 42 columns
df.isnull().any().any(), df.shape
# df.isnull().any(), df.shape # show if each column has null value
# How many data points in each column are null?
df.isnull().sum(axis=0)
# initial number of rows
initial_rows_no = df.shape[0]

# Drop the null rows
df = df.dropna()
# Check if all nulls are gone
df.isnull().any().any(), df.shape
# How many rows were deleted?
initial_rows_no - df.shape[0]
# Shuffle the rows of df
# df = df.reindex(np.random.permutation(df.index))
#df = df.reindex(np.random.permutation(df.id))


# Look at top few rows
df.head(5)
# Look at top few rows
df.shape
df[:10][['penalties', 'overall_rating']]
# Pearson's Correlation Coefficient to see the correlation of features
df['overall_rating'].corr(df['penalties'])
#features = list(df.columns.values)
features = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']

for f in features:
    correlation = df['overall_rating'].corr(df[f])
    print("%s: %f" % (f, correlation))
    
cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']
# Create a list contain Pearson's Correlation between overall_rating with each columns
correlations = [ df['overall_rating'].corr(df[f]) for f in cols]
len(cols), len(correlations)
# Create a dataframe using cols and correlations
df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations})
df2

# Function for plotting a dataframe
def plot_dataframe(df, y_label):
    color = 'coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)
    
    ax = df2.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df2.index)
    ax.set_xticklabels(df2.attributes, rotation=75)
    plt.show()
plot_dataframe(df2, 'Player\'s Overall Rating')
# Select features on which to group players
select5features = ['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']

# Generate a new dataframe with selected features
df_select = df[select5features].copy(deep=True)
df_select.head()
# Perform scaling on the dataframe
data = scale(df_select)

# Define no of clusters
noOfClusters = 4
model = KMeans(init='k-means++', n_clusters=noOfClusters, n_init=20).fit(data)
print(90*'_')
print("\nCount of players in each cluster")
print(90*'_')

pd.value_counts(model.labels_, sort=False)
def pd_centers(featuresUsed, centers):
    from itertools import cycle, islice
    from pandas.tools.plotting import parallel_coordinates
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    colNames = list(featuresUsed)
    colNames.append('prediction')

    # Zip with a column called 'prediction' (index)
    Z = [np.append(A, index) for index, A in enumerate(centers)]

    # Convert to pandas for plotting
    P = pd.DataFrame(Z, columns=colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P
