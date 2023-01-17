#edX week1

#Import all important libraries

import sqlite3

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import scale

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Some Custom function for visualization

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



def parallel_plot(data):

    from itertools import cycle, islice

    from pandas.tools.plotting import parallel_coordinates

    import matplotlib.pyplot as plt



    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))

    plt.figure(figsize=(15,8)).gca().axes.set_ylim([-2.5,+2.5])

    parallel_coordinates(data, 'prediction', color = my_colors, marker='o')
# Create your connection.

cnx = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.columns
df.describe().transpose()
#Total number of rows and columns

df.shape
df.isnull().any().any(), df.shape
df.isnull().sum(axis=0)
df = df.dropna()
df.shape
df.head()
## Randomizing the dataset

df = df.reindex(np.random.permutation(df.index))

df.head()
df[:10][['penalties', 'overall_rating']]
df['overall_rating'].corr(df['penalties'])
potentialFeatures = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']
for f in potentialFeatures:

    related = df['overall_rating'].corr(df[f])

    print("%s: %f" %(f, related))
cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',

       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',

       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',

       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',

       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',

       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',

       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',

       'gk_reflexes']
# create a list containing Pearson's correlation between 'overall_rating' with each column in cols

correlations = [ df['overall_rating'].corr(df[f]) for f in cols ]
len(cols), len(correlations)
# create a function for plotting a dataframe with string columns and numeric values



def plot_dataframe(df, y_label):  

    color='coral'

    fig = plt.gcf()

    fig.set_size_inches(20, 12)

    plt.ylabel(y_label)



    ax = df2.correlation.plot(linewidth=3.3, color=color)

    ax.set_xticks(df2.index)

    ax.set_xticklabels(df2.attributes, rotation=75); #Notice the ; (remove it and see what happens !)

    plt.show()
# create a dataframe using cols and correlations



df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations}) 
# let's plot above dataframe using the function we created

    

plot_dataframe(df2, 'Player\'s Overall Rating')
import pylab as pl

x = range(34)

xTicks = cols

y = correlations

pl.xticks(x, xTicks)

pl.xticks(range(34), xTicks, rotation=45) #writes strings with 45 degree angle

pl.plot(x,y,'*')

pl.show()
df2.index = df2['attributes']
# Define the features you want to use for grouping players



select5features = ['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']

select5features
# Generate a new dataframe by selecting the features you just defined



df_select = df[select5features].copy(deep=True)
df_select.head()
# Perform scaling on the dataframe containing the features



data = scale(df_select)



# Define number of clusters

noOfClusters = 4



# Train a model

model = KMeans(init='k-means++', n_clusters=noOfClusters, n_init=20).fit(data)
print(90*'_')

print("\nCount of players in each cluster")

print(90*'_')



pd.value_counts(model.labels_, sort=False)
# Create a composite dataframe for plotting

# ... Use custom function declared in customplot.py (which we imported at the beginning of this notebook)



P = pd_centers(featuresUsed=select5features, centers=model.cluster_centers_)

P
parallel_plot(P)