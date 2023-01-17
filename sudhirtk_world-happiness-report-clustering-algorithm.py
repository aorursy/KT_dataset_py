# Call Libraries



import numpy as np                   # Data manipulation

import pandas as pd                  # DataFrame manipulation

import time                          # To time processes 

import warnings                      # To suppress warnings

import matplotlib.pyplot as plt      # For Graphics

import seaborn as sns

from sklearn import cluster, mixture              # For clustering

from sklearn.preprocessing import StandardScaler  # For scaling dataset

from sklearn.neighbors import kneighbors_graph

from itertools import cycle, islice



import os                     # For os related operations

import sys                    # For data size



import plotly.plotly as py 

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



# Read the happiness report file

h2017 = pd.read_csv("../input/2017.csv", header=0)

h2016 = pd.read_csv("../input/2016.csv", header=0)

h2015 = pd.read_csv("../input/2015.csv", header=0)

h2017.head()

#Data types in happiness report file

h2017.dtypes

# Heatmap citing correlation

h2017a = h2017[['Happiness.Score','Economy..GDP.per.Capita.','Family','Health..Life.Expectancy.', 'Freedom', 

          'Generosity','Trust..Government.Corruption.','Dystopia.Residual']]

cor = h2017a.corr()

sns.heatmap(cor, square = True)

plt.show()



# Let us try to plot happiness score

sns.kdeplot(h2015['Happiness Score'],color="r")

sns.kdeplot(h2016['Happiness Score'],color="b")

sns.kdeplot(h2017['Happiness.Score'],color="g")

plt.xlabel('Rank of Country', fontsize=16)

plt.ylabel('Happiness Score', fontsize=16)

plt.title('Happiness Score of 2015 (Red) 2016 (Blue) 2017 (Green)', fontsize=18)

plt.show()



#Global happiness of 2017

data = dict(type = 'choropleth', 

           locations = h2017['Country'],

           locationmode = 'country names',

           z = h2017['Happiness.Rank'], 

           text = h2017['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Global Happiness in 2017', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)



# Visualize how happiness compares to Per Capita income in 2017

plt.scatter(h2017['Happiness.Score'],h2017['Economy..GDP.per.Capita.'], marker = 'o')

plt.xlabel('Happiness Score')

plt.ylabel('Government Corrupption')

plt.title("Scatter Plot of Happiness Score Vs Per Capita.GDP ")

plt.legend()

plt.show()



# Visualize how happiness compares to Trust..Govt Corruption in 2017

plt.scatter(h2017['Happiness.Score'],h2017['Trust..Government.Corruption.'], marker = 'o')

plt.xlabel('Happiness Score')

plt.ylabel('Trust..Government Corrupption')

plt.title("Scatter Plot of Happiness Score Vs Corruption ")

plt.legend()

plt.show()



# Its very evident from above, as the percapita increases the happiness definitely goes up:) 

# which kind of makes sense, well being increases happiness..Cant say the same with the corrpution

# figures, there are cases where corruption is high with med-high happiness score.





# Try plotting pandas scatter martrix

pd.scatter_matrix(h2017, alpha = 0.8, figsize = (20,20), diagonal = 'kde');





# Trying another happniess score plot

plt.plot(h2015['Happiness Score'], 'b', label='2015')

plt.plot(h2016['Happiness Score'], 'g', label='2016')

plt.plot(h2017['Happiness.Score'], 'r', label='2017')

plt.title('Happiness Score of 2015 (Blue) & 2016 (Green) & 2017 (Red)', fontsize=18)

plt.xlabel('Rank of Country', fontsize=16)

plt.ylabel('Happiness Score', fontsize=16)



# Perform Clustering

# Clustering methods are unsupervised methods, there is no outcome that we're trying to 

# predict here, but the effort is to see patterns in data that hasnt been observed before

# Perform a K means clustering with 3 centroids..



country=h2017[h2017.columns[0]]

data= h2017.iloc[:,2:]



def normalizedData(x):

    normalised = StandardScaler()

    normalised.fit_transform(x)

    return(x)

    

data = normalizedData(data)    



n_clusters=3

def Kmeans(x, y):

    km= cluster.KMeans(x)

    km_result=km.fit_predict(y)

    return(km_result)

   

km_result = Kmeans(3,data)

data['Kmeans'] = pd.DataFrame(km_result)

plt.scatter(data.iloc[:, 0], data.iloc[:, 9],  c=km_result)

plt.show()      





dataset=pd.concat([data,country],axis=1)



dataPlot = dict(type = 'choropleth', 

           locations = dataset['Country'],

           locationmode = 'country names',

           z = dataset['Kmeans'], 

           text = dataset['Country'],

           colorbar = {'title':'Cluster Group'})

layout = dict(title = 'Kmeans Clustering', 

           geo = dict(showframe = False, 

           projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [dataPlot], layout=layout)

iplot(choromap3)     