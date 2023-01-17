import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
from bubbly.bubbly import bubbleplot 
from __future__ import division
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
IS_LOCAL = False
import os
if(IS_LOCAL):
    PATH="../input/earthquake-database"
else:
    PATH="../input"
print(os.listdir(PATH))
events_df = pd.read_csv(PATH+"/database.csv")
print("Earthaquakes data -  rows:",events_df.shape[0]," columns:", events_df.shape[1])
events_df.head(5)
events_df['Type'].unique()
earthquakes = events_df[events_df['Type']=='Earthquake'].copy()
earthquakes["Year"] = pd.to_datetime(earthquakes['Date']).dt.year
print("Years from:", min(earthquakes['Year']), " to:", max(earthquakes['Year']))
print("Magnitude from:", min(earthquakes['Magnitude']), " to:", max(earthquakes['Magnitude']))
earthquakes["RichterM"] = np.power(earthquakes["Magnitude"],10)
figure = bubbleplot(dataset=earthquakes, x_column='Longitude', y_column='Latitude', color_column = 'Magnitude',
    bubble_column = 'Magnitude', time_column='Year', size_column = 'RichterM',
    x_title='Longitude', y_title='Latitude', 
    title='Earthquakes position (long, lat) and magnitude - from 1965 to 2016', 
    colorscale='Rainbow', colorbar_title='Magnitude', 
    x_range=[-181,181], y_range=[-90,90], scale_bubble=0.5, height=650)
iplot(figure, config={'scrollzoom': True})