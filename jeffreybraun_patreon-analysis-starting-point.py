import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly_express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import folium
from folium import plugins
from plotly.offline import init_notebook_mode, iplot
import os
import base64
init_notebook_mode()

df = pd.read_csv('../input/patreon-top-creators/Patreon1-1000.csv', thousands=',')
print("Number of Patrons Statistics: ")
print(df.Patrons.describe(percentiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]))

fig1 = px.histogram(df, x = 'Patrons', title = 'Distribution of Number of Patrons')
fig1.show()

fig2 = px.histogram(df[10:], x = 'Patrons', title = 'Distribution of Number of Patrons (Excluding top 10 Patrons)')
fig2.show()

print("Number of Days Running Statistics: ")
print(df.DaysRunning.describe(percentiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]))


fig3 = px.histogram(df, x = 'DaysRunning', title = 'Distribution of Days Running (Days Creator Has Been on Patreon)')
fig3.show()
fig4 = px.scatter(df, x = 'DaysRunning', y = 'Patrons')
fig4.show()

print("Correlation between DaysRunning and Patrons")
print(df[['DaysRunning', 'Patrons']].corr())
df['Launched'] = pd.to_datetime(df['Launched'],format='%b-%y')
fig5 = px.scatter(df, x = 'Launched', y = 'Patrons')
fig5.show()

df_bylaunch = df.groupby('Launched').mean().reset_index()
df_bylaunch.rename(columns = {'Patrons':'Mean Patrons'}, inplace=True)
fig6 = px.scatter(df_bylaunch, x = 'Launched', y = 'Mean Patrons')
fig6.show()