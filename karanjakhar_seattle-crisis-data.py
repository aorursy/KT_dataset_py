
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


crisis_data = pd.read_csv('../input/crisis-data.csv')
crisis_data.head(10).T
crisis_data['Reported Date'] = pd.to_datetime(crisis_data['Reported Date'],format = '%Y-%m-%d')
crisis_data[(crisis_data['Reported Date'].dt.year == pd.datetime.now().year) & (crisis_data['Reported Date'].dt.month == pd.datetime.now().month)]['Reported Date'].value_counts().plot.line()
plt.show()
crisis_data[(crisis_data['Reported Date'].dt.year == pd.datetime.now().year) & (crisis_data['Reported Date'].dt.month == pd.datetime.now().month)]['Reported Date'].value_counts().plot.bar()
plt.show()
crisis_data['Officer Gender'].value_counts().plot.pie()
plt.show()
crisis_data['Disposition'].value_counts().plot.bar()
plt.show()
crisis_data['Sector'].value_counts().plot.bar()
plt.show()
crisis_data['Precinct'].value_counts().plot.bar()
plt.show()
north_calls = crisis_data[(crisis_data['Precinct'] == 'NORTH') & (crisis_data['Reported Date'].dt.year == pd.datetime.now().year)&(crisis_data['Reported Date'].dt.month == pd.datetime.now().month)]['Reported Date']
south_calls = crisis_data[(crisis_data['Precinct'] == 'SOUTH') & (crisis_data['Reported Date'].dt.year == pd.datetime.now().year)&(crisis_data['Reported Date'].dt.month == pd.datetime.now().month)]['Reported Date']
west_calls = crisis_data[(crisis_data['Precinct'] == 'WEST') & (crisis_data['Reported Date'].dt.year == pd.datetime.now().year)&(crisis_data['Reported Date'].dt.month == pd.datetime.now().month)]['Reported Date']
east_calls = crisis_data[(crisis_data['Precinct'] == 'EAST') & (crisis_data['Reported Date'].dt.year == pd.datetime.now().year)&(crisis_data['Reported Date'].dt.month == pd.datetime.now().month)]['Reported Date']
north_calls.value_counts().plot.line(label = 'North').legend()
south_calls.value_counts().plot.line(label = 'South').legend()
west_calls.value_counts().plot.line(label = 'West').legend()
east_calls.value_counts().plot.line(label = 'East').legend()
plt.show()
