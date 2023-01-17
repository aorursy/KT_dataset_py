import os
print(os.listdir("../input"))
import pandas as pd
pd.options.display.max_columns = 500
pd.options.display.max_rows = 100
import numpy as np
import seaborn as sns
sns.set(style='ticks', palette='RdBu')
from datetime import datetime
import csv
import matplotlib.pyplot as plt

df = pd.read_csv('../input/DP_LIVE_07012019070411833.csv')
# print (df.columns,'\n',df.head(1)) 
# Index(['LOCATION', 'INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'TIME', 'Value', 'Flag Codes']  
# 0      AUS  GDPLTFORECAST     TOT  MLN_USD         A  1990  479404.856305         NaN  
## from the CSV:
# "WLD","GDPLTFORECAST","TOT","MLN_USD","A","2060",267676800,
# "OECD","GDPLTFORECAST","TOT","MLN_USD","A","1995",31351213.1144145,

# from: https://www.digitalocean.com/community/tutorials/data-analysis-and-visualization-with-pandas-and-jupyter-notebook-in-python-3
df = df.filter(items=['LOCATION', 'TIME', 'Value'])
df_index = df.set_index(['LOCATION', 'TIME']).sort_index()

def location_plot(LOCATION):
    data = df_index.loc[LOCATION]
    plt.plot(data.index, data.values)
    
plt.figure(figsize = (18, 8))
locations = ['WLD', 'OECD', 'AUS', 'USA']
for location in locations:    
    location_plot(location)
plt.legend(locations)