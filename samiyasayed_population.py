import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
datadir='/kaggle/input/world-population/WorldPopulation.csv'
pop_df=pd.read_csv(datadir,index_col='Country')
pop_df
pop_df=pop_df.iloc[:, :-1]
pop_df=pop_df.iloc[:, 3:]
pop_df=pop_df[pop_df.columns[::5]]
pop_bangladesh = pop_df.loc['Bangladesh']
pop_japan = pop_df.loc['Japan']
pop_ethopia = pop_df.loc['Ethiopia']
pop_mexico = pop_df.loc['Mexico']
years=pop_japan.index
years
plt.xticks(rotation=45)
plt.title('Country Populations 1960-2018')
# Plot with differently-colored markers.
plt.plot(years, pop_bangladesh,'r', label='Bangladesh')
plt.plot(years, pop_japan,c='orange', label='Japan')
plt.plot(years, pop_ethopia,'g', label='Ethiopia')
plt.plot(years, pop_mexico,'b', label='Mexico')

# Create legend.
plt.legend(bbox_to_anchor=(1.04,1),loc='upper left')
plt.xlabel('Year')
plt.ylabel('Population (millions)')