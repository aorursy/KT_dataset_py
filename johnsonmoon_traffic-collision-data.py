# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
collision = pd.read_csv('../input/traffic-collision-data-from-2010-to-present/traffic-collision-data-from-2010-to-present.csv')

# Renaming the columns to get rid of spaces
cols = collision.columns
cols = cols.map(lambda x: x.replace(' ', '_'))
collision.columns = cols

collision.head()
collision.shape[0]
collision.columns
# Hint: Use conditioning to filter data from Hollywood
collision[collision.Area_Name == 'Hollywood']
# Hint: Use the conditioning to sort out the victim profile and the time occurred. 
# Hint: Then set 'Area_Name' as the index, and access the first element (the report number) in the row indexed as 'Hollywood'
collision[(collision.Victim_Age == 29.0) & 
          (collision.Victim_Sex == 'F') & 
          (collision.Time_Occurred == 1450)]\
    .set_index('Area_Name')\
    .loc['Hollywood']\
    .iloc[0]
collision[['Time_Occurred','Victim_Age']].mean()
# Hint: First sort the values by 'Time_Occurred' in descending order and count the frequencies of each time. 
# Hint: Select only the top 20 time and plot as a bar graph.

collision.sort_values('Time_Occurred', ascending = False).Time_Occurred.value_counts().iloc[:20].plot.bar()
# Hint: First group the data set by 'Area_Name' and count the frequencies using .size() function.
# Hint: Sort values in ascending order and plot as a horizontal bar graph.

collision.groupby('Area_Name').size().sort_values(ascending = True).plot.barh()
# Hint: Parse the first 4 letters of elements in the 'Date_Occurred' column using .str[] function.

collision['Year'] = collision['Date_Occurred'].str[:4]

collision.groupby('Year').size().plot.bar()
# Adjusting the dimensions of the graph for a better visual view
import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))


collision['Year_Month'] = collision['Date_Occurred'].str[:7]
collision.groupby('Year_Month').size().plot()

