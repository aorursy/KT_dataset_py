# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/fivethirtyeight-police-killings-dataset/police_killings.csv", encoding = "ISO-8859-1")
data['Shooting_Deaths'] = 1

#data.head()
#Interactive Tree MAP of 2015 Shooting Deaths: Click to visualize by city and police department:



data["title"] = "US" # in order to have a single root node

fig = px.treemap(data, path=['title','state', 'city', 'lawenforcementagency'], values='Shooting_Deaths',

                  color='pop', hover_data=['state'], color_continuous_scale='RdBu',

                  color_continuous_midpoint=np.average(data['pop'], weights=data['Shooting_Deaths']), title="Deaths by Police 2015")

fig.show(rendering = "kaggle")
#####TOP POLICE DEPARTMENTS WITH SHOOTING Death Records Greater Than 2 in 2015:####

police_dept_stat=data.groupby(['lawenforcementagency'])['Shooting_Deaths'].sum()

police_dept_stat=police_dept_stat.to_frame()

#police_dept_stat.columns.values[1] = 'Shooting_Deaths'

police_dept_stat=police_dept_stat[police_dept_stat['Shooting_Deaths']>2]



print(police_dept_stat)
import matplotlib.pyplot as plt

pd_stat=police_dept_stat[police_dept_stat['Shooting_Deaths']>2]



pd_stat.plot(kind="bar")

plt.ylabel('Deaths')

plt.title('2015 Police Departments With Most Deaths by Police')

plt.show()
df = data[data['raceethnicity'] == "Black"]
#######Interactive Tree MAP of 2015 Black Shooting Deaths: Click to visualize by city and police department:

df["title"] = "US" # in order to have a single root node

fig = px.treemap(df, path=['title','state', 'city', 'lawenforcementagency'], values='Shooting_Deaths',

                  color='pop', hover_data=['state'], color_continuous_scale='RdBu',

                  color_continuous_midpoint=np.average(data['pop'], weights=data['Shooting_Deaths']), title="African American Deaths by Police 2015")

fig.show(rendering = "kaggle")
#####TOP POLICE DEPARTMENTS WITH Black SHOOTING Death Records Greater Than 1 Compared to Population in 2015:####

police_dept_stat_black_only=df.groupby(['lawenforcementagency'])['Shooting_Deaths'].sum()

police_dept_stat_black_only=police_dept_stat_black_only.to_frame()

police_dept_stat_black_only=police_dept_stat_black_only[police_dept_stat_black_only['Shooting_Deaths']>1]

print(police_dept_stat_black_only)


police_dept_stat_black_only.plot(kind="bar")

plt.ylabel('Deaths')

plt.title('2015 Police Departments With Most African American Deaths by Police')

plt.show()