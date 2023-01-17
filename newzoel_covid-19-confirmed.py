# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
confirmed_path = "/kaggle/input/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
confirmed_case = pd.read_csv(confirmed_path, index_col="Country/Region", parse_dates=True)
df = pd.DataFrame(confirmed_case)
df
location = "Korea, South", "Italy", "Iran", "Indonesia", "Germany", "Spain"

def getDataFrame(location):
    data = df.loc[location,:]
    data = data.drop(columns=['Lat','Long', 'Province/State'])
    return data

def sortby(data, point, sortby):
    sort_data = data.loc[point, sortby:]
    return sort_data

compare_df = getDataFrame(location)
compare_df
startdate = "1/22/20"
loc = ['Indonesia', 'Italy', 'Iran', 'Korea, South', 'Germany', 'Spain']

state_list = []

for state in loc:
    data = sortby(compare_df, state, startdate)
    state_list.append(data)

compare = pd.concat(state_list, axis=1, join='inner')
compare.index.name = 'Date'
compare
state_case = []

#1st day case of each state
ind_tl = pd.DataFrame(state_list[0].iloc[40:])
state_case.append(ind_tl)
italy_tl = pd.DataFrame(state_list[1].iloc[9:])
state_case.append(italy_tl)
iran_tl = pd.DataFrame(state_list[2].iloc[28:])
state_case.append(iran_tl)
s_korea_tl = pd.DataFrame(state_list[3].iloc[0:])
state_case.append(s_korea_tl)
germany_tl = pd.DataFrame(state_list[4].iloc[5:])
state_case.append(germany_tl)
spain_tl = pd.DataFrame(state_list[5].iloc[10:])
state_case.append(spain_tl)
def change_index(data, index_name):
    data.index = np.arange(1, len(data)+1)
    data.index.name = index_name
    
index = 'Day'
    
for state in state_case:
    change_index(state, index)
confirmed_day = pd.concat(state_case, axis=1, join='inner')
confirmed_day
fig,ax = plt.subplots(figsize=(20,20))

plt.title('COVID 19 Comparation')
plt.plot(confirmed_day)
plt.gca().legend(('Korea','Germany','Italy','Iran','Spain','Indonesia'))
plt.xticks(rotation='45')
plt.xlabel("DAY")
plt.ylabel("CASES")
plt.grid()

fig.show()